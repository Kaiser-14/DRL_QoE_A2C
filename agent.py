import environment
import actor
import critic

from environment import consume_kafka, assign_profile

import os
import logging
import time
import numpy as np
import multiprocessing as mp
import tensorflow as tf
import requests
import math

from kafka import KafkaConsumer
from json import loads
from scipy.spatial import distance
from random import randint
from datetime import datetime

NUM_AGENTS = 1
# NUM_AGENTS = multiprocessing.cpu_count()  # Enable to fully process the model
random_seed = 42

NUM_STATES = 5  # Number of possible states: quality, loss rate, resolution, encoding quality, ram usage
LEN_STATES = 8  # Number of states to hold
TRAINING_REPORT = 25  # Batch to write information into the logs
TRAINING_LEN = 500  # Limit of epochs to train the model

# Different profiles combining resolutions and bitrate
# PROFILES = {1: {1080: 50}, 2: {1080: 30}, 3: {1080: 20}, 4: {1080: 15}, 5: {1080: 10}, 6: {1080: 5}, 7: {720: 25},
#             8: {720: 15}, 9: {720: 10}, 10: {720: 7.5}, 11: {720: 5}, 12: {720: 2.5}}
PROFILES = {0: {1080: 10}, 1: {1080: 7.5}, 2: {1080: 5}, 3: {720: 4}, 4: {720: 2.5}, 5: {720: 1}}
RESOLUTIONS = {1080: 1, 720: 0}

# DEFAULT_ACTION = 4  # PROFILES[0][1] 1080p 15Mbps
DEFAULT_ACTION = 1  # PROFILES[0][1] 1080p 6Mbps
MAX_BR = max(list(x.values())[0] for x in list(PROFILES.values()))
MAX_CAPACITY = 20000.0  # in kb -> 15Mb
# DEFAULT_RES = list(PROFILES[4].keys())[0]
# DEFAULT_BITRATE = list(PROFILES[4].values())[0]
NUM_ACTION = len(PROFILES)

# Learning rates
ACTOR_LR = 0.0001
CRITIC_LR = 0.001

# Files
NOW = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
SUMMARY_DIR = './results/' + NOW + '/'  # './results/summary/'
LOGS_DIR = './results/' + NOW + '/'  # './results/logs/'
MODEL_DIR = './results/' + NOW + '/'  # './results/model/'

# Kafka parameters
KAFKA_TOPIC_OUT = 'tfm.probe.out'
KAFKA_SERVER = ['192.168.0.55:9092']

# Docker address
VCE_RES_ADDR = 'localhost'
VCE_BR_ADDR = 'localhost'
PROBE_ADDR = '192.168.0.10'
TB_ADDR = 'localhost'

# Docker ports
VCE_RES_PORT = '3003'
VCE_BR_PORT = '3000'
PROBE_PORT = '3005'
TB_PORT = '3002'

# Other parameters
RAND_ACTION = 1000  # Random value to decide exploratory action
CLEAN = 0  # Change to 1 to delete the results folder and start a fresh modelling
A2C_MODEL = None  # Start a fresh model
# A2C_MODEL = tf.train.latest_checkpoint(MODEL_DIR)  # Load latest trained model


def sup_agent(net_params_queues, exp_queues):  # Supervisor agent
    # Include logging information about system behaviour
    logging.basicConfig(filename=LOGS_DIR + 'log_supervisor', filemode='w', level=logging.INFO)

    with tf.Session() as sess:
        actor_net = actor.Actor(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                learning_rate=ACTOR_LR)
        critic_net = critic.Critic(sess, states_dim=[NUM_STATES, LEN_STATES],
                                   learning_rate=CRITIC_LR)

        model_vars, model_ops = environment.model_summary()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # Training monitor
        saver = tf.train.Saver()  # Save neural net parameters

        # Restore previously NN trained (set in global parameters)
        model = A2C_MODEL
        if model is not None:
            # saver = tf.train.import_meta_graph('my-model-1000.meta')
            saver.restore(sess, model)
            print('\nModel restored')

            # https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125
            # # Let's load a previously saved meta graph in the default graph
            # # This function returns a Saver
            # saver = tf.train.import_meta_graph('results/model.ckpt-1000.meta')
            #
            # # We can now access the default graph where all our metadata has been loaded
            # graph = tf.get_default_graph()
            #
            # # Finally we can retrieve tensors, operations, collections, etc.
            # global_step_tensor = graph.get_tensor_by_name('loss/global_step:0')
            # train_op = graph.get_operation_by_name('loss/train_op')
            # hyperparameters = tf.get_collection('hyperparameters')

        epoch = 0
        while True:
            actor_params = actor_net.get_actor_params()
            critic_params = critic_net.get_critic_params()

            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_params, critic_params])

            total_len = 0.0
            reward_sum = 0.0
            tdloss_sum = 0.0
            entropy_sum = 0.0
            agents_sum = 0.0

            # assemble experiences from the agents
            actor_gradient_matrix = []
            critic_gradient_matrix = []

            for i in range(NUM_AGENTS):
                states_matrix, actions_matrix, rewards_matrix, info = exp_queues[i].get()

                actor_gradients, critic_gradients, td_matrix = environment.compute_gradients(
                        states_matrix=np.stack(states_matrix, axis=0),
                        actions_matrix=np.vstack(actions_matrix),
                        rewards_matrix=np.vstack(rewards_matrix),
                        actor_net=actor_net, critic_net=critic_net)

                actor_gradient_matrix.append(actor_gradients)
                critic_gradient_matrix.append(critic_gradients)

                total_len += len(rewards_matrix)
                reward_sum += np.sum(rewards_matrix)
                tdloss_sum += np.sum(td_matrix)
                entropy_sum += np.sum(info['entropy'])
                agents_sum += 1.0

            assert NUM_AGENTS == len(actor_gradient_matrix)
            assert len(actor_gradient_matrix) == len(critic_gradient_matrix)

            for i in range(len(actor_gradient_matrix)):
                actor_net.apply_gradients(actor_gradient_matrix[i])
                critic_net.apply_gradients(critic_gradient_matrix[i])

            epoch += 1
            print('Epoch {}'.format(epoch))
            print('----------------')
            reward_avg = reward_sum / agents_sum
            td_loss_avg = tdloss_sum / total_len
            entropy_avg = entropy_sum / total_len

            logging.info('Epoch: ' + str(epoch) + ' TD_Loss: ' + str(td_loss_avg) +
                         ' Total_Reward: ' + str(reward_avg) + ' Average_Entropy: ' + str(entropy_avg))

            summary = sess.run(model_ops, feed_dict={
                model_vars[0]: td_loss_avg,
                model_vars[1]: reward_avg,
                model_vars[2]: entropy_avg
            })

            writer.add_summary(summary, epoch)
            writer.flush()

            if epoch % 100 == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, MODEL_DIR + "model_epoch_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)

            if epoch == TRAINING_LEN:
                sess.close()
                exit(0)


def agent(agent_id, net_params_queue, exp_queue, consumer):  # General agent

    with tf.Session() as sess, open(LOGS_DIR + 'metrics_agent_' + str(agent_id+1), 'w') as log_file:
        actor_net = actor.Actor(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                learning_rate=ACTOR_LR)
        critic_net = critic.Critic(sess, states_dim=[NUM_STATES, LEN_STATES],
                                   learning_rate=CRITIC_LR)

        # Initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor_net.set_actor_params(actor_net_params)
        critic_net.set_critic_params(critic_net_params)

        # Initialize action
        action = DEFAULT_ACTION
        last_action = DEFAULT_ACTION

        # Vectors for storing values: states, actions, rewards.
        actions = np.zeros(NUM_ACTION)
        actions[action] = 1
        actions_matrix = [actions]

        states_matrix = [np.zeros((NUM_STATES, LEN_STATES))]
        rewards_matrix = []
        entropy_matrix = []

        # Pointer to change randomly the background traffic injected to the link
        tb_counter = 0
        br_background = 3.0  # Initialize background traffic: 3 Mb

        while True:

            # counter = 0
            # while True:

            print('Consuming data from Kafka...')
            resolution, fps_out, bitrate_out, duration_out, mos_out, timestamp = consume_kafka(consumer)
            # print('Probe bitrate: {}'.format(bitrate_out))

            # resp_get_vce_res = requests.get('http://' + VCE_RES_ADDR + ':' + VCE_RES_PORT).json()
            resp_get_vce_br = requests.get('http://' + VCE_BR_ADDR + ':' + VCE_BR_PORT).json()
            #  "stats": {"id": {"name": "Id", "value": ["02:42:ac:11:00:02", "2198B9A0-D7DA-11DD-8743-BCEE7B897BA7"]},
            #            "utc_time": {"name": "UTC Time [ms]", "value": 1584360198258},
            #            "pid": {"name": "PID", "value": 1100}, "pid_cpu": {"name": "CPU Usage [%]", "value": 58.3},
            #            "pid_ram": {"name": "RAM Usage [byte]", "value": 144613376},
            #            "gop_size": {"name": "GoP Size", "value": 25}, "num_fps": {"name": "Fps", "value": 26},
            #            "num_frame": {"name": "Frame", "value": 1276},
            #            "enc_quality": {"name": "Quality [0-69]", "value": 22},
            #            "enc_dbl_time": {"name": "Encoding Time [s]", "value": 54.08},
            #            "enc_str_time": {"name": "Encoding Time [h:m:s:ms]", "value": "00:00:54.29"},
            #            "max_bitrate": {"name": "Maximum Bitrate [kbps]", "value": 3000},
            #            "avg_bitrate": {"name": "Average Bitrate [kbps]", "value": 2850.4},
            #            "act_bitrate": {"name": "Actual Bitrate [kbps]", "value": 3159.4},
            #            "enc_speed": {"name": "Encoding Speed [x]", "value": 1.12}}}

            bitrate_in = resp_get_vce_br['stats']['act_bitrate']['value']
            max_bitrate = resp_get_vce_br['stats']['max_bitrate']['value']
            ram_in = resp_get_vce_br['stats']['pid_ram']['value']
            encoding_quality = resp_get_vce_br['stats']['enc_quality']['value']

            # profile_in = assign_profile(resolution, bitrate_in)

            # if profile_in == last_action and counter == 5:  # TODO: Check behaviour
            #     break
            # else:
            #     counter += 1
            #     print('Waiting new measurements...')
            #     sleep(1)

            # if counter == 5: break  # TODO: Check if it breaks the whole program

            # Main reward, based on MOS
            # Range between 72.05 and 19.10.
            # Possible values of mos x (exp(x)): 5 = 148; 4.3=73.69; 4=54.6; 4.2=66.6; 0.5=1.64
            # rew_mos = math.exp(4) - math.exp(5-float(mos_out))
            # rew_mos = math.exp(float(mos_out))
            if float(mos_out) > 2.5:
                mos = float(mos_out) - 2.5
                aux = 2.0
            else:
                mos = 2.5 - float(mos_out)
                aux = -2.0
            rew_mos = aux * math.exp(1.5 * mos)

            # Penalization based on losses received by probe in terms of bitrate
            # Range between -12 and -1.
            # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html
            # rew_br = -math.exp(3*distance.canberra(bitrate_in/1000, bitrate_out/1000))
            # New range between -44 and -12 (-1 if same bitrate).
            rew_br = -math.exp(2 * (1 + distance.canberra(float(bitrate_in) / 1000, float(bitrate_out) / 1000)))

            # Penalization when changing abruptly between profiles
            # Range between -20.58 and 0.0. New range between -13.7 and 0.
            # rew_smooth = np.where(distance.canberra(action, last_action) != 0,
            #                       (12-1)*np.log(1-distance.canberra(action, last_action)), 0)
            rew_smooth = 12 * np.log(1 - distance.canberra(action + 1, last_action + 1))

            # Reward higher profiles
            # Range between 22 and 11
            # rew_profile = 2*(12 - action)
            rew_profile = math.pow(2, (4-action))

            # Reward accumulating good results
            # rew_res =

            # https://en.wikipedia.org/wiki/Test_functions_for_optimization
            reward = rew_mos + rew_br + rew_smooth + rew_profile
            print('Rew_MOS: {}'.format(rew_mos))
            print('Rew_BR: {}'.format(rew_br))
            print('Rew_Smooth: {}'.format(rew_smooth))
            print('Rew_Profile: {}'.format(rew_profile))
            print('Total Reward: {}'.format(reward))
            rewards_matrix.append(reward)

            last_action = action

            if len(states_matrix) == 0:
                curr_state = [np.zeros((NUM_STATES, LEN_STATES))]
            else:
                curr_state = np.array(states_matrix[-1], copy=True)

            curr_state = np.roll(curr_state, -1, axis=1)  # Keep last 8 states, moving the first one to the end

            # Model states
            curr_state[0, -1] = float(bitrate_out) / MAX_BR  # Quality of the streaming
            curr_state[1, -1] = (float(max_bitrate) - float(bitrate_out)) / MAX_CAPACITY  # Loss rate
            # curr_state[1, -1] = RESOLUTIONS[int(resolution)]  # Resolution: 1080 (1) or 720 (0)
            # curr_state[2, -1] = br_background / (MAX_CAPACITY/1000)  # Current traffic background
            curr_state[2, -1] = float(encoding_quality) / 69  # Streaming encoding quality [0, 69]
            curr_state[3, -1] = float(ram_in) / 5000.0  # Ram_usage. Adapt based on computer

            predictions = actor_net.predict(np.reshape(curr_state, (1, NUM_STATES, LEN_STATES)))
            # print('\nAction predicted: ', predictions)
            predictions_cumsum = np.cumsum(predictions)
            # print('\nAction cumsum: ', predictions_cumsum)
            action = (predictions_cumsum > np.random.randint(1, RAND_ACTION) / float(RAND_ACTION)).argmax()
            # print('Last action: {0}. New action: {1}.'.format(last_action, action))

            entropy_matrix.append(environment.compute_entropy(predictions[0]))

            br_predicted = list(PROFILES[action].values())[0]
            res_predicted = list(PROFILES[action].keys())[0]

            print('Previous Profile-> Bitrate: {}. Resolution: {}'.format(list(PROFILES[last_action].values())[0],
                                                                          list(PROFILES[last_action].keys())[0]))
            print('Predicted Profile-> Bitrate: {}. Resolution: {}'.format(br_predicted, res_predicted))

            if res_predicted == 1080:
                res_out = 'high'
            else:
                res_out = 'low'

            # Send request to change resolution
            # if RESOLUTIONS[res_predicted] != RESOLUTIONS[int(resolution)]:
            #     print('Changing resolution...')
            #     resp_post_vco_res = requests.post('http://' + VCE_RES_ADDR +':' + VCE_RES_PORT + '/resolution/' + res_out)
            #     if resp_post_vco_res.status_code == 200:  # if response:
            #         print('Report to the vCE_resolution success')
            #     elif resp_post_vco_res.status_code == 404:  # else:
            #         print('Report to the vCE_resolution not found')
            #
            #     time.sleep(3)
            #     requests.get('http://' + VCE_BR_ADDR + ':' + VCE_BR_PORT + '/refresh/')
            #     requests.get('http://' + PROBE_ADDR + ':' + PROBE_PORT + '/refresh/')
            #     time.sleep(8)

            # Send request to change bitrate
            resp_post_vco_br = requests.post('http://' + VCE_BR_ADDR + ':' + VCE_BR_PORT + '/bitrate/' + str(br_predicted*1000))
            if resp_post_vco_br.status_code == 200:  # if response:
                print('Report to the vCE_bitrate success')
            elif resp_post_vco_br.status_code == 404:  # else:
                print('Report to the vCE_bitrate not found')

            # print(int(resolution))
            # print('Resolution actual: {}'.format(int(resolution)))
            # print('Resolution predicted: {}'.format(res_predicted))

            # To receive status from the vco-resolution
            # resp_get_vco_res = requests.get('http://' + VCE_RES_ADDR + ':' + VCE_RES_PORT + '/resolution/' + res_out).json()
            # {"status":true,"stats":{"frame":"2635","fps":"26","q":"1.0","size":"68432","time":"00:01:48.98","bitrate":"5143.9kbits/s","speed":"1.06x"}}'

            if tb_counter == 1:
                br_background = randint(1, MAX_CAPACITY/1000 - 2)  # Generate randomly the traffic background
                requests.post('http://' + TB_ADDR + ':' + TB_PORT + '/bitrate/' + str(br_background*1000))
                tb_counter = 0
                # requests.get('http://' + VCE_RES_ADDR + ':' + VCE_RES_PORT + '/refresh/')

            tb_counter += 1

            print('----------------')
            time.sleep(4)

            log_file.write(str(timestamp) + '\t' +
                           str(reward) + '\t' +
                           str(max_bitrate) + '\t' +
                           str(bitrate_in) + '\t' +
                           str(bitrate_out) + '\t' +
                           str(resolution) + '\t' +
                           str(float(mos_out)) + '\t' +
                           str(br_background*1000) + '\t' +
                           str(action) +
                           '\n'
                           )

            log_file.flush()

            # Report experience to the coordinator
            if len(rewards_matrix) >= TRAINING_REPORT:
                exp_queue.put([states_matrix, actions_matrix, rewards_matrix, {'entropy': entropy_matrix}])

                # Sync information
                actor_net_params, critic_net_params = net_params_queue.get()
                actor_net.set_actor_params(actor_net_params)
                critic_net.set_critic_params(critic_net_params)

                del states_matrix[:]
                del actions_matrix[:]
                del rewards_matrix[:]
                del entropy_matrix[:]

                log_file.write('\n')

                # requests.get('http://' + VCE_RES_ADDR + ':' + VCE_RES_PORT + '/refresh/')

            states_matrix.append(curr_state)

            actions = np.zeros(NUM_ACTION)
            actions[action] = 1
            actions_matrix.append(actions)


def main():

    # Remove results folder (summaries, logs and trained models) to start creating a new model
    if CLEAN > 0:
        # os.system('rm ./results/*')  # Avoid using this line in order to preserve models
        os.system('rm ./results/summary')
        os.system('rm ./results/logs')
        os.system('rm ./results/model')

    np.random.seed(random_seed)

    # Directory to place results and logs
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # Kafka consumer to get data sent from probe
    consumer = KafkaConsumer(
        KAFKA_TOPIC_OUT,
        bootstrap_servers=KAFKA_SERVER,
        auto_offset_reset='latest',  # Collect at the end of the log. To collect every message: 'earliest'
        enable_auto_commit=True,
        value_deserializer=lambda x: loads(x.decode('utf-8')))

    print('Starting the program...')

    # Create a coordinator and multiple agent processes. This projects aims to train only one agent
    coordinator = mp.Process(target=sup_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for id_agent in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent, args=(id_agent, net_params_queues[id_agent], exp_queues[id_agent],
                                                     consumer)))
    for id_agent in range(NUM_AGENTS):
        agents[id_agent].start()

    # Training done
    coordinator.join()
    consumer.close()
    print("------------TRAINING DONE-----------------")


if __name__ == '__main__':
    main()
