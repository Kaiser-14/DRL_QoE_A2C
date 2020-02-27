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

from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from time import sleep
from json import dumps
from pymongo import MongoClient
from json import loads

# TODO: Check CUDA
# os.environ['CUDA_VISIBLE_DEVICES']=''
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


NUM_AGENTS = 1
# NUM_AGENTS = multiprocessing.cpu_count()  # Enable to fully process the model
random_seed = 42

# TODO: Tune parameters
NUM_STATES = 3  # Number of possible states, e.g., bitrate
LEN_STATES = 8  # Number of frames in the past
TRAINING_REPORT = 100  # Batch to write information into the logs

# Different profiles combining resolutions and bitrate
PROFILES = {1: {1080: 50}, 2: {1080: 30}, 3: {1080: 20}, 4: {1080: 15}, 5: {1080: 10}, 6: {1080: 5}, 7: {720: 25},
            8: {720: 15}, 9: {720: 10}, 10: {720: 7.5}, 11: {720: 5}, 12: {720: 2.5}}
RESOLUTIONS = {1080: 1, 720: 0}

DEFAULT_ACTION = 4  # PROFILES[0][1] 1080p 15Mbps
MAX_BITRATE = max(list(x.values())[0] for x in list(PROFILES.values()))
MAX_CAPACITY = 20000000.0  # 20MB
# DEFAULT_RES = list(PROFILES[4].keys())[0]
# DEFAULT_BITRATE = list(PROFILES[4].values())[0]
NUM_ACTION = len(PROFILES)

# Learning rates
ACTOR_LR = 0.0001
CRITIC_LR = 0.001

# Files
SUMMARY_DIR = './results/summary/'
LOGS_DIR = './results/logs/'
MODEL_DIR = './results/model/'


# Kafka parameters
# KAFKA_URL = /api/datasources/proxy/:datasourceId/*  # Check URL
# KAFKA_PORT = {'address':"address"}
KAFKA_TOPIC = 'a2c.mod'
KAFKA_SERVER = ['localhost:2181']

# Other parameters
# TODO: Tune parameters
RAND_ACTION = 1000  # Random value to decide exploratory action
alpha = 0.8  # Reward tuning: pMOS
beta = 0.8  # Reward tuning: Usage CPU
gamma = 0.8  # Reward tuning: Bitrate

# Predefined parameters
CLEAN = 0  # Change to 1 to delete the results folder and start a fresh modelling
# TODO: Check correct restoring of pretrained model
A2C_MODEL = None  # Start a fresh model
# A2C_MODEL = tf.train.latest_checkpoint(MODEL_DIR)  # Load latest trained model


def sup_agent(net_params_queues, exp_queues):  # Supervisor agent
    # FIXME: Need it?
    logging.basicConfig(filename=LOGS_DIR + 'log_supervisor', filemode='w', level=logging.INFO)

    # FIXME: Insert the code to check GPU device working.
    # with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess, open(LOGS_DIR + '_test', 'w')
    # as log_test_file:
    with tf.Session() as sess, open(LOGS_DIR + 'log_test', 'w') as log_test_file:
        log_test_file.flush()  # Only to silent the python-check
        actor_net = actor.Actor(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                learning_rate=ACTOR_LR)
        critic_net = critic.Critic(sess, states_dim=[NUM_STATES, LEN_STATES],  actions_dim=NUM_ACTION,
                                   learning_rate=CRITIC_LR)

        model_vars, model_ops = environment.model_summary()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # Training monitor
        saver = tf.train.Saver()  # Save neural net parameters

        # Restore previously NN trained (set in global parameters)
        model = A2C_MODEL
        if model is not None:
            saver.restore(sess, model)
            print('\nModel restored')

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
                states_matrix, actions_matrix, rewards_matrix, terminal, info = exp_queues[i].get()

                actor_gradients, critic_gradients, td_matrix = environment.compute_gradients(
                        states_matrix=np.stack(states_matrix, axis=0),
                        actions_matrix=np.vstack(actions_matrix),
                        rewards_matrix=np.vstack(rewards_matrix),
                        terminal=terminal, actor_net=actor_net, critic_net=critic_net)

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
            reward_avg = reward_sum / agents_sum
            td_loss_avg = tdloss_sum / total_len
            entropy_avg = entropy_sum / total_len

            logging.info('Epoch: ' + str(epoch) + ' TD_Loss: ' + str(td_loss_avg) +
                         ' Average_Reward: ' + str(reward_avg) + ' Average_Entropy: ' + str(entropy_avg))

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
                # testing(epoch,
                #     SUMMARY_DIR + '/model/' + "/model_epoch_" + str(epoch) + ".ckpt",
                #     log_test_file)
                log_test_file.write(str(epoch) + '\t' +
                               str(td_loss_avg) + '\t' +
                               str(reward_avg) + '\t' +
                               str(entropy_avg) + '\n')
                log_test_file.flush()


def agent(agent_id, net_params_queue, exp_queue):  # General agent

    with tf.Session() as sess, open(LOGS_DIR + 'agent' + str(agent_id), 'w') as log_file:
        actor_net = actor.Actor(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                learning_rate=ACTOR_LR)
        critic_net = critic.Critic(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                   learning_rate=CRITIC_LR)

        # Initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor_net.set_actor_params(actor_net_params)
        critic_net.set_critic_params(critic_net_params)

        # Initialize action
        action = DEFAULT_ACTION
        last_action = action

        # Vectors for storing values: states, actions, rewards.
        actions = np.zeros(NUM_ACTION)
        actions[action] = 1
        actions_matrix = [actions]

        states_matrix = [np.zeros((NUM_STATES, LEN_STATES))]
        rewards_matrix = []
        entropy_matrix = []

        #timestamp = 0
        #time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

        while True:

            # TODO: Streaming calls. Should I wait until coincide bitrate_tx with previous bitrate_rx?
            # Server
            # free_capacity = API_CALL()
            while True:
                timestamp, bitrate_tx, bitrate_rx, resolution, pMOS, result = consume_kafka()
                if result:
                    break;
                else:
                    sleep(1)

            #bitrate_in = list(PROFILES[action].values())[0]  # Received from entry
            #bitrate_out = list(PROFILES[action].values())[0]  # Predicted bitrate

            # resolution_out = list(PROFILES[1].keys())[0]  # Resolution

            # TODO: create correctly reward function after Kafka
            # reward = alpha*pMOS - beta*usage_CPU - gamma*(bitrate_in/bitrate_out)
            reward = alpha*pMOS - beta*(bitrate_tx / bitrate_rx)
            rewards_matrix.append(reward)

            if len(states_matrix) == 0:
                curr_state = [np.zeros((NUM_STATES, LEN_STATES))]
            else:
                curr_state = np.array(states_matrix[-1], copy=True)

            curr_state = np.roll(curr_state, -1, axis=1)  # Keep last 8 states, moving the first one to the end

            # FIXME: Control states based on information received by server (cpu_usage, memory, cpu_number, blockiness, blur, blockloss)
            curr_state[0, -1] = bitrate_rx / MAX_BITRATE  # Quality
            curr_state[1, -1] = 1 / MAX_CAPACITY  # Available capacity: limited to 50Mbps #TODO: Define it
            curr_state[2, -1] = RESOLUTIONS[resolution]  # Resolution: 1080 or 720
            #curr_state[3, -1] = cpu_usage;

            predictions = actor_net.predict(np.reshape(curr_state, (1, NUM_STATES, LEN_STATES)))
            print('\nAction predicted: ', predictions)
            predictions_cumsum = np.cumsum(predictions)
            print('\nAction cumsum: ', predictions_cumsum)
            action = (predictions_cumsum > np.random.randint(1, RAND_ACTION) / float(RAND_ACTION)).argmax()
            print('Last action: {0}. New action: {1}.'.format(last_action, action))

            states_matrix.append(curr_state)
            last_action = action

            # profile_in = assign_profile(resolution, bitrate_rx)
            # br = list(PROFILES[profile_in].values())[0]


            # TODO: Report action (resolution and bitrate) to vCE
            # Resolution
            # curl -XPOST http://ip:3000/bitrate/150000
            # https://curl.trillworks.com/#
            # response = requests.get('https://api.test.com/', auth=('some_username', 'some_password'))



            #VCE_IP = '1.1.1.1'
            #POST_MESSAGE = 'http://' + VCE_IP + '/bitrate/' + bitrate_rx
            #response = requests.post(POST_MESSAGE)
            # https://realpython.com/python-requests/
            #if response.status_code == 200:  # if response:
            #    print('Report to the vCE success')
            #elif response.status_code == 404:  # else:
            #    print('Report to the vCE success not found')



            # Bitrate. Change trough SSH
            # https://stackoverflow.com/questions/3586106/perform-commands-over-ssh-with-python
            #post /resolution/low
            #post /resolution/high

            # FIXME: Adapt timer to the behaviour of the system
            time.sleep(1)

            entropy_matrix.append(environment.compute_entropy(predictions[0]))

            # TODO: Add more information to the logs
            log_file.write(str(timestamp) + '\t' +
                           str(reward) + '\t' +
                           str(bitrate_tx) + '\t' +
                           str(bitrate_rx) + '\t' +
                           str(resolution) + '\t' +
                           str(pMOS) +
                           '\n'
                           )

            log_file.flush()

            # Report experience to the coordinator
            if len(rewards_matrix) >= TRAINING_REPORT:
                exp_queue.put([states_matrix[1:], actions_matrix[1:], rewards_matrix[1:], {'entropy': entropy_matrix}])

                # Sync information
                actor_net_params, critic_net_params = net_params_queue.get()
                actor_net.set_actor_params(actor_net_params)
                critic_net.set_critic_params(critic_net_params)

                del states_matrix[:]
                del actions_matrix[:]
                del rewards_matrix[:]
                del entropy_matrix[:]

                # log_file.write('\n')


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

    # Kafka
    # https://towardsdatascience.com/kafka-python-explained-in-10-lines-of-code-800e3e07dad1
    # TODO: Tune parameters
    # consumer = KafkaConsumer(
    #     KAFKA_TOPIC,
    #     bootstrap_servers=KAFKA_SERVER,
    #     auto_offset_reset='latest',  # Collect at the end of the log. To collect every message: 'earliest'
    #     enable_auto_commit=True,
    #     group_id=None,
    #     value_deserializer=lambda x: loads(x.decode('utf-8')))

    #client = MongoClient('localhost:27017')
    #collection = client.drltfm.drltfm

    # producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
    #                          value_serializer=lambda x:
    #                          dumps(x).encode('utf-8'))

    # Get last message to initialize parameters
    # try:
    #     with open('kafka_metrics.log', 'r') as kafka_log:
    #         for line in kafka_log:
    #             values = line.split()  # Possible values: timestamp, bitrate sent and received, resolution, block_loss
    #             a = values[0]
    #             b = values[1]

    # except Exception as ex:
    #     kafka_log = open('kafka_metrics.log', 'w')
    #     kafka_log.close()



    # Create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=sup_agent, args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for id_agent in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent, args=(id_agent, net_params_queues[id_agent], exp_queues[id_agent])))
    for id_agent in range(NUM_AGENTS):
        agents[id_agent].start()

    # Training done
    #coordinator.join()
    #consumer.close()
    # producer.close()
    print("------------DONE-----------------")


if __name__ == '__main__':
    main()
