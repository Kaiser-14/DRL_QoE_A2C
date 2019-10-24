import environment
import actor
import critic

import os
import logging
import time
import numpy as np
import multiprocessing as mp
import tensorflow as tf

import logging

# TODO: Check CUDA
# os.environ['CUDA_VISIBLE_DEVICES']=''


NUM_AGENTS = 4
# NUM_AGENTS = multiprocessing.cpu_count()  # Enable to fully process the model
random_seed = 42

# TODO: Tune parameters
NUM_STATES = 3  # Number of possible states, e.g., bitrate
LEN_STATES = 8  # Number of frames in the past
TRAINING_REPORT = 100  # Batch to write information into the logs

# Different profiles combining resolutions and bitrates
PROFILES = {1: {1080: 50}, 2: {1080: 30}, 3: {1080: 20}, 4: {1080: 15}, 5: {1080: 10}, 6: {1080: 5}, 7: {720: 25},
            8: {720: 15}, 9: {720: 10}, 10: {720: 7.5}, 11: {720: 5}, 12: {720: 2.5}}

DEFAULT_ACTION = 4  #PROFILES[0][1] 1080p 15Mbps
# DEFAULT_RES = list(PROFILES[4].keys())[0]
# DEFAULT_BITRATE = list(PROFILES[4].values())[0]
NUM_ACTION = len(PROFILES)

# Learning rates
ACTOR_LR = 0.0001
CRITIC_LR = 0.001

# Files
SUMMARY_DIR = './results/summary/'
TRACES_DIR = './traces/'
LOGS_DIR = './results/logs/'


# Kafka parameters
# KAFKA_URL = /api/datasources/proxy/:datasourceId/*  # Check URL
# KAFKA_PORT = {'address':"address"}

# Other parameters
# TODO: Tune parameters
RAND_ACTION = 1000  # Random value to decide exploratory action
alpha = 0.8  # Reward tuning: pMOS
beta = 0.8  # Reward tuning: Usage CPU
gamma = 0.8  # Reward tuning: Bitrate

# Neural Network Model
# TODO: Clean these lines
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
# NN_MODEL = LOGS_DIR + 'pretrain_linear_reward.ckpt'
# NN_MODEL = latest = tf.train.latest_checkpoint(checkpoint_dir) #Use the previous trained model
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models
NN_MODEL = None

# Predefined parameters
CLEAN = 0  # Change to 1 to delete the results folder and start a fresh modelling


def sup_agent(net_params_queues, exp_queues):  # Supervisor agent
    # FIXME: Need it?
    logging.basicConfig(filename=LOGS_DIR + 'log_supervisor', filemode='w', level=logging.INFO)

    with tf.Session() as sess, open(LOGS_DIR + '_test', 'w') as log_test_file:
        log_test_file.flush()  # Only to silent the python-check
        actor_net = actor.Actor(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                learning_rate=ACTOR_LR)
        critic_net = critic.Critic(sess, states_dim=[NUM_STATES, LEN_STATES], learning_rate=CRITIC_LR)

        model_vars, model_ops = environment.model_summary()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)  # Training monitor
        saver = tf.train.Saver()  # Save neural net parameters

        # Restore previously NN trained (set in global parameters)
        model = NN_MODEL
        if model is not None:
            saver.restore(sess, model)
            print('\nModel restored')

        epoch = 0
        while True:
            actor_params = actor_net.get_actor_params()
            critic_params = critic_net.get_critic_params()

            for i in range(NUM_AGENTS):
                net_params_queues[i].put([actor_params, critic_params])  # TODO: Do not forget to check asynchronous way

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

            # compute aggregated gradient
            # TODO: Check it
            assert NUM_AGENTS == len(actor_gradient_matrix)
            assert len(actor_gradient_matrix) == len(critic_gradient_matrix)

            # TODO: Check it
            for i in range(len(actor_gradient_matrix)):
                actor_net.apply_gradients(actor_gradient_matrix[i])
                critic_net.apply_gradients(critic_gradient_matrix[i])

            epoch += 1
            reward_avg = reward_sum / agents_sum
            td_loss_avg = tdloss_sum / total_len
            entropy_avg = entropy_sum / total_len

            logging.info('Epoch: ' + str(epoch) + ' TD Loss: ' + str(td_loss_avg) +
                         ' Average Reward: ' + str(reward_avg) + ' Average Entropy: ' + str(entropy_avg))

            summary = sess.run(model_ops, feed_dict={
                model_vars[0]: td_loss_avg,
                model_vars[1]: reward_avg
            })

            writer.add_summary(summary, epoch)
            writer.flush()

            if epoch % 100 == 0:
                # Save the neural net parameters to disk.
                save_path = saver.save(sess, SUMMARY_DIR + '/model/' + "/model_epoch_" +
                                       str(epoch) + ".ckpt")
                logging.info("Model saved in file: " + save_path)
                # testing(epoch,
                #     SUMMARY_DIR + '/model/' + "/model_epoch_" + str(epoch) + ".ckpt",
                #     log_test_file)


def agent(agent_id, traces, net_params_queue, exp_queue):  # General agent
    # FIXME: Remove it?
    # env = environment.Environment()

    # TODO: Check possibility to open another file to see current environment working
    with tf.Session() as sess, open(LOGS_DIR + 'agent' + str(agent_id), 'w') as log_file:
        actor_net = actor.Actor(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                learning_rate=ACTOR_LR)
        critic_net = critic.Critic(sess, states_dim=[NUM_STATES, LEN_STATES], actions_dim=NUM_ACTION,
                                   learning_rate=CRITIC_LR)

        # Initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor_net.set_actor_params(actor_net_params)
        critic_net.set_critic_params(critic_net_params)

        # TODO: Check where to include last_action parameter
        # Initialize some parameters
        last_action = DEFAULT_ACTION
        action = last_action

        # Vectors for storing values: states, actions, rewards.
        actions = np.zeros(NUM_ACTION)
        actions[action] = 1
        actions_matrix = [actions]

        states_matrix = [np.zeros((NUM_STATES, LEN_STATES))]
        rewards_matrix = []
        entropy_matrix = []

        time_stamp = 0

        while True:

            # TODO: Think to better handle action variable

            # TODO: Kafka calls
            # Kafka Server

            profile = action
            bitrate_in = list(PROFILES[action].values())[0]  # Received from Kafka
            bitrate_out = list(PROFILES[profile].values())[0]  # Predicted bitrate

            # TODO: create correctly reward function after Kafka
            # reward = alpha*pMOS - beta*usage_CPU - gamma*(bitrate_in/bitrate_out)
            reward = 0
            rewards_matrix.append(reward)

            last_action = action

            if len(states_matrix) == 0:
                curr_state = [np.zeros((NUM_STATES, LEN_STATES))]
            else:
                curr_state = np.array(states_matrix[-1], copy=True)

            curr_state = np.roll(curr_state, -1, axis=1)  # Keep last 8 states, moving the first one to the end

            curr_state[0, -1] = 1  # Bitrate
            curr_state[1, -1] = 2  # Rescaling? Yes or nope
            curr_state[2, -1] = 3  # Usage of CPU above %
            curr_state[3, -1] = 3  # Maybe CRF to discover the next bitrate
            curr_state[4, -1] = 3  # Current bitrate of traffic manager or percentage of tunnel occupation
            curr_state[5, -1] = 3  # Current output bitrate

            #TODO: Control states
            # curr_state[0, -1] = video_bitrates[bitrate_rec]  #Latest profile
            # curr_state[1, -1] = float(mean_free_capacity_frac)  #Free capacity of the link
            # curr_state[2, -1] = float(mean_loss_rate_frac)  #Check frac

            predictions = actor_net.predict(np.reshape(curr_state, (1, NUM_STATES, LEN_STATES)))
            # print('\nAction predicted: ', action_predicted)
            predictions_cumsum = np.cumsum(predictions)
            # print('\nAction cumsum: ', action_cumsum)
            action = (predictions_cumsum > np.random.randint(1, RAND_ACTION) / float(RAND_ACTION)).argmax()

            # TODO: Report action (resolution and bitrate) to vCE
            # REST/api:.../bitrate/+bitrate
            # REST/api:.../bitrate/+resolution

            # FIXME: Add here wait() and last_action
            last_action = action
            time.sleep(5)

            entropy_matrix.append(environment.compute_entropy(predictions[0]))

            # TODO: Add more information to the logs
            log_file.write(str(time_stamp) + '\t' +
                           str(reward) # + '\t' # +
                           # str(video_bitrates[bit_rate]) + '\t' + #Check how to decide
                           # str(mean_free_capacity) + '\t' +
                           # str(mean_loss_rate * 8.0 / M_IN_K) + '\t' +
                           # str(video_count) + '\n'
                            )

            log_file.flush()

            # Report experience to the coordinator
            if len(rewards_matrix) >= TRAINING_REPORT:
                exp_queue.put([states_matrix[1:],
                               actions_matrix[1:],
                               rewards_matrix[1:],
                               {'entropy': entropy_matrix}
                              ])

                # Sync information
                actor_net_params, critic_net_params = net_params_queue.get()
                actor_net.set_network_params(actor_net_params)
                critic_net.set_network_params(critic_net_params)

                del states_matrix[:]
                del actions_matrix[:]
                del rewards_matrix[:]
                del entropy_matrix[:]

                # log_file.write('\n')


def main():

    if CLEAN > 0:
        os.system("rm ./results/*")

    np.random.seed(random_seed)

    # Directory to place results
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # Inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # Create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=sup_agent, args=(net_params_queues, exp_queues))  # TODO: Improve it
    coordinator.start()

    agents = []
    for id_agent in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent, args=(id_agent, net_params_queues[id_agent], exp_queues[i])))
    for id_agent in range(NUM_AGENTS):
        agents[id_agent].start()

    # Training done
    coordinator.join()
    print("------------DONE-----------------")


if __name__ == '__main__':
    main()
