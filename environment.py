import numpy as np
import tensorflow as tf
import datetime
import json

# Different profiles combining resolutions and bitrate
PROFILES = {1: {1080: 50}, 2: {1080: 30}, 3: {1080: 20}, 4: {1080: 15}, 5: {1080: 10}, 6: {1080: 5}, 7: {720: 25},
            8: {720: 15}, 9: {720: 10}, 10: {720: 7.5}, 11: {720: 5}, 12: {720: 2.5}}

RANDOM_SEED = 42
GAMMA = 0.99


class Environment:
    def __init__(self):

        np.random.seed(RANDOM_SEED)


def model_summary():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    total_reward = tf.Variable(0.)
    tf.summary.scalar("Total Reward", total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    model_vars = [td_loss, total_reward, avg_entropy]
    model_ops = tf.summary.merge_all()

    return model_vars, model_ops


def compute_gradients(states_matrix, actions_matrix, rewards_matrix, actor_net, critic_net):
    assert states_matrix.shape[0] == actions_matrix.shape[0]
    assert states_matrix.shape[0] == rewards_matrix.shape[0]

    ba_size = states_matrix.shape[0]

    v_matrix = critic_net.predict(states_matrix)

    r_matrix = np.zeros(rewards_matrix.shape)

    # if terminal:
    #     r_matrix[-1, 0] = 0  # terminal state
    # else:
    #     r_matrix[-1, 0] = v_matrix[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        r_matrix[t, 0] = rewards_matrix[t] + GAMMA * r_matrix[t + 1, 0]

    td_matrix = r_matrix - v_matrix

    actor_gradients = actor_net.get_gradients(states_matrix, actions_matrix, td_matrix)
    critic_gradients = critic_net.get_gradients(states_matrix, r_matrix)

    return actor_gradients, critic_gradients, td_matrix


def compute_entropy(info):
    entropy = 0.0
    for i in range(len(info)):
        if 0 < info[i] < 1:
            entropy -= info[i] * np.log(info[i])
    return entropy


# TODO: Check good behaviour, based on result
def consume_kafka(consumer):

    for message in consumer:
        content = message.value

        resolution = content['value']['resolution']
        frame_rate = content['value']['frame_rate']
        bitrate = content['value']['bitrate']
        duration = content['value']['duration']
        mos = content['value']['mos']
        break

    return resolution, frame_rate, bitrate, duration, mos


def assign_profile(resolution, bitrate):
    bitrate_list = np.reshape([list(item.values())[0] for item in list(PROFILES.values())], (2, 6))

    if resolution == '1080':
        comparison_list = abs(bitrate_list[0] - bitrate)
        profile = np.argmin(comparison_list) + 1

    elif resolution == '720':
        comparison_list = abs(bitrate_list[1] - bitrate)
        profile = np.argmin(comparison_list) + 7

    else:  # TODO: Control resolutions
        print('Wrong resolution')

    return profile
