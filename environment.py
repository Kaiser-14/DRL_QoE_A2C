import numpy as np
import tensorflow as tf

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


def compute_gradients(states_matrix, actions_matrix, rewards_matrix, terminal, actor_net, critic_net):
    assert states_matrix.shape[0] == actions_matrix.shape[0]
    assert states_matrix.shape[0] == rewards_matrix.shape[0]

    ba_size = states_matrix.shape[0]

    v_matrix = critic_net.predict(states_matrix)

    r_matrix = np.zeros(rewards_matrix.shape)

    if terminal:
        r_matrix[-1, 0] = 0  # terminal state
    else:
        r_matrix[-1, 0] = v_matrix[-1, 0]  # boot strap from last state

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
