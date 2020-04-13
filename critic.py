import tflearn
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.merge import Add
from keras.optimizers import Adam


class Critic(object):
    def __init__(self, sess, states_dim, learning_rate):
        self.sess = sess
        self.states_dim = states_dim
        # self.actions_dim = actions_dim
        self.learning_rate = learning_rate

        # Creating the full critic network
        # self.state_input, self.action_input, self.model_critic = create_critic(self)  # Keras
        self.input, self.output = self.create_critic_tf()
        # Get all network parameters
        self.critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_critic_params = []
        for param in self.critic_params:
            self.input_critic_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_critic_params_op = []
        for idx, param in enumerate(self.input_critic_params):
            self.set_critic_params_op.append(self.critic_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.output)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.output)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.critic_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).\
            apply_gradients(zip(self.critic_gradients, self.critic_params))

    def create_critic(self):
        state_input = Input(shape=[None, self.states_dim[0], self.states_dim[1]])
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.actions_dim)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model_critic = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model_critic.compile(loss="mse", optimizer=adam)

        return state_input, action_input, model_critic

    def create_critic_tf(self):
        with tf.variable_scope('critic'):
            input_critic = tflearn.input_data(shape=[None, self.states_dim[0], self.states_dim[1]])

            split_0 = tflearn.fully_connected(input_critic[:, 0:1, -1], 128, activation='relu')
            # split_1 = tflearn.fully_connected(input_critic[:, 1:2, -1], 128, activation='relu')
            split_2 = tflearn.conv_1d(input_critic[:, 1:2, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(input_critic[:, 2:3, :], 128, 4, activation='relu')
            # split_4 = tflearn.conv_1d(input_critic[:, 2:3, :], 128, 4, activation='relu')
            # split_5 = tflearn.fully_connected(input_critic[:, 2:3, -1], 128, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            # split_4_flat = tflearn.flatten(split_4)

            # merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')
            # merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_4_flat, split_5], 'concat')
            merge_net = tflearn.merge([split_0, split_2_flat, split_3_flat], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 64, activation='relu')
            output_critic = tflearn.fully_connected(dense_net_0, 1, activation='linear')

            return input_critic, output_critic

    def create_critic1(self):
        input_critic = tflearn.input_data(shape=[None, self.states_dim[0], self.states_dim[1]])

        h0 = tflearn.fully_connected(input_critic, 128, activation='relu')
        h1 = tflearn.conv_1d(h0, 64, 4, activation='relu')
        h2 = tflearn.conv_1d(h1, 64, 4, activation='relu')
        dense_net_0 = tflearn.fully_connected(h2, 128, activation='relu')

        output_critic = tflearn.fully_connected(dense_net_0, 1, activation='linear')

        return input_critic, output_critic

    def train(self, input_tf, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.input: input_tf,
            self.td_target: td_target
        })

    def predict(self, input_tf):
        return self.sess.run(self.output, feed_dict={
            self.input: input_tf
        })

    def get_td(self, input_tf, td_target):
        return self.sess.run(self.td, feed_dict={
            self.input: input_tf,
            self.td_target: td_target
        })

    def get_gradients(self, input_tf, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.input: input_tf,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_critic_params(self):
        return self.sess.run(self.critic_params)

    def set_critic_params(self, input_critic_params):
        self.sess.run(self.set_critic_params_op, feed_dict={
            i: d for i, d in zip(self.input_critic_params, input_critic_params)
        })
