import tflearn
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

ENTROPY_WEIGHT = 0.5
ENTROPY_EPS = 1e-6


class Actor(object):
    def __init__(self, sess, states_dim, actions_dim, learning_rate):
        self.sess = sess
        self.states_dim = states_dim
        self.actions_dim = actions_dim
        self.learning_rate = learning_rate

        # Creating the full actor network
        self.input, self.output = self.create_actor_tf()

        # Get all network parameters
        self.actor_params = \
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_actor_params = []
        for param in self.actor_params:
            self.input_actor_params.append(
                tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_actor_params_op = []
        for idx, param in enumerate(self.input_actor_params):
            self.set_actor_params_op.append(self.actor_params[idx].assign(param))

        self.actions = tf.placeholder(tf.float32, [None, self.actions_dim])

        # This gradient will be provided by the critic network
        self.actor_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.objective = tf.reduce_sum(tf.multiply(tf.log(tf.reduce_sum(
            tf.multiply(self.output, self.actions),
            reduction_indices=1, keep_dims=True)),
            -self.actor_weights)) + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.output,
                                                                               tf.log(self.output + ENTROPY_EPS)))

        self.actor_grads = tf.gradients(self.objective, self.actor_params)

        # FIXME: Test other optimizer
        self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).apply_gradients(zip(self.actor_grads,
                                                                                          self.actor_params))

    # Using Keras
    def create_actor(self):
        input_actor = Input(shape=[None, self.states_dim[0], self.states_dim[1]])
        h1 = Dense(24, activation='relu')(input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output_actor = Dense(self.actions_dim, activation='relu')(h3)

        model_actor = Model(input=input, output=output_actor)
        adam = Adam(lr=0.001)
        model_actor.compile(loss="mse", optimizer=adam)

        return input_actor, output_actor, model_actor

    def create_actor_tf(self):
        with tf.variable_scope('actor'):
            input_actor = tflearn.input_data(shape=[None, self.states_dim[0], self.states_dim[1]])

            split_0 = tflearn.fully_connected(input_actor[:, 0:1, -1], 128, activation='relu')
            split_1 = tflearn.fully_connected(input_actor[:, 1:2, -1], 128, activation='relu')
            split_2 = tflearn.conv_1d(input_actor[:, 2:3, :], 128, 4, activation='relu')
            split_3 = tflearn.conv_1d(input_actor[:, 3:4, :], 128, 4, activation='relu')
            split_4 = tflearn.conv_1d(input_actor[:, 4:5, : self.actions_dim], 128, 4, activation='relu')
            split_5 = tflearn.fully_connected(input_actor[:, 4:5, -1], 128, activation='relu')

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merge_net = tflearn.merge([split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5], 'concat')

            dense_net_0 = tflearn.fully_connected(merge_net, 128, activation='relu')
            output_actor = tflearn.fully_connected(dense_net_0, self.actions_dim, activation='softmax')

            return input_actor, output_actor

    def train(self, input, actions, actor_weights):
        self.sess.run(self.optimize, feed_dict={self.input: input,
                                                self.actions: actions,
                                                self.actor_weights: actor_weights
                                                })

    def predict(self, input):
        return self.sess.run(self.output, feed_dict={
            self.input: input
        })

    # Could be set in code
    def get_gradients(self, input, actions, actor_weights):
        return self.sess.run(self.actor_grads, feed_dict={
                                                          self.input: input,
                                                          self.actions: actions,
                                                          self.actor_weights: actor_weights
                                                          })

    def apply_gradients(self, actor_grads):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_grads, actor_grads)
        })

    def get_actor_params(self):
        return self.sess.run(self.actor_params)

    def set_actor_params(self, input_actor_params):
        self.sess.run(self.set_actor_params_op, feed_dict={
            i: d for i, d in zip(self.input_actor_params, input_actor_params)
        })
