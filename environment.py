import numpy as np
import tensorflow as tf

VIDEO_BITRATE = [3, 5, 10, 15, 20, 50]  # Mbps. Consider in providing it in kbps
# VIDEO_BITRATE = [3000, 5000, 10000, 15000, 20000, 50000]  # Kbps
NUM_ACTION = len(VIDEO_BITRATE)
RANDOM_SEED = 42
VIDEO_CHUNKS = 48#Why? FPS?
GAMMA = 0.99

class Environment:
    def __init__(self, traces):

        np.random.seed(RANDOM_SEED)
        self.video_pointer = 0

        #Check if needed to include separate or join files
        #self.time_all = traces[0]
        #self.bw_all = traces[1]
        #self.crf_all = traces[2]
        #self.rescaling_all = traces[3]
        #self.cpu_all = traces[4]

        # Selecting one specific trace file
        #index_traces = np.random.randint(len(self.time_all))
        #self.time = self.time_all(index_traces)
        #self.bw = self.bw_all(index_traces)
        #self.crf = self.crf_all(index_traces)
        #self.rescaling = self.rescaling_all(index_traces)
        #self.cpu = self.cpu_all(index_traces)

        #self.video_chunk_counter = 0
        #self.buffer_size = 0

        # Check it if necessary. We only have two resolutions, and five bitrates
        #self.video_size = {}
        #for bitrate in range(BITRATES):
        #    self.video_size[bitrate] = []
        #    filename = 'video_size_' + str(bitrate)
        #    with open(filename) as f:
        #        for line in f:
        #            self.video_size[bitrate].append(int(line.split()[0]))

    def get_info(self, video_bitrates, quality, link_capacity):
        #chunk_size = self.video_size[bitrate][self.video_pointer]
        self.video_pointer += 1

        video = 0.0
        free_capacity = 0
        lossrate = 0

        mean_free_capacity = []
        #mean_free_capacity _frac = []
        mean_loss_rate = []
        #mean_loss_rate_frac = []


        while True:
            # TODO: Create background traffic
            #   background = self.get_traffic_background()
            video = video_bitrates[quality] * 1000  #Convert to bps. TODO: Check random normal

            free_capacity = link_capacity - video - background_traffic
            if free_capacity < 0: free_capacity = 0.0
            free_capacity_list.append(free_capacity)
            #free_capacity_frac_list.append(free_capacity/link_capacity)

            loss_rate = video/video_rx  # TODO: Check it
            if loss_rate < 0: loss_rate = 0
            loss_rate_list.append(loss_rate)
            #loss_rate_frac_list.append(loss_rate/)

        #mean_free_capacity = np.mean(free_capacity_list)
        #mean_free_capacity_frac = np.mean(free_capacity_frac_list)
        #mean_loss_rate = np.mean(loss_rate_list)
        #mean_loss_rate_frac = np.mean(loss_rate_frac_list)

        #end_of_video = False
        #if.self.video_pointer >= VIDEO_CHUNKS:
        #    end_of_video = True
        #    self.video_pointer = True

        #return mean_free_capacity, \
            #mean_free_capacity_frac, \
        #    mean_loss_rate, \
            #mean_loss_rate_frac, \
        #    end_of_video
    return profile, pMOS, usage_CPU , end_of_video

def model_summary():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    total_reward = tf.Variable(0.)
    tf.summary.scalar("Total Reward", total_reward)
    #avg_entropy = tf.Variable(0.)
    #tf.summary.scalar("Avg_entropy", avg_entropy)

    model_vars = [td_loss, total_reward]
    model_ops = tf.summary.merge_all()

    return model_vars, model_ops


def compute_gradients(states_matrix, actions_matrix, rewards_matrix, terminal, actor_net, critic_net):
    assert states_matrix.shape[0] == actions_matrix.shape[0]
    assert states_matrix.shape[0] == rewards_matrix.shape[0]

    ba_size = states_matrix.shape[0]

    v_matrix = critic_net.predict(states_matrix)

    R_matrix = np.zeros(rewards_matrix.shape)

    if terminal:
        R_matrix[-1, 0] = 0  # terminal state
    else:
        R_matrix[-1, 0] = v_matrix[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_matrix[t, 0] = rewards_matrix[t] + GAMMA * R_matrix[t + 1, 0]

    td_matrix = R_matrix - v_matrix

    actor_gradients = actor_net.get_gradients(states_matrix, actions_matrix, td_matrix)
    critic_gradients = critic_net.get_gradients(states_matrix, R_matrix)

    return actor_gradients, critic_gradients, td_matrix

def compute_entropy(x):
    H = 0.0
    for i in range(len(x)):
        if 0 < x[i] < 1:
            H -= x[i] * np.log(x[i])
    return H