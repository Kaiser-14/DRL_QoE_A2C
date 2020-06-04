import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# filename = './results/Useful_Training/2020-04-11-12:09:13/metrics_agent_1'
filename = './metrics_agent_1'
# time, reward, max_br, br_in, br_out, resolution, mos, br_bg, action = np.loadtxt(filename, unpack=True)
data = pd.read_csv(filename, sep="\t")

data.columns = ['time', 'reward', 'max_br', 'br_in', 'br_out', 'resolution', 'mos', 'br_bg', 'action']

print('Length of the dataset: {}'.format(data.shape))

print(data.head())

"""Histogram of profiles"""

data['action'].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.xlabel('Actions')
plt.show()

data['action'].iloc[-100:].value_counts().sort_index(ascending=True).plot(kind='bar')
plt.xlabel('Actions')
plt.show()

"""Histogram of MOS"""

plt.hist(data['mos'], bins=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
plt.xlabel('MOS')
plt.show()

plt.hist(data['mos'].iloc[-100:], bins=[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
plt.xlabel('MOS')
plt.show()

"""Histogram of Rewards"""

plt.hist(data['reward'])
plt.xlabel('Reward')
plt.show()

plt.hist(data['reward'].iloc[-5000:])
plt.xlabel('Reward')
plt.show()