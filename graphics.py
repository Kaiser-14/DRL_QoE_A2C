import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = './results/Useful_Training/2020-04-11-12:09:13/metrics_agent_1'
# time, reward, max_br, br_in, br_out, resolution, mos, br_bg, action = np.loadtxt(filename, unpack=True)
data = pd.read_csv(filename, sep="\t", header=None)

data.columns = ['time', 'reward', 'max_br', 'br_in', 'br_out', 'resolution', 'mos', 'br_bg', 'action']

# Info about data
print('Length of the dataset: {}'.format(data.shape))

# print(data['time'].iloc[:27])

# print(data['action'].value_counts())

plt.bar(range(data['action'].value_counts()), list(char_count.values()), align='center')
plt.xticks(range(len(char_count)), list(char_count.keys()), rotation=90)
plt.ylabel('counts')
plt.show()



# histo = data['mos'].hist()
# plt.show()
#
# histo = data['action'].hist()
# plt.show()
#
# histo = data['reward'].hist()
# plt.show()





# time = float(time.strftime('%Y-%m-%d-%H:%M:%S'))

# plt.plot(time, br_out,  ls='--', c='r', lw=0.5, label='br_out')
# plt.plot(time, mos, ls='--', c='g', lw=0.5, label='mos')
# x1,x2,y1,y2 = plt.axis()
# # plt.axis([0, x2, 0, y2])
# plt.xlabel('Videos (#)')
# plt.ylabel('Rates (Kbps)')
# plt.title('BitRate and LossRate', fontsize=14, color='black')
# # plt.yscale('log')
# plt.grid(True)
# plt.legend()
# plt.show()
