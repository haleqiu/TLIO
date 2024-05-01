import os.path

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

filter_data_path = "not_vio_state.txt.npy"
filter_data = np.load(filter_data_path)
print(filter_data.shape)
print(filter_data)

ground_truth_path = "imu0_resampled.npy"
ground_truth = np.load(ground_truth_path)
print(ground_truth.shape)
print(ground_truth)

filter_t = filter_data[:, 27]
print(filter_t)
filter_p_x = filter_data[:, 12]
filter_p_y = filter_data[:, 13]
filter_p_z = filter_data[:, 14]

ground_truth_p_x = ground_truth[:, -6]
ground_truth_p_y = ground_truth[:, -5]
ground_truth_p_z = ground_truth[:, -4]

filter_t_stamp = np.array(range(0, filter_data.shape[0]))
print(filter_t_stamp.shape)
print(filter_t_stamp)

ax0 = plt.figure().add_subplot(projection='3d')
ax0.plot(xs=filter_p_x, ys=filter_p_y, zs=filter_p_z, c='b', label='filter_result'
         , linewidth=0.5, linestyle='solid')
ax0.plot(xs=ground_truth_p_x, ys=ground_truth_p_y, zs=ground_truth_p_z, c='r', label='ground_truth'
         , linewidth=0.5, linestyle='solid')
plt.legend(loc='best')
plt.savefig('gt_and_filter.png', dpi=1000)

ax1 = plt.figure().add_subplot(projection='3d')
ax1.plot(xs=filter_p_x, ys=filter_p_y, zs=filter_p_z, c='b', label='filter_result'
         , linewidth=0.5, linestyle='solid')
plt.legend(loc='best')
plt.savefig('filter_result.png', dpi=1000)

ax2 = plt.figure().add_subplot(projection='3d')
ax2.plot(xs=ground_truth_p_x, ys=ground_truth_p_y, zs=ground_truth_p_z, c='r', label='ground_truth'
         , linewidth=0.5, linestyle='solid')
plt.legend(loc='best')
plt.savefig('ground_truth.png', dpi=1000)

plt.show()
