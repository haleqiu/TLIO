import os.path

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    filter_outputs_folder = "filter_outputs/EuRoc_dataset"
    visualization_outputs_folder = "visualization_of_EKF"
    transformed_dataset_path = "local_data/EuRoc_transformed"

    for root, dirs, files in os.walk(filter_outputs_folder):

        if not os.path.exists(visualization_outputs_folder):
            os.makedirs(visualization_outputs_folder)

        for subset in dirs:

            if not os.path.exists(os.path.join(root, subset)):
                os.makedirs(os.path.join(root, subset))

            filter_data_path = os.path.join(root, subset, "not_vio_state.txt.npy")
            filter_data = np.load(filter_data_path)

            ground_truth_path = os.path.join(transformed_dataset_path, subset, "imu0_resampled.npy")
            ground_truth = np.load(ground_truth_path)

            filter_t = filter_data[:, 27]
            filter_p_x = filter_data[:, 12]
            filter_p_y = filter_data[:, 13]
            filter_p_z = filter_data[:, 14]

            ground_truth_p_x = ground_truth[:, -6]
            ground_truth_p_y = ground_truth[:, -5]
            ground_truth_p_z = ground_truth[:, -4]

            filter_t_stamp = np.array(range(0, filter_data.shape[0]))

            if not os.path.exists(os.path.join(visualization_outputs_folder, subset)):
                os.makedirs(os.path.join(visualization_outputs_folder, subset))

            ax0 = plt.figure().add_subplot(projection='3d')
            ax0.plot(xs=filter_p_x, ys=filter_p_y, zs=filter_p_z, c='b', label='filter_result'
                     , linewidth=0.5, linestyle='solid')
            ax0.plot(xs=ground_truth_p_x, ys=ground_truth_p_y, zs=ground_truth_p_z, c='r', label='ground_truth'
                     , linewidth=0.5, linestyle='solid')
            plt.legend(loc='best')
            plt.savefig(os.path.join(visualization_outputs_folder, subset, 'gt_and_filter.png'), dpi=1000)

            ax1 = plt.figure().add_subplot(projection='3d')
            ax1.plot(xs=filter_p_x, ys=filter_p_y, zs=filter_p_z, c='b', label='filter_result'
                     , linewidth=0.5, linestyle='solid')
            plt.legend(loc='best')
            plt.savefig(os.path.join(visualization_outputs_folder, subset, 'filter_result.png'), dpi=1000)

            ax2 = plt.figure().add_subplot(projection='3d')
            ax2.plot(xs=ground_truth_p_x, ys=ground_truth_p_y, zs=ground_truth_p_z, c='r', label='ground_truth'
                     , linewidth=0.5, linestyle='solid')
            plt.legend(loc='best')
            plt.savefig(os.path.join(visualization_outputs_folder, subset, 'ground_truth.png'), dpi=1000)

