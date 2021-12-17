"""
Pytorch dataloader for FB dataset
"""

import random
from abc import ABC, abstractmethod
from os import path as osp

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class SubtSequence():
    def __init__(self, data_path, args, data_window_config, **kwargs):
        super(SubtSequence, self).__init__()
        (
            self.ts,
            self.features,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (None, None, None, None, None, None)

        self.target_dim = args.output_dim
        self.imu_freq = args.imu_freq
        self.imu_base_freq = args.imu_base_freq
        self.interval = data_window_config["window_size"]
        self.mode = kwargs.get("mode", "train")

        if data_path is not None:
            self.load(data_path)

    def load(self, data_path):

        with h5py.File(osp.join(data_path, "data.hdf5"), "r") as f:
            ts = np.copy(f["ts"])
            int_q = np.copy(f["integrated_q_wxyz"])
            int_p = np.copy(f["integrated_p"])
            gyro = np.copy(f["gyro_dcalibrated"])
            accel = np.copy(f["accel_dcalibrated"])
            ### For comparision 
            # integ_q = np.copy(f["integration_q_wxyz"])
            # filter_q = np.copy(f["filter_q_wxyz"])

        # subsample from IMU base rate:
        subsample_factor = int(np.around(self.imu_base_freq / self.imu_freq)) #500/200
        ts = ts[::subsample_factor]
        int_q = int_q[::subsample_factor, :]
        int_p = int_p[::subsample_factor, :]
        gyro = gyro[::subsample_factor, :]
        acce = accel[::subsample_factor, :]
        # integ_q = integ_q[::subsample_factor,:]
        # filter_q = filter_q[::subsample_factor, :]

        # ground truth displacement
        gt_disp = int_p[self.interval :] - int_p[: -self.interval]

        # rotation in the world frame in quaternions
        ## TODO: save it
        ori_R_int = Rotation.from_quat(int_q) ### N * (x y z w)
        ori_R_imu = Rotation.from_quat(int_q)
        if self.mode in ["train", "val"]:
            ori_R = ori_R_int
        elif self.mode in ["test", "eval"]:
            ori_R = ori_R_imu
        # TODO: setup the test and eval set, try the ground truth init testing.
        # elif self.mode in ["test", "eval"]:
        #     ori_R = Rotation.from_quat(filter_q[:, [1, 2, 3, 0]])
        #     ori_R_vio_z = Rotation.from_euler("z", ori_R_vio.as_euler("xyz")[0, 2]) # gt rotation
        #     ori_R_z = Rotation.from_euler("z", ori_R.as_euler("xyz")[0, 2]) # filter z
        #     dRz = ori_R_vio_z * ori_R_z.inv()
        #     ori_R = dRz * ori_R

        # in the world coordinate
        glob_gyro = np.einsum("tip,tp->ti", ori_R.as_matrix(), gyro)
        glob_acce = np.einsum("tip,tp->ti", ori_R.as_matrix(), acce)

        self.ts = ts  # ts of the beginning of each window
        self.features = np.concatenate([glob_gyro, glob_acce], axis=1)
        self.orientations = ori_R.as_quat()
        self.gt_pos = int_p
        self.gt_ori = ori_R_int.as_quat()
        # disp from the beginning to + interval
        # does not have the last interval of data
        self.targets = gt_disp[:, : self.target_dim]

    def get_feature(self):
        return self.features

    def get_target(self):
        ## 3d displacement
        return self.targets

    def get_aux(self):
        return np.concatenate(
            [self.ts[:, None], self.orientations, self.gt_pos, self.gt_ori], axis=1
        )


class SubtSequneceDataset(Dataset):
    def __init__(self, root_dir, data_list, args, data_window_config, **kwargs):
        super(SubtSequneceDataset, self).__init__()

        self.window_size = data_window_config["window_size"]
        self.past_data_size = data_window_config["past_data_size"]
        self.future_data_size = data_window_config["future_data_size"]
        self.step_size = data_window_config["step_size"]

        self.do_bias_shift = args.do_bias_shift
        self.accel_bias_range = args.accel_bias_range
        self.gyro_bias_range = args.gyro_bias_range
        self.perturb_gravity = args.perturb_gravity
        self.perturb_gravity_theta_range = args.perturb_gravity_theta_range

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform = False, False ## what's the point
        if self.mode == "train":
            self.shuffle = True
            self.transform = True
        elif self.mode == "val":
            self.shuffle = True
        elif self.mode == "test":
            self.shuffle = False
        elif self.mode == "eval":
            self.shuffle = False

        self.index_map = []
        self.ts, self.orientations, self.gt_pos, self.gt_ori = [], [], [], []
        self.features, self.targets = [], []
        for i in range(len(data_list)):
            seq = SubtSequence(
                osp.join(root_dir, data_list[i]), args, data_window_config, **kwargs
            )
            feat, targ, aux = seq.get_feature(), seq.get_target(), seq.get_aux()
            self.features.append(feat)
            self.targets.append(targ)
            self.ts.append(aux[:, 0])
            self.orientations.append(aux[:, 1:5])
            self.gt_pos.append(aux[:, 5:8])
            self.gt_ori.append(aux[:, 8:12])
            self.index_map += [
                [i, j]
                for j in range(
                    0 + self.past_data_size,
                    self.targets[i].shape[0] - self.future_data_size,
                    self.step_size,
                )
            ]

        if self.shuffle:
            random.shuffle(self.index_map)

    def __getitem__(self, item):
        seq_id, frame_id = self.index_map[item][0], self.index_map[item][1]

        # in the world frame
        feat = self.features[seq_id][
            frame_id
            - self.past_data_size : frame_id
            + self.window_size
            + self.future_data_size
        ]
        targ = self.targets[seq_id][frame_id]  # the beginning of the sequence

        if self.mode in ["train", "eval"]:
            ## transform, random transform the yaw plane
            if self.transform:
                # rotate in the yaw plane
                angle = np.random.random() * (2 * np.pi)
                rm = np.array(
                    [[np.cos(angle), -(np.sin(angle))], [np.sin(angle), np.cos(angle)]]
                )
                feat_aug = np.copy(feat)
                targ_aug = np.copy(targ)
                feat_aug[:, 0:2] = np.matmul(rm, feat[:, 0:2].T).T
                feat_aug[:, 3:5] = np.matmul(rm, feat[:, 3:5].T).T
                targ_aug[0:2] = np.matmul(rm, targ[0:2].T).T
                feat = feat_aug
                targ = targ_aug
            if self.do_bias_shift:
                # shift in the accel and gyro bias terms
                random_bias = [
                    (random.random() - 0.5) * self.accel_bias_range / 0.5,
                    (random.random() - 0.5) * self.accel_bias_range / 0.5,
                    (random.random() - 0.5) * self.accel_bias_range / 0.5,
                    (random.random() - 0.5) * self.gyro_bias_range / 0.5,
                    (random.random() - 0.5) * self.gyro_bias_range / 0.5,
                    (random.random() - 0.5) * self.gyro_bias_range / 0.5,
                ]
                feat[:, 0] += random_bias[0]
                feat[:, 1] += random_bias[1]
                feat[:, 2] += random_bias[2]
                feat[:, 3] += random_bias[3]
                feat[:, 4] += random_bias[4]
                feat[:, 5] += random_bias[5]
            if self.perturb_gravity:
                # get rotation vector of random horizontal direction
                angle_rand = random.random() * np.pi * 2
                vec_rand = np.array([np.cos(angle_rand), np.sin(angle_rand), 0])
                theta_rand = (
                    random.random() * np.pi * self.perturb_gravity_theta_range / 180.0
                )
                rvec = theta_rand * vec_rand
                r = Rotation.from_rotvec(rvec)
                R_mat = r.as_matrix()
                feat[:, 0:3] = np.matmul(R_mat, feat[:, 0:3].T).T
                feat[:, 3:6] = np.matmul(R_mat, feat[:, 3:6].T).T

        return feat.astype(np.float32).T, targ.astype(np.float32), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)