"""
Pytorch dataloader for FB dataset
"""

import random, os
from abc import ABC, abstractmethod

import torch
import pypose as pp
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class EurocSequence():
    """
    Output:
    acce: the accelaration in **world frame**
    """
    def __init__(self, data_path, intepolate = False, calib = False, load_vicon = False, glob = False, **kwargs):
        super(EurocSequence, self).__init__()
        (
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (dict(), None, None, None, None, None)

        self.vicon_ext_R =  np.array([[0.33638, -0.01749,  0.94156],[-0.02078, -0.99972, -0.01114],[0.94150, -0.01582, -0.33665]])
        self.vicon_ext_T =  np.array([0.06901, -0.02781,-0.12395])
        self.ext_R = pp.mat2SO3(self.vicon_ext_R, check= False)
        self.ext_t = torch.tensor(self.vicon_ext_T)

        self.load_imu(data_path)
        self.load_gt(data_path)
        if load_vicon:
            self.load_vicon(data_path)
        
        # inteporlate the ground truth pose
        if intepolate:
            self.data["gt_orientation"] = self.interp_rot(self.data['time'], self.data['gt_time'], self.data['pose'][:,3:])
            self.data["gt_translation"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data['pose'][:,:3])
            self.data["b_acc"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["b_acc"])
            self.data["b_gyro"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["b_gyro"])
            self.data["velocity"] = self.interp_xyz(self.data['time'], self.data['gt_time'], self.data["velocity"])
        else:
            self.data["gt_orientation"] = pp.SO3(torch.tensor(self.data['pose'][:,3:]))
            self.data['gt_translation'] = torch.tensor(self.data['pose'][:,:3])
        
        # move the time to torch
        self.data["time"] = torch.tensor(self.data["time"])
        self.data['dt'] = (self.data["time"][1:] - self.data["time"][:-1])[:,None]

        if calib == "head":
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - self.data["b_gyro"][0]
            self.data["acc"] = torch.tensor(self.data["acc"]) - self.data["b_acc"][0]
        elif calib == "debug":
            ave_acc_b = torch.tensor([-0.014258, 0.104451, 0.076776])
            ave_gyro_b = torch.tensor([0.002153, 0.020744, 0.075806])
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - ave_gyro_b
            self.data["acc"] = torch.tensor(self.data["acc"]) - ave_acc_b
        elif calib == "full":
            self.data["gyro"] = torch.tensor(self.data["gyro"]) - self.data["b_gyro"]
            self.data["acc"] = torch.tensor(self.data["acc"]) - self.data["b_acc"]
        else:
            print("Invalid calibration type....")
            self.data["gyro"] = torch.tensor(self.data["gyro"])
            self.data["acc"] = torch.tensor(self.data["acc"])
        
        # change the acc and gyro scope into the global coordinate.  
        if glob:
            self.data['acc'] = self.data["gt_orientation"] * self.data['acc']
            self.data['gyro'] = self.data["gt_orientation"] * self.data['gyro']
            print("global coordinate")

        print("loaded: ", data_path, "calib: ", calib)

    def get_length(self):
        return self.data['time'].shape[0]

    def load_imu(self, folder):
        imu_data = np.loadtxt(os.path.join(folder, "mav0/imu0/data.csv"), dtype=float, delimiter=',')
        self.data["time"] = imu_data[:,0] * 1e-9
        self.data["gyro"] = imu_data[:,1:4] # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.data["acc"] = imu_data[:,4:]# acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self, folder):
        gt_data = np.loadtxt(os.path.join(folder, "mav0/state_groundtruth_estimate0/data.csv"), dtype=float, delimiter=',')
        self.data["gt_time"] = gt_data[:,0] * 1e-9
        self.data["pose"] = np.zeros([self.data["gt_time"].shape[0],7])
        self.data["pose"][:,:3] = gt_data[:,1:4]

        quat =  gt_data[:,4:8] # w, x, y, z
        self.data["pose"][:,3:6] = quat[:,1:] 
        self.data["pose"][:,6] = quat[:,0] #set xyzw

        self.data["b_acc"] = gt_data[:,-3:]
        self.data["b_gyro"] = gt_data[:,-6:-3]
        self.data["velocity"] = gt_data[:,-9:-6]

    def load_vicon(self, folder):
        vicon_data = np.loadtxt(os.path.join(folder, "mav0/vicon0/data.csv"), dtype=float, delimiter=',')
        self.data["opti_time"] = vicon_data[:,0] * 1e-9
        self.data["vicon_pose"] = np.zeros([self.data["opti_time"].shape[0],7])
        self.data["vicon_pose"][:,:3] = vicon_data[:,1:4]

        quat =  vicon_data[:,4:8] # w, x, y, z
        self.data["vicon_pose"][:,3:6] = quat[:,1:] 
        self.data["vicon_pose"][:,6] = quat[:,0] #set xyzw

    def interp_rot(self, time, opt_time, r):
        # interpolation in the log space
        so3 = pp.SO3(torch.tensor(r)).Log().data.numpy()

        # torch have no interp
        intep_so3_x = np.interp(time, xp=opt_time, fp = so3[:,0])
        intep_so3_y = np.interp(time, xp=opt_time, fp = so3[:,1])
        intep_so3_z = np.interp(time, xp=opt_time, fp = so3[:,2])
        intep_so3 = np.stack([intep_so3_x, intep_so3_y, intep_so3_z]).transpose()
        intep_so3 = pp.so3(torch.tensor(intep_so3))

        return intep_so3.Exp()

    def interp_xyz(self, time, opt_time, xyz):
        
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)


class EurocDataset(Dataset):
    def __init__(self, root_dir, data_list, args, data_window_config, **kwargs):
        super().__init__()

        self.window_size = data_window_config["window_size"]
        self.past_data_size = data_window_config["past_data_size"]
        self.future_data_size = data_window_config["future_data_size"]
        self.step_size = data_window_config["step_size"]

        self.do_bias_shift = args.do_bias_shift
        self.accel_bias_range = args.accel_bias_range
        self.gyro_bias_range = args.gyro_bias_range
        self.perturb_gravity = args.perturb_gravity
        self.perturb_gravity_theta_range = args.perturb_gravity_theta_range

        print("do bias shift: ", self.do_bias_shift, " gravity: ", self.perturb_gravity)

        self.mode = kwargs.get("mode", "train")
        self.shuffle, self.transform = False, False
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
            seq = EurocSequence(
                os.path.join(root_dir, data_list[i]), intepolate = True, glob=True, **data_window_config, **kwargs
            )
            if "time" in seq.data.keys():
                self.ts.append(seq.data["time"].numpy())
            self.features.append(torch.cat([seq.data["gyro"], seq.data["acc"]], dim = -1))

            targets = seq.data["gt_translation"][self.window_size :] - seq.data["gt_translation"][: -self.window_size]
            self.targets.append(targets)
            self.gt_ori.append(seq.data["gt_orientation"])
            self.orientations.append(seq.data["gt_orientation"])# Temporary
            self.gt_pos.append(seq.data["gt_translation"].numpy())
            
            self.index_map += [
                [i, j]
                for j in range(
                    0 + self.past_data_size,
                    targets.shape[0] - self.future_data_size,
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

        return feat.T.float(), targ.float(), seq_id, frame_id

    def __len__(self):
        return len(self.index_map)
