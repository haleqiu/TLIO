import numpy as np
import os
import json
import math


def interp_xyz(time, opt_time, xyz):
    intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
    intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
    intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return inte_xyz


def interp_quat(time, opt_time, xyzw):
    intep_x = np.interp(time, xp=opt_time, fp=xyzw[:, 0])
    intep_y = np.interp(time, xp=opt_time, fp=xyzw[:, 1])
    intep_z = np.interp(time, xp=opt_time, fp=xyzw[:, 2])
    intep_w = np.interp(time, xp=opt_time, fp=xyzw[:, 3])

    inte_xyzw = np.stack([intep_x, intep_y, intep_z, intep_w]).transpose()
    return inte_xyzw


raw_data_path = os.path.join('imu0', 'data.csv')
raw_IMUdata = np.loadtxt(raw_data_path, delimiter=',')
print(raw_IMUdata)
print(raw_IMUdata.shape)
print(type(raw_IMUdata))
raw_rows = np.size(raw_IMUdata, axis=0)
raw_cols = np.size(raw_IMUdata, axis=1)

imu0_samples_rows = raw_rows
imu0_samples_cols = raw_cols + 1

imu0_samples = np.zeros((imu0_samples_rows, imu0_samples_cols), dtype=np.float64)
imu0_samples[:, 0] = raw_IMUdata[:, 0]
imu0_samples[:, 1] = 0
imu0_samples[:, 2:imu0_samples_cols] = raw_IMUdata[:, 1:raw_cols]
np.savetxt("imu_samples_0.csv", imu0_samples, delimiter=",")

gt_file_path = os.path.join('state_groundtruth_estimate0', 'data.csv')
gt_data = np.loadtxt(gt_file_path, delimiter=',')
calib_path = "calibration.json"
with open(calib_path) as f:
    calib = json.load(f)
acc_bias = calib['Accelerometer']['Bias']['Offset']
gyro_bias = calib['Gyroscope']['Bias']['Offset']

t_start = np.max([raw_IMUdata[0, 0], gt_data[0, 0]])
t_end = np.min([raw_IMUdata[-1, 0], gt_data[-1, 0]])

idex_t_start_raw = np.searchsorted(raw_IMUdata[:, 0], t_start)
idex_t_end_raw = np.searchsorted(raw_IMUdata[:, 0], t_end, side='right')

idex_t_start_gt = np.searchsorted(gt_data[:, 0], t_start)
idex_t_end_gt = np.searchsorted(gt_data[:, 0], t_end, 'right')

calibrated_rows_gt = idex_t_end_gt - idex_t_start_gt
calibrated_rows_raw = idex_t_end_raw - idex_t_start_raw
calibrated_cols = 17
if calibrated_rows_raw == calibrated_rows_gt:

    imu0_resampled = np.zeros((calibrated_rows_gt, calibrated_cols), dtype=np.float64)

    imu0_resampled[:, 0] = np.trunc(gt_data[idex_t_start_gt:idex_t_end_gt, 0] / 1e3)
    imu0_resampled[:, 1:4] = raw_IMUdata[idex_t_start_raw:idex_t_end_raw, 1:4] + gyro_bias
    imu0_resampled[:, 4:7] = raw_IMUdata[idex_t_start_raw:idex_t_end_raw, 4:7] + acc_bias
    imu0_resampled[:, 7:10] = gt_data[:, 5:8]
    imu0_resampled[:, 10] = gt_data[:, 4]
    imu0_resampled[:, 11:14] = gt_data[:, 1:4]
    imu0_resampled[:, 14:17] = gt_data[:, 8:11]

    np.save("imu0_resampled.npy", imu0_resampled)

    imu0_resampled_description = {
        "columns_name(width)": [
            "ts_us(1)",
            "gyr_compensated_rotated_in_World(3)",
            "acc_compensated_rotated_in_World(3)",
            "qxyzw_World_Device(4)",
            "pos_World_Device(3)",
            "vel_World(3)"
        ],
        "num_rows": calibrated_rows_gt.item(),
        "approximate_frequency_hz": 200.0,
        "t_start_us": imu0_resampled[0, 0],
        "t_end_us": imu0_resampled[-1, 0]
    }

else:
    imu0_resampled = np.zeros((calibrated_rows_gt, calibrated_cols), dtype=np.float64)

    imu0_resampled[:, 0] = np.trunc(gt_data[idex_t_start_gt:idex_t_end_gt, 0] / 1e3)
    imu0_resampled[:, 1:4] = interp_xyz(gt_data[idex_t_start_gt:idex_t_end_gt, 0],
                                        raw_IMUdata[idex_t_start_raw:idex_t_end_raw, 0],
                                        raw_IMUdata[idex_t_start_raw:idex_t_end_raw, 1:4] + gyro_bias)
    imu0_resampled[:, 4:7] = interp_xyz(gt_data[idex_t_start_gt:idex_t_end_gt, 0],
                                        raw_IMUdata[idex_t_start_raw:idex_t_end_raw, 0],
                                        raw_IMUdata[idex_t_start_raw:idex_t_end_raw, 4:7] + acc_bias)
    imu0_resampled[:, 7:10] = gt_data[idex_t_start_gt:idex_t_end_gt, 5:8]
    imu0_resampled[:, 10] = gt_data[idex_t_start_gt:idex_t_end_gt, 4]
    imu0_resampled[:, 11:14] = gt_data[idex_t_start_gt:idex_t_end_gt, 1:4]
    imu0_resampled[:, 14:17] = gt_data[idex_t_start_gt:idex_t_end_gt, 8:11]

    np.save("imu0_resampled.npy", imu0_resampled)

    imu0_resampled_description = {
        "columns_name(width)": [
            "ts_us(1)",
            "gyr_compensated_rotated_in_World(3)",
            "acc_compensated_rotated_in_World(3)",
            "qxyzw_World_Device(4)",
            "pos_World_Device(3)",
            "vel_World(3)"
        ],
        "num_rows": calibrated_rows_gt.item(),
        "approximate_frequency_hz": 200.0,
        "t_start_us": imu0_resampled[0, 0],
        "t_end_us": imu0_resampled[-1, 0]
    }

with open("imu0_resampled_description.json", "w") as file:
    json.dump(imu0_resampled_description, file, indent=4)
