import numpy as np
import os
import json
import math


def interp_xyz(time, opt_time, xyz):
    interp_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
    interp_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
    interp_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

    interp_xyz = np.stack([interp_x, interp_y, interp_z]).transpose()
    return interp_xyz


def interp_quat(time, opt_time, xyzw):
    interp_x = np.interp(time, xp=opt_time, fp=xyzw[:, 0])
    interp_y = np.interp(time, xp=opt_time, fp=xyzw[:, 1])
    interp_z = np.interp(time, xp=opt_time, fp=xyzw[:, 2])
    interp_w = np.interp(time, xp=opt_time, fp=xyzw[:, 3])

    interp_xyzw = np.stack([interp_x, interp_y, interp_z, interp_w]).transpose()
    return interp_xyzw


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--EuRoC_path', type=str, default='local_data/EuRoc_dataset',
                        help='path to the EuRoC dataset')
    parser.add_argument('--transformed_path', type=str, default='local_data/EuRoc_transformed',
                        help='path to the transformed EuRoC dataset')
    args = parser.parse_args()

    EuRoc_path = args.EuRoC_path
    transformed_path = args.transformed_path

    if not os.path.exists(transformed_path):
        os.makedirs(transformed_path)

    train_list = open("train_list.txt", 'w')
    train_list.write("MH_01" + '\n'
                     + "MH_03" + '\n'
                     + "MH_05" + '\n'
                     + "V1_01_easy" + '\n'
                     + "V1_03_difficult" + '\n'
                     + "V2_01_easy" + '\n'
                     + "V2_03_difficult" + '\n')
    train_list.close()
    test_list = open("test_list.txt", 'w')
    test_list.write("MH_04"+'\n'+'V1_02_medium'+'\n')
    test_list.close()
    val_list = open("val_list.txt", 'w')
    val_list.write("MH_02"+'\n'+'V2_02_medium'+'\n')

    for root, subsets, files in os.walk(EuRoc_path):

        for subset in subsets:

            if not subset.startswith("MH") and not subset.startswith("V"):
                continue

            subset_path = os.path.join(EuRoc_path, subset)

            if not os.path.exists(os.path.join(transformed_path, subset)):
                os.makedirs(os.path.join(transformed_path, subset))

            raw_data_path = os.path.join(subset_path, "mav0", "imu0", "data.csv")
            raw_IMUdata = np.loadtxt(raw_data_path, delimiter=',')

            raw_rows = np.size(raw_IMUdata, axis=0)
            raw_cols = np.size(raw_IMUdata, axis=1)

            imu0_samples_rows = raw_rows
            imu0_samples_cols = raw_cols + 1

            imu0_samples = np.zeros((imu0_samples_rows, imu0_samples_cols), dtype=np.float64)
            imu0_samples[:, 0] = raw_IMUdata[:, 0]
            imu0_samples[:, 1] = 0
            imu0_samples[:, 2:imu0_samples_cols] = raw_IMUdata[:, 1:raw_cols]
            np.savetxt(os.path.join(transformed_path, subset, "imu_samples_0.csv"), imu0_samples, delimiter=",",
                       header="timestamp [ns], temperature [degC], w_RS_S_x [rad s^-1], w_RS_S_y [rad s^-1],"
                              " w_RS_S_z [rad s^-1], a_RS_S_x [m s^-2], a_RS_S_y [m s^-2], a_RS_S_z [m s^-2]")

            gt_file_path = os.path.join(subset_path, "mav0", 'state_groundtruth_estimate0', 'data.csv')
            gt_data = np.loadtxt(gt_file_path, delimiter=',')

            t_start = np.max([raw_IMUdata[0, 0], gt_data[0, 0]])
            t_end = np.min([raw_IMUdata[-1, 0], gt_data[-1, 0]])

            index_t_start_raw = np.searchsorted(raw_IMUdata[:, 0], t_start)
            index_t_end_raw = np.searchsorted(raw_IMUdata[:, 0], t_end, side='right')

            index_t_start_gt = np.searchsorted(gt_data[:, 0], t_start)
            index_t_end_gt = np.searchsorted(gt_data[:, 0], t_end, 'right')

            calibrated_rows_gt = index_t_end_gt - index_t_start_gt
            calibrated_rows_raw = index_t_end_raw - index_t_start_raw
            calibrated_cols = 17

            imu0_resampled = np.zeros((calibrated_rows_gt, calibrated_cols), dtype=np.float64)

            imu0_resampled[:, 0] = np.trunc(gt_data[index_t_start_gt:index_t_end_gt, 0] / 1e3)
            if calibrated_rows_raw == calibrated_rows_gt:
                imu0_resampled[:, 1:4] = raw_IMUdata[index_t_start_raw:index_t_end_raw, 1:4]
                imu0_resampled[:, 4:7] = raw_IMUdata[index_t_start_raw:index_t_end_raw, 4:7]
            else:
                imu0_resampled[:, 1:4] = interp_xyz(gt_data[index_t_start_gt:index_t_end_gt, 0],
                                                    raw_IMUdata[index_t_start_raw:index_t_end_raw, 0],
                                                    raw_IMUdata[index_t_start_raw:index_t_end_raw, 1:4])
                imu0_resampled[:, 4:7] = interp_xyz(gt_data[index_t_start_gt:index_t_end_gt, 0],
                                                    raw_IMUdata[index_t_start_raw:index_t_end_raw, 0],
                                                    raw_IMUdata[index_t_start_raw:index_t_end_raw, 4:7])
            imu0_resampled[:, 7:10] = gt_data[:, 5:8]
            imu0_resampled[:, 10] = gt_data[:, 4]
            imu0_resampled[:, 11:14] = gt_data[:, 1:4]
            imu0_resampled[:, 14:17] = gt_data[:, 8:11]

            np.save(os.path.join(transformed_path, subset, "imu0_resampled.npy"), imu0_resampled)

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
                "t_start_us": float(imu0_resampled[0, 0]),
                "t_end_us": float(imu0_resampled[-1, 0])
            }

            with open(os.path.join(transformed_path, subset, "imu0_resampled_description.json"), "w") as file:
                json.dump(imu0_resampled_description, file, indent=4)

            with open(os.path.join(transformed_path, subset, "calibration.json"), "w") as file:
                calibration_info = {
                    "Accelerometer": {
                        "Bias": {
                            "Name": "Constant",
                            "Offset": [
                                0.0,
                                0.0,
                                0.0
                            ]
                        },
                        "Model": {
                            "Name": "Linear",
                            "RectificationMatrix": [
                                [
                                    1.0,
                                    0.0,
                                    0.0
                                ],
                                [
                                    0.0,
                                    1.0,
                                    0.0
                                ],
                                [
                                    0.0,
                                    0.0,
                                    1.0
                                ]
                            ]
                        },
                        "TimeOffsetSec_Device_Accel": 0.0
                    },
                    "Calibrated": True,
                    "Gyroscope": {
                        "Bias": {
                            "Name": "Constant",
                            "Offset": [
                                0.0,
                                0.0,
                                0.0
                            ]
                        },
                        "Model": {
                            "Name": "Linear",
                            "RectificationMatrix": [
                                [
                                    1.0,
                                    0.0,
                                    0.0
                                ],
                                [
                                    0.0,
                                    1.0,
                                    0.0
                                ],
                                [
                                    0.0,
                                    0.0,
                                    1.0
                                ]
                            ]
                        },
                        "TimeOffsetSec_Device_Gyro": 0.0
                    },
                    "Label": "unlabeled_imu_0",
                    "SerialNumber": "rift://",
                    "T_Device_Imu": {
                        "Translation": [
                            0.0,
                            0.0,
                            0.0
                        ],
                        "UnitQuaternion": [
                            1.0,
                            [
                                0.0,
                                0.0,
                                0.0
                            ]
                        ]
                    }
                }

                json.dump(calibration_info, file, indent=4)
