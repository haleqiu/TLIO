import numpy as np
import os
import json
import math

with open("imu0_resampled_description.json", "r") as file:
    data = json.load(file)

imu0_resampled_description = {
        "columns_name(width)": [
            "ts_us(1)",
            "gyr_compensated_rotated_in_World(3)",
            "acc_compensated_rotated_in_World(3)",
            "qxyzw_World_Device(4)",
            "pos_World_Device(3)",
            "vel_World(3)"
        ],
        "num_rows": int(data["num_rows"]),
        "approximate_frequency_hz": 200.0,
        "t_start_us": float(data["t_start_us"]),
        "t_end_us": float(data["t_end_us"])
    }

with open("imu0_resampled_description.json", "w") as file:
    json.dump(imu0_resampled_description, file, indent=4)