#!/usr/bin/env python3
"""
Generate GPS-like data for a flock of sheep.
One sheep at a time walks out and back, others stay near base.
Output: CSV file 'sheep_gps_data.csv' WITHOUT phase or distance.
"""

import numpy as np
import pandas as pd
import math
from math import cos, radians, ceil

# ---------------- PARAMETERS ----------------
np.random.seed(42)

num_sheep = 10
interval_seconds = 30
step_m_per_interval = 50
max_distance_m = 2000
stay_seconds = 120
jitter_std_m = 3
active_jitter_m = 2
num_cycles = 1

# Base location
base_lat = 31.0000
base_lon = -7.0000
start_time = pd.Timestamp("2025-08-27 10:00:00")
# ---------------------------------------------------------

def m_to_deg_lat(meters: float) -> float:
    return meters / 111_320.0

def m_to_deg_lon(meters: float, lat_deg: float) -> float:
    return meters / (111_320.0 * cos(radians(lat_deg)))

travel_out_frames = int(ceil(max_distance_m / step_m_per_interval))
stay_frames = int(ceil(stay_seconds / interval_seconds))
frames_per_sheep = travel_out_frames + stay_frames + travel_out_frames

start_frames_per_sheep = [
    [(cycle * num_sheep + i) * frames_per_sheep for cycle in range(num_cycles)]
    for i in range(num_sheep)
]

total_frames = (num_cycles * num_sheep) * frames_per_sheep
print(f"Total frames: {total_frames} -> total time {total_frames*interval_seconds/60:.1f} minutes")

angles = np.random.uniform(0, 2*math.pi, size=num_sheep)
dir_x = np.cos(angles)
dir_y = np.sin(angles)

rows = []
for t in range(total_frames):
    ts = start_time + pd.Timedelta(seconds=t * interval_seconds)

    for i in range(num_sheep):
        k = None
        for sf in start_frames_per_sheep[i]:
            rel = t - sf
            if 0 <= rel < frames_per_sheep:
                k = rel
                break

        if k is None:
            # idle sheep
            jitter_n = np.random.normal(0, jitter_std_m)
            jitter_e = np.random.normal(0, jitter_std_m)
            lat = base_lat + m_to_deg_lat(jitter_n)
            lon = base_lon + m_to_deg_lon(jitter_e, base_lat)
        else:
            # active sheep
            if k < travel_out_frames:
                distance_m = min((k + 1) * step_m_per_interval, max_distance_m)
            elif k < travel_out_frames + stay_frames:
                distance_m = max_distance_m
            else:
                ret_step = k - (travel_out_frames + stay_frames) + 1
                distance_m = max_distance_m - min(ret_step * step_m_per_interval, max_distance_m)

            drift_e = dir_x[i] * distance_m
            drift_n = dir_y[i] * distance_m
            jitter_n = np.random.normal(0, active_jitter_m)
            jitter_e = np.random.normal(0, active_jitter_m)
            lat = base_lat + m_to_deg_lat(drift_n + jitter_n)
            lon = base_lon + m_to_deg_lon(drift_e + jitter_e, base_lat)

        rows.append([ts, f"Sheep_{i+1}", lat, lon])

# Save
df = pd.DataFrame(rows, columns=["timestamp", "sheep_id", "latitude", "longitude"])
out_file = "sheep_gps_data.csv"
df.to_csv(out_file, index=False)
print(f"✅ Saved {out_file} with {len(df)} rows (only timestamp, sheep_id, latitude, longitude)")
