# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 22:16:45 2026

@author: deleo
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import glob
from matplotlib.ticker import FuncFormatter, MultipleLocator

# ==========================================
# LOAD DATA
# ==========================================
data_dict = {}

colors = ['red','darkorange','yellow','green',
          'lime','purple','magenta','cadetblue',
          'silver','cyan','gray','blue',
          'honeydew','palevioletred','tan','brown','mediumorchid']

idx = -1

data_path = "C:/Users/deleo/Documents/BNP_daily_v3_allfields_with_rayleigh"
json_files = glob.glob(f'{data_path}/BNP*.json')

for file in json_files:

    with open(file, "r") as f:
        data = json.load(f)

    idx += 1
    
    day = data["date"]
    time = np.array(data["LocalTime(hh:mm:ss)"])
    sza = np.array(data["sun_zenith_deg"])
    gza = np.array(data["grasp_np_za_355nm"])

    correction_factor = 0.59*(np.array(data["acquisition"]))
    
    uza = np.array(data["np_zenith_deg"]) - correction_factor
    ray_zen = np.array(data["rayleigh_np_za_355nm"])

    data_dict[day] = {
        "time": time,
        "sun_zenith": sza,
        "ultra_zen": uza,
        "ray_zen": ray_zen
    }

# ==========================================
# BUILD ARRAYS
# ==========================================
time_sec = []
ray_diff = []
sza_all = []

for day, values in data_dict.items():

    # ---- Skip October ----
    month = day.split("_")[1]
    if month in ["10"]:
        continue

    for t, uz, rz, sza in zip(values["time"],
                               values["ultra_zen"],
                               values["ray_zen"],
                               values["sun_zenith"]):

        h, m, s = map(int, t.split(":"))
        t_seconds = h*3600 + m*60 + s

        time_sec.append(t_seconds)
        ray_diff.append(uz - rz)
        sza_all.append(sza)

time_sec = np.array(time_sec)
ray_diff = np.array(ray_diff)
sza_all = np.array(sza_all)

# ==========================================
# SORT BY TIME
# ==========================================
sort_idx = np.argsort(time_sec)

time_sorted = time_sec[sort_idx]
ray_diff_sorted = ray_diff[sort_idx]
sza_sorted = sza_all[sort_idx]

# ==========================================
# FORMAT FUNCTION
# ==========================================
def sec_to_hhmm(x, pos):
    h = int(x // 3600)
    m = int((x % 3600) // 60)
    return f"{h:02d}:{m:02d}"

# ==========================================
# PLOT
# ==========================================
plt.figure(figsize=(12,5))
ax = plt.gca()

plt.scatter(time_sorted, ray_diff_sorted, s=20, color='black')

# ---- Axis labels ----
plt.xlabel("Time of Day (HH:MM)", fontsize=15)
plt.ylabel("(Ultrasip - Rayleigh)", fontsize=15)

# ---- X ticks: every hour ----
ax.xaxis.set_major_locator(MultipleLocator(3600))   # 1 hour
ax.xaxis.set_minor_locator(MultipleLocator(1800))   # 30 min
ax.xaxis.set_major_formatter(FuncFormatter(sec_to_hhmm))

# ---- Y axis ----
#ax.set_ylim(-6.5, 8)
ax.yaxis.set_major_locator(MultipleLocator(5))

# ---- Tick label size ----
ax.tick_params(axis='both', which='major', labelsize=13)
ax.tick_params(axis='both', which='minor', labelsize=11)

# ---- Gridlines ----
ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.6)
ax.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.4)

# ---- Horizontal zero line ----
plt.axhline(0, linestyle='-', linewidth=1.5, color='black')


# ==========================================
# TOP AXIS: SZA aligned with TIME TICKS
# ==========================================
ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())

# Get the EXACT time tick locations from bottom axis
time_ticks = ax.get_xticks()

# Interpolate SZA at those exact time positions
sza_at_ticks = np.interp(time_ticks, time_sorted, sza_sorted)

# Apply to top axis
ax_top.set_xticks(time_ticks)
ax_top.set_xticklabels([f"{val:.1f}" for val in sza_at_ticks])

ax_top.set_xlabel("Sun Zenith Angle (deg)", fontsize=15)
ax_top.tick_params(axis='x', labelsize=13)


# ---- Final formatting ----
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

