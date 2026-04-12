# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 10:01:23 2026

@author: ULTRASIP_1
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =========================
# Folder path
# =========================
folder_path = r"C:/Users/ULTRASIP_1/OneDrive/Desktop/Analyzed_Data_JSON"

dates = ['2025_05_28','2025_06_04','2025_06_07','2025_06_09','2025_06_10',
         '2025_06_13','2025_06_14','2025_06_18','2025_06_23','2025_06_24',
         '2025_06_25','2025_06_26','2025_06_30','2025_07_01',
         '2025_07_08','2025_07_09','2025_07_10','2025_07_13',
         '2025_07_17','2025_07_18','2025_07_21',
         '2025_10_22','2025_10_23']



# # #OMIT "OUTLIERS"
# dates = ['2025_05_28','2025_06_04','2025_06_07','2025_06_09','2025_06_10',
#           '2025_06_13','2025_06_14','2025_06_18','2025_06_23',
#           '2025_06_25','2025_06_26','2025_06_30','2025_07_01',
#           '2025_07_08','2025_07_09','2025_07_10','2025_07_13',
#           '2025_07_17','2025_07_18','2025_07_21',
#           '2025_10_22','2025_10_23']

#OMITTING DAYS WITHOUT AERONET
dates = ['2025_06_04','2025_06_09','2025_06_10',
          '2025_06_18','2025_06_23','2025_06_24',
          '2025_06_25','2025_06_30','2025_07_01',
          '2025_07_08','2025_07_09','2025_07_10','2025_07_13',
          '2025_07_17','2025_07_18', '2025_07_21']

#OMITTING DAYS WITHOUT SIM
dates = ['2025_06_04','2025_06_09','2025_06_10',
          '2025_06_18','2025_06_23','2025_06_24',
          '2025_06_25','2025_06_30','2025_07_01',
          '2025_07_08','2025_07_09','2025_07_10',
          '2025_07_17','2025_07_18']


daily_AODs = [0.2067,
0.14871,
0.074018,
0.17344,
0.15351,
0.35358,
0.27248,
0.11352,
0.15591,
0.1675,
0.17168,
0.16671,
0.21415,
0.26384]


# #OMITTING DAYS WITHOUT AERONET and JUNE 24
# dates = ['2025_06_04','2025_06_09','2025_06_10',
#           '2025_06_18','2025_06_23',
#           '2025_06_25','2025_06_30','2025_07_01',
#           '2025_07_08','2025_07_09','2025_07_10','2025_07_13',
#           '2025_07_17','2025_07_18','2025_07_21']





slopes = []
intercepts = []
valid_dates = []

# =========================
# Load data
# =========================
for date in dates:
    json_path = os.path.join(folder_path, f"BNP_observations_{date}_v3.json")

    if not os.path.exists(json_path):
        continue

    if os.path.getsize(json_path) == 0:
        continue

    with open(json_path, "r") as f:
        data = json.load(f)

    if "delta_vs_sza_slope" not in data:
        continue

    slopes.append(data["delta_vs_sza_slope"])
    intercepts.append(data["delta_vs_sza_intercept"])
    valid_dates.append(date)

# Convert for plotting
x = np.arange(len(valid_dates))

# Colormap (each date different color)
colors = cm.tab20(np.linspace(0,1,len(valid_dates)))

# =========================
# Create figure
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14,8), sharex=True
)
ax1.plot(x, slopes, '-', color='black', linewidth=1.5)

# -------------------------
# Top: slope vs date
# -------------------------
for i in range(len(valid_dates)):
    ax1.plot(
        x[i], slopes[i],
        'o',
        markersize=15,
        color=colors[i],
        markeredgecolor='black',
        markeredgewidth=1.2
    )

ax1.set_ylabel("Slope", fontsize=18)
ax1.tick_params(labelsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

# -------------------------
# Bottom: intercept vs date
# -------------------------
ax2.plot(x, intercepts, '-', color='black', linewidth=1.5)

for i in range(len(valid_dates)):
    ax2.plot(
        x[i], intercepts[i],
        'o',
        markersize=15,
        color=colors[i],
        markeredgecolor='black',
        markeredgewidth=1.2
    )

ax2.set_ylabel("Intercept", fontsize=18)
ax2.set_xlabel("Date", fontsize=18)

ax2.tick_params(labelsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)

# -------------------------
# Format x-axis as dates
# -------------------------
ax2.set_xticks(x)
ax2.set_xticklabels(valid_dates, rotation=45, ha='right')

plt.tight_layout()
plt.show()
