# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 12:50:35 2026
@author: ULTRASIP_1
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import pearsonr

plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20
})


# =========================
# Folder path
# =========================
folder_path = r"C:/Users/ULTRASIP_1/OneDrive/Desktop/Analyzed_Data_JSON"

# dates = ['2025_06_04','2025_06_09','2025_06_10',
#           '2025_06_18','2025_06_23','2025_06_24',
#           '2025_06_25','2025_06_30','2025_07_01',
#           '2025_07_08','2025_07_09','2025_07_10',
#           '2025_07_17','2025_07_18']

# daily_AODs = [0.2067,0.14871,0.074018,0.17344,0.15351,0.35358,
#               0.27248,0.11352,0.15591,0.1675,0.17168,
#               0.16671,0.21415,0.26384]



# sim_slopes = np.array([
#     -0.31431837, -0.27209096, -0.31884777, -0.31325142,
#     -0.29047546, -0.31698982, -0.31428989, -0.31714076,
#     -0.31786798, -0.32490378, -0.31469065, -0.31175251,
#     -0.33739854
# ])

# sim_intercepts = np.array([
#     -1.54800416, -4.56735061, -1.31336578, -1.56310746,
#     -3.21131144, -1.37344459, -1.54777989, -1.36778198,
#     -1.36960829, -0.98396621, -1.53349682, -1.71953006,
#     -0.40837062
# ])

dates = [
    '2025_06_10',
    '2025_06_30',
    '2025_06_09',
    '2025_06_23',
    '2025_07_01',
    '2025_07_10',
    '2025_07_08',
    '2025_07_09',
    '2025_06_18',
    '2025_06_04',
    '2025_07_17',
    '2025_07_18',
    '2025_06_25',
    '2025_06_24'
]

daily_AODs = [
    0.074018,
    0.11352,
    0.14871,
    0.15351,
    0.15591,
    0.16671,
    0.1675,
    0.17168,
    0.17344,
    0.2067,
    0.21415,
    0.26384,
    0.27248,
    0.35358
]


sim_slopes = np.array([
    -0.31884777,  # 2025_06_10
    -0.31714076,  # 2025_06_30
    -0.27209096,  # 2025_06_09
    -0.29047546,  # 2025_06_23
    -0.31786798,  # 2025_07_01
    -0.31175251,  # 2025_07_10
    -0.32490378,  # 2025_07_08
    -0.31469065,  # 2025_07_09
    -0.31325142,  # 2025_06_18
    -0.31431837,  # 2025_06_04
    -0.33739854,  # 2025_07_17
    -0.31428989,  # 2025_07_18
    -0.31698982   # 2025_06_25
])

sim_intercepts = np.array([
    -1.31336578,
    -1.36778198,
    -4.56735061,
    -3.21131144,
    -1.36960829,
    -1.71953006,
    -0.98396621,
    -1.53349682,
    -1.56310746,
    -1.54800416,
    -0.40837062,
    -1.54777989,
    -1.37344459
])


# =========================
# Load observational data
# =========================
slopes = []
intercepts = []
valid_dates = []
valid_AODs = []

for date, aod in zip(dates, daily_AODs):

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
    valid_AODs.append(aod)

slopes = np.array(slopes)
intercepts = np.array(intercepts)

x = np.arange(len(valid_dates))
colors = cm.tab20(np.linspace(0, 1, len(valid_dates)))

# =========================
# Pearson correlations
# =========================
r_slope, p_slope = pearsonr(slopes, sim_slopes[:len(slopes)])
r_intercept, p_intercept = pearsonr(intercepts, sim_intercepts[:len(intercepts)])

# =========================
# Create figure
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14, 8), sharex=True
)

# -------------------------
# SLOPE PANEL
# -------------------------
ax1.plot(x, slopes, '-', color='black', linewidth=1.5)
ax1.plot(x, sim_slopes, '-', color='black', linewidth=1.5)


for i in range(len(valid_dates)):
    ax1.plot(x[i], slopes[i], 'o',
             markersize=15,
             color=colors[i],
             markeredgecolor='black',
             markeredgewidth=1.2)

    ax1.plot(x[i], sim_slopes[i], 's',
             markersize=12,
             color=colors[i],
             markeredgecolor='black',
             markeredgewidth=1.2)

ax1.set_ylabel("Slope", fontsize=20)
ax1.tick_params(labelsize=18)
ax1.grid(True, linestyle='--', alpha=0.6)

# Pearson textbox (lower left)
# ax1.text(
#     0.02, 0.05,
#     f"Correlation Coefficient = {r_slope:.3f}",
#     transform=ax1.transAxes,
#     fontsize=14,
#     verticalalignment='bottom',
#     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
# )

# Ensure full y-range coverage
ax1.set_ylim(-0.45, -0.15)

ax1.set_ylabel("Slope")
ax1.grid(True, linestyle='--', alpha=0.6)


# -------------------------
# INTERCEPT PANEL
# -------------------------
ax2.plot(x, intercepts, '-', color='black', linewidth=1.5)
ax2.plot(x, sim_intercepts, '-', color='black', linewidth=1.5)


for i in range(len(valid_dates)):
    ax2.plot(x[i], intercepts[i], 'o',
             markersize=15,
             color=colors[i],
             markeredgecolor='black',
             markeredgewidth=1.2)

    ax2.plot(x[i], sim_intercepts[i], 's',
             markersize=12,
             color=colors[i],
             markeredgecolor='black',
             markeredgewidth=1.2)

ax2.set_ylabel("Intercept", fontsize=20)
ax2.set_xlabel("Date (MM_DD)", fontsize=20)
ax2.tick_params(labelsize=18)
ax2.grid(True, linestyle='--', alpha=0.6)

# Pearson textbox (upper left)
# ax2.text(
#     0.02, 0.95,
#     f"Correlation Coefficient = {r_intercept:.3f}",
#     transform=ax2.transAxes,
#     fontsize=14,
#     verticalalignment='top',
#     bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
# )
ax2.set_ylim(-10, 10)

# -------------------------
# X-axis formatting
# -------------------------
ax2.set_xticks(x)
short_dates = [d.replace("2025_", "") for d in valid_dates]
ax2.set_xticklabels(short_dates, rotation=45, ha='right')

# -------------------------
# AOD top axis
# -------------------------
ax_top = ax1.twiny()
ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks(x)
ax_top.set_xticklabels([f"{aod:.3f}" for aod in valid_AODs],
                       rotation=45, ha='left')
ax_top.set_xlabel("AOD (355 nm)", fontsize=20)
ax_top.tick_params(labelsize=18)

plt.tight_layout()
plt.show()
