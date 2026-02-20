# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 08:20:46 2026

@author: ULTRASIP_1
"""

# -*- coding: utf-8 -*-
"""
Load BNP JSON file
Compute deltas
Make required plots
Save fit parameters back to JSON
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

sim_sun_zen = np.array([
    59.277323,
    42.592545,
    44.085867,
    55.451474,
    61.006963,
    71.415686,
    76.39244
])

sim_delta = np.array([
    -20.39164929,
    -14.76625695,
    -15.28140779,
    -19.14911842,
    -20.91928794,
    -24.00322845,
    -25.31664981
])
# =========================
# File path
# =========================
json_path = r"C:/Users/ULTRASIP_1/OneDrive/Desktop/Analyzed_Data_JSON/BNP_observations_2025_07_10_v3.json"

# =========================
# Load JSON
# =========================
with open(json_path, "r") as f:
    data = json.load(f)

# Pull date from JSON if available
date_str = data.get("date", "")

# Convert to numpy arrays
sun_zen = np.array(data["sun_zenith_deg"])
sun_az  = np.array(data["sun_azimuth_deg"])
np_zen  = np.array(data["np_zenith_deg"])
np_az   = np.array(data["np_azimuth_deg"])

np_zen_err = np.array(data["np_zen_error_arcsec"])
np_az_err  = np.array(data["np_az_error_arcsec"])

# =========================
# Compute Deltas
# =========================
delta_zen = np_zen - sun_zen
delta_az  = np_az  - sun_az

# Save to dictionary
data["delta_zenith_deg"] = delta_zen.tolist()
data["delta_azimuth_deg"] = delta_az.tolist()

# =========================
# 1) delta vs SZA
# =========================
fig1, ax1 = plt.subplots(figsize=(6,5))

# --- Main data (purple, no connecting line)
ax1.plot(
    sun_zen,
    delta_zen,
    'o',
    color='purple',
    markersize=10,
    label=r"$\delta$ Zenith"
)

ax1.set_xlabel("Sun Zenith  [$\circ$]", fontsize=16)
ax1.set_ylabel(r"$\delta$ Zenith [$\circ$]", fontsize=16, color='purple')
# ax1.set_ylim(-20, -5)   # LEFT axis limits

ax1.tick_params(axis='both', labelsize=14)
ax1.tick_params(axis='y', colors='purple')

ax1.grid(True, which='major', axis='both',
         linestyle='--', linewidth=1, alpha=0.8)

# --- Second axis (errors in red)
ax2 = ax1.twinx()

ax2.errorbar(
    sun_zen,
    np_zen_err,
    fmt='P',
    color='red',
    markersize=8,
    capsize=4,
    label="NP Zenith Error"
)

ax2.set_ylabel(r"NP Zenith Error [arcseconds]", fontsize=16, color='red')
ax2.tick_params(axis='y', labelsize=14, colors='red')
# ax2.set_ylim(5, 45)     # RIGHT axis limits

plt.title(date_str, fontsize=20)


plt.tight_layout()
plt.show()


# =========================
# 2) Azimuth difference vs SZA
# =========================
fig2, ax1 = plt.subplots(figsize=(8,5))

# --- Left axis (delta azimuth, purple, no connecting line)
ax1.plot(
    sun_zen,
    delta_az,
    'o',
    color='purple',
    markersize=9
)

# Zero reference line
ax1.axhline(
    0,
    color='black',
    linewidth=1.5,
    linestyle='-'
)

ax1.set_xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
ax1.set_ylabel(r"$\delta$ Azimuth [$^\circ$]", fontsize=18, color='purple')

# ax1.set_ylim(-1.5, 2)
ax1.tick_params(axis='both', labelsize=16)
ax1.tick_params(axis='y', colors='purple')

# Make left axis spine purple
ax1.spines['left'].set_color('purple')

# --- Right axis (az error, red)
ax2 = ax1.twinx()

ax2.errorbar(
    sun_zen,
    np_az_err,
    fmt='P',
    color='red',
    markersize=9,
    capsize=4
)

ax2.set_ylabel(r"NP Azimuth Error [arcseconds]", fontsize=18, color='red')
# ax2.set_ylim(5, 45)

ax2.tick_params(axis='y', labelsize=16, colors='red')

# Make right axis spine red
ax2.spines['right'].set_color('red')

plt.grid(True, linestyle='--', alpha=0.8)

plt.title(date_str, fontsize=20)

plt.tight_layout()
plt.show()


# =========================
# 3) Linear Fit: delta vs SZA
# =========================
# Linear regression
slope, intercept, r_value, p_value, std_err = linregress(sun_zen, delta_zen)
fit_line = slope * sun_zen + intercept



fig3 = plt.figure(figsize=(8,6))  # wider figure

# Observed data (purple markers)
plt.plot(
    sun_zen,
    delta_zen,
    'o',
    color='olive',
    markeredgecolor='black',
    markersize=12,
    label="Observed"
)

# Linear fit
plt.plot(
    sun_zen,
    fit_line,
    '-',
    linewidth=2,
    color='black',
    label=(
        f"Slope = {slope:.4f}\n"
        f"Intercept = {intercept:.4f}\n"
        # f"$R^2$ = {r_value**2:.4f}\n"
    )
)

# Gridlines
plt.grid(True, linestyle='--', alpha=0.8)

plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
plt.ylabel(r"$\delta$ Zenith [$^\circ$]", fontsize=18)

plt.title(date_str, fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# plt.ylim(-25, -5)

plt.legend(
    fontsize=16,
    handlelength=2,
    borderpad=1,
    labelspacing=1.0,
)    
plt.tight_layout()
plt.show()


fig4 = plt.figure(figsize=(8,6))  

# Observed data (purple markers)
plt.plot(
    sun_zen,
    delta_zen,
    'o',
    color='red',
    markeredgecolor='black',
    markersize=12,
    label="Observed"
)

plt.plot(
    sim_sun_zen,
    sim_delta,
    's',
    color='red',
    markeredgecolor='black',
    markersize=12,
    label="Simulated"
)


# Gridlines
plt.grid(True, linestyle='--', alpha=0.8)

plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
plt.ylabel(r"$\delta$ Zenith [$^\circ$]", fontsize=18)

plt.title(date_str, fontsize=20)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# plt.ylim(-25, -5)

# plt.legend(
#     fontsize=16,
#     handlelength=2,
#     borderpad=1,
#     labelspacing=1.0,
# )    
# plt.tight_layout()
# plt.show()


# # =========================
# # Save Fit Parameters
# # =========================
# data["delta_vs_sza_slope"] = float(slope)
# data["delta_vs_sza_intercept"] = float(intercept)
# data["delta_vs_sza_r_squared"] = float(r_value**2)

# # =========================
# # Write Updated JSON
# # =========================
# with open(json_path, "w") as f:
#     json.dump(data, f, indent=2)

# print("Updated JSON file saved successfully.")
