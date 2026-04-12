# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:44:03 2026

@author: deleo
"""

#Correction and aerosol trend plots 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.agreement import mean_diff_plot
import numpy as np
import json
import glob
import pyCompare
from scipy.stats import spearmanr, kendalltau, pearsonr
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
from scipy.stats import linregress


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
    saz = np.array(data["sun_azimuth_deg"])
    gza = np.array(data["grasp_np_za_355nm"])
    aqnum = np.array(data["acquisition"])

    
    uza = np.array(data["np_zenith_deg"]) - 3.16
    uaz = np.array(data["np_azimuth_deg"])
    ray_zen = np.array(data["rayleigh_np_za_355nm"])

    data_dict[day] = {
        "time": time,
        "sun_zenith": sza,
        "ultra_zen": uza,
        "ray_zen": ray_zen,
        "grasp_zen": gza,
        "aqnum": aqnum,
        "sphericity": data["Sphericity_Factor(%)"],
        "ssa": data["Single_Scattering_Albedo[440nm]"],
        "aod": data["AOD_Extinction-Total[440nm]"],
        "g": data["Asymmetry_Factor-Total[440nm]"],
        "ae": data["Extinction_Angstrom_Exponent_440-870nm-Total"],
        "marker_color": colors[idx]
    }


#------------Comparison at certain zenith range---------------------#

selected_days = [
    "2025_10_22",
    "2025_10_23",
    "2025_10_24"
]

# --------------------------------
# Determine common SZA overlap
# --------------------------------
mins = []
maxs = []

for day in selected_days:
    sza = np.array(data_dict[day]["sun_zenith"])
    mins.append(np.min(sza))
    maxs.append(np.max(sza))

sza_min = max(mins)
sza_max = min(maxs)

print(f"\nCommon SZA overlap: {sza_min:.1f}–{sza_max:.1f}")

plt.figure(figsize=(10,6))

# --------------------------------
# Loop through days
# --------------------------------
for day in selected_days:

    values = data_dict[day]

    sza = np.array(values["sun_zenith"])
    dd = np.array(values["ray_zen"])-np.array(values["ultra_zen"])
    #ray = np.array(values["ray_delta"])
    print("obs og",len(dd))
    print("sza range og",np.max(sza)-np.min(sza))

    #delta_delta = delta - ray

    color = values["marker_color"]

    # mask overlap region
    mask = (sza >= sza_min) & (sza <= sza_max)

    sza_overlap = sza[mask]
    delta_overlap = dd[mask]
    
    print(np.std(sza_overlap))
    
    print("obs mask",len(delta_overlap))

    mean_val = np.mean(delta_overlap)
    std_val = np.std(delta_overlap)

    # improved legend formatting
    label_text = f"{day}   $\\bar{{\\mu}}_{{\\Delta\\delta}}$={mean_val:.2f}°,  $\\sigma_{{\\Delta\\delta}}$={std_val:.2f}°"

    # --------------------------------
    # plot all points (faint)
    # --------------------------------
    plt.scatter(
        sza,
        dd,
        color=color,
        alpha=0.25,
        s=80
    )

    # --------------------------------
    # plot overlap points (bold)
    # --------------------------------
    plt.scatter(
        sza_overlap,
        delta_overlap,
        color=color,
        edgecolor='black',
        s=120,
        label=label_text
    )

# --------------------------------
# overlap region shading
# --------------------------------
plt.axvspan(sza_min, sza_max, color='gray', alpha=0.12)

#plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel(r"$\theta_s$ [$^\circ$]", fontsize=18)
plt.ylabel(r"$\Delta\delta$ [$^\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

# plt.xlim([55,80])
# plt.ylim([-2,3])

plt.legend(fontsize=14)



plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()