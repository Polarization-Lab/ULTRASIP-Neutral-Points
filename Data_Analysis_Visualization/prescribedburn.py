# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:05:34 2026

@author: deleo
Burn day analysis
"""


import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
import h5py
import json
import cmocean.cm as cmo
import glob
import os
import matplotlib.animation as animation
from scipy.stats import linregress
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# ==========================================
# LOAD DATA
# ==========================================
data_dict = {}

# colors = ['red','darkorange','yellow','green',
#           'lime','purple','magenta','cadetblue',
#           'silver','cyan','gray','blue',
#           'honeydew','palevioletred','tan','brown','mediumorchid']

colors = ['tan','brown','mediumorchid']

idx = -1

data_path = "C:/Users/deleo/Documents/BNP_daily_v3_allfields_with_rayleigh"
json_files = glob.glob(f'{data_path}/BNP*.json')

oct22 = json_files[15]
oct23 = json_files[16]
oct24 = json_files[0]

oct_files = [oct22,oct23,oct24]

for file in oct_files:

    with open(file, "r") as f:
        data = json.load(f)

    idx += 1
    
    day = data["date"]
    #time = np.array(data["LocalTime(hh:mm:ss)"])
    sza = np.array(data["sun_zenith_deg"])
    saz = np.array(data["sun_azimuth_deg"])
    aqnum = np.array(data["acquisition"])

    
    uza = np.array(data["np_zenith_deg"]) - 3.16
    uaz = np.array(data["np_azimuth_deg"])
    ray_zen = np.array(data["rayleigh_np_za_355nm"])

    data_dict[day] = {
        "sun_zenith": sza,
        "ultra_zen": uza,
        "ray_zen": ray_zen,
        "aqnum": aqnum,
        "marker_color": colors[idx]
    }
    
    
# ==========================================
# PLOT: (rayleigh - ultra) vs sun zenith
# ==========================================
fig, ax = plt.subplots(figsize=(8,6))

for day, d in data_dict.items():
    
    sza = d["sun_zenith"]
    delta = d["ray_zen"] - d["ultra_zen"]
    color = d["marker_color"]
    
    # --------------------------------------
    # SPECIAL CASE: Oct 24 → last 2 are stars
    # --------------------------------------
    if day == "2025_10_24":   # make sure this matches your JSON
        
        # First points (circles)
        ax.scatter(sza[:-2], delta[:-2],
                   color=color,
                   marker='o',
                   s=80,
                   edgecolor='black',
                   label=day)
        
        # Last two points (stars)
        ax.scatter(sza[-2:], delta[-2:],
                   color=color,
                   marker='*',
                   s=160,  # bigger so stars stand out
                   edgecolor='black')
    else:
        ax.scatter(sza, delta,
                   color=color,
                   marker='o',
                   s=80,
                   edgecolor='black',
                   label=day)

# ==========================================
# FORMAT
# ==========================================
ax.set_xlabel("Sun Zenith Angle (deg)")
ax.set_ylabel("$\Delta\delta$ (deg)")

ax.axhline(0, linestyle='--', color='black', linewidth=1)
ax.grid(True, linestyle='--', alpha=0.5)

ax.legend()
plt.tight_layout()
plt.show()