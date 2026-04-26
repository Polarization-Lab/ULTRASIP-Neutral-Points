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

colors = ['tan','brown','magenta',"lime"]

idx = -1

data_path = "C:/Users/deleo/Documents/BNP_daily_v3_allfields_with_rayleigh"
json_files = glob.glob(f'{data_path}/BNP*.json')

oct22 = json_files[15]
oct23 = json_files[16]
oct24 = json_files[0]
oct24_reg = json_files[17]

oct_files = [oct22,oct23,oct24]

labels = ["2025_10_22", "2025_10_23", "2025_10_24", "2025_10_24_B"]

for i, file in enumerate(oct_files):

    with open(file, "r") as f:
        data = json.load(f)

    day = labels[i]   # ← force unique label

    sza = np.array(data["sun_zenith_deg"])
    uza = np.array(data["np_zenith_deg"]) - 3.16
    ray_zen = np.array(data["rayleigh_np_za_355nm"])

    data_dict[day] = {
        "sun_zenith": sza,
        "ultra_zen": uza,
        "ray_zen": ray_zen,
        "marker_color": colors[i]
    }

    
    
# ==========================================
# PLOT: (rayleigh - ultra) vs sun zenith
# ==========================================
fig, ax = plt.subplots(figsize=(12,6))

for day, d in data_dict.items():
    
    sza = d["sun_zenith"]
    delta = d["ray_zen"] - d["ultra_zen"]
    color = d["marker_color"]
    
    print(len(sza))
    
    # --------------------------------------
    # SPECIAL CASE: Oct 24 → last 2 are stars
    # --------------------------------------
    if labels == "2025_10_24_B":   # make sure this matches your JSON
        
        # First points (circles)
        ax.scatter(sza[:-2], delta[:-2],
                   color=color,
                   marker='o',
                   s=100,
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
                   s=100,
                   edgecolor='black',
                   label=day)

# ==========================================
# FORMAT
# ==========================================
ax.set_xlabel("$\\theta_s [\\circ]$",fontsize=20)
ax.set_ylabel("$\Delta\delta [\\circ]$ ",fontsize=20)
ax.set_ylim([-1.5,1.5])
ax.set_xlim([55,80])
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.tick_params(axis='both', which='major', labelsize=20)

ax.axhline(0,color="royalblue",zorder=0)
ax.grid(True, linestyle='--', alpha=0.5)

ax.legend(fontsize=18)
plt.tight_layout()
plt.show()