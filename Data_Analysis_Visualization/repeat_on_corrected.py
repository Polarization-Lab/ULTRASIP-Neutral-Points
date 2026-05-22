# -*- coding: utf-8 -*-
"""
Correction and aerosol trend plots
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.agreement import mean_diff_plot
import numpy as np
import json
import glob
import pyCompare
import cmocean.cm as cm
from scipy.stats import linregress
from matplotlib.lines import Line2D

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

json_files = glob.glob(f'{data_path}/BNP*v3.json')
corrected_json_files = glob.glob(f'{data_path}/BNP*v4_Corrected.json')

# ==========================================
# LOAD V3 (BASE DATA)
# ==========================================
for file in json_files:

    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        print(f"Skipping unreadable file: {file}")
        continue

    idx += 1

    day = data["date"]

    data_dict[day] = {
        "time": np.array(data["LocalTime(hh:mm:ss)"]),
        "sun_zenith": np.array(data["sun_zenith_deg"]),
        "ray_zen": np.array(data["rayleigh_np_za_355nm"]),
        "grasp_zen": np.array(data["grasp_np_za_355nm"]),
        "aqnum": np.array(data["acquisition"]),
        "sphericity": np.array(data["Sphericity_Factor(%)"]),
        "ssa": np.array(data["Single_Scattering_Albedo[440nm]"]),
        "aod": np.array(data["AOD_Extinction-Total[440nm]"]),
        "g": np.array(data["Asymmetry_Factor-Total[440nm]"]),
        "ae": np.array(data["Extinction_Angstrom_Exponent_440-870nm-Total"]),
        "marker_color": colors[idx % len(colors)]
    }

# ==========================================
# LOAD V4 (CORRECTED ULTRASIP DATA)
# ==========================================
for file in corrected_json_files:

    try:
        with open(file, "r") as f:
            data = json.load(f)
    except:
        print(f"Skipping unreadable corrected file: {file}")
        continue

    day = data["date"]
    uza = np.array(data["np_zenith_deg"])

    if day in data_dict:
        data_dict[day]["ultra_zen"] = uza 
    else:
        print(f"Warning: {day} not found in v3 dataset")

# ==========================================
# BUILD ARRAYS
# ==========================================
time_sec = []
gzen = []
rzen = []
uzen = []
sza_all = []
aods = []
sphers = []
ssas = []
gs = []
aes = []
steps = []

for day, values in data_dict.items():

    # ---- Skip October ----
    month = day.split("_")[1]
    if month not in ["10"]:
        continue

    # ---- Ensure corrected data exists ----
    if "ultra_zen" not in values:
        print(f"Skipping {day}: no corrected data")
        continue

    # ---- Ensure equal lengths ----
    n = min(len(values["time"]),
            len(values["ultra_zen"]),
            len(values["ray_zen"]),
            len(values["sun_zenith"]),
            len(values["grasp_zen"]))

    for i in range(n):

        t = values["time"][i]
        uz = values["ultra_zen"][i]
        rz = values["ray_zen"][i]
        sza = values["sun_zenith"][i]
        gz = values["grasp_zen"][i]
        aod = values["aod"][i]
        ssa = values["ssa"][i]
        g = values["g"][i]
        ae = values["ae"][i]
        spher = values["sphericity"][i]
        step = values["aqnum"][i]

        # Convert time to seconds
        try:
            h, m, s = map(int, t.split(":"))
            t_seconds = h*3600 + m*60 + s
        except:
            continue

        time_sec.append(t_seconds)
        gzen.append(gz)
        rzen.append(rz)
        uzen.append(uz)
        sza_all.append(sza)
        aods.append(aod)
        sphers.append(spher)
        ssas.append(ssa)
        gs.append(g)
        aes.append(ae)
        steps.append(step)

# ==========================================
# CONVERT TO NUMPY
# ==========================================
time_sec = np.array(time_sec)
sza_all = np.array(sza_all)
uzen = np.array(uzen)
gzen = np.array(gzen)
rzen = np.array(rzen)

aods = np.array(aods)
sphers = np.array(sphers)
ssas = np.array(ssas)
gs = np.array(gs)
aes = np.array(aes)
steps = np.array(steps)

print("Done. Arrays built:")
print(f"Total points: {len(uzen)}")

#------------Comparison at certain zenith range---------------------#

selected_days = [
    "2025_10_22",
    "2025_10_23"
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

sza_min = 55 #max(mins)
sza_max = 80 #min(maxs)

print(f"\nCommon SZA overlap: {sza_min:.1f}–{sza_max:.1f}")

plt.figure(figsize=(10,6))

# --------------------------------
# Loop through days
# --------------------------------
for day in selected_days:

    values = data_dict[day]

    sza = np.array(values["sun_zenith"])
    dd = np.array(values["ray_zen"])-(np.array(values["ultra_zen"])-3.29)
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
    label_text = f"{day}  " # $\\bar{{\\mu}}_{{\\Delta\\delta}}$={mean_val:.2f}°,  $\\sigma_{{\\Delta\\delta}}$={std_val:.2f}°"

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
#plt.axvspan(sza_min, sza_max, color='gray', alpha=0.12)
plt.scatter(
        65.01608,
        -0.25806,
        color='magenta',
        edgecolor='black',
        s=120,
        label = '2025_10_24')
#plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.axhline(0,color="royalblue",zorder=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel(r"$\theta_s$ [$^\circ$]", fontsize=18)
plt.ylabel(r"$\Delta\delta$ [$^\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim([55,80])
plt.ylim([-1.5,1.5])

plt.legend(fontsize=14)



plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()
