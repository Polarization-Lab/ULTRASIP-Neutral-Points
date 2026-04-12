# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:21:43 2026

@author: deleo
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
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe


data_dict = {}
date = []
sun_zenith = []
grasp_delta = []
ultrasip_delta = []
sphericity = []
ssa = []
aod = []

colors = ['red','darkorange','yellow','green',
          'lime','purple','magenta','cadetblue',
          'silver','cyan','gray','blue',
          'honeydew','palevioletred','tan','brown','mediumorchid']

idx=-1

# Load JSON Files
data_path = "C:/Users/deleo/Documents/BNP_daily_v3_allfields_with_rayleigh"
json_files = glob.glob(f'{data_path}/BNP*.json')

# Output folder
figure_dir = os.path.join(data_path, "figures")
os.makedirs(figure_dir, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
for file in json_files:

    with open(file, "r") as f:
        data = json.load(f)

    idx += 1

    day = data["date"]
    aquisition = 0.59*(np.array(data["acquisition"]))
    sza = np.array(data["sun_zenith_deg"])
    ultra_np = np.array(data["np_zenith_deg"]) -aquisition
    delta_obs = ultra_np-sza
    delta_sim = np.array(data["GRASP_delta"])
    

    ray_zen = np.array(data["rayleigh_np_za_355nm"])
    delta_ray = ray_zen - sza

    # NEW variable: deviation from Rayleigh
    delta_delta_obs = delta_obs - delta_ray
    delta_delta_sim = delta_sim - delta_ray

    slope_obs, intercept_obs, r_value, p_value, std_err = linregress(sza, delta_obs)
    slope_sim, intercept_sim, r_value, p_value, std_err = linregress(sza, delta_sim)
    slope_ray, intercept_ray, r_value, p_value, std_err = linregress(sza, delta_ray)

    data_dict[day] = {
        "sun_zenith": sza,
        "ray_zen": ray_zen,
        "ultrasip_delta": delta_obs,
        "grasp_delta": delta_sim,
        "ray_delta": delta_ray,
        "delta_delta_obs": delta_delta_obs,
        "delta_delta_sim": delta_delta_sim,
        "sphericity": data["Sphericity_Factor(%)"],
        "ssa": data["Single_Scattering_Albedo[440nm]"],
        "aod": data["AOD_Extinction-Total[440nm]"],
        "g": data["Asymmetry_Factor-Total[440nm]"],
        "ae": data["Extinction_Angstrom_Exponent_440-870nm-Total"],
        "slope_obs": slope_obs,
        "intercept_obs": intercept_obs,
        "slope_sim": slope_sim,
        "intercept_sim": intercept_sim,
        "slope_ray": slope_ray,
        "intercept_ray": intercept_ray,
        "marker_color": colors[idx]
    }
    
sza = []
delta_ultrasip = []
delta_grasp=[]
ray_delta= []
aod = []
g= []
ae = []
    
for day, values in data_dict.items():

    # ---- Skip October ----
    month = day.split("_")[1]   # assumes format YYYY_MM_DD
    if month in ["10"]:
        continue
    
    values = data_dict[day]

    sza = np.append(sza,np.array(values["sun_zenith"]))
    delta_ultrasip = np.append(delta_ultrasip,np.array(values["ultrasip_delta"]))
    delta_grasp = np.append(delta_grasp,np.array(values["grasp_delta"]))
    ray_delta = np.append(ray_delta,np.array(values["ray_delta"]))
    aod = np.append(aod,np.array(values["aod"]))
    ssa = np.append(ssa,np.array(values["ssa"]))
    g = np.append(g,np.array(values["g"]))
    ae = np.append(ae,np.array(values["ae"]))

avg_diff = np.average(np.abs(delta_ultrasip-delta_grasp))
print(f"avg absolute difference: {avg_diff}")

std_diff = np.std(np.abs(delta_ultrasip-delta_grasp))
print(f"std of diff: {std_diff}")

med_diff = np.median(np.abs(delta_ultrasip-delta_grasp))
print(f"median of diff: {med_diff}")


delta_delta_ultrasip = delta_ultrasip - ray_delta 
delta_delta_grasp = delta_grasp - ray_delta


rho,_ = pearsonr(sza,delta_ultrasip)
print("rho",rho)
plt.figure(figsize=(12,6))
plt.scatter(sza,delta_ultrasip,label = f"ULTRASIP (Observed), $\\rho$={rho:.3f}",color='black',s=50)
plt.grid(True)
plt.tick_params(axis='both', labelsize=15) # Change font size for both x and y axes
plt.xlabel("Sun Zenith Angle [$\circ$]",fontsize=20)
plt.ylabel("$\delta$ [$\circ$]",fontsize=20)
plt.legend(fontsize=18,loc='upper right',ncol=2)
plt.ylim([-35,-5])
plt.xlim([25,90])
plt.xticks(np.arange(25, 91, 5))
plt.show()

rho,_ = pearsonr(sza,delta_delta_ultrasip)
print("rho",rho)
delmu = np.average(delta_delta_ultrasip)
delsd = np.std(delta_delta_ultrasip)
plt.figure(figsize=(12,6))
plt.scatter(sza,delta_delta_ultrasip,label = f"ULTRASIP (Observed), $\\rho$={rho:.3f}\n $\\mu = {delmu:.2f}^\\circ,SD = {delsd:.2f}^\\circ$",color='black',s=50)
plt.grid(True)
plt.tick_params(axis='both', labelsize=15) # Change font size for both x and y axes
plt.xlabel("Sun Zenith Angle [$\circ$]",fontsize=20)
plt.ylabel("$\Delta_\delta$ [$\circ$]",fontsize=20)
plt.legend(fontsize=15,loc='upper right',ncol=2)
plt.ylim([-6,6])
plt.xlim([25,90])
plt.xticks(np.arange(25, 91, 5))
plt.show()


rho,_ = pearsonr(sza,delta_delta_grasp)
print("rho",rho)
plt.figure(figsize=(12,6))
plt.scatter(sza,delta_delta_grasp,label = f"GRASP-AERONET (Simulated), $\\rho$={rho:.3f}",color='black',s=50,marker='s')
plt.grid(True)
plt.tick_params(axis='both', labelsize=15) # Change font size for both x and y axes
plt.xlabel("Sun Zenith Angle [$\circ$]",fontsize=20)
plt.ylabel("$\Delta_\delta$ [$\circ$]",fontsize=20)
plt.legend(fontsize=18,loc='upper right')
plt.ylim([-1,3])
plt.xlim([25,90])
plt.xticks(np.arange(25, 91, 5))
plt.show()



#downselect
dd_ultra = []
s_sun = []
s_aod=[]
s_ssa = []
s_g = []
s_ae = []
idx=-1

lower = 40
upper= 75
for zen in sza:
    idx=idx+1
    if lower <= zen <= upper: 
        dd_ultra = np.append(dd_ultra, delta_delta_ultrasip[idx])
        s_sun = np.append(s_sun, sza[idx])
        s_aod = np.append(s_aod,aod[idx])
        s_ssa = np.append(s_ssa,ssa[idx])
        s_g = np.append(s_g,g[idx])
        s_ae = np.append(s_ae,ae[idx])
        
smin = np.min(s_sun)
smax= np.max(s_sun)



# Compute N once
N = len(dd_ultra)

# Create figure and axes (2x2)
fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row')

# Flatten for easy looping
axs = axs.ravel()

# Data + labels
x_data = [s_ssa, s_aod, s_g, s_ae]
x_labels = ["SSA", "AOD", "g", "AE"]

for i, (x, label) in enumerate(zip(x_data, x_labels)):
    ax = axs[i]
    
    rho, _ = pearsonr(x, dd_ultra)
    
    ax.scatter(x, dd_ultra,
               label=f"$\\rho$ = {rho:.3f}",
               color='black', s=60)
    
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=13)
    ax.set_xlabel(label, fontsize=16)
    
    # Only left column gets y-label
    if i % 2 == 0:
        ax.set_ylabel("$\\Delta_\\delta$ [$^\\circ$]", fontsize=16)
    
    ax.set_ylim([-6, 4])
    ax.set_yticks(np.arange(-6, 4, 1))
    
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    
    ax.legend(fontsize=16, loc='upper right')

# Suptitle with N and SZA range
fig.suptitle(
    f"N = {N} | SZA Range: {smin:.2f}$^\\circ$–{smax:.2f}$^\\circ$",
    fontsize=20
)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
plt.show()

# ---------------- SETTINGS ----------------
zmin = 25
zmax = 90
bin_size = 25   # <-- change this freely

# Create bin edges
bins = np.arange(zmin, zmax, bin_size)

# ---------------- LOOP OVER BINS ----------------
for lower in bins:
    upper = lower + bin_size

    # Downselect using boolean mask (much cleaner)
    mask = (sza >= lower) & (sza < upper)

    dd_ultra = delta_delta_ultrasip[mask]
    s_sun   = sza[mask]
    s_aod   = aod[mask]
    s_ssa   = ssa[mask]
    s_g     = g[mask]
    s_ae    = ae[mask]

    # Skip empty bins
    if len(dd_ultra) == 0:
        continue

    smin = np.min(s_sun)
    smax = np.max(s_sun)
    N = len(dd_ultra)

    # ---------------- PLOTTING ----------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
    axs = axs.ravel()

    x_data = [s_ssa, s_aod, s_g, s_ae]
    x_labels = ["SSA", "AOD", "g", "AE"]

    for i, (x, label) in enumerate(zip(x_data, x_labels)):
        ax = axs[i]

        # Handle constant arrays safely
        if len(x) > 1 and np.std(x) > 0:
            rho, _ = pearsonr(x, dd_ultra)
            SD = np.std(dd_ultra)
            mu = np.average(dd_ultra)
        else:
            rho = np.nan
            SD = np.nan
            mu = np.nan

        ax.scatter(x, dd_ultra,
                   label=f"$\\rho$ = {rho:.3f}",
                   color='black', s=60)

        ax.grid(True)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xlabel(label, fontsize=16)

        if i % 2 == 0:
            ax.set_ylabel("$\\Delta_\\delta$ [$^\\circ$]", fontsize=16)

        ax.set_ylim([-6, 4])
        ax.set_yticks(np.arange(-6, 4, 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax.legend(fontsize=14, loc='upper right')

    # Suptitle per bin
    fig.suptitle(
        f"SZA Range: {smin:.2f}$^\\circ$–{smax:.2f}$^\\circ$ | N = {N}\n SD = {SD:.2f}$^\\circ$, $\\mu = {mu:.2f}^\\circ$",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
# ---------------- SETTINGS ----------------
zmin = 25
zmax = 90
bin_size = 25   # <-- change this freely

# Create bin edges
bins = np.arange(zmin, zmax, bin_size)

# ---------------- LOOP OVER BINS ----------------
for lower in bins:
    upper = lower + bin_size

    # Downselect using boolean mask (much cleaner)
    mask = (sza >= lower) & (sza < upper)

    dd_grasp = delta_delta_grasp[mask]
    s_sun   = sza[mask]
    s_aod   = aod[mask]
    s_ssa   = ssa[mask]
    s_g     = g[mask]
    s_ae    = ae[mask]

    # Skip empty bins
    if len(dd_grasp) == 0:
        continue

    smin = np.min(s_sun)
    smax = np.max(s_sun)
    N = len(dd_grasp)

    # ---------------- PLOTTING ----------------
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharey='row')
    axs = axs.ravel()

    x_data = [s_ssa, s_aod, s_g, s_ae]
    x_labels = ["SSA", "AOD", "g", "AE"]

    for i, (x, label) in enumerate(zip(x_data, x_labels)):
        ax = axs[i]

        # Handle constant arrays safely
        if len(x) > 1 and np.std(x) > 0:
            rho, _ = pearsonr(x, dd_grasp)
            SD = np.std(dd_grasp)
            mu = np.average(dd_grasp)
        else:
            rho = np.nan
            SD = np.nan
            mu = np.nan

        ax.scatter(x, dd_grasp,
                   label=f"$\\rho$ = {rho:.3f}",
                   color='black', s=60,marker='s')

        ax.grid(True)
        ax.tick_params(axis='both', labelsize=13)
        ax.set_xlabel(label, fontsize=16)

        if i % 2 == 0:
            ax.set_ylabel("$\\Delta_\\delta$ [$^\\circ$]", fontsize=16)

        ax.set_ylim([-6, 4])
        ax.set_yticks(np.arange(-6, 4, 1))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        ax.legend(fontsize=14, loc='upper right')

    # Suptitle per bin
    fig.suptitle(
        f"SZA Range: {smin:.2f}$^\\circ$–{smax:.2f}$^\\circ$ | N = {N}\n SD = {SD:.2f}$^\\circ$, $\\mu = {mu:.2f}^\\circ$",
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
# Define SZA bins 
# Create green gradient (light → dark)
greens = plt.cm.Greens(np.linspace(0.15, 1,13))

# Rebuild bins with new colors
# bins = [
#     (25, 30, greens[0]),
#     (30, 35, greens[1]),
#     (35, 40, greens[2]),
#     (40, 45, greens[3]),
#     (45, 50, greens[4]),
#     (50, 55, greens[5]),
#     (55, 60, greens[6]),
#     (60, 65, greens[7]),
#     (65, 70, greens[8]),
#     (70, 75, greens[9]),
#     (75, 80, greens[10]),
#     (80, 85, greens[11]),
#     (85, 90, greens[12]),
# ]
bins = [
    (25, 45, greens[0]),
    (45, 75, greens[6]),
    (75, 90, greens[12]),
]
plt.figure(figsize=(12,6))

# Loop through each SZA region
for i, (lower, upper, color) in enumerate(bins):
    
    # Mask for this bin
    mask = (sza >= lower) & (sza <= upper)
    
    # Downselected data
    dd_ultra = delta_delta_ultrasip[mask]
    s_aod_bin = aod[mask]
    
    # Skip empty bins
    if len(dd_ultra) < 2:
        continue
    
    # Correlation
    rho, _ = pearsonr(s_aod_bin, dd_ultra)
    N = len(dd_ultra)
    
    # Scatter plot for this bin
    plt.scatter(
        s_aod_bin,
        dd_ultra,
        color=color,
        s=60,
        alpha=0.9,
        edgecolor="black",
        label=f"{lower}-{upper}° (N={N}, ρ= {rho:.2f})"
    )

# Formatting
plt.grid(True)
plt.tick_params(axis='both', labelsize=15)
plt.xlabel("AOD", fontsize=20)
plt.ylabel("$\Delta_\\delta$ [$\\circ$]", fontsize=20)
plt.ylim([-6,4])
plt.xticks(np.arange(0.05, 0.25, 0.02))
# Cleaner legend
plt.legend(
    fontsize=14,
    ncol=3,  # fewer columns = easier reading
    frameon=True,
    facecolor='lightgray',
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02)
)

plt.show()

bins = [
    (25, 45, greens[0]),
    (45, 75, greens[6]),
    (75, 90, greens[12]),
]
plt.figure(figsize=(12,6))

# Loop through each SZA region
for i, (lower, upper, color) in enumerate(bins):
    
    # Mask for this bin
    mask = (sza >= lower) & (sza <= upper)
    
    # Downselected data
    dd_ultra = delta_delta_ultrasip[mask]
    s_ssa_bin = ssa[mask]
    
    # Skip empty bins
    if len(dd_ultra) < 2:
        continue
    
    # Correlation
    rho, _ = pearsonr(s_ssa_bin, dd_ultra)
    N = len(dd_ultra)
    
    # Scatter plot for this bin
    plt.scatter(
        s_ssa_bin,
        dd_ultra,
        color=color,
        s=60,
        alpha=0.9,
        edgecolor="black",
        label=f"{lower}-{upper}° (N={N}, ρ= {rho:.2f})"
    )

# Formatting
plt.grid(True)
plt.tick_params(axis='both', labelsize=15)
plt.xlabel("SSA", fontsize=20)
plt.ylabel("$\Delta_\\delta$ [$\\circ$]", fontsize=20)
plt.ylim([-6,4])
#plt.xticks(np.arange(0.05, 0.25, 0.02))
# Cleaner legend
plt.legend(
    fontsize=14,
    ncol=3,  # fewer columns = easier reading
    frameon=True,
    facecolor='lightgray',
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02)
)

plt.show()


bins = [
    (25, 45, greens[0]),
    (45, 75, greens[6]),
    (75, 90, greens[12]),
]
plt.figure(figsize=(12,6))

# Loop through each SZA region
for i, (lower, upper, color) in enumerate(bins):
    
    # Mask for this bin
    mask = (sza >= lower) & (sza <= upper)
    
    # Downselected data
    dd_ultra = delta_delta_ultrasip[mask]
    s_g_bin = g[mask]
    
    # Skip empty bins
    if len(dd_ultra) < 2:
        continue
    
    # Correlation
    rho, _ = pearsonr(s_g_bin, dd_ultra)
    N = len(dd_ultra)
    
    # Scatter plot for this bin
    plt.scatter(
        s_g_bin,
        dd_ultra,
        color=color,
        s=60,
        alpha=0.9,
        edgecolor="black",
        label=f"{lower}-{upper}° (N={N}, ρ= {rho:.2f})"
    )

# Formatting
plt.grid(True)
plt.tick_params(axis='both', labelsize=15)
plt.xlabel("g", fontsize=20)
plt.ylabel("$\Delta_\\delta$ [$\\circ$]", fontsize=20)
plt.ylim([-6,4])
#plt.xticks(np.arange(0.05, 0.25, 0.02))
# Cleaner legend
plt.legend(
    fontsize=14,
    ncol=3,  # fewer columns = easier reading
    frameon=True,
    facecolor='lightgray',
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02)
)

plt.show()




bins = [
    (25, 45, greens[0]),
    (45, 75, greens[6]),
    (75, 90, greens[12]),
]
plt.figure(figsize=(12,6))

# Loop through each SZA region
for i, (lower, upper, color) in enumerate(bins):
    
    # Mask for this bin
    mask = (sza >= lower) & (sza <= upper)
    
    # Downselected data
    dd_ultra = delta_delta_ultrasip[mask]
    s_ae_bin = ae[mask]
    
    # Skip empty bins
    if len(dd_ultra) < 2:
        continue
    
    # Correlation
    rho, _ = pearsonr(s_ae_bin, dd_ultra)
    N = len(dd_ultra)
    
    # Scatter plot for this bin
    plt.scatter(
        s_ae_bin,
        dd_ultra,
        color=color,
        s=60,
        alpha=0.9,
        edgecolor="black",
        label=f"{lower}-{upper}° (N={N}, ρ= {rho:.2f})"
    )

# Formatting
plt.grid(True)
plt.tick_params(axis='both', labelsize=15)
plt.xlabel("AE", fontsize=20)
plt.ylabel("$\Delta_\\delta$ [$\\circ$]", fontsize=20)
plt.ylim([-6,4])
#plt.xticks(np.arange(0.05, 0.25, 0.02))
# Cleaner legend
plt.legend(
    fontsize=14,
    ncol=3,  # fewer columns = easier reading
    frameon=True,
    facecolor='lightgray',
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02)
)

plt.show()










