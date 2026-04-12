# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 12:48:28 2026

@author: deleo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 12:47:44 2026

@author: C.M.DeLeon
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
import imageio.v2 as imageio


data_dict = {}
date = []
sun_zenith = []
grasp_delta = []
ultrasip_delta = []
sphericity = []
ssa = []
aod = []

colors = ['red','darkorange','yellow','green',
          'blue','purple','magenta','cadetblue',
          'silver','cyan','gray','lime',
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
    sza = np.array(data["sun_zenith_deg"])

    delta_obs = np.array(data["ultrasip_delta"])
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

# ==========================================
# ORIGINAL DAILY PLOTS
# ==========================================
for day, values in data_dict.items():

    sun_zen = np.array(values["sun_zenith"])
    delta_ultrasip = np.array(values["ultrasip_delta"])
    delta_grasp = np.array(values["grasp_delta"])
    delta_ray = np.array(values["ray_delta"])

    slope_obs = values["slope_obs"]
    intercept_obs = values["intercept_obs"]

    slope_sim = values["slope_sim"]
    intercept_sim = values["intercept_sim"]

    aod_val = np.average(values["aod"])
    color = values["marker_color"]
    
    rho_delta, p_value = pearsonr(delta_ultrasip,delta_grasp)
    rho_ultraray, p_value = pearsonr(delta_ultrasip,delta_ray)
    rho_graspray, p_value = pearsonr(delta_grasp,delta_ray)

    fig = plt.figure(figsize=(10,6))

    # Observed
    plt.plot(
        sun_zen,
        delta_ultrasip,
        'o',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Observed (ULTRASIP)"
    )

    # Simulated
    plt.plot(
        sun_zen,
        delta_grasp,
        's',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Simulated (GRASP)"
    )

    # Rayleigh
    plt.plot(
        sun_zen,
        delta_ray,
        'v',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Rayleigh"
    )

    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
    plt.ylabel(r"$\delta$ Zenith [$^\circ$]", fontsize=18)

    plt.title(
        f"{day}\n Daily Average AOD(440nm) = {aod_val:.3f}",
        fontsize=20
    )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper right')

    plt.ylim([-30,-5])
    plt.xlim([24,89])
    
    plt.text(
        30,
        -27,
        f"$\\rho_{{obs,sim}}$ = {rho_delta:.4f}\n $\\rho_{{obs,ray}}$ = {rho_ultraray:.4f}\n$\\rho_{{sim,ray}}$ = {rho_graspray:.4f}",
        fontsize=16,
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round'
        )
    )

    plt.tight_layout()

    fig_path = os.path.join(figure_dir, f"{day}.png")
    plt.savefig(fig_path, dpi=400)
    plt.close(fig)

# ==========================================
# DELTA FROM RAYLEIGH PLOTS
# ==========================================
for day, values in data_dict.items():

    sun_zen = np.array(values["sun_zenith"])

    delta_delta_obs = np.array(values["delta_delta_obs"])
    delta_delta_sim = np.array(values["delta_delta_sim"])
    
    rho, p_value = pearsonr(delta_delta_obs,delta_delta_sim)

    aod_val = np.average(values["aod"])
    color = values["marker_color"]

    std_obs = np.std(delta_delta_obs)
    std_sim = np.std(delta_delta_sim)

    fig = plt.figure(figsize=(10,6))

    plt.plot(
        sun_zen,
        delta_delta_obs,
        'o',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Observed (ULTRASIP)"
    )

    plt.plot(
        sun_zen,
        delta_delta_sim,
        's',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Simulated (GRASP)"
    )

    plt.axhline(
        0,
        color='black',
        linestyle='--',
        linewidth=2,
        label="Rayleigh Reference"
    )

    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
    plt.ylabel(r"$\Delta_{\delta}$  [$^\circ$]", fontsize=18)

    plt.title(
        f"{day}\n Daily Average AOD(440nm) = {aod_val:.3f}",
        fontsize=20
    )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper right', ncol=3)

    plt.xlim([24,89])
    plt.ylim([-3,8])

    plt.text(
        0.98, 0.05,
        f"$SD_{{obs}}$ = {std_obs:.4f}°\n$SD_{{sim}}$ = {std_sim:.4f}°\n$\\rho_{{obs,sim}}$ = {rho:.4f}",
        transform=plt.gca().transAxes,
        fontsize=14,
        ha='right',
        va='bottom',
        bbox=dict(
            boxstyle='round',
            facecolor='white',
            edgecolor='black',
            alpha=0.9
        )
    )

    plt.tight_layout()

    fig_path = os.path.join(figure_dir, f"{day}_delta_from_rayleigh.png")
    plt.savefig(fig_path, dpi=400)
    plt.close(fig)


# ==========================================
# CREATE MOVIE 1: ORIGINAL DAILY SERIES
# ==========================================

daily_images = sorted(glob.glob(os.path.join(figure_dir, "*.png")))

movie_path = os.path.join(figure_dir, "daily_series.mp4")

with imageio.get_writer(movie_path, fps=2) as writer:
    for filename in daily_images:
        if "delta_from_rayleigh" not in filename:
            image = imageio.imread(filename)
            writer.append_data(image)

print("Daily movie saved:", movie_path)


# ==========================================
# CREATE MOVIE 2: DELTA FROM RAYLEIGH SERIES
# ==========================================

delta_images = sorted(glob.glob(os.path.join(figure_dir, "*delta_from_rayleigh.png")))

movie_path = os.path.join(figure_dir, "delta_from_rayleigh_series.mp4")

with imageio.get_writer(movie_path, fps=2) as writer:
    for filename in delta_images:
        image = imageio.imread(filename)
        writer.append_data(image)

print("Delta-from-Rayleigh movie saved:", movie_path)