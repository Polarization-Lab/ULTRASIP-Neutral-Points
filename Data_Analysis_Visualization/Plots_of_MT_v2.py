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

    plt.tight_layout()
    plt.show()

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

    aod_val = np.average(values["aod"])
    color = values["marker_color"]

    # ----------------------------------
    # Standard deviations
    # ----------------------------------
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

    # Rayleigh reference
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

    # ----------------------------------
    # Report standard deviation
    # ----------------------------------
    plt.text(
        0.98, 0.05,
        f"$\\sigma_{{obs}}$ = {std_obs:.2f}°\n$\\sigma_{{sim}}$ = {std_sim:.2f}°",
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
    plt.show()

    fig_path = os.path.join(figure_dir, f"{day}_delta_from_rayleigh.png")
    plt.savefig(fig_path, dpi=400)
    plt.close(fig)

# ==========================================
# SLOPE / INTERCEPT SUMMARY PLOT
# ==========================================

valid_dates = sorted([
    d for d in data_dict.keys()
    if "2025_10_" not in d
])

slopes_obs = []
intercepts_obs = []
slopes_sim = []
intercepts_sim = []
slopes_ray = []
intercepts_ray = []
plot_colors = []

for day in valid_dates:

    slopes_obs.append(data_dict[day]["slope_obs"])
    intercepts_obs.append(data_dict[day]["intercept_obs"])
    slopes_sim.append(data_dict[day]["slope_sim"])
    intercepts_sim.append(data_dict[day]["intercept_sim"])
    slopes_ray.append(data_dict[day]["slope_ray"])
    intercepts_ray.append(data_dict[day]["intercept_ray"])
    plot_colors.append(data_dict[day]["marker_color"])

slopes_obs = np.array(slopes_obs)
intercepts_obs = np.array(intercepts_obs)
slopes_sim = np.array(slopes_sim)
intercepts_sim = np.array(intercepts_sim)
slopes_ray = np.array(slopes_ray)
intercepts_ray = np.array(intercepts_ray)

r_slope, _ = pearsonr(slopes_obs, slopes_sim)
r_intercept, _ = pearsonr(intercepts_obs, intercepts_sim)

valid_dates = sorted(
    valid_dates,
    key=lambda d: np.average(data_dict[d]["aod"])
)

aod_values = np.array([np.average(data_dict[d]["aod"]) for d in valid_dates])
x = np.arange(len(valid_dates))

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14,8), sharex=True
)

ax1.plot(x, slopes_obs, '-', color='black', linewidth=1.5)
ax1.plot(x, slopes_sim, '-', color='black', linewidth=1.5)
ax1.plot(x, slopes_ray, '-', color='black', linewidth=1.5)

for i in range(len(valid_dates)):

    ax1.plot(x[i], slopes_obs[i],'o',markersize=15,color=plot_colors[i],markeredgecolor='black')
    ax1.plot(x[i], slopes_sim[i],'s',markersize=15,color=plot_colors[i],markeredgecolor='black')
    ax1.plot(x[i], slopes_ray[i],'v',markersize=15,color=plot_colors[i],markeredgecolor='black')

ax1.set_ylabel("Slope", fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.plot(x, intercepts_obs, '-', color='black', linewidth=1.5)
ax2.plot(x, intercepts_sim, '-', color='black', linewidth=1.5)
ax2.plot(x, intercepts_ray, '-', color='black', linewidth=1.5)

for i in range(len(valid_dates)):

    ax2.plot(x[i], intercepts_obs[i],'o',markersize=15,color=plot_colors[i],markeredgecolor='black')
    ax2.plot(x[i], intercepts_sim[i],'s',markersize=15,color=plot_colors[i],markeredgecolor='black')
    ax2.plot(x[i], intercepts_ray[i],'v',markersize=15,color=plot_colors[i],markeredgecolor='black')

ax2.set_ylabel("Intercept", fontsize=18)
ax2.set_xlabel("Date", fontsize=18)

ax2.set_xticks(x)
ax2.set_xticklabels(valid_dates, rotation=45, ha='right')

ax_top = ax1.twiny()
ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks(x)
ax_top.set_xticklabels([f"{a:.3f}" for a in aod_values], rotation=45, ha='left')
ax_top.set_xlabel("Daily Average AOD (440 nm)", fontsize=18)

legend_elements = [
    Line2D([0],[0],marker='o',color='black',linestyle='None',markersize=12,label='Observed'),
    Line2D([0],[0],marker='s',color='black',linestyle='None',markersize=12,label='Simulated'),
    Line2D([0],[0],marker='v',color='black',linestyle='None',markersize=12,label='Rayleigh')
]

fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5,1.1),
    ncol=3,
    fontsize=16
)

plt.tight_layout()
plt.show()

# ==========================================
# MULTI-DAY Δδ COMPARISON
# ==========================================

# Choose subset of days
selected_days = [
    "2025_06_23",
    "2025_06_24",
    "2025_06_25"
]

plt.figure(figsize=(10,6))

for day in selected_days:

    values = data_dict[day]

    sza = np.array(values["sun_zenith"])

    delta_obs = np.array(values["ultrasip_delta"])
    delta_sim = np.array(values["grasp_delta"])
    delta_ray = np.array(values["ray_delta"])

    # Δδ relative to each day's Rayleigh
    delta_delta_obs = delta_obs - delta_ray
    delta_delta_sim = delta_sim - delta_ray

    color = values["marker_color"]

    # Observations
    plt.plot(
        sza,
        delta_delta_obs,
        'o',
        markersize=10,
        color=color,
        markeredgecolor='black',
        label=f"{day} Obs"
    )

    # Simulations
    plt.plot(
        sza,
        delta_delta_sim,
        's',
        markersize=10,
        color=color,
        markeredgecolor='black',
        label=f"{day} Sim"
    )

# Rayleigh reference
plt.axhline(
    0,
    color='black',
    linestyle='--',
    linewidth=2,
    label="Rayleigh"
)

# Formatting
plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
plt.ylabel(r"$\Delta_{\delta}$ [$^\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim([24,89])
plt.ylim([-3,8])

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.legend(fontsize=12)

plt.tight_layout()
plt.show()


#------------Comparison at certain zenith range---------------------#

selected_days = [
    "2025_06_24",
    "2025_06_25",
    "2025_07_18"
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
    delta = np.array(values["ultrasip_delta"])
    ray = np.array(values["ray_delta"])

    delta_delta = delta - ray

    color = values["marker_color"]

    # mask overlap region
    mask = (sza >= sza_min) & (sza <= sza_max)

    sza_overlap = sza[mask]
    delta_overlap = delta_delta[mask]

    mean_val = np.mean(delta_overlap)
    std_val = np.std(delta_overlap, ddof=1)

    # improved legend formatting
    label_text = f"{day}   μ={mean_val:.2f}°,  σ={std_val:.2f}°"

    # --------------------------------
    # plot all points (faint)
    # --------------------------------
    plt.scatter(
        sza,
        delta_delta,
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

plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
plt.ylabel(r"$\Delta_{\delta}$ [$^\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim([24,89])
plt.ylim([-3,8])

plt.legend(fontsize=14)

plt.title(
    f"$\Delta_\delta$ Comparison Across Days\nCommon SZA: {sza_min:.1f}–{sza_max:.1f}",
    fontsize=20
)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

#-----------------Compare Sim across zenith range------------------#

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
    delta = np.array(values["grasp_delta"])
    ray = np.array(values["ray_delta"])

    delta_delta = delta - ray

    color = values["marker_color"]

    # mask overlap region
    mask = (sza >= sza_min) & (sza <= sza_max)

    sza_overlap = sza[mask]
    delta_overlap = delta_delta[mask]

    mean_val = np.mean(delta_overlap)
    std_val = np.std(delta_overlap, ddof=1)

    # improved legend formatting
    label_text = f"{day}   μ={mean_val:.2f}°,  σ={std_val:.2f}°"

    # --------------------------------
    # plot all points (faint)
    # --------------------------------
    plt.scatter(
        sza,
        delta_delta,
        marker='s',
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
        marker='s',
        color=color,
        edgecolor='black',
        s=120,
        label=label_text
    )

# --------------------------------
# overlap region shading
# --------------------------------
plt.axvspan(sza_min, sza_max, color='gray', alpha=0.12)

plt.axhline(0, color='black', linestyle='--', linewidth=2)

plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
plt.ylabel(r"$\Delta_{\delta}$ [$^\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim([24,89])
plt.ylim([-3,8])

plt.legend(fontsize=14)

plt.title(
    f"$\Delta_\delta$ Comparison Across Days\nCommon SZA: {sza_min:.1f}–{sza_max:.1f}",
    fontsize=20
)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()