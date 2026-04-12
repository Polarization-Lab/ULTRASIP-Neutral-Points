# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 10:12:07 2026

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
# GLOBAL ABSOLUTE DIFFERENCE ANALYSIS (EXCLUDING OCTOBER)
# ==========================================

all_abs_diff = []   # |GRASP - ULTRASIP|
all_days = []       # track day for each point
all_sza = []        # track SZA

for day, values in data_dict.items():

    # ---- Skip October ----
    month = day.split("_")[1]   # assumes format YYYY_MM_DD
    if month == "10":
        continue

    delta_obs = np.array(values["ultrasip_delta"])
    delta_sim = np.array(values["grasp_delta"])
    sza = np.array(values["sun_zenith"])

    abs_diff = np.abs(delta_sim - delta_obs)

    all_abs_diff.extend(abs_diff)
    all_days.extend([day]*len(abs_diff))
    all_sza.extend(sza)

# Convert to numpy arrays
all_abs_diff = np.array(all_abs_diff)
all_days = np.array(all_days)
all_sza = np.array(all_sza)

# ---- Global statistics ----
max_abs_diff = np.max(all_abs_diff)
min_abs_diff = np.min(all_abs_diff)

max_idx = np.argmax(all_abs_diff)
min_idx = np.argmin(all_abs_diff)

mean_abs_diff = np.mean(all_abs_diff)
std_abs_diff = np.std(all_abs_diff)

print("==========================================")
print("GLOBAL |Δ (GRASP - ULTRASIP)| STATISTICS")
print("EXCLUDING OCTOBER")
print("==========================================")

print(f"Maximum |difference|: {max_abs_diff:.3f} deg")
print(f"   Day: {all_days[max_idx]}")
print(f"   SZA: {all_sza[max_idx]:.2f}")

print(f"\nMinimum |difference|: {min_abs_diff:.3f} deg")
print(f"   Day: {all_days[min_idx]}")
print(f"   SZA: {all_sza[min_idx]:.2f}")

print(f"\nMean |difference|: {mean_abs_diff:.3f} deg")
print(f"Std dev: {std_abs_diff:.3f} deg")

# ==========================================
# UNIQUE VALUES OF AOD AND SSA (PER DAY)
# ==========================================

print("==========================================")
print("UNIQUE AOD & SSA VALUES PER DAY")
print("==========================================")

for day, values in data_dict.items():

    aod_vals = np.array(values["aod"])
    ssa_vals = np.array(values["ssa"])

    # ---- Raw unique values ----
    unique_aod = np.unique(aod_vals)
    unique_ssa = np.unique(ssa_vals)
    
    delta_obs = np.array(values["ultrasip_delta"])


    # ---- Rounded (recommended for floats) ----
    unique_aod_round = np.unique(np.round(aod_vals, 3))
    unique_ssa_round = np.unique(np.round(ssa_vals, 3))
    
    n_delta = len(delta_obs)
    
    print(f"  Number of δ_obs points: {n_delta}")

    print(f"\nDay: {day}")
    print(f"  AOD unique (raw): {len(unique_aod)}")
    print(f"  SSA unique (raw): {len(unique_ssa)}")

    print(f"  AOD unique (rounded, 3dp): {len(unique_aod_round)}")
    print(f"  SSA unique (rounded, 3dp): {len(unique_ssa_round)}")

    # Optional: ranges 
    print(f"  AOD range: {aod_vals.min():.3f} → {aod_vals.max():.3f}")
    print(f"  SSA range: {ssa_vals.min():.3f} → {ssa_vals.max():.3f}")
# ==========================================
# GROUPED MULTI-DAY PLOT
# ==========================================

# ---- SELECT DAYS TO PLOT ----
selected_days = [
    "2025_06_09"
    ]

fig, ax = plt.subplots(figsize=(10,6))

legend_elements = []

# Loop through selected days
for i, day in enumerate(selected_days):

    if day not in data_dict:
        print(f"{day} not found, skipping")
        continue

    values = data_dict[day]

    sun_zen = np.array(values["sun_zenith"])
    delta_ultrasip = np.array(values["ultrasip_delta"])
    delta_grasp = np.array(values["grasp_delta"])
    ray_delta = np.array(values["ray_delta"])

    color = values["marker_color"]

    # ---- Correlation ----
    rho, _ = pearsonr(delta_ultrasip, delta_grasp)

    # ---- Plot lines ----

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
        ray_delta,
        'v',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Simulated (GRASP)"
    )
    
    # Simulated
    plt.plot(
        sun_zen,
        ray_delta,
        'v',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Simulated (GRASP)"
    )

    legend_elements.append(
        Line2D(
            [0], [0],
            color=color,
            lw=3,
            label=f"{day}  ($\\rho$={rho:.3f})",
            path_effects=[
                pe.Stroke(linewidth=5, foreground='black'),  
                pe.Normal()
                ]
            )
        )

# ==========================================
# AXES FORMATTING
# ==========================================
ax.set_xlabel(r"Sun Zenith Angle [$\circ$]", fontsize=18)
ax.set_ylabel(r"$\delta$ Zenith [$\circ$]", fontsize=18)

ax.set_ylim([-30, -4])
ax.set_xlim([24, 89])

ax.grid(True, linestyle='--', alpha=0.8)

ax.tick_params(labelsize=16)

# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.xaxis.set_major_locator(MultipleLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(4))

# ---- Minor ticks (grid every 2 degrees) ----
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(2))

# ---- Grid styling ----
ax.grid(which='major', linestyle='--', alpha=0.8)
ax.grid(which='minor', linestyle=':', alpha=0.5)

# ==========================================
# LEGENDS
# ==========================================

# # ==========================================
# # MARKER LEGEND (TOP, OUTSIDE AXES)
# # ==========================================

# marker_handles = [
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='black',
#         linestyle='None',
#         markersize=14,   
#         label='Observed'
#     ),
#     Line2D(
#         [0], [0],
#         marker='s',
#         color='black',
#         linestyle='None',
#         markersize=14,   
#         label='Simulated'
#     )
# ]

# marker_legend = ax.legend(
#     handles=marker_handles,
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.15),  
#     ncol=2,
#     fontsize=18,
#     frameon=True,
#     edgecolor='black'
# )

#ax.add_artist(marker_legend)

# Colored line legend (inside plot)
ax.legend(
    handles=legend_elements,
    loc='upper right',
    fontsize=18,
    frameon=True,
    facecolor='white',
    edgecolor='black'
)

plt.tight_layout()
plt.show()

# ==========================================
# GROUPED MULTI-DAY PLOT
# ==========================================

# ---- SELECT DAYS TO PLOT ----
selected_days = [
    "2025_06_04",
    "2025_06_09",
    "2025_06_18",
    "2025_06_23",
    "2025_06_24",
    "2025_06_25",
    "2025_06_30",
    "2025_07_01",
    "2025_07_08",
    "2025_07_09",
    "2025_07_10",
    "2025_07_13",
    "2025_07_17",
    "2025_07_18"
]

fig, ax = plt.subplots(figsize=(10,6))

legend_elements = []

sd_table = []

# Loop through selected days
for i, day in enumerate(selected_days):

    if day not in data_dict:
        print(f"{day} not found, skipping")
        continue

    values = data_dict[day]

    sun_zen = np.array(values["sun_zenith"])
    delta_ultrasip = np.array(values["ultrasip_delta"])
    delta_grasp = np.array(values["grasp_delta"])
    ray_delta = np.array(values["ray_delta"])
    
    delta_delta_ultrasip = delta_ultrasip - ray_delta
    delta_delta_grasp = delta_grasp - ray_delta
    
    # ---- Standard deviations ----
    sd_obs = np.std(delta_delta_ultrasip)
    sd_sim = np.std(delta_delta_grasp)

    sd_table.append((day, sd_obs, sd_sim))

    color = values["marker_color"]

    # ---- Correlation ----
    #rho, _ = pearsonr(delta_delta_ultrasip, delta_delta_grasp)
    rho,_ = pearsonr(delta_delta_ultrasip,sun_zen)
    print("rho",rho)

    # ---- Plot lines ----

    # Observed
    plt.plot(
        sun_zen,
        delta_delta_ultrasip,
        'o',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Observed (ULTRASIP)"
    )

    # # Simulated
    # plt.plot(
    #     sun_zen,
    #     delta_delta_grasp,
    #     's',
    #     color=color,
    #     markeredgecolor='black',
    #     markersize=10,
    #     label="Simulated (GRASP)"
    # )
    
    plt.axhline(0, color='black', linestyle='-', linewidth=2)

    # ---- Legend entry (NO SD now) ----
    legend_elements.append( 
        Line2D( [0], [0], color=color, lw=3, 
               label=( f"{day} ($\\rho$={rho:.3f})\n" f"$SD_{{obs}}$={sd_obs:.2f}, $SD_{{sim}}$={sd_sim:.2f}" ), 
               path_effects=[ pe.Stroke(linewidth=5, foreground='black'), pe.Normal() 
        ] ) )

# ==========================================
# AXES FORMATTING
# ==========================================
ax.set_xlabel(r"Sun Zenith Angle [$\circ$]", fontsize=18)
ax.set_ylabel(r"$\Delta_\delta$ [$\circ$]", fontsize=18)

ax.set_ylim([-1, 8])
ax.set_xlim([24, 89])

ax.tick_params(labelsize=16)

ax.xaxis.set_major_locator(MultipleLocator(4))
ax.yaxis.set_major_locator(MultipleLocator(1))

# ---- Minor ticks ----
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

# ---- Grid ----
ax.grid(which='major', linestyle='--', alpha=0.8)
ax.grid(which='minor', linestyle=':', alpha=0.5)

# ==========================================
# LEGEND
# ==========================================
# ax.legend(
#     handles=legend_elements,
#     loc='upper right',
#     fontsize=18,
#     frameon=True,
#     facecolor='white',
#     edgecolor='black'
# )

plt.tight_layout()
plt.show()

# ==========================================
# PRINT SD TABLE
# ==========================================

print("\n==========================================")
print("STANDARD DEVIATION OF Δδ (PER DAY)")
print("==========================================")
print(f"{'Date':<12}{'SD_obs':<12}{'SD_sim':<12}")

for day, sd_obs, sd_sim in sd_table:
    print(f"{day:<12}{sd_obs:<12.2f}{sd_sim:<12.2f}")

# ==========================================
# LATEX TABLE OUTPUT
# ==========================================

print("\nLaTeX Table:\n")
print("\\begin{tabular}{ccc}")
print("\\hline")
print("Date & $SD_{obs}$ & $SD_{sim}$ \\\\")
print("\\hline")

for day, sd_obs, sd_sim in sd_table:
    mmdd = f"{day.split('_')[1]}/{day.split('_')[2]}"
    print(f"{mmdd} & {sd_obs:.2f} & {sd_sim:.2f} \\\\")

print("\\hline")
print("\\end{tabular}")

#------------Comparison at certain zenith range---------------------#

selected_days = [
    "2025_06_04",
    "2025_06_09",
    "2025_06_18",
    "2025_06_23",
    "2025_06_24",
    "2025_06_25",
    "2025_06_30",
    "2025_07_01",
    "2025_07_08",
    "2025_07_09",
    "2025_07_10",
    "2025_07_13",
    "2025_07_17",
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

sza_min = 40 #max(mins)
sza_max = 80 #min(maxs)

print(f"\nCommon SZA overlap: {sza_min:.1f}–{sza_max:.1f}")

plt.figure(figsize=(10,6))

# --------------------------------
# Loop through days
# --------------------------------
for day in selected_days:

    values = data_dict[day]

    sza = np.array(values["sun_zenith"])
    delta = np.array(values["ultrasip_delta"]) 
    #ray = np.array(values["ray_delta"])
    print("obs og",len(delta))
    print("sza range og",np.max(sza)-np.min(sza))

    #delta_delta = delta - ray

    color = values["marker_color"]

    # mask overlap region
    mask = (sza >= sza_min) & (sza <= sza_max)

    sza_overlap = sza[mask]
    delta_overlap = delta[mask]
    
    print("obs mask",len(delta_overlap))

    mean_val = np.mean(delta_overlap)
    std_val = np.std(delta_overlap)

    # improved legend formatting
    label_text = f"{day}   μ={mean_val:.2f}°,  SD={std_val:.2f}°"

    # --------------------------------
    # plot all points (faint)
    # --------------------------------
    plt.scatter(
        sza,
        delta,
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

plt.xlabel(r"Sun Zenith Angle [$\circ$]", fontsize=18)
plt.ylabel(r"$\delta$ [$\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim([24,88])
plt.ylim([-30,-4])

#plt.legend(fontsize=14)

plt.title(
    f"$\delta$ Comparison Across Days\nCommon SZA: {sza_min:.2f}–{sza_max:.2f}",
    fontsize=20
)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))

# --------------------------------
# Loop through days
# --------------------------------
for day in selected_days:

    values = data_dict[day]

    sza = np.array(values["sun_zenith"])
    delta = np.array(values["ultrasip_delta"]) - np.array(values["ray_delta"])
    #ray = np.array(values["ray_delta"])
    print("obs og",len(delta))
    print("sza range og",np.max(sza)-np.min(sza))

    #delta_delta = delta - ray

    color = values["marker_color"]

    # mask overlap region
    mask = (sza >= sza_min) & (sza <= sza_max)

    sza_overlap = sza[mask]
    delta_overlap = delta[mask]
    
    print("obs mask",len(delta_overlap))

    mean_val = np.mean(delta_overlap)
    std_val = np.std(delta_overlap)

    # improved legend formatting
    label_text = f"{day}   μ={mean_val:.2f}°,  SD={std_val:.2f}°"

    # --------------------------------
    # plot all points (faint)
    # --------------------------------
    plt.scatter(
        sza,
        delta,
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

plt.xlabel(r"Sun Zenith Angle [$\circ$]", fontsize=18)
plt.ylabel(r"$\Delta_\delta$ [$\circ$]", fontsize=18)

plt.grid(True, linestyle='--', alpha=0.7)

plt.xlim([24, 89])
plt.ylim([1,8])

plt.legend(fontsize=14)

plt.title(
    f"$\delta$ Comparison Across Days\nCommon SZA: {sza_min:.2f}–{sza_max:.2f}",
    fontsize=20
)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.show()

# # ==========================================
# # STANDALONE MARKER LEGEND FIGURE
# # ==========================================

# fig_legend = plt.figure(figsize=(10,4))
# ax_leg = fig_legend.add_subplot(111)

# # Remove axes completely
# ax_leg.axis('off')

# # Create large black marker handles
# legend_handles = [
#     Line2D(
#         [0], [0],
#         marker='o',
#         color='black',
#         linestyle='None',
#         markersize=16,
#         label='ULTRASIP (Observed)'
#     ),
#     Line2D(
#         [0], [0],
#         marker='s',
#         color='black',
#         linestyle='None',
#         markersize=16,
#         label='GRASP-AERONET (Simulated)'
#     )
# ]

# # Centered legend
# ax_leg.legend(
#     handles=legend_handles,
#     loc='center',
#     ncol=3,
#     fontsize=18,
#     frameon=True,
#     edgecolor='black'
# )

# plt.tight_layout()
# plt.show()

# # Save
# legend_path = os.path.join(figure_dir, "marker_legend.png")
# fig_legend.savefig(legend_path, dpi=400, bbox_inches='tight')
# plt.close(fig_legend)