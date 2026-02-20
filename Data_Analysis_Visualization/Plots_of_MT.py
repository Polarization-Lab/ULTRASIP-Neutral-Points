# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 12:04:57 2026
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

colors = ['red','orange','yellow','green',
          'blue','purple','magenta','olive',
          'magenta','cyan','gray','goldenrod',
          'lightsalmon','deeppink','tan','brown','wheat']
idx=-1
# Load in JSON Files
data_path = "C:/Users/ULTRASIP_1/OneDrive/Desktop/BNP_daily_v3_allfields"
json_files = glob.glob(f'{data_path}/BNP*.json')

# =========================
# Output folder for daily figures
# =========================
figure_dir = os.path.join(data_path, "daily_figures")
os.makedirs(figure_dir, exist_ok=True)

for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        
        idx = idx+1

        day = data["date"]
        sza= data["sun_zenith_deg"]
        delta_obs = data["ultrasip_delta"]
        delta_sim = data["GRASP_delta"]
        
        slope_obs, intercept_obs, r_value, p_value, std_err = linregress(sza, delta_obs)
        slope_sim, intercept_sim, r_value, p_value, std_err = linregress(sza, delta_sim)
        
        data_dict[day] = {
            "sun_zenith": data["sun_zenith_deg"],
            "grasp_delta": data["GRASP_delta"],
            "ultrasip_delta": data["ultrasip_delta"],
            "sphericity": data["Sphericity_Factor(%)"][0],
            "ssa": data["Single_Scattering_Albedo[440nm]"][0] ,
            "aod": data["AOD_Extinction-Total[440nm]"][0],
            "slope_obs": slope_obs,
            "intercept_obs": intercept_obs,
            "slope_sim": slope_sim,
            "intercept_sim": intercept_sim,
            "marker_color": colors[idx]
        }
        
#For each day plot grasp_delta (squares) and ultrasip_deltas (circle) versus sza with each color
# ==========================================
# Plot each day separately
# ==========================================
for day, values in data_dict.items():
    
    sun_zen = np.array(values["sun_zenith"])
    delta_ultrasip = np.array(values["ultrasip_delta"])
    delta_grasp = np.array(values["grasp_delta"])
    
    slope_obs = values["slope_obs"]
    intercept_obs = values["intercept_obs"]
    
    slope_sim = values["slope_sim"]
    intercept_sim = values["intercept_sim"]
    
    aod_val = values["aod"]
    color = values["marker_color"]
    
    # Create new figure for each day
    fig = plt.figure(figsize=(10,6))
    
    # -------------------------
    # Observed (circles)
    # -------------------------
    plt.plot(
        sun_zen,
        delta_ultrasip,
        'o',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Observed (ULTRASIP)"
    )
    
    # Regression line (Observed)
    x_fit = np.linspace(min(sun_zen), max(sun_zen), 100)
    y_fit_obs = slope_obs * x_fit + intercept_obs
    
    
    # -------------------------
    # Simulated (squares)
    # -------------------------
    plt.plot(
        sun_zen,
        delta_grasp,
        's',
        color=color,
        markeredgecolor='black',
        markersize=10,
        label="Simulated (GRASP)"
    )
    
    # # Regression line (Simulated)
    # y_fit_sim = slope_sim * x_fit + intercept_sim
    

    
    # -------------------------
    # Formatting
    # -------------------------
    plt.grid(True, linestyle='--', alpha=0.8)

    plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
    plt.ylabel(r"$\delta$ Zenith [$^\circ$]", fontsize=18)

    plt.title(
        f"{day}\nAOD(440nm) = {aod_val:.3f}",
        fontsize=20
    )

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.legend(fontsize=14, loc='upper right')
    
    plt.ylim([-30,-5])
    
    # Define limits
    xmin = np.min(sun_zen) - 1
    xmax = np.max(sun_zen) + 1

    plt.xlim([xmin, xmax])

    # 10 evenly spaced ticks
    xticks = np.linspace(xmin, xmax, 10)

    # Truncate decimals (no rounding)
    plt.xticks(xticks, [f"{int(x)}" for x in xticks])
    plt.tight_layout()
    plt.show()
    # Save figure (no changes to appearance)
    fig_path = os.path.join(figure_dir, f"{day}.png")
    plt.savefig(fig_path, dpi=400)
    plt.close(fig)



# ==========================================
# Slope & Intercept summary plot (ALL DAYS)
# ==========================================

# ------------------------------------------
# Remove October dates
# ------------------------------------------
valid_dates = sorted([
    d for d in data_dict.keys()
    if "2025_10_" not in d   # removes October (YYYY-10-DD format)
])

slopes_obs = []
intercepts_obs = []
slopes_sim = []
intercepts_sim = []
plot_colors = []

for day in valid_dates:
    slopes_obs.append(data_dict[day]["slope_obs"])
    intercepts_obs.append(data_dict[day]["intercept_obs"])
    slopes_sim.append(data_dict[day]["slope_sim"])
    intercepts_sim.append(data_dict[day]["intercept_sim"])
    plot_colors.append(data_dict[day]["marker_color"])

slopes_obs = np.array(slopes_obs)
intercepts_obs = np.array(intercepts_obs)
slopes_sim = np.array(slopes_sim)
intercepts_sim = np.array(intercepts_sim)

# -------------------------
# Pearson correlations
# -------------------------
r_slope, _ = pearsonr(slopes_obs, slopes_sim)
r_intercept, _ = pearsonr(intercepts_obs, intercepts_sim)

x = np.arange(len(valid_dates))

# =========================
# Create figure
# =========================
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(14,8), sharex=True
)

# -------------------------
# Top: slope vs date
# -------------------------
ax1.plot(x, slopes_obs, '-', color='black', linewidth=1.5)
ax1.plot(x, slopes_sim, '-', color='black', linewidth=1.5)


for i in range(len(valid_dates)):
    # Observed (circles)
    ax1.plot(
        x[i], slopes_obs[i],
        'o',
        markersize=15,
        color=plot_colors[i],
        markeredgecolor='black',
        markeredgewidth=1.2
    )
    # Simulated (squares)
    ax1.plot(
        x[i], slopes_sim[i],
        's',
        markersize=15,
        color=plot_colors[i],
        markeredgecolor='black',
        markeredgewidth=1.2
    )

ax1.set_ylabel("Slope", fontsize=18)
ax1.tick_params(labelsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

ax1.text(
    0.98, 0.05,
    f"r = {r_slope:.3f}",
    transform=ax1.transAxes,
    fontsize=16,
    ha='right',
    va='bottom',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        edgecolor='black',
        alpha=0.9
    )
)

# -------------------------
# Bottom: intercept vs date
# -------------------------
ax2.plot(x, intercepts_obs, '-', color='black', linewidth=1.5)
ax2.plot(x, intercepts_sim, '-', color='black', linewidth=1.5)


for i in range(len(valid_dates)):
    # Observed (circles)
    ax2.plot(
        x[i], intercepts_obs[i],
        'o',
        markersize=15,
        color=plot_colors[i],
        markeredgecolor='black',
        markeredgewidth=1.2
    )
    # Simulated (squares)
    ax2.plot(
        x[i], intercepts_sim[i],
        's',
        markersize=15,
        color=plot_colors[i],
        markeredgecolor='black',
        markeredgewidth=1.2
    )

ax2.set_ylabel("Intercept", fontsize=18)
ax2.set_xlabel("Date", fontsize=18)

ax2.tick_params(labelsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)

# -------------------------
# Format x-axis as dates
# -------------------------
ax2.set_xticks(x)
ax2.set_xticklabels(valid_dates, rotation=45, ha='right')

ax2.text(
    0.98, 0.05,
    f"r = {r_intercept:.3f}",
    transform=ax2.transAxes,
    fontsize=16,
    ha='right',
    va='bottom',
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        edgecolor='black',
        alpha=0.9
    )
)


# ------------------------------------------
# Universal Legend (black markers only)
# ------------------------------------------
legend_elements = [
    Line2D(
        [0], [0],
        marker='o',
        color='black',
        linestyle='None',
        markersize=12,
        markeredgecolor='black',
        label='Observed'
    ),
    Line2D(
        [0], [0],
        marker='s',
        color='black',
        linestyle='None',
        markersize=12,
        markeredgecolor='black',
        label='Simulated'
    )
]

fig.legend(
    handles=legend_elements,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),  # move higher (increase second number)
    ncol=2,
    fontsize=16,
    frameon=True
)

plt.tight_layout()
plt.show()


# # =========================================
# # Movie Writer Setup (HIGH RESOLUTION)
# # ==========================================
# Writer = animation.FFMpegWriter
# writer = Writer(fps=1, bitrate=5000)  # high bitrate for sharp output

# # Dates to separate
# october_dates = ["2025_10_22", "2025_10_23", "2025_10_24"]

# # Sort chronologically
# all_days = sorted(data_dict.keys())

# non_oct_days = [d for d in all_days if d not in october_dates]
# oct_days = [d for d in all_days if d in october_dates]


# # ==========================================
# # Function to render movie directly
# # ==========================================
# def make_movie(day_list, output_name):

#     if len(day_list) == 0:
#         print(f"No frames for {output_name}")
#         return

#     fig = plt.figure(figsize=(10,6), dpi=500)

#     with writer.saving(fig, output_name, dpi=500):

#         for day in day_list:

#             plt.clf()

#             values = data_dict[day]

#             sun_zen = np.array(values["sun_zenith"])
#             delta_ultrasip = np.array(values["ultrasip_delta"])
#             delta_grasp = np.array(values["grasp_delta"])
#             aod_val = values["aod"]
#             color = values["marker_color"]

#             # Observed
#             plt.plot(
#                 sun_zen,
#                 delta_ultrasip,
#                 'o',
#                 color=color,
#                 markeredgecolor='black',
#                 markersize=10,
#                 label="Observed (ULTRASIP)"
#             )

#             # Simulated
#             plt.plot(
#                 sun_zen,
#                 delta_grasp,
#                 's',
#                 color=color,
#                 markeredgecolor='black',
#                 markersize=10,
#                 label="Simulated (GRASP)"
#             )

#             plt.grid(True, linestyle='--', alpha=0.8)
#             plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
#             plt.ylabel(r"$\delta$ Zenith [$^\circ$]", fontsize=18)
#             plt.title(f"{day}\nAOD(440nm) = {aod_val:.3f}", fontsize=20)

#             plt.xticks(fontsize=16)
#             plt.yticks(fontsize=16)
#             plt.legend(fontsize=14, loc='upper right')

#             plt.ylim([-30, -5])

#             xmin = np.min(sun_zen) - 1
#             xmax = np.max(sun_zen) + 1
#             plt.xlim([xmin, xmax])

#             xticks = np.linspace(xmin, xmax, 10)
#             plt.xticks(xticks, [f"{int(x)}" for x in xticks])

#             plt.tight_layout()

#             writer.grab_frame()

#     plt.close(fig)
#     print(f"Saved: {output_name}")

# # ==========================================
# # Create AOD-Sorted Movie (Non-October Only)
# # ==========================================

# # Sort non-October days by increasing AOD
# non_oct_days_sorted_aod = sorted(
#     non_oct_days,
#     key=lambda d: data_dict[d]["aod"]
# )

# make_movie(non_oct_days_sorted_aod,
#            "Daily_All_EXCEPT_October_Sorted_By_AOD.mp4")

# # ==========================================
# # Create Both Movies
# # ==========================================
# make_movie(non_oct_days, "Daily_All_EXCEPT_October.mp4")
# make_movie(oct_days, "Daily_October_Only.mp4")
