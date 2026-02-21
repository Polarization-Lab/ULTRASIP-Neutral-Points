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

colors = ['red','darkorange','yellow','green',
          'blue','purple','magenta','cadetblue',
          'silver','cyan','gray','lime',
          'honeydew','palevioletred','tan','brown','mediumorchid']
idx=-1
# Load in JSON Files
#data_path = "C:/Users/ULTRASIP_1/OneDrive/Desktop/BNP_daily_v3_allfields"
data_path ="C:/Users/deleo/Downloads/BNP_daily_v3_allfields/BNP_daily_v3_allfields"
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
            "sphericity": data["Sphericity_Factor(%)"],
            "ssa": data["Single_Scattering_Albedo[440nm]"],
            "aod": data["AOD_Extinction-Total[440nm]"],
            "g": data["Asymmetry_Factor-Total[440nm]"],
            "ae": data["Extinction_Angstrom_Exponent_440-870nm-Total"],
            "slope_obs": slope_obs,
            "intercept_obs": intercept_obs,
            "slope_sim": slope_sim,
            "intercept_sim": intercept_sim,
            "marker_color": colors[idx]
        }
        
# #For each day plot grasp_delta (squares) and ultrasip_deltas (circle) versus sza with each color
# # ==========================================
# # Plot each day separately
# # ==========================================
# for day, values in data_dict.items():
    
#     sun_zen = np.array(values["sun_zenith"])
#     delta_ultrasip = np.array(values["ultrasip_delta"])
#     delta_grasp = np.array(values["grasp_delta"])
    
#     slope_obs = values["slope_obs"]
#     intercept_obs = values["intercept_obs"]
    
#     slope_sim = values["slope_sim"]
#     intercept_sim = values["intercept_sim"]
    
#     aod_val = np.average(values["aod"])
#     color = values["marker_color"]
    
#     # Create new figure for each day
#     fig = plt.figure(figsize=(10,6))
    
#     # -------------------------
#     # Observed (circles)
#     # -------------------------
#     plt.plot(
#         sun_zen,
#         delta_ultrasip,
#         'o',
#         color=color,
#         markeredgecolor='black',
#         markersize=10,
#         label="Observed (ULTRASIP)"
#     )
    
#     # Regression line (Observed)
#     x_fit = np.linspace(min(sun_zen), max(sun_zen), 100)
#     y_fit_obs = slope_obs * x_fit + intercept_obs
    
    
#     # -------------------------
#     # Simulated (squares)
#     # -------------------------
#     plt.plot(
#         sun_zen,
#         delta_grasp,
#         's',
#         color=color,
#         markeredgecolor='black',
#         markersize=10,
#         label="Simulated (GRASP)"
#     )
    
#     # # Regression line (Simulated)
#     # y_fit_sim = slope_sim * x_fit + intercept_sim
    

    
#     # -------------------------
#     # Formatting
#     # -------------------------
#     plt.grid(True, linestyle='--', alpha=0.8)

#     plt.xlabel(r"Sun Zenith Angle [$^\circ$]", fontsize=18)
#     plt.ylabel(r"$\delta$ Zenith [$^\circ$]", fontsize=18)

#     plt.title(
#         f"{day}\n Daily Average AOD(440nm) = {aod_val:.3f}",
#         fontsize=20
#     )

#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)

#     plt.legend(fontsize=14, loc='upper right')
    
#     plt.ylim([-30,-5])

    
#     # Define limits
#     xmin = np.min(sun_zen) - 1
#     xmax = np.max(sun_zen) + 1

#     plt.xlim([xmin, xmax])

#     # 10 evenly spaced ticks
#     xticks = np.linspace(xmin, xmax, 10)

#     # Truncate decimals (no rounding)
#     plt.xticks(xticks, [f"{int(x)}" for x in xticks])
#     plt.tight_layout()
#     plt.show()
#     # Save figure (no changes to appearance)
#     fig_path = os.path.join(figure_dir, f"{day}.png")
#     plt.savefig(fig_path, dpi=400)
#     plt.close(fig)


# ==========================================
# Polar Plot Each Day
# Angle  = Sun Zenith Angle
# ==========================================

for day, values in data_dict.items():
    
    sun_zen = np.array(values["sun_zenith"])
    delta_ultrasip = np.array(values["ultrasip_delta"])
    delta_grasp = np.array(values["grasp_delta"])
    
    aod_val = np.average(values["aod"])
    color = values["marker_color"]
    
    # Convert degrees to radians
    theta = np.deg2rad(sun_zen)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.subplot(111, projection='polar')
    
    # -------------------------
    # Observed (ULTRASIP)
    # -------------------------
    ax.scatter(
        theta,
        delta_ultrasip,
        c=color,
        edgecolors='black',
        s=120,
        marker='o',
        label="Observed (ULTRASIP)"
    )
    
    # -------------------------
    # Simulated (GRASP)
    # -------------------------
    ax.scatter(
        theta,
        delta_grasp,
        c=color,
        edgecolors='black',
        s=120,
        marker='s',
        label="Simulated (GRASP)"
    )
    
    # -------------------------
    # Formatting
    # -------------------------
    
    ax.set_title(
        f"{day}\nDaily Average AOD(440nm) = {aod_val:.3f}",
        fontsize=18,
        pad=20
    )
    
    ax.set_xlim(0, np.pi) 
    
    #Angular ticks (SZA)
    xmin = 0
    xmax = 90
    ang_ticks = np.linspace(xmin, xmax, 10)
    
    ax.set_xticks(np.deg2rad(ang_ticks))
    ax.set_xticklabels([f"{int(x)}" for x in ang_ticks])
    
    
    
    # TRUE radial limits (negative values preserved)
    ax.set_ylim(-30, -5)
    
    # Radial ticks (no rounding, no decimals)
    rticks = np.linspace(-30, -5, 6)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{int(r)}" for r in rticks])
    
    ax.grid(True, linestyle='--', alpha=0.8)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.1), fontsize=12)
    
    plt.tight_layout()
    plt.show()
    

# ==========================================
# Polar Plot
# Radius  = δ
# Angle   = Observation number (sorted by AOD)
# North   = lowest AOD
# Fixed radial limits [-30, -5]
# ==========================================

for day, values in data_dict.items():

    delta = np.array(values["ultrasip_delta"])
    delta_sim = np.array(values["grasp_delta"])
    aod = np.array(values["aod"])
    ssa = np.array(values["ssa"])
    g = np.array(values["g"])
    ae = np.array(values["ae"])
    sph = np.array(values["sphericity"])

    color = values["marker_color"]

    # ---------------------------------
    # SORT BY AOD (LOWEST → HIGHEST)
    # ---------------------------------
    sort_idx = np.argsort(aod)

    delta = delta[sort_idx]
    delta_sim = delta_sim[sort_idx]
    aod   = aod[sort_idx]
    ssa   = ssa[sort_idx]
    g     = g[sort_idx]
    ae    = ae[sort_idx]
    sph   = sph[sort_idx]

    n = len(delta)

    # Now theta follows sorted order
    theta = np.linspace(0, 2*np.pi, n, endpoint=False)

    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111, projection='polar')

    # ---------------------------------
    # Connect observations
    # ---------------------------------
    ax.plot(
        theta,
        delta,
        color='black',
        linewidth=2,
        alpha=0.8
    )

    ax.scatter(
        theta,
        delta,
        s=120,
        marker='o',
        color=color,
        edgecolor='black',
        zorder=3
    )

    # ---------------------------------
    # Connect sim
    # ---------------------------------
    ax.plot(
        theta,
        delta_sim,
        color='black',
        linewidth=2,
        alpha=0.8
    )

    ax.scatter(
        theta,
        delta_sim,
        s=120,
        marker='s',
        color=color,
        edgecolor='black',
        zorder=3
    )

    # ---------------------------------
    # Fixed radial limits
    # ---------------------------------
    ax.set_ylim(-30, -5)

    delta_ticks = np.linspace(-30, -5, 6)
    ax.set_yticks(delta_ticks)
    ax.set_yticklabels([f"{v:.0f}" for v in delta_ticks], fontsize=12)

    # ---------------------------------
    # Angular formatting
    # ---------------------------------
    ax.set_theta_zero_location("N")   # North = lowest AOD
    ax.set_theta_direction(-1)        # clockwise increasing

    ax.set_xticks(theta)
    ax.set_xticklabels([])

    rmax = -1

    for i in range(n):

        label_text = (
            f"AOD: {aod[i]:.2f}\n"
            f"SSA: {ssa[i]:.2f}\n"
            f"g: {g[i]:.2f}\n"
            f"AE: {ae[i]:.2f}\n"
            f"Sph: {sph[i]:.0f}"
        )

        ax.text(
            theta[i],
            rmax,
            label_text,
            rotation=0,
            rotation_mode='anchor',
            ha='center',
            va='center',
            fontsize=13
        )

    ax.tick_params(axis='x', pad=55)

    ax.set_title(
        f"{day}",
        fontsize=16,
        pad=100
    )

    plt.tight_layout()
    plt.show()

# ==========================================
# Slope & Intercept summary plot (ALL DAYS)
# ==========================================

# ------------------------------------------
# Remove October dates
# ------------------------------------------
valid_dates = sorted([
    d for d in data_dict.keys()
    if "2025_10_"  not in d   # removes October (YYYY-10-DD format)
])

#valid_dates = [d for d in valid_dates if d not in ["2025_06_04","2025_06_23","2025_06_24","2025_07_01"]]

# # Sort  by increasing AOD
valid_dates = sorted(
    valid_dates,
    key=lambda d: np.average(data_dict[d]["aod"])
)
aod_values = np.array([np.average(data_dict[d]["aod"]) for d in valid_dates])

# # Sort  by increasing SSA
# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["ssa"])
# )
# ssa_values = np.array([np.average(data_dict[d]["ssa"]) for d in valid_dates])

# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["ae"])
# )
# ae_values = np.array([np.average(data_dict[d]["ae"]) for d in valid_dates])

# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["g"])
# )
# g_values = np.array([np.average(data_dict[d]["g"]) for d in valid_dates])

# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["sphericity"])
# )
# sphericity_values = np.array([np.average(data_dict[d]["sphericity"]) for d in valid_dates])

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

# ------------------------------------------
# Top x-axis: AOD values
# ------------------------------------------
ax_top = ax1.twiny()

ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks(x)

# Format AOD values (3 decimals — change if desired)
ax_top.set_xticklabels([f"{a:.3f}" for a in aod_values], rotation=45, ha='left')

#Format SSA values (3 decimals — change if desired)
#ax_top.set_xticklabels([f"{a:.3f}" for a in sphericity_values], rotation=45, ha='left')

ax_top.set_xlabel("Daily Average AOD (440 nm)", fontsize=18)
ax_top.tick_params(labelsize=14)

ax2.text(
    0.98, 0.9,
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


# ==========================================
# Slope & Intercept summary plot (ALL DAYS)
# ==========================================

# ------------------------------------------
# Remove October dates
# ------------------------------------------
valid_dates = sorted([
    d for d in data_dict.keys()
    if "2025_10_"  not in d   # removes October (YYYY-10-DD format)
])

#valid_dates = [d for d in valid_dates if d not in ["2025_06_04","2025_06_23","2025_06_24","2025_07_01"]]

# Sort  by increasing AOD
valid_dates = sorted(
    valid_dates,
    key=lambda d: np.average(data_dict[d]["aod"])
)
aod_values = np.array([np.average(data_dict[d]["aod"]) for d in valid_dates])

# #Sort  by increasing SSA
# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["ssa"])
# )
# ssa_values = np.array([np.average(data_dict[d]["ssa"]) for d in valid_dates])

# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["ae"])
# )
# ae_values = np.array([np.average(data_dict[d]["ae"]) for d in valid_dates])

# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["g"])
# )
# g_values = np.array([np.average(data_dict[d]["g"]) for d in valid_dates])

# valid_dates = sorted(
#     valid_dates,
#     key=lambda d: np.average(data_dict[d]["sphericity"])
# )
# sphericity_values = np.array([np.average(data_dict[d]["sphericity"]) for d in valid_dates])

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

slope_diff = slopes_obs-slope_sim
int_diff = intercepts_obs-intercepts_sim


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

ax1.plot(x, slope_diff, '-', color='black', linewidth=1.5)


for i in range(len(valid_dates)):
    # Observed (circles)
    ax1.plot(
        x[i], slope_diff[i],
        '^',
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
ax2.plot(x, int_diff, '-', color='black', linewidth=1.5)


for i in range(len(valid_dates)):
    # Observed (circles)
    ax2.plot(
        x[i], int_diff[i],
        '^',
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

# ------------------------------------------
# Top x-axis: AOD values
# ------------------------------------------
ax_top = ax1.twiny()

ax_top.set_xlim(ax1.get_xlim())
ax_top.set_xticks(x)

# Format AOD values (3 decimals — change if desired)
ax_top.set_xticklabels([f"{a:.3f}" for a in aod_values], rotation=45, ha='left')

#Format SSA values (3 decimals — change if desired)
#ax_top.set_xticklabels([f"{a:.3f}" for a in sphericity_values], rotation=45, ha='left')

ax_top.set_xlabel("Daily Average AOD (440 nm)", fontsize=18)
ax_top.tick_params(labelsize=14)

ax2.text(
    0.98, 0.9,
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
        marker='^',
        color='black',
        linestyle='None',
        markersize=12,
        markeredgecolor='black',
        label='Difference'
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

# ------------------------------------------
# Choose what to sort by (currently AOD)
# ------------------------------------------
valid_dates = sorted(
    valid_dates,
    key=lambda d: np.average(data_dict[d]["aod"])
)

n_days = len(valid_dates)

# Extract slope + aerosol averages
slopes_obs = np.array([data_dict[d]["slope_obs"] for d in valid_dates])
slopes_sim = np.array([data_dict[d]["slope_sim"] for d in valid_dates])

aod_avg = np.array([np.average(data_dict[d]["aod"]) for d in valid_dates])
ssa_avg = np.array([np.average(data_dict[d]["ssa"]) for d in valid_dates])
g_avg   = np.array([np.average(data_dict[d]["g"]) for d in valid_dates])
ae_avg  = np.array([np.average(data_dict[d]["ae"]) for d in valid_dates])
sph_avg = np.array([np.average(data_dict[d]["sphericity"]) for d in valid_dates])

# Angular positions
theta = np.linspace(0, 2*np.pi, n_days, endpoint=False)

# ==========================================
# Create Polar Plot
# ==========================================
fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111, projection='polar')


# Scatter markers
for i in range(n_days):

    color = data_dict[valid_dates[i]]["marker_color"]

    # Observed
    ax.plot(
        theta[i], slopes_obs[i],
        'o',
        markersize=12,
        color=color,
        markeredgecolor='black'
    )

    # Simulated
    ax.plot(
        theta[i], slopes_sim[i],
        's',
        markersize=12,
        color=color,
        markeredgecolor='black'
    )

# ------------------------------------------
# Radial Formatting (Slope axis)
# ------------------------------------------
rmin = -0.45
rmax=-0.15
ax.set_ylim(rmin, rmax)

rticks = np.linspace(rmin, rmax, 6)
ax.set_yticks(rticks)
ax.set_yticklabels([f"{v:.3f}" for v in rticks], fontsize=15)

# ------------------------------------------
# Angular formatting
# ------------------------------------------
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

ax.set_xticks(theta)
ax.set_xticklabels([])

# Move labels outside circle
label_radius = 0

for i in range(n_days):

    label_text = (
        f"AOD: {aod_avg[i]:.3f}\n"
        f"SSA: {ssa_avg[i]:.3f}\n"
        f"g: {g_avg[i]:.3f}\n"
        f"AE: {ae_avg[i]:.3f}\n"
        f"Sph: {sph_avg[i]:.0f}"
    )

    ax.text(
        theta[i],
        label_radius,
        label_text,
        rotation=0,
        ha='center',
        va='center',
        fontsize=25
    )

ax.tick_params(axis='x', pad=60)


# ------------------------------------------
# Legend: Observed / Simulated markers
# ------------------------------------------
legend_elements_main = [
    Line2D([0],[0],
           marker='o',
           color='black',
           linestyle='None',
           markersize=10,
           label='Observed'),
    Line2D([0],[0],
           marker='s',
           color='black',
           linestyle='None',
           markersize=10,
           label='Simulated')
]

# ------------------------------------------
# Legend: Dates as short colored lines
# ------------------------------------------
legend_elements_dates = []

for i, day in enumerate(valid_dates):
    color = data_dict[day]["marker_color"]
    
    legend_elements_dates.append(
        Line2D([0],[0],
               color=color,
               linewidth=4,
               label=day)
    )

# Combine legends
all_legend_elements = legend_elements_main + legend_elements_dates

ax.legend(
    handles=all_legend_elements,
    loc='center',
    bbox_to_anchor=(1.45, 1.15),
    fontsize=11,
    frameon=True,
    ncol=5
)

plt.tight_layout()
plt.show()


# # =========================================
# # Movie Writer Setup (HIGH RESOLUTION)
# # ==========================================
# Writer = animation.FFMpegWriter
# writer = Writer(fps=0.5, bitrate=5000)  # high bitrate for sharp output

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
#             aod_val = np.average(values["aod"])
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
#             plt.title(f"{day}\n Daily Average AOD(440nm) = {aod_val:.3f}", fontsize=20)

#             plt.xticks(fontsize=16)
#             plt.yticks(fontsize=16)
#             plt.legend(fontsize=14, loc='upper right')

#             plt.ylim([-30, -5])

#             xmin = np.min(sun_zen) - 1
#             xmax = np.max(sun_zen) + 1
#             plt.xlim([xmin, xmax])

#             xticks = np.linspace(xmin, xmax, 10)
#             plt.xticks(xticks, [f"{int(x)}" for x in xticks])


#             writer.grab_frame()

#     plt.close(fig)
#     print(f"Saved: {output_name}")

# # ==========================================
# # Create AOD-Sorted Movie (Non-October Only)
# # ==========================================

# # Sort non-October days by increasing AOD
# non_oct_days_sorted_aod = sorted(
#     non_oct_days,
#     key=lambda d: np.average(data_dict[d]["aod"])
# )

# make_movie(non_oct_days_sorted_aod,"Daily_All_EXCEPT_October_Sorted_By_AOD.mp4")

# # ==========================================
# # Create Both Movies
# # ==========================================
# make_movie(non_oct_days, "Daily_All_EXCEPT_October.mp4")
# make_movie(oct_days, "Daily_October_Only.mp4")
