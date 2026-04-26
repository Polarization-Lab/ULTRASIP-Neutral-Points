# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 12:09:41 2026

@author: cdeleon
Dissertation plots over aerosol parameter
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.agreement import mean_diff_plot
import numpy as np
import json
import glob
import pyCompare
import cmocean.cm as cm
from scipy.stats import spearmanr, kendalltau, pearsonr
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
from scipy.stats import linregress
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf

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

    
    uza = np.array(data["np_zenith_deg"]) 
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

# ==========================================
# BUILD ARRAYS
# ==========================================
time_sec = []
uzen = []
gzen = []
rzen = []
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
    if month in ["10"]:
        continue

    for t, uz, rz, sza,  gz,aod,ssa,g,ae,spher,step in zip(values["time"],
                               values["ultra_zen"],
                               values["ray_zen"],
                               values["sun_zenith"],
                               values["grasp_zen"],
                               values["aod"],
                               values["ssa"],
                               values["g"],
                               values["ae"],
                               values["sphericity"],
                               values["aqnum"]):

        h, m, s = map(int, t.split(":"))
        t_seconds = h*3600 + m*60 + s

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

steps = np.array(steps)
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

#Deltas
d_ray = rzen - sza_all
d_sim = gzen - sza_all
d_obs = uzen - sza_all
dd_sim = rzen - gzen 
dd_obs = rzen - uzen 

#Standard deviations 
SDSim = np.std(d_sim)
SDdSim = np.std(dd_sim)
SDObs = np.std(d_obs)
SDdObs = np.std(dd_obs)

#----------------------------------------Correct observations
d_obs_corr = (uzen - np.average(dd_sim-dd_obs)) - sza_all
dd_obs_corr = rzen - (uzen - np.average(dd_sim-dd_obs))
SDObsc = np.std(d_obs_corr)
SDdObsc = np.std(dd_obs_corr)

#-----------------------------only select dd_obs_corr and dd_sim values with 2 standard deviations of the mean dd_obs_corr value
# ------------------------------------------
# 2σ FILTER (based on dd_obs_corr)
# ------------------------------------------
mean_dd = np.mean(dd_obs_corr)
std_dd = np.std(dd_obs_corr)

mask = np.abs(dd_obs_corr - mean_dd) <= 2 * std_dd

# Apply mask to data
dd_obs_corr_f = dd_obs_corr[mask]
dd_sim_f = dd_sim[mask]
#Aerosol properties
gs_f = gs[mask]             #asymmetry parameter
aods_f = aods[mask]        # aerosol optical depth
aes_f = aes[mask]          #angstrom exponent 
ssas_f = ssas[mask]       #Single scattering albedo
sphers_f = sphers[mask]    #sphericity 

#plot dd_sim_f versus dd_obs_corr_f with a 1:1 line, and colorbar showing value of aerosol property

parameter = ssas_f
xaxlabel = 'SSA at 440 nm'
uncert=0.02
xmin= 0.88
xmax =0.98

# ------------------------------------------
# CREATE PLOT
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(11, 6))

# ------------------------------------------
# SCATTER PLOTS
# ------------------------------------------
sc1 = ax.scatter(
    parameter,
    dd_obs_corr_f,
    c=aods_f,
    vmin=np.min(aods_f),
    vmax=np.max(aods_f),
    cmap=cm.ice,
    s=80,
    edgecolor='black',
)

sc2 = ax.scatter(
    parameter,
    dd_sim_f,
    c=aods_f,
    vmin=np.min(aods_f),
    vmax=np.max(aods_f),
    cmap=cm.ice,
    s=80,
    edgecolor='black',
    marker='s'
)

# ------------------------------------------
# LINEAR FITS
# ------------------------------------------
slope_sim, intercept_sim, r_sim, p_sim, _ = linregress(parameter, dd_sim_f)
slope_obs, intercept_obs, r_obs, p_obs, _ = linregress(parameter, dd_obs_corr_f)

fit_sim = intercept_sim + slope_sim * parameter
fit_obs = intercept_obs + slope_obs * parameter

line_obs, = ax.plot(parameter, fit_obs, color='lime', linewidth=3)
line_sim, = ax.plot(parameter, fit_sim, color='magenta', linewidth=4)

# ------------------------------------------
# LEGEND LABELS (COMBINED WITH STATS)
# ------------------------------------------
label_obs = (
    f"ULTRASIP Fit")
    
#     \n"
#     f"slope= {slope_obs:.2f}, intercept= {intercept_obs:.2f}°\n"
#     f"R²={r_obs**2:.2f}, p={p_obs:.2f}"
# )

label_sim = (
    f"GRASP Fit")
    
#     \n"
#     f"slope= {slope_sim:.2f}, intercept= {intercept_sim:.2f}°\n"
#     f"R²={r_sim**2:.2f}, p={p_sim:.2f}"
# )

# ------------------------------------------
# LEGEND (MARKERS + LINES COMBINED)
# ------------------------------------------
handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black', markeredgecolor='black',
           markersize=10, label="ULTRASIP"),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor='black', markeredgecolor='black',
           markersize=10, label='GRASP (AERONET)'),
    Line2D([0], [0], color='lime', lw=3, label=label_obs),
    Line2D([0], [0], color='magenta', lw=4, label=label_sim)
]

# legend = ax.legend(
#     handles=handles,
#     loc='upper center',
#     bbox_to_anchor=(0.5, 1.2),
#     ncol=4,
#     fontsize=19,
#     frameon=True,
#     fancybox=True,
#     framealpha=0.9,
#     handlelength=2.5
# )

# # Align multiline text nicely
# for text in legend.get_texts():
#     text.set_multialignment('left')

# ------------------------------------------
# COLORBAR
# ------------------------------------------
# cbar = plt.colorbar(sc1, ax=ax,pad=0.015)
# cbar.set_label('AOD at 440 nm', fontsize=16)
# cbar.ax.tick_params(labelsize=14)

# ticks = np.linspace(np.min(aods_f), np.max(aods_f), 11)
# cbar.set_ticks(ticks)
# cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

# ------------------------------------------
# AXES FORMATTING
# ------------------------------------------
ax.set_ylim([-3, 1.5])
ax.set_xlim([xmin, xmax])

ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel('$\\Delta\\delta$ [$^\\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(uncert))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.tick_params(axis='both', which='major', labelsize=19)
ax.grid(True, linestyle='--', alpha=0.6)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()



#------------------NO COLORBAR-----------------------------------#

parameter = ssas_f
xaxlabel = 'SSA at 440 nm'
uncert=0.01
xmin= 0.88
xmax = 0.97


# ------------------------------------------
# CREATE PLOT
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(16, 8))

# ------------------------------------------
# SCATTER PLOTS
# ------------------------------------------
sc1 = ax.scatter(
    parameter,
    dd_obs_corr_f,
    s=100,
    edgecolor='black',
    color="green"
)

sc2 = ax.scatter(
    parameter,
    dd_sim_f,
    s=100,
    edgecolor='black',
    marker='s',
    color='purple'
)

# ------------------------------------------
# LINEAR FITS
# ------------------------------------------
slope_sim, intercept_sim, r_sim, p_sim, _ = linregress(parameter, dd_sim_f)
slope_obs, intercept_obs, r_obs, p_obs, _ = linregress(parameter, dd_obs_corr_f)

fit_sim = intercept_sim + slope_sim * parameter
fit_obs = intercept_obs + slope_obs * parameter

line_obs, = ax.plot(parameter, fit_obs, color='lime', linewidth=3)
line_sim, = ax.plot(parameter, fit_sim, color='magenta', linewidth=4)

# ------------------------------------------
# LEGEND LABELS (COMBINED WITH STATS)
# ------------------------------------------
label_obs = (
    f"ULTRASIP Fit\n"
    f"slope= {slope_obs:.2f}, intercept= {intercept_obs:.2f}°\n"
    f"R²={r_obs**2:.2f}, p={p_obs:.2f}"
)

label_sim = (
    f"GRASP Fit\n"
    f"slope= {slope_sim:.2f}, intercept= {intercept_sim:.2f}°\n"
    f"R²={r_sim**2:.2f}, p={p_sim:.2f}"
)

# ------------------------------------------
# LEGEND (MARKERS + LINES COMBINED)
# ------------------------------------------
handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='green', markeredgecolor='black',
           markersize=10, label="ULTRASIP"),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor='purple', markeredgecolor='black',
           markersize=10, label='GRASP (AERONET)'),
    Line2D([0], [0], color='lime', lw=3, label=label_obs),
    Line2D([0], [0], color='magenta', lw=4, label=label_sim)
]

legend = ax.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.2),
    ncol=3,
    fontsize=16,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    handlelength=2.5
)

# Align multiline text nicely
for text in legend.get_texts():
    text.set_multialignment('left')


# ------------------------------------------
# AXES FORMATTING
# ------------------------------------------
ax.set_ylim([-3, 1.5])
ax.set_xlim([xmin, xmax])

ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel('$\\Delta\\delta$ [$^\\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(uncert))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

# ------------------ less stats ------------------ #

parameter = aods_f
xaxlabel = 'AOD at 440 nm'
uncert = 0.02
xmin = 0.04
xmax = .24

# ------------------------------------------
# CREATE PLOT
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

# ------------------------------------------
# SCATTER PLOTS
# ------------------------------------------
ax.scatter(
    parameter,
    dd_obs_corr_f,
    s=100,
    edgecolor='black',
    color="green"
)

ax.scatter(
    parameter,
    dd_sim_f,
    s=100,
    edgecolor='black',
    marker='s',
    color='purple'
)

# ------------------------------------------
# LINEAR FITS
# ------------------------------------------
slope_sim, intercept_sim, r_sim, p_sim, _ = linregress(parameter, dd_sim_f)
slope_obs, intercept_obs, r_obs, p_obs, _ = linregress(parameter, dd_obs_corr_f)


# ------------------------------------------
# FORMAT STATS (2 DECIMAL PLACES)
# ------------------------------------------
stats_df = pd.DataFrame({
    "Dataset": ["ULTRASIP", "GRASP"],
    "Slope": [slope_obs, slope_sim],
    "Intercept": [intercept_obs, intercept_sim],
    "R": [r_obs, r_sim],
    "p-value": [p_obs, p_sim]
})

# Round to 2 decimals
stats_df = stats_df.round(2)

print("\nRegression Statistics:")
print(stats_df.to_string(index=False))


fit_sim = intercept_sim + slope_sim * parameter
fit_obs = intercept_obs + slope_obs * parameter

ax.plot(parameter, fit_obs, color='lime', linewidth=3)
ax.plot(parameter, fit_sim, color='magenta', linewidth=4)

# ------------------------------------------
# SLOPE DIFFERENCE
# ------------------------------------------
delta_slope = slope_obs - slope_sim

# ------------------------------------------
# LEGEND LABELS
# ------------------------------------------
label_obs = "ULTRASIP Fit"
label_sim = "GRASP Fit"

# ------------------------------------------
# LEGEND
# ------------------------------------------
handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='green', markeredgecolor='black',
           markersize=10, label="ULTRASIP"),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor='purple', markeredgecolor='black',
           markersize=10, label='GRASP (AERONET)'),
    Line2D([0], [0], color='lime', lw=3, label=label_obs),
    Line2D([0], [0], color='magenta', lw=4, label=label_sim)
]

legend = ax.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.1),
    ncol=4,
    fontsize=18,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    handlelength=2.5
)

for text in legend.get_texts():
    text.set_multialignment('left')

# ------------------------------------------
# AXES FORMATTING
# ------------------------------------------
ax.set_ylim([-3, 1.5])
ax.set_xlim([xmin, xmax])

ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\Delta\delta$ [$^\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(uncert))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()


# ------------------------------------------
# CREATE FIGURE
# ------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_visible(False)

# ------------------------------------------
# DUMMY SCALAR MAPPABLE (uses your data directly)
# ------------------------------------------
sm = plt.cm.ScalarMappable(cmap=cm.ice)
sm.set_array(aods_f)  # <-- this is the key (no normalization step)

# ------------------------------------------
# COLORBAR
# ------------------------------------------
cbar = plt.colorbar(
    sm,
    ax=ax,
    orientation='horizontal',
    pad=0.2
)

# ------------------------------------------
# FORMAT (same as your original)
# ------------------------------------------
cbar.ax.tick_params(labelsize=25)

ticks = np.linspace(np.min(aods_f), np.max(aods_f), 11)
cbar.set_ticks(ticks)
cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

plt.tight_layout()
plt.show()


# ------------------------------------------
# CREATE HANDLES (same as your plot)
# ------------------------------------------
handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor='black', markeredgecolor='black',
           markersize=16, label="ULTRASIP"),
    Line2D([0], [0], marker='s', color='w',
           markerfacecolor='black', markeredgecolor='black',
           markersize=16, label='GRASP (AERONET)'),
    Line2D([0], [0], color='lime', lw=3, label="ULTRASIP Fit"),
    Line2D([0], [0], color='magenta', lw=4, label="GRASP Fit")
]

# ------------------------------------------
# CREATE FIGURE
# ------------------------------------------
fig, ax = plt.subplots(figsize=(6, 1.5))
ax.axis('off')  # no axes

# ------------------------------------------
# ADD LEGEND
# ------------------------------------------
legend = ax.legend(
    handles=handles,
    loc='center',
    ncol=4,
    fontsize=25,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    handlelength=2.5
)

plt.tight_layout()
plt.show()
# plt.savefig('legend_only.png', dpi=300, bbox_inches='tight')
