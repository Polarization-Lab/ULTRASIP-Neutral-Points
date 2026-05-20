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
from scipy.stats import spearmanr, kendalltau, pearsonr


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
    if month in ["10"]:
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

#Uncorrected dataset 
# ------------------------------------------
# CREATE STACKED SUBPLOTS (SHARED X-AXIS)
# ------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# ==========================================
# TOP PLOT: delta_obs
# ==========================================
axs[0].scatter(sza_all, d_obs, s=100, color='green', zorder=2,edgecolor='black',label="Obs (ULTRASIP)")
axs[0].scatter(sza_all, d_sim, s=100, color='purple', zorder=2,marker='s',edgecolor='black',label="Sim(GRASP-AERONET)")
axs[0].scatter(sza_all, d_sim, s=100, color='skyblue', zorder=2,marker='^',edgecolor='black',label="Sim (GRASP-Molecular)")

axs[0].set_ylim([-30, -3])
axs[0].set_ylabel('$\\delta$ [$^\\circ$]', fontsize=20)

axs[0].yaxis.set_major_locator(MultipleLocator(5))
axs[0].tick_params(axis='both', which='major', labelsize=16)

axs[0].grid(True, linestyle='--', alpha=0.6)
# axs[0].legend(fontsize=13)

# Correlation textbox
axs[0].text(0.05, 0.15,
            f'$\sigma_{{obs}}:{SDObs:.2f}^\\circ$,$\sigma_{{sim}}:{SDSim:.2f}^\\circ$',
            transform=axs[0].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ==========================================
# BOTTOM PLOT: delta_delta_obs
# ==========================================
axs[1].scatter(sza_all, dd_obs, s=100, color='green', zorder=2,edgecolor='black')
axs[1].scatter(sza_all, dd_sim, s=100, color='purple', zorder=2,marker='s',edgecolor='black')
axs[1].axhline(0,color="royalblue",zorder=0)

axs[1].set_ylim([-4, 4])
axs[1].set_xlim([20, 90])

axs[1].set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
axs[1].set_ylabel('$\\Delta\\delta$ [$^\\circ$]', fontsize=20)

axs[1].xaxis.set_major_locator(MultipleLocator(5))
axs[1].yaxis.set_major_locator(MultipleLocator(1))

axs[1].tick_params(axis='both', which='major', labelsize=16)

axs[1].grid(True, linestyle='--', alpha=0.6)

# Correlation textbox
axs[1].text(0.05, 0.12,
            f'$\sigma_{{obs}}:{SDdObs:.2f}^\\circ$,$\sigma_{{sim}}:{SDdSim:.2f}^\\circ$',
            transform=axs[1].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
#plt.suptitle('Uncorrected',fontsize=16)
plt.tight_layout()
plt.show()


# ------------------------------------------
# DIFFERENCE PLOT
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_all, dd_sim-dd_obs, s=200, color='black',
           zorder=2, edgecolor='gray',marker='d')

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1)


# ==========================================
# AXIS FORMATTING
# ==========================================
#ax.set_ylim([-4, 4])
ax.set_xlim([20, 90])

ax.set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
ax.set_ylabel('$\\Delta\\delta_{diff}$ [$^\\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# LEGEND
# ==========================================
#ax.legend(fontsize=16, frameon=True)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

d_obs_corr = (uzen - np.average(dd_sim-dd_obs)) - sza_all
dd_obs_corr = rzen - (uzen - np.average(dd_sim-dd_obs))

r_p_obs, p_p_obs = pearsonr(sza_all,d_sim)
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

# ------------------------------------------
# CREATE SINGLE PLOT (Δδ ONLY)
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_all, dd_obs_corr, s=100, color='green',
           zorder=2, edgecolor='black', label="ULTRASIP")

ax.scatter(sza_all, dd_sim, s=100, color='purple',
           zorder=2, marker='s', edgecolor='black',
           label="GRASP (AERONET)")

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1) #,label="GRASP (Molecular)")

# ==========================================
# SHADED REGION (mean ± 2σ)
# ==========================================
mean_dd = np.mean(dd_obs_corr)
std_dd = np.std(dd_obs_corr)
n_std = 2

upper = mean_dd + n_std * std_dd
lower = mean_dd - n_std * std_dd

x_min = np.min(sza_all)
x_max = np.max(sza_all)

ax.fill_between(
    [20, 90],
    lower,
    upper,
    color='gray',
    alpha=0.2,
    zorder=0,
    label=rf'$\bar{{\mu}} \pm {n_std}\sigma$'
)
# ==========================================
# AXIS FORMATTING
# ==========================================
ax.set_xlim([20, 90])

ax.set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
ax.set_ylabel('$\\Delta\\delta$ [$^\\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# LEGEND
# ==========================================
ax.legend(fontsize=16, frameon=True)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

#plot dd_sim_f versus dd_obs_corr_f with a 1:1 line, and colorbar showing value of aerosol property

parameter = sphers_f
xaxlabel = '$Sph_\%$'
uncert=10
xmin= 0
xmax =100

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
ax.axhline(0,color="black",zorder=0)
ax.grid(True, linestyle='--', alpha=0.5)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()



#------------------NO COLORBAR-----------------------------------#

# parameter = ssas_f
# xaxlabel = 'SSA at 440 nm'
# uncert=0.02
# xmin= 0.88
# xmax =0.98



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
ax.set_ylim([-8, 5])
ax.set_xlim([xmin, xmax])

ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel('$\\Delta\\delta$ [$^\\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(uncert))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)

ax.axhline(0,color="royalblue",zorder=0)
ax.grid(True, linestyle='--', alpha=0.5)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

# ------------------ less stats ------------------ #

parameter = aods_f
xaxlabel = 'AOD at 440 nm'
uncert=0.02
xmin= 0.04
xmax =0.24


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
    "R^2": [r_obs**2, r_sim**2],
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
ax.axhline(0,color="black",zorder=0)
ax.grid(True, linestyle='--', alpha=0.5)

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