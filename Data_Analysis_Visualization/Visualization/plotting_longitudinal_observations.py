# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 11:52:41 2026

@author: cdeleo
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
import pyCompare
import cmocean.cm as cm
from scipy.stats import linregress,pearsonr
from matplotlib.lines import Line2D
import pandas as pd
from tabulate import tabulate

#--------------------------Plot GRASP and ULTRASIP Dataset -----------------------#
# ==========================================
# LOAD DATA
# ==========================================
data_dict = {}

colors = ['red','darkorange','yellow','green',
          'lime','purple','magenta','cadetblue',
          'silver','cyan','gray','blue',
          'honeydew','palevioletred','tan','brown','mediumorchid']

idx = -1

#Load in GRASP values from jsons
data_path = "C:/Users/deleo/Documents/JSON_Files"
json_files = glob.glob(f'{data_path}/BNP*v3.json')

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
# LOAD (CORRECTED ULTRASIP DATA)
# ==========================================
corrected_json_files = glob.glob(f'{data_path}/BNP*v4_Corrected.json')

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
        except Exception as e:
            print(f"Bad time on {day}, index {i}: {t}")
            continue

        time_sec.append(t_seconds)
        gzen.append(gz)
        rzen.append(rz)
        uzen.append(uz)
        # print(day,len(values["ultra_zen"]))

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

#---------------------Plot of original data-----------------------# 
#Plot molecular, GRASP-sim, and ULTRASIP 
# ------------------------------------------
# CREATE STACKED SUBPLOTS (SHARED X-AXIS)
# ------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ==========================================
# TOP PLOT: delta_obs
# ==========================================
axs[0].scatter(sza_all, d_obs, s=100, color='green', zorder=2,edgecolor='black',label="Obs (ULTRASIP)")
axs[0].scatter(sza_all, d_sim, s=100, color='purple', zorder=2,marker='s',edgecolor='black',label="Sim(GRASP-AERONET)")
axs[0].scatter(sza_all, d_ray, s=100, color='skyblue', zorder=2,marker='^',edgecolor='black',label="Sim (GRASP-Molecular)")

axs[0].set_ylim([-30, -3])
axs[0].set_ylabel('$\\theta_{BNP}-\\theta_s$ [$^\\circ$]', fontsize=20)

axs[0].yaxis.set_major_locator(MultipleLocator(5))
axs[0].tick_params(axis='both', which='major', labelsize=16)

axs[0].grid(True, linestyle='--', alpha=0.6)
# axs[0].legend(fontsize=13)

# Correlation textbox
axs[0].text(0.7, 0.95,
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

axs[1].set_ylim([-6, 2])
axs[1].set_xlim([20, 90])

axs[1].set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
axs[1].set_ylabel('$\\delta$ [$^\\circ$]', fontsize=20)

axs[1].xaxis.set_major_locator(MultipleLocator(5))
axs[1].yaxis.set_major_locator(MultipleLocator(1))

axs[1].tick_params(axis='both', which='major', labelsize=16)

axs[1].grid(True, linestyle='--', alpha=0.6)

# Correlation textbox
axs[1].text(0.7, 0.95,
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


#Screen for outliers from observations 
# ------------------------------------------
# 2σ FILTER (based on dd_obs_corr)
# ------------------------------------------
mean_dd = np.mean(dd_obs)
std_dd = np.std(dd_obs)

mask = np.abs(dd_obs - mean_dd) <= 2 * std_dd

# Apply mask to data
d_ray_f = d_ray[mask]
d_obs_f = d_obs[mask]
dd_obs_f = dd_obs[mask] 
d_sim_f = d_sim[mask]
dd_sim_f = dd_sim[mask]
sza_f = sza_all[mask]
#Aerosol properties
gs_f = gs[mask]             #asymmetry parameter
aods_f = aods[mask]        # aerosol optical depth
aes_f = aes[mask]          #angstrom exponent 
ssas_f = ssas[mask]       #Single scattering albedo
sphers_f = sphers[mask]    #sphericity 

print('N=', len(dd_obs_f))

# ------------------------------------------
# CREATE SINGLE PLOT (Δδ ONLY)
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_all, dd_obs, s=100, color='green',
           zorder=2, edgecolor='black', label="ULTRASIP")

ax.scatter(sza_all, dd_sim, s=100, color='purple',
           zorder=2, marker='s', edgecolor='black',
           label="GRASP (AERONET)")

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1) #,label="GRASP (Molecular)")

# ==========================================
# SHADED REGION (mean ± 2σ)
# ==========================================
mean_dd = np.mean(dd_obs)
std_dd = np.std(dd_obs)
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
    # label=rf'$\bar{{\mu}} \pm {n_std}\sigma$'
)
# ==========================================
# AXIS FORMATTING
# ==========================================
ax.set_xlim([20, 90])
ax.set_ylim([-6,2])

ax.set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
ax.set_ylabel('$\\delta$ [$^\\circ$]', fontsize=20)

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


#-----------------------------NOW USING FILTERED DATASET---------------------#

# ------------------------------------------
# DIFFERENCE PLOT
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_f, dd_sim_f-dd_obs_f, s=200, color='black',
           zorder=2, edgecolor='gray',marker='d')

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1)


# ==========================================
# AXIS FORMATTING
# ==========================================
ax.set_ylim([-1, 6])
ax.set_xlim([20, 90])

ax.set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
ax.set_ylabel('$\\delta_{diff}$ [$^\\circ$]', fontsize=20)

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


#--------Corrected datset--------------------# 
bias = np.abs(np.average(dd_sim_f-dd_obs_f))
d_obs_f = (uzen[mask]-bias) - sza_f
dd_obs_f = rzen[mask] - (uzen[mask]-bias) 

#Standard deviations 
SDSimf = np.std(d_sim_f)
SDdSimf = np.std(dd_sim_f)
SDObsf = np.std(d_obs_f)
SDdObsf = np.std(dd_obs_f)

#---------------------Plot of corrected data-----------------------# 
#Plot molecular, GRASP-sim, and ULTRASIP 
# ------------------------------------------
# CREATE STACKED SUBPLOTS (SHARED X-AXIS)
# ------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# ==========================================
# TOP PLOT: delta_obs
# ==========================================
axs[0].scatter(sza_f, d_obs_f, s=100, color='green', zorder=2,edgecolor='black',label="Obs (ULTRASIP)")
axs[0].scatter(sza_f, d_sim_f, s=100, color='purple', zorder=2,marker='s',edgecolor='black',label="Sim(GRASP-AERONET)")
axs[0].scatter(sza_f, d_ray_f, s=100, color='skyblue', zorder=2,marker='^',edgecolor='black',label="Sim (GRASP-Molecular)")
axs[0].set_ylim([-30, -3])
axs[0].set_ylabel('$\\theta_{BNP}-\\theta_s$ [$^\\circ$]', fontsize=20)

axs[0].yaxis.set_major_locator(MultipleLocator(5))
axs[0].tick_params(axis='both', which='major', labelsize=16)

axs[0].grid(True, linestyle='--', alpha=0.6)
# axs[0].legend(fontsize=13)

# Correlation textbox
axs[0].text(0.7, 0.95,
            f'$\sigma_{{obs}}:{SDObsf:.2f}^\\circ$,$\sigma_{{sim}}:{SDSimf:.2f}^\\circ$',
            transform=axs[0].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ==========================================
# BOTTOM PLOT: delta_delta_obs
# ==========================================
axs[1].scatter(sza_f, dd_obs_f, s=100, color='green', zorder=2,edgecolor='black')
axs[1].scatter(sza_f, dd_sim_f, s=100, color='purple', zorder=2,marker='s',edgecolor='black')
axs[1].axhline(0,color="royalblue",zorder=0)

axs[1].set_ylim([-3, 2])
axs[1].set_xlim([20, 90])

axs[1].set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
axs[1].set_ylabel('$\\delta$ [$^\\circ$]', fontsize=20)

axs[1].xaxis.set_major_locator(MultipleLocator(5))
axs[1].yaxis.set_major_locator(MultipleLocator(1))

axs[1].tick_params(axis='both', which='major', labelsize=16)

axs[1].grid(True, linestyle='--', alpha=0.6)

# Correlation textbox
axs[1].text(0.7, 0.95,
            f'$\sigma_{{obs}}:{SDdObsf:.2f}^\\circ$,$\sigma_{{sim}}:{SDdSimf:.2f}^\\circ$',
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
ax.scatter(sza_f, dd_sim_f-dd_obs_f, s=200, color='black',
           zorder=2, edgecolor='gray',marker='d')

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1)


# ==========================================
# AXIS FORMATTING
# ==========================================
ax.set_ylim([-3, 3])
ax.set_xlim([20, 90])

ax.set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
ax.set_ylabel('$\\delta_{diff}$ [$^\\circ$]', fontsize=20)

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

# ------------------------------------------
# 1-1 PLOT
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 12))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(dd_sim_f, dd_obs_f,
           s=200,
           color='black',
           edgecolor='gray',
           marker='s',
           zorder=2)

# Standard deviation of observations
sigma = np.nanstd(dd_obs_f)

# 1:1 line
x = np.linspace(*ax.get_xlim(), 1000)
y = x

ax.plot(x, y, 'r--', linewidth=2, label='1:1 Line', zorder=3)

# ±2σ shaded region
ax.fill_between(
    x,
    y - sigma,
    y + sigma,
    color='red',
    alpha=0.2,
    label=r'$\pm 2\sigma_{obs}$',
    zorder=1
)

# ==========================================
# 1:1 LINE
# ==========================================
lims = [
    -3,
    3
]

ax.plot(lims, lims, 'r--', linewidth=2, zorder=1)

# ==========================================
# AXIS FORMATTING
# ==========================================
ax.set_xlim(lims)
ax.set_ylim(lims)

# Force equal scaling
ax.set_aspect('equal', adjustable='box')

ax.set_xlabel(r'$\delta_{GRASP}$ [$^\circ$]', fontsize=20)
ax.set_ylabel(r'$\delta_{ULTRASIP}$ [$^\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(0.5))

ax.tick_params(axis='both', which='major', labelsize=16)
ax.grid(True, linestyle='--', alpha=0.6)


# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

tol = 2*sigma  # degrees

pct_within = (
    np.sum(np.abs(dd_obs_f - dd_sim_f) <= tol)
    / len(dd_obs_f)
    * 100
)
print(f"{pct_within:.1f}% of points are within ±{tol}° of the 1:1 line")

# Difference
delta_diff = dd_sim_f - dd_obs_f


# ------------------------------------------
# PARAMETERS
# ------------------------------------------
parameter = sza_f
ydata = delta_diff

xaxlabel = '$\\theta_s$'
bin_size = 25
xmin = 20
xmax = 95
# ------------------------------------------
# CREATE BINS
# ------------------------------------------
bins = np.arange(xmin, xmax + bin_size, bin_size)

box_data = []
labels = []


# ------------------------------------------
# COMPUTE BIN STATISTICS
# ------------------------------------------
box_data = []
labels = []
medians = []
ns = []
stats_rows = []

for low, high in zip(bins[:-1], bins[1:]):

    mask = (parameter >= low) & (parameter < high)

    if np.sum(mask) > 0:
        data = ydata[mask]

        box_data.append(data)
        labels.append(f'{low:.2f}\n{high:.2f}')
        medians.append(np.median(data))
        ns.append(len(data))
        
        stats_rows.append({
            'Bin Range': f'{low:.2f}-{high:.2f}',
            'n': len(data),
            'Median': np.median(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data, ddof=1)
            })

# ------------------------------------------
# PLOT
# ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    widths=0.7,
    showfliers=True
)

# Style boxes
for box in bp['boxes']:
    box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

for median in bp['medians']:
    median.set(color='red', linewidth=2)

# ------------------------------------------
# MEDIAN TREND LINE
# ------------------------------------------
positions = np.arange(1, len(medians) + 1)

ax.plot(
    positions,
    medians,
    '-o',
    color='royalblue',
    linewidth=2.5,
    markersize=8,
)

# ------------------------------------------
# INDIVIDUAL POINTS (optional)
# ------------------------------------------
for i, data in enumerate(box_data, start=1):
    x = np.random.normal(i, 0.04, len(data))
    ax.scatter(
        x,
        data,
        s=15,
        color='black',
        alpha=0.25,
        zorder=1
    )



# ------------------------------------------
# LABELS
# ------------------------------------------
ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\delta_{diff}$ [$^\circ$]', fontsize=20)

ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.yaxis.set_major_locator(MultipleLocator(1))


plt.tight_layout()
plt.show()

# ------------------------------------------
# PRINT BIN STATISTICS TABLE
# ------------------------------------------

df_stats = pd.DataFrame(stats_rows)

print(
    tabulate(
        df_stats.round(3),
        headers='keys',
        tablefmt='grid',
        showindex=False
    )
)


# ------------------------------------------
# PARAMETERS
# ------------------------------------------
parameter = aods_f
ydata = delta_diff

xaxlabel = 'AOD at 440 nm'
bin_size = 0.02
xmin = 0.04
xmax = 0.24

# ------------------------------------------
# CREATE BINS
# ------------------------------------------
bins = np.arange(xmin, xmax + bin_size, bin_size)

box_data = []
labels = []


# ------------------------------------------
# COMPUTE BIN STATISTICS
# ------------------------------------------
box_data = []
labels = []
medians = []
ns = []
stats_rows = []

for low, high in zip(bins[:-1], bins[1:]):

    mask = (parameter >= low) & (parameter < high)

    if np.sum(mask) > 0:
        data = ydata[mask]

        box_data.append(data)
        labels.append(f'{low:.2f}\n{high:.2f}')
        medians.append(np.median(data))
        ns.append(len(data))
        
        stats_rows.append({
            'Bin Range': f'{low:.2f}-{high:.2f}',
            'n': len(data),
            'Median': np.median(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data, ddof=1)
            })

# ------------------------------------------
# PLOT
# ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    widths=0.7,
    showfliers=True
)

# Style boxes
for box in bp['boxes']:
    box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

for median in bp['medians']:
    median.set(color='red', linewidth=2)

# ------------------------------------------
# MEDIAN TREND LINE
# ------------------------------------------
positions = np.arange(1, len(medians) + 1)

ax.plot(
    positions,
    medians,
    '-o',
    color='royalblue',
    linewidth=2.5,
    markersize=8,
)

# ------------------------------------------
# INDIVIDUAL POINTS (optional)
# ------------------------------------------
for i, data in enumerate(box_data, start=1):
    x = np.random.normal(i, 0.04, len(data))
    ax.scatter(
        x,
        data,
        s=15,
        color='black',
        alpha=0.25,
        zorder=1
    )



# ------------------------------------------
# LABELS
# ------------------------------------------
ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\delta_{diff}$ [$^\circ$]', fontsize=20)

ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.yaxis.set_major_locator(MultipleLocator(1))


plt.tight_layout()
plt.show()

# ------------------------------------------
# PRINT BIN STATISTICS TABLE
# ------------------------------------------


df_stats = pd.DataFrame(stats_rows)

print(
    tabulate(
        df_stats.round(3),
        headers='keys',
        tablefmt='grid',
        showindex=False
    )
)

# ------------------------------------------
# PARAMETERS
# ------------------------------------------
parameter = ssas_f
ydata = delta_diff

xaxlabel = 'SSA at 440 nm'
bin_size = 0.02
xmin = 0.86
xmax = 0.98

# ------------------------------------------
# CREATE BINS
# ------------------------------------------
bins = np.arange(xmin, xmax + bin_size, bin_size)

box_data = []
labels = []


# ------------------------------------------
# COMPUTE BIN STATISTICS
# ------------------------------------------
box_data = []
labels = []
medians = []
ns = []
stats_rows = []

for low, high in zip(bins[:-1], bins[1:]):

    mask = (parameter >= low) & (parameter < high)

    if np.sum(mask) > 0:
        data = ydata[mask]

        box_data.append(data)
        labels.append(f'{low:.2f}\n{high:.2f}')
        medians.append(np.median(data))
        ns.append(len(data))
        
        stats_rows.append({
            'Bin Range': f'{low:.2f}-{high:.2f}',
            'n': len(data),
            'Median': np.median(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data, ddof=1)
            })

# ------------------------------------------
# PLOT
# ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    widths=0.7,
    showfliers=True
)

# Style boxes
for box in bp['boxes']:
    box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

for median in bp['medians']:
    median.set(color='red', linewidth=2)

# ------------------------------------------
# MEDIAN TREND LINE
# ------------------------------------------
positions = np.arange(1, len(medians) + 1)

ax.plot(
    positions,
    medians,
    '-o',
    color='royalblue',
    linewidth=2.5,
    markersize=8,
)

# ------------------------------------------
# INDIVIDUAL POINTS (optional)
# ------------------------------------------
for i, data in enumerate(box_data, start=1):
    x = np.random.normal(i, 0.04, len(data))
    ax.scatter(
        x,
        data,
        s=15,
        color='black',
        alpha=0.25,
        zorder=1
    )



# ------------------------------------------
# LABELS
# ------------------------------------------
ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\delta_{diff}$ [$^\circ$]', fontsize=20)

ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.yaxis.set_major_locator(MultipleLocator(1))


plt.tight_layout()
plt.show()

# ------------------------------------------
# PRINT BIN STATISTICS TABLE
# ------------------------------------------

df_stats = pd.DataFrame(stats_rows)

print(
    tabulate(
        df_stats.round(3),
        headers='keys',
        tablefmt='grid',
        showindex=False
    )
)

# ------------------------------------------
# PARAMETERS
# ------------------------------------------
parameter = aes_f
ydata = delta_diff

xaxlabel = 'AE'
bin_size = 0.1
xmin = 1.25
xmax = 1.95

# ------------------------------------------
# CREATE BINS
# ------------------------------------------
bins = np.arange(xmin, xmax + bin_size, bin_size)

box_data = []
labels = []


# ------------------------------------------
# COMPUTE BIN STATISTICS
# ------------------------------------------
box_data = []
labels = []
medians = []
ns = []
stats_rows = []

for low, high in zip(bins[:-1], bins[1:]):

    mask = (parameter >= low) & (parameter < high)

    if np.sum(mask) > 0:
        data = ydata[mask]

        box_data.append(data)
        labels.append(f'{low:.2f}\n{high:.2f}')
        medians.append(np.median(data))
        ns.append(len(data))
        
        stats_rows.append({
            'Bin Range': f'{low:.2f}-{high:.2f}',
            'n': len(data),
            'Median': np.median(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data, ddof=1)
            })

# ------------------------------------------
# PLOT
# ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    widths=0.7,
    showfliers=True
)

# Style boxes
for box in bp['boxes']:
    box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

for median in bp['medians']:
    median.set(color='red', linewidth=2)

# ------------------------------------------
# MEDIAN TREND LINE
# ------------------------------------------
positions = np.arange(1, len(medians) + 1)

ax.plot(
    positions,
    medians,
    '-o',
    color='royalblue',
    linewidth=2.5,
    markersize=8,
)

# ------------------------------------------
# INDIVIDUAL POINTS (optional)
# ------------------------------------------
for i, data in enumerate(box_data, start=1):
    x = np.random.normal(i, 0.04, len(data))
    ax.scatter(
        x,
        data,
        s=15,
        color='black',
        alpha=0.25,
        zorder=1
    )



# ------------------------------------------
# LABELS
# ------------------------------------------
ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\delta_{diff}$ [$^\circ$]', fontsize=20)

ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.yaxis.set_major_locator(MultipleLocator(1))


plt.tight_layout()
plt.show()

# ------------------------------------------
# PRINT BIN STATISTICS TABLE
# ------------------------------------------

df_stats = pd.DataFrame(stats_rows)

print(
    tabulate(
        df_stats.round(3),
        headers='keys',
        tablefmt='grid',
        showindex=False
    )
)

# ------------------------------------------
# PARAMETERS
# ------------------------------------------
parameter = gs_f
ydata = delta_diff

xaxlabel = '$g$'
bin_size = 0.02
xmin = 0.62
xmax = 0.72

# ------------------------------------------
# CREATE BINS
# ------------------------------------------
bins = np.arange(xmin, xmax + bin_size, bin_size)

box_data = []
labels = []


# ------------------------------------------
# COMPUTE BIN STATISTICS
# ------------------------------------------
box_data = []
labels = []
medians = []
ns = []
stats_rows = []

for low, high in zip(bins[:-1], bins[1:]):

    mask = (parameter >= low) & (parameter < high)

    if np.sum(mask) > 0:
        data = ydata[mask]

        box_data.append(data)
        labels.append(f'{low:.2f}\n{high:.2f}')
        medians.append(np.median(data))
        ns.append(len(data))
        
        stats_rows.append({
            'Bin Range': f'{low:.2f}-{high:.2f}',
            'n': len(data),
            'Median': np.median(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data, ddof=1)
            })

# ------------------------------------------
# PLOT
# ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    widths=0.7,
    showfliers=True
)

# Style boxes
for box in bp['boxes']:
    box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

for median in bp['medians']:
    median.set(color='red', linewidth=2)

# ------------------------------------------
# MEDIAN TREND LINE
# ------------------------------------------
positions = np.arange(1, len(medians) + 1)

ax.plot(
    positions,
    medians,
    '-o',
    color='royalblue',
    linewidth=2.5,
    markersize=8,
)

# ------------------------------------------
# INDIVIDUAL POINTS (optional)
# ------------------------------------------
for i, data in enumerate(box_data, start=1):
    x = np.random.normal(i, 0.04, len(data))
    ax.scatter(
        x,
        data,
        s=15,
        color='black',
        alpha=0.25,
        zorder=1
    )



# ------------------------------------------
# LABELS
# ------------------------------------------
ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\delta_{diff}$ [$^\circ$]', fontsize=20)

ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.yaxis.set_major_locator(MultipleLocator(1))


plt.tight_layout()
plt.show()

# ------------------------------------------
# PRINT BIN STATISTICS TABLE
# ------------------------------------------


df_stats = pd.DataFrame(stats_rows)

print(
    tabulate(
        df_stats.round(3),
        headers='keys',
        tablefmt='grid',
        showindex=False
    )
)

# ------------------------------------------
# PARAMETERS
# ------------------------------------------
parameter = sphers_f
ydata = delta_diff

xaxlabel = '$Sph_\%$'
bin_size = 10
xmin = 0
xmax = 100
# ------------------------------------------
# CREATE BINS
# ------------------------------------------
bins = np.arange(xmin, xmax + bin_size, bin_size)

box_data = []
labels = []


# ------------------------------------------
# COMPUTE BIN STATISTICS
# ------------------------------------------
box_data = []
labels = []
medians = []
ns = []
stats_rows = []

for low, high in zip(bins[:-1], bins[1:]):

    mask = (parameter >= low) & (parameter < high)

    if np.sum(mask) > 0:
        data = ydata[mask]

        box_data.append(data)
        labels.append(f'{low:.2f}\n{high:.2f}')
        medians.append(np.median(data))
        ns.append(len(data))
        
        stats_rows.append({
            'Bin Range': f'{low:.2f}-{high:.2f}',
            'n': len(data),
            'Median': np.median(data),
            'Mean': np.mean(data),
            'Std Dev': np.std(data, ddof=1)
            })

# ------------------------------------------
# PLOT
# ------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

bp = ax.boxplot(
    box_data,
    labels=labels,
    patch_artist=True,
    widths=0.7,
    showfliers=True
)

# Style boxes
for box in bp['boxes']:
    box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

for median in bp['medians']:
    median.set(color='red', linewidth=2)

# ------------------------------------------
# MEDIAN TREND LINE
# ------------------------------------------
positions = np.arange(1, len(medians) + 1)

ax.plot(
    positions,
    medians,
    '-o',
    color='royalblue',
    linewidth=2.5,
    markersize=8,
)

# ------------------------------------------
# INDIVIDUAL POINTS (optional)
# ------------------------------------------
for i, data in enumerate(box_data, start=1):
    x = np.random.normal(i, 0.04, len(data))
    ax.scatter(
        x,
        data,
        s=15,
        color='black',
        alpha=0.25,
        zorder=1
    )



# ------------------------------------------
# LABELS
# ------------------------------------------
ax.set_xlabel(xaxlabel, fontsize=20)
ax.set_ylabel(r'$\delta_{diff}$ [$^\circ$]', fontsize=20)

ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
ax.yaxis.set_major_locator(MultipleLocator(1))


plt.tight_layout()
plt.show()

# ------------------------------------------
# PRINT BIN STATISTICS TABLE
# ------------------------------------------

df_stats = pd.DataFrame(stats_rows)

print(
    tabulate(
        df_stats.round(3),
        headers='keys',
        tablefmt='grid',
        showindex=False
    )
)



def quartile_boxplot(parameter, ydata, n_bins=3, xmin=None, xmax=None,
                     xlabel="", ylabel=r'$\delta_{diff}$ [$^\circ$]',
                     figsize=(12, 6), show_points=True):

    # ------------------------------------------
    # FILTER DOMAIN
    # ------------------------------------------
    if xmin is None: xmin = np.nanmin(parameter)
    if xmax is None: xmax = np.nanmax(parameter)

    mask0 = (parameter >= xmin) & (parameter <= xmax)
    x = parameter[mask0]
    y = ydata[mask0]

    # ------------------------------------------
    # QUARTILE EDGES
    # ------------------------------------------
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))

    box_data = []
    medians = []
    stats_rows = []

    # ------------------------------------------
    # BINNING
    # ------------------------------------------
    for i in range(len(bins) - 1):

        low, high = bins[i], bins[i + 1]

        if i == len(bins) - 2:
            mask = (x >= low) & (x <= high)  # include last bin upper edge
        else:
            mask = (x >= low) & (x < high)
        data = y[mask]

        if len(data) == 0:
            continue

        box_data.append(data)
        medians.append(np.median(data))

        stats_rows.append({
            "Bin Range": f"{low:.3f}-{high:.3f}",
            "n": len(data),
            "Median": np.median(data),
            "Mean": np.mean(data),
            "Std Dev": np.std(data, ddof=1)
        })

    # ------------------------------------------
    # PLOT
    # ------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        box_data,
        patch_artist=True,
        showfliers=True,
        widths=0.7
    )

    for box in bp['boxes']:
        box.set(facecolor='lightgray', edgecolor='black', linewidth=1.5)

    for med in bp['medians']:
        med.set(color='red', linewidth=2)

    # median trend line
    pos = np.arange(1, len(medians) + 1)

    ax.plot(pos, medians, '-o',
            color='royalblue',
            linewidth=2.5,
            markersize=8)

    # jittered points
    if show_points:
        for i, data in enumerate(box_data, start=1):
            xj = np.random.normal(i, 0.04, len(data))
            ax.scatter(xj, data, s=15, color='black', alpha=0.25, zorder=1)

    # ------------------------------------------
    # FORMATTING
    # ------------------------------------------
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------
    # TABLE OUTPUT
    # ------------------------------------------
    df = pd.DataFrame(stats_rows)

    print(tabulate(df.round(3),
                   headers='keys',
                   tablefmt='grid',
                   showindex=False))

    return df

# ------------------------------------------
# Δδ vs SZA
# ------------------------------------------
df_sza = quartile_boxplot(
    parameter=sza_f,
    ydata=delta_diff,
    n_bins=3,
    xmin=20,
    xmax=90,
    xlabel=r'$\theta_s$'
)

# ------------------------------------------
# AOD
# ------------------------------------------
df_aod = quartile_boxplot(
    parameter=aods_f,
    ydata=delta_diff,
    n_bins=3,
    xmin=0.04,
    xmax=0.24,
    xlabel='AOD at 440 nm'
)

# ------------------------------------------
# SSA
# ------------------------------------------
df_ssa = quartile_boxplot(
    parameter=ssas_f,
    ydata=delta_diff,
    n_bins=3,
    xmin=0.86,
    xmax=0.98,
    xlabel='SSA at 440 nm'
)

# ------------------------------------------
# AE
# ------------------------------------------
df_ae = quartile_boxplot(
    parameter=aes_f,
    ydata=delta_diff,
    n_bins=3,
    xmin=1.25,
    xmax=1.95,
    xlabel='AE'
)

# ------------------------------------------
# g
# ------------------------------------------
df_g = quartile_boxplot(
    parameter=gs_f,
    ydata=delta_diff,
    n_bins=3,
    xmin=0.62,
    xmax=0.72,
    xlabel='$g$'
)

# ------------------------------------------
# Sphericity
# ------------------------------------------
df_sph = quartile_boxplot(
    parameter=sphers_f,
    ydata=delta_diff,
    n_bins=3,
    xmin=0,
    xmax=100,
    xlabel='Sphericity (%)'
)





def binned_delta_vs_parameter(dd_sim_f, dd_obs_f, parameter,
                              n_bins=4, xmin=None, xmax=None,
                              xlabel="SZA",
                              ylabel=r'$\delta$',
                              figsize=(10, 6),
                              plot=True):

    # ------------------------------------------
    # CLEAN INPUTS
    # ------------------------------------------
    dd_sim_f = np.asarray(dd_sim_f)
    dd_obs_f = np.asarray(dd_obs_f)
    parameter = np.asarray(parameter)

    if xmin is None:
        xmin = np.nanmin(parameter)
    if xmax is None:
        xmax = np.nanmax(parameter)

    mask = (parameter >= xmin) & (parameter <= xmax) & \
           np.isfinite(dd_sim_f) & np.isfinite(dd_obs_f)

    x = parameter[mask]
    sim = dd_sim_f[mask]
    obs = dd_obs_f[mask]

    # ------------------------------------------
    # QUARTILE BINS
    # ------------------------------------------
    bins = np.quantile(x, np.linspace(0, 1, n_bins + 1))

    rows = []
    sim_fits = []
    obs_fits = []
    bin_centers = []

    # ------------------------------------------
    # BIN LOOP
    # ------------------------------------------
    for i in range(len(bins) - 1):

        low, high = bins[i], bins[i + 1]

        if i == len(bins) - 2:
            m = (x >= low) & (x <= high)
        else:
            m = (x >= low) & (x < high)

        xs = x[m]
        ys_sim = sim[m]
        ys_obs = obs[m]

        if len(xs) < 3:
            continue

        center = np.nanmean(xs)
        bin_centers.append(center)

        # ------------------------------------------
        # LINEAR FITS PER BIN
        # ------------------------------------------
        sim_fit = linregress(xs, ys_sim)
        obs_fit = linregress(xs, ys_obs)

        sim_fits.append(sim_fit)
        obs_fits.append(obs_fit)

        rows.append({
            "Bin Range": f"{low:.3f}-{high:.3f}",
            "n": len(xs),

            "SIM slope": sim_fit.slope,
            "SIM intercept": sim_fit.intercept,
            "SIM r2": sim_fit.rvalue**2,

            "OBS slope": obs_fit.slope,
            "OBS intercept": obs_fit.intercept,
            "OBS r2": obs_fit.rvalue**2
        })

    df = pd.DataFrame(rows)

    # ------------------------------------------
    # PLOT
    # ------------------------------------------
    if plot:
        fig, ax = plt.subplots(figsize=figsize)

        # ------------------------------------------
        # DATA (ONLY ONCE FOR LEGEND)
        # ------------------------------------------
        ax.scatter(x, sim, s=12, color='purple', marker='s',alpha=0.25, label='GRASP (AERONET)')
        ax.scatter(x, obs, s=12, color='green', marker='o',alpha=0.25, label='ULTRASIP')

        colors = plt.cm.jet(np.linspace(0, 1, len(bin_centers)))

        # ------------------------------------------
        # BIN FIT LINES
        # ------------------------------------------
        for i, (fit_s, fit_o, c) in enumerate(zip(sim_fits, obs_fits, colors)):

            xs_plot = np.linspace(bins[i], bins[i + 1], 50)

            # GRASP (dashed)
            ax.plot(xs_plot,
                    fit_s.slope * xs_plot + fit_s.intercept,
                    linestyle='--', color=c)

            # ULTRASIP (solid)
            ax.plot(xs_plot,
                    fit_o.slope * xs_plot + fit_o.intercept,
                    linestyle='-', color=c)

        # ------------------------------------------
        # FORMATTING
        # ------------------------------------------
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.set_ylim([-3, 1])

        # ------------------------------------------
        # SINGLE CLEAN LEGEND
        # ------------------------------------------
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='s', color='w',
                   markerfacecolor='purple', markersize=6,
                   label='GRASP (AERONET)'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='green', markersize=6,
                   label='ULTRASIP'),
            Line2D([0], [0], linestyle='--', color='black',
                   label='GRASP (AERONET) fit'),
            Line2D([0], [0], linestyle='-', color='black',
                   label='ULTRASIP fit')
        ]

        ax.legend(handles=legend_elements, loc='best')

        plt.tight_layout()
        plt.show()

    # ------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------
    print(tabulate(df.round(3),
                   headers='keys',
                   tablefmt='grid',
                   showindex=False))

    return df

df = binned_delta_vs_parameter(
    dd_sim_f=dd_sim_f,
    dd_obs_f=dd_obs_f,
    parameter=sza_f,
    n_bins=3,
    xlabel="$\\theta_s [\circ]$",
    ylabel=r'$\delta [\circ]$'
)

df = binned_delta_vs_parameter(
    dd_sim_f=dd_sim_f,
    dd_obs_f=dd_obs_f,
    parameter=aods_f,
    n_bins=3,
    xlabel="AOD at 440 nm",
    ylabel=r'$\delta [\circ]$'
)

df = binned_delta_vs_parameter(
    dd_sim_f=dd_sim_f,
    dd_obs_f=dd_obs_f,
    parameter=ssas_f,
    n_bins=3,
    xlabel="SSA at 440 nm",
    ylabel=r'$\delta [\circ]$'
)

df = binned_delta_vs_parameter(
    dd_sim_f=dd_sim_f,
    dd_obs_f=dd_obs_f,
    parameter=aes_f,
    n_bins=3,
    xlabel="AE",
    ylabel=r'$\delta [\circ]$'
)

df = binned_delta_vs_parameter(
    dd_sim_f=dd_sim_f,
    dd_obs_f=dd_obs_f,
    parameter=gs_f,
    n_bins=3,
    xlabel="$g$",
    ylabel=r'$\delta [\circ]$'
)

df = binned_delta_vs_parameter(
    dd_sim_f=dd_sim_f,
    dd_obs_f=dd_obs_f,
    parameter=sphers_f,
    n_bins=3,
    xlabel="$Spher_\%$",
    ylabel=r'$\delta [\circ]$'
)