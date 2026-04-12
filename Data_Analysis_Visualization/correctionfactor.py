# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:09:21 2026

@author: deleo
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:45:44 2026

@author: deleo
"""

import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.agreement import mean_diff_plot
import numpy as np
import json
import glob
import pyCompare
from scipy.stats import pearsonr
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
from scipy.stats import linregress


def refined_index_of_agreement(obs, pred):
    """
    Compute refined index of agreement (Willmott et al., 2012)

    Parameters
    ----------
    obs : array-like
        Observations
    pred : array-like
        Predictions (or model values)

    Returns
    -------
    dr : float
        Refined index of agreement (-1 to 1)
    """

    obs = np.asarray(obs)
    pred = np.asarray(pred)

    # Remove NaNs if present
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]
    pred = pred[mask]

    mean_obs = np.mean(obs)

    numerator = np.sum(np.abs(pred - obs))
    denominator = 2 * np.sum(np.abs(obs - mean_obs))

    if denominator == 0:
        return np.nan  # avoid division by zero

    if numerator <= denominator:
        dr = 1 - (numerator / denominator)
    else:
        dr = (denominator / numerator) - 1

    return dr

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
json_files = glob.glob(f'{data_path}/BNP*.json')

for file in json_files:

    with open(file, "r") as f:
        data = json.load(f)

    idx += 1
    
    day = data["date"]
    time = np.array(data["LocalTime(hh:mm:ss)"])
    sza = np.array(data["sun_zenith_deg"])
    saz = np.array(data["sun_azimuth_deg"])
    gza = np.array(data["grasp_np_za_355nm"])

    correction_factor = 3.16 #0.59 *(np.array(data["acquisition"]))
    
    uza = np.array(data["np_zenith_deg"]) #- correction_factor
    uaz = np.array(data["np_azimuth_deg"])
    ray_zen = np.array(data["rayleigh_np_za_355nm"])

    data_dict[day] = {
        "time": time,
        "sun_zenith": sza,
        "sun_azimuth": saz,
        "ultra_zen": uza,
        "ray_zen": ray_zen,
        "grasp_zen": gza,
        "ultra_az":uaz
    }

# ==========================================
# BUILD ARRAYS
# ==========================================
time_sec = []
uzen = []
uzenog = []
gzen = []
rzen = []
ray_diff = []
sza_all = []
saz_all = []
for day, values in data_dict.items():

    # ---- Skip October ----
    month = day.split("_")[1]
    if month in ["10"]:
        continue

    for t, uz, rz, sza, saz, gz,uaz in zip(values["time"],
                               values["ultra_zen"],
                               values["ray_zen"],
                               values["sun_zenith"],
                               values["sun_azimuth"],
                               values["grasp_zen"],
                               values["ultra_az"]):

        h, m, s = map(int, t.split(":"))
        t_seconds = h*3600 + m*60 + s

        time_sec.append(t_seconds)
        gzen.append(gz)
        rzen.append(rz)
        uzen.append(uz)
        ray_diff.append(rz - uz)
        sza_all.append(sza)
        saz_all.append(saz)

time_sec = np.array(time_sec)
sza_all = np.array(sza_all)
saz_all = np.array(saz_all)
uzen = np.array(uzen) 
gzen = np.array(gzen) 
rzen = np.array(rzen)



#--------------------Compare agreement with Simulations
#------------------------------------------------------
diff = (rzen-gzen)-(rzen-uzen)
md = np.mean(diff)
sd = np.std(diff, axis=0)
print(f"std diff: {sd}")

avgos = np.mean([rzen-gzen, rzen-uzen], axis=0)

#--------------------LINEAR FIT (proportional bias)
#------------------------------------------------------
slope, intercept, r_value, p_value, std_err = linregress(avgos, diff)

print(f"Slope (proportional bias): {slope}")
print(f"Intercept (constant bias): {intercept}")
print(f"R^2: {r_value**2}")
print(f"p-value: {p_value}")

# Fitted line
fit_line = intercept + slope * avgos

#--------------------PLOT
#------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,6))

# Scatter
ax.scatter(avgos, diff, s=35, color='sienna', alpha=0.45, marker='D')
#ax.scatter(avgos, diff_corrected, s=35, color='sienna', alpha=0.45, marker='D', label='Data')


# Mean bias
ax.axhline(md, color='green', linestyle='-', label=f'Mean:{md:.2f}')

# Limits of agreement
ax.axhline(md + 2*sd, color='gray', linestyle='--', label=f'±2SD ({sd:.2f})')
ax.axhline(md - 2*sd, color='gray', linestyle='--')

# Regression line (proportional bias)
ax.plot(avgos, fit_line, color='red', linewidth=2, label='Fit Line')

# Annotation box
textstr = '\n'.join((
    f'slope = {slope:.3f}',
    f'intercept = {intercept:.3f}',
    f'$R^2$ = {r_value**2:.3f}',
    f'p = {p_value:.2e}'
))
ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Labels
ax.set_xlabel('Average of $\Delta\delta_{sim}, \Delta\delta_{obs}$ [$\\circ$]', fontsize=16)
ax.set_ylabel('$\Delta\delta_{sim}-\Delta\delta_{obs}$ [$\\circ$]', fontsize=16)

# Ticks/grid
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(axis='x', which='major', linestyle='--', alpha=0.6)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend()

plt.tight_layout()
plt.show()

#--------------------CORRECTED DIFFERENCE
#------------------------------------------------------
diff_corrected = diff - (intercept+ slope * avgos)   # remove proportional bias only

md = np.mean(diff_corrected)
sd = np.std(diff_corrected, axis=0)
print(f"std diff: {sd}")

slope_corr, intercept_corr, r_value_corr, p_value_corr, std_err_corr = linregress(avgos, diff_corrected)

print(f"Slope (proportional bias): {slope_corr}")
print(f"Intercept (constant bias): {intercept_corr}")
print(f"R^2: {r_value_corr**2}")
print(f"p-value: {p_value_corr}")

# Fitted line
fit_line_corr = intercept_corr + slope_corr * avgos


#--------------------PLOT
#------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,6))

# Scatter
ax.scatter(avgos, diff_corrected, s=35, color='sienna', alpha=0.45, marker='D')
#ax.scatter(avgos, diff_corrected, s=35, color='sienna', alpha=0.45, marker='D', label='Data')


# Mean bias
ax.axhline(md, color='green', linestyle='-', label=f'Mean:{md:.2f}')

# Limits of agreement
ax.axhline(md + 2*sd, color='gray', linestyle='--', label=f'±2SD ({sd:.2f})')
ax.axhline(md - 2*sd, color='gray', linestyle='--')

# Regression line (proportional bias)
ax.plot(avgos, fit_line_corr, color='red', linewidth=2, label='Fit Line')

# Annotation box
textstr = '\n'.join((
    f'slope = {slope_corr:.3f}',
    f'intercept = {intercept_corr:.3f}',
    f'$R^2$ = {r_value_corr**2:.3f}',
    f'p = {p_value_corr:.2e}'
))
ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# Labels
ax.set_xlabel('Average of $\Delta\delta_{sim}, \Delta\delta_{obs}$ [$\\circ$]', fontsize=16)
ax.set_ylabel('Corrected $\Delta\delta_{sim}-\Delta\delta_{obs}$ [$\\circ$]', fontsize=16)

# Ticks/grid
ax.xaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.grid(axis='x', which='major', linestyle='--', alpha=0.6)

ax.tick_params(axis='both', which='major', labelsize=14)
ax.legend()

plt.tight_layout()
plt.show()



#------------------------Correcting nominal values--------------------------
# ORIGINAL quantities
delta_obs =rzen - uzen
delta_sim = rzen - gzen

# Bland–Altman terms
diff = delta_sim - delta_obs
avgos = np.mean([delta_sim, delta_obs], axis=0)

# FIT (on ORIGINAL data)
slope, intercept, *_ = linregress(avgos, diff)

# CORRECTION 
delta_obs_corr = delta_obs + (intercept + slope * avgos)
uzen_corr = -(delta_obs_corr-rzen)
diff_new = delta_sim - delta_obs_corr
diff_new_mean = np.average(diff_new)
diff_new_std = np.std(diff_new)
print("Delta")
print(f"avg difference {diff_new_mean:.2f}\n std differences {diff_new_std:.2f} ")

# ------------------------------------------
# COMPUTE CORRELATIONS
# ------------------------------------------
r1, p1 = pearsonr(uzen_corr - sza_all, sza_all)
r2, p2 = pearsonr(rzen - uzen_corr, sza_all)

r1sim, p1 = pearsonr(gzen - sza_all, sza_all)
r2sim, p2 = pearsonr(rzen - gzen, sza_all)


SDObs = np.std(uzen_corr-sza_all)
SDSim = np.std(gzen-sza_all)

SDO = np.std(rzen-uzen_corr)
SDS = np.std(rzen-gzen)

# ------------------------------------------
# CREATE STACKED SUBPLOTS (SHARED X-AXIS)
# ------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# ==========================================
# TOP PLOT: delta_obs
# ==========================================
axs[0].scatter(sza_all, uzen_corr - sza_all, s=30, color='black', zorder=2)
axs[0].scatter(sza_all, gzen - sza_all, s=30, color='magenta', zorder=2,marker='s')

axs[0].set_ylim([-35, -5])
axs[0].set_ylabel('Corrected $\\delta$ [$^\\circ$]', fontsize=16)

axs[0].yaxis.set_major_locator(MultipleLocator(5))
axs[0].tick_params(axis='both', which='major', labelsize=14)

axs[0].grid(True, linestyle='--', alpha=0.6)

# Correlation textbox
axs[0].text(0.05, 0.15,
            f'$\\rho_{{obs}} = {r1:.3f}$,$\\rho_{{sim}} = {r1sim:.3f}$\n$SD_{{obs}}:{SDObs:.2f}^\\circ$,$SD_{{sim}}:{SDSim:.2f}^\\circ$',
            transform=axs[0].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ==========================================
# BOTTOM PLOT: delta_delta_obs
# ==========================================
axs[1].scatter(sza_all, rzen - uzen_corr, s=30, color='black', zorder=2)
axs[1].scatter(sza_all, rzen - gzen, s=30, color='magenta', zorder=2,marker='s')
axs[1].axhline(0,color="black")

axs[1].set_ylim([-3, 3])
axs[1].set_xlim([20, 90])

axs[1].set_xlabel('SZA [$^\\circ$]', fontsize=16)
axs[1].set_ylabel('Corrected $\\Delta\\delta$ [$^\\circ$]', fontsize=16)

axs[1].xaxis.set_major_locator(MultipleLocator(5))
axs[1].yaxis.set_major_locator(MultipleLocator(1))

axs[1].tick_params(axis='both', which='major', labelsize=14)

axs[1].grid(True, linestyle='--', alpha=0.6)

# Correlation textbox
axs[1].text(0.05, 0.95,
            f'$\\rho_{{obs}} = {r2:.3f}$,$\\rho_{{sim}} = {r2sim:.3f}$\n$SD_{{obs}}:{SDO:.2f}^\\circ$,$SD_{{sim}}:{SDS:.2f}^\\circ$',
            transform=axs[1].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

slope, intercept, r_value, p_value, std_err = linregress(sza_all,diff_new)
fit_line = intercept + slope * sza_all
diff_newstd = np.std(diff_new)

plt.figure(figsize=(12,6))

plt.scatter(sza_all, diff_new, zorder=2, s=35, color='black',  marker='D')
plt.plot(sza_all, fit_line, color='red', linewidth=2, label='Fit (proportional bias)')

plt.ylim([-4, 2])
plt.xlim([20, 90])

plt.xlabel('SZA [$^\\circ$]', fontsize=16)
plt.ylabel('Corrected $\\Delta\\delta_{{sim}}-\\Delta\\delta_{{obs}}$ [$^\\circ$]', fontsize=16)

# plt.set_major_locator(MultipleLocator(5))
# plt.set_major_locator(MultipleLocator(2))

#plt.ticks(axis='both', which='major', labelsize=14)

plt.grid(True, linestyle='--', alpha=0.6)

textstr = '\n'.join((
    f'slope = {slope:.3f}',
    f'intercept = {intercept:.3f}',
    f'$R^2$ = {r_value**2:.3f}',
    f'p = {p_value:.2e}',
    f'SD = {diff_newstd:.2f}'
))
# Correlation textbox
plt.text(0.05, 0.35,
            textstr,
            transform=axs[1].transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

plt.scatter(sza_all, rzen - uzen_corr, s=30, color='black', zorder=2)

plt.axhline(0, color="black")

plt.ylim([-3, 1])
plt.xlim([20, 90])

plt.xlabel('SZA [$^\\circ$]', fontsize=16)
plt.ylabel('Corrected $\\Delta\\delta$ [$^\\circ$]', fontsize=16)

plt.gca().xaxis.set_major_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(1))

plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

# Correlation textbox
# plt.text(0.05, 0.95,
#          f'$\\rho_{{obs}} = {r2:.3f}$,$\\rho_{{sim}} = {r2sim:.3f}$\n'
#          f'$SD_{{obs}}:{SDO:.2f}^\\circ$,$SD_{{sim}}:{SDS:.2f}^\\circ$',
#          transform=plt.gca().transAxes,
#          fontsize=14,
#          verticalalignment='top',
#          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

#Refined index of agreement
dr = refined_index_of_agreement((rzen-uzen),(rzen-gzen))
print('dd no correction',dr)
dr = refined_index_of_agreement((uzen-sza_all),(gzen-sza_all))
print('d no correction',dr)

dr = refined_index_of_agreement((rzen-uzen_corr),(rzen-gzen))
print('dd correction',dr)
dr = refined_index_of_agreement((uzen_corr-sza_all),(gzen-sza_all))
print('d correction',dr)

