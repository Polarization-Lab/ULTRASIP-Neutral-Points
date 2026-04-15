# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 14:44:03 2026

@author: deleo
"""

#Correction and aerosol trend plots 
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.agreement import mean_diff_plot
import numpy as np
import json
import glob
import pyCompare
from scipy.stats import spearmanr, kendalltau, pearsonr
from matplotlib.ticker import FuncFormatter, MultipleLocator
import pandas as pd
from scipy.stats import linregress

# ==========================================
# FUNCTION: Full Correlation Table
# ==========================================
def correlation_table(dd_obs, dd_sim, param_dict, filter_outliers=False, n_std=2):
    """
    Computes Pearson, Spearman, and Kendall correlations between ΔΔ and aerosol parameters.
    """

    results = []

    # ------------------------------------------
    # Optional filtering (same as plots)
    # ------------------------------------------
    if filter_outliers:
        mean = np.mean(dd_obs)
        std = np.std(dd_obs)

        lower = mean - n_std * std
        upper = mean + n_std * std

        mask = (dd_obs >= lower) & (dd_obs <= upper)

        dd_obs_use = dd_obs[mask]
        dd_sim_use = dd_sim[mask]

        filtered_params = {k: v[mask] for k, v in param_dict.items()}
    else:
        dd_obs_use = dd_obs
        dd_sim_use = dd_sim
        filtered_params = param_dict

    # ------------------------------------------
    # Loop through parameters
    # ------------------------------------------
    for name, param in filtered_params.items():

        # ---- OBS ----
        r_p_obs, p_p_obs = pearsonr(param, dd_obs_use)
        r_s_obs, p_s_obs = spearmanr(param, dd_obs_use)
        r_k_obs, p_k_obs = kendalltau(param, dd_obs_use)

        # ---- SIM ----
        r_p_sim, p_p_sim = pearsonr(param, dd_sim_use)
        r_s_sim, p_s_sim = spearmanr(param, dd_sim_use)
        r_k_sim, p_k_sim = kendalltau(param, dd_sim_use)

        results.append({
            "Parameter": name,

            # Pearson
            "Pearson r (Obs)": r_p_obs,
            #"p (Obs, Pearson)": p_p_obs,
            "Pearson r (Sim)": r_p_sim,
            #"p (Sim, Pearson)": p_p_sim,

            # Spearman
            "Spearman ρ (Obs)": r_s_obs,
            #"p (Obs, Spearman)": p_s_obs,
            "Spearman ρ (Sim)": r_s_sim,
            #"p (Sim, Spearman)": p_s_sim,

            # Kendall
            "Kendall τ (Obs)": r_k_obs,
            #"p (Obs, Kendall)": p_k_obs,
            "Kendall τ (Sim)": r_k_sim,
            #"p (Sim, Kendall)": p_k_sim,
        })

    df = pd.DataFrame(results)

    # ------------------------------------------
    # Print nicely
    # ------------------------------------------
    print("\n" + "="*80)
    print("Correlation Table (Δδ vs Aerosol Parameters)")
    print("="*80)
    print(df.to_string(index=False, float_format="%.3f"))

    return df


def plot_dd_vs_parameterstd(x, dd_obs, dd_sim, 
                         xlabel="AOD at 440 nm",
                         ylabel=r'$\Delta\delta$ [$^\circ$]',
                         title=None,
                         filter_outliers=False,
                         n_std=2):
    """
    Plot Δδ vs parameter with optional outlier filtering.

    Parameters
    ----------
    x : array-like
    dd_obs : array-like
    dd_sim : array-like
    filter_outliers : bool
        If True, removes points outside ±n_std of dd_obs
    n_std : int or float
        Number of standard deviations for filtering
    """

    x = np.asarray(x)
    dd_obs = np.asarray(dd_obs)
    dd_sim = np.asarray(dd_sim)

    # ==========================================
    # Optional filtering
    # ==========================================
    if filter_outliers:
        mean = np.mean(dd_obs)
        std = np.std(dd_obs)

        lower = mean - n_std * std
        upper = mean + n_std * std

        mask = (dd_obs >= lower) & (dd_obs <= upper)

        x_plot = x[mask]
        dd_obs_plot = dd_obs[mask]
        dd_sim_plot = dd_sim[mask]

        n_removed = np.sum(~mask)
    else:
        x_plot = x
        dd_obs_plot = dd_obs
        dd_sim_plot = dd_sim
        n_removed = 0

    print(n_removed)
    # ==========================================
    # Linear fits
    # ==========================================
    slope_sim, intercept_sim, r_sim, _, _ = linregress(x_plot, dd_sim_plot)
    slope_obs, intercept_obs, r_obs, _, _ = linregress(x_plot, dd_obs_plot)

    fit_sim = intercept_sim + slope_sim * x_plot
    fit_obs = intercept_obs + slope_obs * x_plot

    # ==========================================
    # Plot
    # ==========================================
    fig, ax = plt.subplots(figsize=(14, 8))

    # ==========================================
    # Scatter
    # ==========================================
    ax.scatter(x_plot, dd_obs_plot, s=100, color='green', edgecolor='black',
               label='ULTRASIP')
    
    ax.plot(x_plot, fit_obs, color='lime', linewidth=3,
            label='Linear Fit')

    ax.scatter(x_plot, dd_sim_plot, s=100, color='purple', marker='s',
               edgecolor='black', label='GRASP (AERONET)')

    # ==========================================
    # Fit lines
    # ==========================================
    ax.plot(x_plot, fit_sim, color='magenta', linewidth=4, 
            label='Linear Fit')



    # ==========================================
    # Axes styling
    # ==========================================
    ax.axhline(0, color="royalblue")


    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)

    ax.set_ylim([-4, 4])
    ax.yaxis.set_major_locator(MultipleLocator(1))

    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.grid(True, linestyle='--', alpha=0.6)

    if title:
        ax.set_title(title, fontsize=18)

    # ==========================================
    # Structured textbox
    # ==========================================
    colw = 10

    textstr = (
    f"{'':27s}{'Slope':>{colw}s}{'Intercept':>{colw+4}s}{'R²':>{colw}s}\n"
    f"{'-'*(14+5*colw)}\n"
    f"{'ULTRASIP':22s}{slope_obs:{colw}.3f}{intercept_obs:{colw}.3f}$^\circ${r_obs**2:{colw}.3f}\n"
    f"{'GRASP (AERONET)':4s}{slope_sim:{colw}.3f}{intercept_sim:{colw}.3f}$^\circ${r_sim**2:{colw}.3f}"
)
    ax.text(0.5, 0.98, textstr, 
            transform=ax.transAxes, fontsize=17, 
            verticalalignment='top', family='Sans Serif', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # ==========================================
    # Legend ABOVE plot
    # # ==========================================
    # ax.legend(loc='lower center',
    #           bbox_to_anchor=(0.5, 1.5),
    #           ncol=4,
    #           fontsize=18,
    #           markerscale=2,
    #           frameon=False)

    plt.tight_layout()
    plt.show()
    
    print("std:",np.std(x_plot),"mean:",np.mean(x_plot),"unique:",len(np.unique(x_plot)),"min/max",np.min(x_plot),np.max(x_plot))


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

axs[0].set_ylim([-30, -5])
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

axs[1].set_ylim([-7, 1])
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


#----------------------------------------Correct observations
d_obs_corr = (uzen - np.average(dd_sim-dd_obs)) - sza_all
dd_obs_corr = rzen - (uzen - np.average(dd_sim-dd_obs))

SDObsc = np.std(d_obs_corr)
SDdObsc = np.std(dd_obs_corr)


# ------------------------------------------
# CREATE STACKED SUBPLOTS (SHARED X-AXIS)
# ------------------------------------------
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# ==========================================
# TOP PLOT: delta_obs
# ==========================================
axs[0].scatter(sza_all, d_obs_corr, s=100, color='green', zorder=2,edgecolor='black',label="ULTRASIP")
axs[0].scatter(sza_all, d_sim, s=100, color='purple', zorder=2,marker='s',edgecolor='black',label="GRASP (AERONET)")
axs[0].scatter(sza_all, d_sim, s=100, color='skyblue', zorder=2,marker='^',edgecolor='black',label=" GRASP (Molecular)")

axs[0].set_ylim([-30, -5])
axs[0].set_ylabel('$\\delta$ [$^\\circ$]', fontsize=20)

axs[0].yaxis.set_major_locator(MultipleLocator(5))
axs[0].tick_params(axis='both', which='major', labelsize=16)

axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(
    fontsize=20,
    ncol=3,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.5),
    markerscale=2,
    frameon=False
)
# Correlation textbox
axs[0].text(0.05, 0.15,
            f'$\sigma_{{obs}}:{SDObsc:.2f}^\\circ$,$\sigma_{{sim}}:{SDSim:.2f}^\\circ$',
            transform=axs[0].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ==========================================
# BOTTOM PLOT: delta_delta_obs
# ==========================================
axs[1].scatter(sza_all, dd_obs_corr, s=100, color='green', zorder=2,edgecolor='black')
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
axs[1].text(0.55, 0.12,
            f'$\sigma_{{obs}}:{SDdObsc:.2f}^\\circ$,$\sigma_{{sim}}:{SDdSim:.2f}^\\circ$',
            transform=axs[1].transAxes,
            fontsize=16,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
#plt.suptitle('Corrected',fontsize=16)
plt.tight_layout()
plt.show()


# #-------------------Choose data within 2SDs-----------------------------------#
# filtered=True
# plot_dd_vs_parameterstd(
#     aods,
#     dd_obs_corr,
#     dd_sim,
#     xlabel="AOD at 440 nm",
#     title="",
#     filter_outliers=filtered,
#     n_std=1.5
# )
# plot_dd_vs_parameterstd(
#     ssas,
#     dd_obs_corr,
#     dd_sim,
#     xlabel="SSA at 440 nm",
#     title="",
#     filter_outliers=filtered,
#     n_std=1.5
# )

# plot_dd_vs_parameterstd(
#     gs,
#     dd_obs_corr,
#     dd_sim,
#     xlabel="g",
#     title="",
#     filter_outliers=filtered,
#     n_std=1.5
# )

# plot_dd_vs_parameterstd(
#     aes,
#     dd_obs_corr,
#     dd_sim,
#     xlabel="AE",
#     title="",
#     filter_outliers=filtered,
#     n_std=1.5
# )

# plot_dd_vs_parameterstd(
#     sphers,
#     dd_obs_corr,
#     dd_sim,
#     xlabel="Spher$_{\%}$",
#     title="",
#     filter_outliers=filtered,
#     n_std=1.5
# )



# # ==========================================
# # PARAMETER DICTIONARY
# # ==========================================
# param_dict = {
#     "SZA": sza_all,
#     "AOD (440 nm)": aods,
#     "SSA (440 nm)": ssas,
#     "g": gs,
#     "AE (440–870 nm)": aes,
#     "Sphericity (%)": sphers
# }

# # ==========================================
# # RUN
# # ==========================================
# df_corr = correlation_table(
#     dd_obs_corr,
#     dd_sim,
#     param_dict,
#     filter_outliers=False,
#     n_std=2
# )

# ------------------------------------------
# CREATE SINGLE PLOT (Δδ ONLY)
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_all, dd_obs_corr, s=100, color='green',
           zorder=2, edgecolor='black', label="Observations (ULTRASIP)")

ax.scatter(sza_all, dd_sim, s=100, color='purple',
           zorder=2, marker='s', edgecolor='black',
           label="Simulations (GRASP-AERONET)")

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1)

# ==========================================
# SHADED ±1.5σ REGION (OBS)
# ==========================================
sigma = SDdObsc
upper = 1.5 * sigma
lower = -1.5 * sigma

ax.fill_between(
    [20, 90],  # x-range
    lower,
    upper,
    color='gray',
    alpha=0.2,
    zorder=0,
    label=r'$\pm1.5\sigma$'
)

# ==========================================
# AXIS FORMATTING
# ==========================================
ax.set_ylim([-4, 4])
ax.set_xlim([20, 90])

ax.set_xlabel('$\\theta_s$ [$^\\circ$]', fontsize=20)
ax.set_ylabel('$\\Delta\\delta$ [$^\\circ$]', fontsize=20)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(1))

ax.tick_params(axis='both', which='major', labelsize=16)

ax.grid(True, linestyle='--', alpha=0.6)

# ==========================================
# TEXTBOX
# ==========================================
ax.text(0.05, 0.12,
        f'$\sigma_{{obs}}:{SDdObsc:.2f}^\\circ$,$\sigma_{{sim}}:{SDdSim:.2f}^\\circ$',
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# ==========================================
# LEGEND (NO BOX)
# ==========================================
ax.legend(fontsize=16, frameon=True)

# ------------------------------------------
# FINAL LAYOUT
# ------------------------------------------
plt.tight_layout()
plt.show()

#----------------------------------SZA Filtering--------------------------#
# ==========================================
# FILTER: ±nσ in θ_s (SZA)
# ==========================================
def sza_std_mask(sza, n_std=2):
    mean_sza = np.mean(sza)
    std_sza = np.std(sza)

    lower = mean_sza - n_std * std_sza
    upper = mean_sza + n_std * std_sza

    mask = (sza >= lower) & (sza <= upper)

    print(f"SZA filter: μ={mean_sza:.2f}, σ={std_sza:.2f}, "
          f"range=[{lower:.2f}, {upper:.2f}]")
    print(f"Removed {np.sum(~mask)} points")

    return mask

# ==========================================
# APPLY SZA FILTER
# ==========================================
use_sza_filter = False
n_std_sza = 2

if use_sza_filter:
    mask_sza = sza_std_mask(sza_all, n_std=n_std_sza)

    sza_f = sza_all[mask_sza]
    dd_obs_corr_f = dd_obs_corr[mask_sza]
    dd_sim_f = dd_sim[mask_sza]

    # also filter parameters
    aods_f = aods[mask_sza]
    ssas_f = ssas[mask_sza]
    gs_f = gs[mask_sza]
    aes_f = aes[mask_sza]
    sphers_f = sphers[mask_sza]

else:
    sza_f = sza_all
    dd_obs_corr_f = dd_obs_corr
    dd_sim_f = dd_sim
    aods_f = aods
    ssas_f = ssas
    gs_f = gs
    aes_f = aes
    sphers_f = sphers
    
#-------------------Choose data within 2SDs-----------------------------------#
filtered=True
n_std=2
plot_dd_vs_parameterstd(
    aods_f,
    dd_obs_corr_f,
    dd_sim_f,
    xlabel="AOD at 440 nm",
    title="",
    filter_outliers=filtered,
    n_std=n_std
)
plot_dd_vs_parameterstd(
    ssas_f,
    dd_obs_corr_f,
    dd_sim_f,
    xlabel="SSA at 440 nm",
    title="",
    filter_outliers=filtered,
    n_std=n_std
)

plot_dd_vs_parameterstd(
    gs_f,
    dd_obs_corr_f,
    dd_sim_f,
    xlabel="g",
    title="",
    filter_outliers=filtered,
    n_std=n_std
)

plot_dd_vs_parameterstd(
    aes_f,
    dd_obs_corr_f,
    dd_sim_f,
    xlabel="AE",
    title="",
    filter_outliers=filtered,
    n_std=n_std
)

plot_dd_vs_parameterstd(
    sphers_f,
    dd_obs_corr_f,
    dd_sim_f,
    xlabel="Spher$_{\%}$",
    title="",
    filter_outliers=filtered,
    n_std=n_std
)

# ------------------------------------------
# CREATE SINGLE PLOT (Δδ ONLY)
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_all, dd_obs_corr, s=100, color='green',
           zorder=2, edgecolor='black', label="Obs (ULTRASIP)")

ax.scatter(sza_all, dd_sim, s=100, color='purple',
           zorder=2, marker='s', edgecolor='black',
           label="Sim (GRASP-AERONET)")

# Zero reference line
ax.axhline(0, color="royalblue", zorder=1)

# ==========================================
# SHADED REGION (mean ± 2σ)
# ==========================================
mean_dd = np.mean(dd_obs_corr)
std_dd = np.std(dd_obs_corr)

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
ax.set_ylim([-4, 4])
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

# ------------------------------------------
# CREATE SINGLE PLOT (Δδ ONLY)
# ------------------------------------------
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# ==========================================
# SCATTER PLOTS
# ==========================================
ax.scatter(sza_all, dd_sim-dd_obs, s=200, color='black',
           zorder=2, edgecolor='gray',marker='d', label="Obs (ULTRASIP)")

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