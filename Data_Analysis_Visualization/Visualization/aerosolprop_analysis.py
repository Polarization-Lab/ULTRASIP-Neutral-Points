# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 22:26:12 2026

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

    correction_factor = 0.59 *(np.array(data["acquisition"]))
    
    uza = np.array(data["np_zenith_deg"]) - correction_factor
    uaz = np.array(data["np_azimuth_deg"])
    ray_zen = np.array(data["rayleigh_np_za_355nm"])

    data_dict[day] = {
        "time": time,
        "sun_zenith": sza,
        "ultra_zen": uza,
        "ray_zen": ray_zen,
        "grasp_zen": gza,
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


for day, values in data_dict.items():

    # ---- Skip October ----
    month = day.split("_")[1]
    if month in ["10"]:
        continue

    for t, uz, rz, sza,  gz,aod,ssa,g,ae,spher in zip(values["time"],
                               values["ultra_zen"],
                               values["ray_zen"],
                               values["sun_zenith"],
                               values["grasp_zen"],
                               values["aod"],
                               values["ssa"],
                               values["g"],
                               values["ae"],
                               values["sphericity"]):

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


#------------------------Correcting nominal values--------------------------
# ORIGINAL quantities
delta_obs = rzen - uzen
delta_sim = rzen - gzen

# CORRECTION 
delta_obs_corr = delta_obs -3.16

delta_delta = delta_obs_corr

#----------------Organize into bins by SZA--------------------------------#
#Group one: delta_obs values from first quartile
#Group two: delta_obs values from second quartile
#Group three: delta_obs values from three quartile



def partial_corr(x, y, z):
    """Compute partial correlation r(x,y|z) using residual method."""
    zc = sm.add_constant(z)

    res_x = sm.OLS(x, zc).fit().resid
    res_y = sm.OLS(y, zc).fit().resid

    r, p = pearsonr(res_x, res_y)
    return r, p


def plot_binned_partial_corr(
    sza,
    x,              # e.g., delta_delta
    param,          # e.g., aods, sphers, ssas
    param_name="Param",
    n_bins=3
):
    """
    Creates scatter plot per SZA bin and reports partial correlation in legend.

    x          : dependent variable (Δδ)
    param      : variable of interest (AOD, sphericity, etc.)
    sza        : control variable
    """



    # ------------------------------------------
    # BINNING (equal-frequency)
    # ------------------------------------------
    bins = pd.qcut(sza, q=n_bins, labels=False, duplicates="drop")

    fig, ax = plt.subplots(figsize=(8,6))

    colors = ["tab:orange", "tab:green", "tab:red"]

    for i in range(n_bins):
        mask = bins == i

        x_bin = x[mask]
        p_bin = param[mask]
        sza_bin = sza[mask]

        if len(x_bin) < 3:
            continue  # avoid unstable stats

        # ------------------------------------------
        # PARTIAL CORRELATION
        # ------------------------------------------
        r_part, pval = partial_corr(x_bin, p_bin, sza_bin)

        # ------------------------------------------
        # LINEAR FIT (optional but nice)
        # ------------------------------------------
        m, b = np.polyfit(p_bin, x_bin, 1)

        # ------------------------------------------
        # PLOT
        # ------------------------------------------
        ax.scatter(p_bin, x_bin, alpha=0.7, color=colors[i],label=f"Bin {i+1} (ρₚ= {r_part:.2f})")

        x_sorted = np.sort(p_bin)
        # ax.plot(x_sorted, m*x_sorted + b, color=colors[i],
        #         label=f"Bin {i+1} (ρₚ={r_part:.2f})")

    ax.set_xlabel(param_name)
    ax.set_ylabel(r"$\Delta\delta$")
    ax.legend(fontsize=12,ncol=3,bbox_to_anchor=(0., 1.1),loc='upper left')
    plt.tight_layout()
    plt.show()
    
plot_binned_partial_corr(
    sza=sza_all,
    x=delta_delta,
    param=aods,
    param_name="AOD"
)

plot_binned_partial_corr(
    sza=sza_all,
    x=delta_delta,
    param=ssas,
    param_name="SSA"
)

plot_binned_partial_corr(
    sza=sza_all,
    x=delta_delta,
    param=gs,
    param_name="g"
)

plot_binned_partial_corr(
    sza=sza_all,
    x=delta_delta,
    param=aes,
    param_name="AE"
)

plot_binned_partial_corr(
    sza=sza_all,
    x=delta_delta,
    param=sphers,
    param_name="Spher %"
)


def plot_binned_pearson(
    sza,
    x,              # Δδ
    param,          # aerosol parameter
    param_name="Param",
    n_bins=3
):
    """
    Scatter plot per SZA bin with Pearson r in legend.
    Prints std table separately.
    """

    # ------------------------------------------
    # BINNING (equal-frequency)
    # ------------------------------------------
    bins = pd.qcut(sza, q=n_bins, labels=False, duplicates="drop")

    fig, ax = plt.subplots(figsize=(8,6))
    colors = ["tab:orange", "tab:green", "tab:red"]

    results = []

    # ------------------------------------------
    # PER BIN
    # ------------------------------------------
    for i in range(n_bins):
        mask = bins == i

        x_bin = x[mask]
        p_bin = param[mask]

        if len(x_bin) < 3:
            continue

        # Pearson correlation
        r, _ = pearsonr(x_bin, p_bin)

        # Standard deviations (store for table)
        std_x = np.std(x_bin)
        std_p = np.std(p_bin)
        
        slope = r*(std_x/std_p)

        results.append([i+1, len(x_bin), r, std_x, std_p,slope])

        # ------------------------------------------
        # PLOT
        # ------------------------------------------
        ax.scatter(
            p_bin,
            x_bin,
            alpha=0.7,
            color=colors[i],
            label=f"Bin {i+1} (ρ = {r:.2f})"
        )
        ax.tick_params(axis='both', labelsize=14)   # tick labels
        ax.set_xlabel(param_name, fontsize=16)
        ax.set_ylabel(r"$\Delta\delta$", fontsize=16)


    # ------------------------------------------
    # OVERALL CORRELATION
    # ------------------------------------------
    r_all, _ = pearsonr(x, param)

    # ------------------------------------------
    # LABELS
    # ------------------------------------------
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"$\Delta\delta$")
    ax.legend(fontsize=12, ncol=3, bbox_to_anchor=(0., 1.2), loc='upper left')
    ax.set_title(f"Overall ρ = {r_all:.2f}")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------
    # PRINT TABLE (separate from figure)
    # ------------------------------------------
    df = pd.DataFrame(results, columns=[
        "Bin", "N", "Pearson r", "std Δδ", f"std {param_name}", "Slope"
    ])

    print("\n" + "="*50)
    print(f"{param_name} — Metrics per Bin")
    print("="*50)
    print(df.to_string(index=False))
    
    # ------------------------------------------
    # OVERALL METRICS
    # ------------------------------------------
    std_x_all = np.std(x)
    std_p_all = np.std(param)

    # slope (overall)
    slope_all  = r_all*(std_x_all/std_p_all)

    # ------------------------------------------
    # PRINT OVERALL
    # ------------------------------------------
    print("="*50)
    print(f"{param_name} — Metrics Overall")
    print("="*50)

    df_all = pd.DataFrame([[
        len(x),
        r_all,
        std_x_all,
        std_p_all,
        slope_all
        ]], columns=[
            "N",
            "Pearson r",
            "std Δδ",
            f"std {param_name}",
            "Slope"
            ])

    print(df_all.to_string(index=False))

    return df

plot_binned_pearson(sza_all, delta_delta, aods, "AOD 440")
plot_binned_pearson(sza_all, delta_delta, ssas, "SSA 440")
plot_binned_pearson(sza_all, delta_delta, gs, "g")
plot_binned_pearson(sza_all, delta_delta, aes, "AE")
plot_binned_pearson(sza_all, delta_delta, sphers, "Sphericity [%]")