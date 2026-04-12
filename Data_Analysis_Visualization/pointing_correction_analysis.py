# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:09:12 2026

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
    aqnum = np.array(data["acquisition"])

    correction_factor = 0.59 * aqnum 
    
    uza = np.array(data["np_zenith_deg"]) - 3.16
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

diff_delta = (rzen-gzen)-(rzen-uzen)

slope, intercept, r_value, p_value, std_err = linregress(steps,diff_delta)
fit_line = intercept + slope * steps

plt.figure()
plt.scatter(steps,diff_delta,zorder=2)
plt.plot(steps, fit_line, color='red', linewidth=2, label='Fit Line')
plt.axhline(0,color='black')
plt.ylim([-4,4])
plt.xlim([2,14])
textstr = '\n'.join((
    f'slope = {slope:.3f}',
    f'intercept = {intercept:.3f}',
    f'$R^2$ = {r_value**2:.3f}',
    f'p = {p_value:.2e}',
))
plt.text(11, 8, textstr, 
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True)
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(2))
plt.xlabel("Scan above Sun")
plt.ylabel("$\Delta\delta_{sim}-\Delta\delta_{obs}$")
plt.title("No Correction Applied")