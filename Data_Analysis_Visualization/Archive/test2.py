
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# =========================
# Days to include
# =========================
dates = ['2025_06_04','2025_06_09','2025_06_10',
         '2025_06_18','2025_06_23','2025_06_24',
         '2025_06_25','2025_06_30','2025_07_01',
         '2025_07_08','2025_07_09','2025_07_10',
         '2025_07_17','2025_07_18']

# Convert to datetime.date format
dates = [pd.to_datetime(d, format="%Y_%m_%d").date() for d in dates]

# =========================
# Load file
# =========================
file_path = r"C:/Users/ULTRASIP_1/Downloads/20250501_20251231_Bozeman_npza_21dates_npza_2025-06-04_2025-07-21_2.json"

with open(file_path, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

df["Local_datetime"] = pd.to_datetime(df["Local_datetime"])
df["date"] = df["Local_datetime"].dt.date

# Compute delta
df["delta"] = df["grasp_np_za_355nm"] - (90-df["sun_alt"])

# =========================
# Storage arrays
# =========================
sim_slopes = []
sim_intercepts = []

for day in dates:

    day_df = df[df["date"] == day]

    if len(day_df) < 2:
        print(f"Skipping {day} (not enough data)")
        continue

    sun_alt = 90 - day_df["sun_alt"].values
    delta = day_df["delta"].values

    # -----------------------------
    # PRINT VALUES
    # -----------------------------
    print("\n==============================")
    print(f"Date: {day}")
    print("Sun Alt (deg):")
    print(sun_alt)
    print("Delta (deg):")
    print(delta)

    # -----------------------------
    # Linear regression
    # -----------------------------
    slope, intercept, r_value, p_value, std_err = linregress(sun_alt, delta)
    fit_line = slope * sun_alt + intercept

    sim_slopes.append(slope)
    sim_intercepts.append(intercept)

    # Plot
    plt.figure(figsize=(6,5))
    plt.scatter(sun_alt, delta)
    plt.plot(sun_alt, fit_line)

    plt.xlabel("Sun Altitude (deg)")
    plt.ylabel("Δ = NP ZA (355 nm) − Sun Alt (deg)")
    plt.title(str(day))
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Convert to numpy arrays
sim_slopes = np.array(sim_slopes)
sim_intercepts = np.array(sim_intercepts)

print("Sim Slopes:", sim_slopes)
print("Sim Intercepts:", sim_intercepts)
