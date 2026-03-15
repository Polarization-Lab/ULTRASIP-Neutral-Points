# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:06:12 2026

@author: ULTRASIP_1
"""

import pandas as pd

# Load dataset
file = "D:/Data/2025_10_22/2025_10_22_neutral_point_fit_rotation_analysis.csv"
df = pd.read_csv(file)

# Filter for good linear fits
df_good = df[df["u_r2"] > 0.9]

# Group by file and compute standard deviation
std_table = (
    df_good
    .groupby("file")["abs_az_diff_deg"]
    .std()
    .reset_index()
)

# Rename column for clarity
std_table = std_table.rename(columns={"abs_az_diff_deg": "std_abs_az_diff_deg"})

print(std_table)
