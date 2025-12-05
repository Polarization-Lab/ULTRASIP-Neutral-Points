# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:58:03 2025

@author: ULTRASIP_1

"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


date = np.array(["6/30", "7/1", "7/8", "7/9", "7/10", "7/13", "7/17", "7/18"])


r1 = np.array([0.163, 0.137, 0.144, 0.133, 0.145, 0.155, 0.149, 0.138])
sigma1 = np.array([0.542, 0.516, 0.511, 0.528, 0.489, 0.523, 0.468, 0.452])
r2 = np.array([3.582, 3.494, 3.784, 4.123, 3.627, 3.401, 3.770, 3.585])
sigma2 = np.array([0.640, 0.599, 0.607, 0.639, 0.615, 0.695, 0.618, 0.587])
C_ratio = np.array([0.746, 0.556, 0.932, 0.566, 0.434, 0.674, 0.850, 1.293])
sphericity = np.array([0.75, 0.24, 0.96, 0.43, 0.36, 0.68, 0.97, 0.63])
n_440 = np.array([1.53, 1.54, 1.48, 1.53, 1.56, 1.54, 1.50, 1.47])
k_440 = np.array([0.012, 0.006, 0.008, 0.006, 0.007, 0.012, 0.008, 0.007])
AOD_355 = np.array([0.114, 0.156, 0.168, 0.172, 0.167, 0.111, 0.214, 0.264])
SSA_355 = np.array([0.766, 0.817, 0.798, 0.799, 0.785, 0.770, 0.800, 0.844])

# Slopes
slope_obs = np.array([-0.321, -0.249, -0.379, -0.390, -0.325, -0.376, -0.383, -0.382])
slope_sim = np.array([-0.316, -0.284, -0.316, -0.342, -0.266, -0.329, -0.298, -0.290])
slope_delta = slope_sim - slope_obs   

# Intercepts
intercept_obs = np.array([ 2.316, -0.383, 5.904, 6.725, 3.030, 4.921, 6.270, 6.779])
intercept_sim = np.array([-0.891, -2.748, -0.924, 0.163, -3.612, -0.300, -1.885, -2.351])
intercept_delta = intercept_sim - intercept_obs  

df = pd.DataFrame({
   # 'date': date,
    "$m_o$": slope_obs,
    "$b_o$": intercept_obs,
    "$r_1$": r1,
    "$\sigma_1$": sigma1,
    "$r_2$": r2,
    "$\sigma_2$": sigma2,
    "$C_r$": C_ratio,
    "$Sph\%$": sphericity,
    "$n_{440}$": n_440,
    "$\kappa_{440}$": k_440,
    "$AOD_{355}$": AOD_355,
    "$SSA_{355}$": SSA_355,
})

df.corr(method='pearson')

plt.figure(figsize = (12,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()


corr = df.corr()

# Create annotation matrix and remove diagonal labels
annot = corr.round(2).astype(str)
np.fill_diagonal(annot.values, "")

# Mask upper triangle (keep diagonal)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(17,12))

hm = sns.heatmap(
    corr,
    mask=mask,
    annot=annot,
    fmt="",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    linewidths=1.3,
    linecolor="white",
    square=True,
    annot_kws={"size":18, "weight":"bold"},
    cbar=False      # ← important
)

# Create a new axis *above* the heatmap for the colorbar
cbar_ax = fig.add_axes([0.25, 0.90, 0.5, 0.02])  
# [left, bottom, width, height]

# Add colorbar manually
cbar = fig.colorbar(hm.collections[0], cax=cbar_ax, orientation="horizontal")

# Put ticks ABOVE the colorbar
cbar.ax.xaxis.set_label_position('top')
cbar.ax.xaxis.tick_top()

#cbar.set_label("Correlation", fontsize=14, weight="bold", labelpad=8)
cbar.ax.tick_params(labelsize=20)

# Axes labels formatting
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=20)
ax.set_yticklabels(ax.get_yticklabels(),rotation =35, fontsize=20)

plt.show()

corr = df.corr()

# Create annotation matrix and remove diagonal labels
annot = corr.round(2).astype(str)
np.fill_diagonal(annot.values, "1")

# Mask upper triangle (keep diagonal)
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(17,13))

hm = sns.heatmap(
    corr,
    mask=mask,
    annot=annot,
    fmt="",
    cmap="coolwarm",
    vmin=-1, vmax=1,
    linewidths=1.3,
    linecolor="white",
    square=True,
    annot_kws={"size":18, "weight":"bold"},
    cbar=False
)

# # Create colorbar on top
# cbar_ax = fig.add_axes([0.25, 0.90, 0.9, 0.02])
# cbar = fig.colorbar(hm.collections[0], cax=cbar_ax, orientation="horizontal")
# cbar.ax.xaxis.set_label_position('top')
# cbar.ax.xaxis.tick_top()
# cbar.ax.tick_params(labelsize=25)

# Axis labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=25)
ax.set_yticklabels(ax.get_yticklabels(), rotation=35, fontsize=25)

# ----------------------------------------------------------
# ADD TEXTBOX (adjust text and position as needed)
# ----------------------------------------------------------

textbox_text = (
    "Parameter Definitions\n"
    "$r_1$, $r_2$: Fine/coarse mode median radius [$\mu m$]\n"
    "$\sigma_1\,\sigma_2$: Width of fine/coarse mode \n size distribution [$\mu m$]\n"
    "$C_r$: Coarse/Fine Mode Ratio\n"
    "$Sph\%$: Percent of Spherical Aerosols\n"
    "$n_{400}$,$\kappa_{400}$: Complex refractive index at 440nm\n"
    "$AOD_{355}$: Aerosol Optical Depth at 355nm\n"
    "$SSA_{355}$: Single Scatter Albedo at 355nm\n"
    "$m_o$/$b_o$: Observed regression slope/intercept"
)

fig.text(
    0.63, 0.66,             # (x, y) position in figure coords
    textbox_text,
    fontsize=25,
    va="center",
    ha="left",
    linespacing=1.8,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="black",
        linewidth=2
    )
)

plt.show()

