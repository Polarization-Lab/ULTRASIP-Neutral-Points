# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 15:58:03 2025

@author: ULTRASIP_1

"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols, mixedlm



dates = np.array(["6/30", "7/1", "7/8", "7/9", "7/10", "7/13", "7/17", "7/18"])


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

dates_dt = pd.to_datetime("2024-" + dates)
dates_num = dates_dt.map(pd.Timestamp.toordinal)

df = pd.DataFrame({
    "date": dates_num,
    "slope_obs": slope_obs,
    "intercept_obs": intercept_obs,
    "r1": r1,
    "sigma1": sigma1,
    "r2": r2,
    "sigma2": sigma2,
    "C_ratio": C_ratio,
    "sphericity": sphericity,
    "n440": n_440,
    "k440": k_440,
    "AOD355": AOD_355,
    "SSA355": SSA_355,
})



df.corr(method='pearson')

plt.figure(figsize = (15,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Fit the fixed effects model
fixed_effects_model = ols('slope_obs ~ sphericity + r2', data=df).fit()


# Summary of the fixed effects model
print(fixed_effects_model.summary())

