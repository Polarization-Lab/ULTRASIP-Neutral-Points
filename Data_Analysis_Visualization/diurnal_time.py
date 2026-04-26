# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:46:42 2026

@author: deleo
"""

import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates

data_path = "C:/Users/deleo/Documents/BNP_daily_v3_allfields_with_rayleigh"
json_files = glob.glob(f'{data_path}/BNP*.json')
# Load JSON
with open(json_files[16], 'r') as f:
    data = json.load(f)

# Extract time and variables
time_str = data['LocalTime(hh:mm:ss)']
np_zen = np.array(data['np_zenith_deg'])
sun_zen = np.array(data['sun_zenith_deg'])
sun_az = np.array(data['sun_azimuth_deg'])

diff =  np_zen - sun_zen

# Convert time to datetime objects
time = [datetime.strptime(t, "%H:%M:%S") for t in time_str]


# Plot
plt.figure(figsize=(8,5))

plt.scatter(sun_az[0:20], np_zen[0:20], s=80, color='red', edgecolor='black', linewidths=1.5,label='BNP')
plt.scatter(sun_az[0:20], sun_zen[0:20], s=80, color='orange', edgecolor='black',linewidths=1.5, label='Sun')

plt.xlabel('Azimuth Angle [$\circ$]',fontsize=15)
plt.ylabel('Zenith Angle [$\circ$]',fontsize=15)

# ---- FIX TIME AXIS ----
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)

plt.grid(True)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


plt.figure(figsize=(8,5))

plt.scatter(sun_zen, diff, s=80, color='red', edgecolor='black', linewidths=1.5,label='BNP')

plt.xlabel('Sun Zenith Angle [$\circ$]',fontsize=15)
plt.ylabel('$\delta [\circ]$',fontsize=15)
ax=plt.gca()
ax.tick_params(axis='both', which='major', labelsize=14)


# ---- FIX TIME AXIS ----
ax = plt.gca()

plt.grid(True)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()