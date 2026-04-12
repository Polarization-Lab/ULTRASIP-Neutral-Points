# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 13:14:08 2026

@author: ULTRASIP_1
"""

#Scan test

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:16:33 2026

@author: ULTRASIP_1
"""

import matplotlib.pyplot as plt
import numpy as np
import h5py
import cmocean.cm as cmo
import os,cv2
import glob
import os
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

img_x = 2848
img_y = 2848
#Custom colormap for Q and U
#Blue to Red Color scale for S1 and S2
colmap = np.zeros((255,3));
# Red
colmap[126:183,0]= np.linspace(0,1,57);
colmap[183:255,0]= 1;
# Green
colmap[0:96,1] = np.linspace(1,0,96);
colmap[158:255,1]= np.linspace(0,1,97);
# Blue
colmap[0:71,2] = 1;
colmap[71:128,2]= np.linspace(1,0,57);
colmap2 = colmap[128:,:]
colmap = ListedColormap(colmap)

# =========================
# Load observations
# =========================
date = '2025_06_25'
basepath = 'D:/Data'
differences = []
sun_zeniths = []
folderdate = os.path.join(basepath, date)
file = glob.glob(f'{folderdate}/*.h5')
idx = np.arange(len(file))

for num in idx:
    try:
        f = h5py.File(file[num], 'r')
    except OSError as e:
        print(f"Skipping bad file: {file[num]}")
        print(e)
        continue
    
    if "Center Pixel (x,y)" in f["Measurement_Metadata"].attrs:
        center_x,center_y = f["Measurement_Metadata"].attrs["Center Pixel (x,y)"]
    else: 
        continue


    for aqnum in range(0, 2):

        print("Processing acquisition:", aqnum)
        key = f"Aquistion_{aqnum}"

        aq = f.get(key)
        if aq is None:
            continue

    # safe to use aq here
        view_az = aq['UV Image Data/view_az'][:]
        view_zen = aq['UV Image Data/view_zen'][:]
        sun_zen = aq['UV Image Data/sun_zen'][()]


        I = aq['UV Image Data/I_corrected'][:]
        Q = aq['UV Image Data/Q_corrected'][:]
        U = aq['UV Image Data/U_corrected'][:]

        q = Q / I
        u = U / I

        dolp = np.sqrt(q**2 + u**2) * 100
        aolp = 0.5 * np.arctan2(U, Q)
        aolp = np.mod(np.degrees(aolp), 180)
    
        # Apply a manual threshold (e.g., 90% of max intensity)
        thresh_val = 0.95 * np.max(I.flatten())
        _, thresh = cv2.threshold(I, thresh_val, 1, cv2.THRESH_BINARY)
        
        thresh_uint8 = (thresh * 255).astype(np.uint8)
    
        # Compute moments
        M = cv2.moments(thresh_uint8)
        # Calculate centroid
        if M["m00"] != 0:
            x_center = int(M["m10"] / M["m00"])
            y_center = int(M["m01"] / M["m00"])

            # print(f"Sun center: (x={x_center}, y={y_center})")
        else:
            print("Could not detect the sun.")
            continue
        
        # if aqnum == 0:
        #     view_zen = 90 - view_zen 

        diff = (view_zen[center_y,center_x] - view_zen[y_center,x_center]) 
        fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)
        plt.imshow(I, cmap = 'gray',
                    interpolation='none')
        plt.scatter(center_x,center_y,s=20,color='green')
        plt.scatter(x_center,y_center,s=20,color='red')
        plt.title(f"{date},{num},{diff}")
        plt.show()
    
        print(num)
        print(view_zen[y_center,x_center])
        print(view_zen[center_y,center_x])
        print("diff:",diff)
        
        # if aqnum == 1: 
        #     differences = np.append(differences,diff)
        #     sun_zeniths = np.append(sun_zeniths,sun_zen)
        user_input = input("Save Value? (yes/no): ")

        if user_input.lower() in ["yes", "y", "ye"]:
            print("Continuing...")
            differences = np.append(differences,diff)
            sun_zeniths = np.append(sun_zeniths,sun_zen)
            # Place the code you want to run next here
        else:
            print("Skipping...")     
            continue

plt.figure()
plt.scatter(sun_zeniths, differences)
plt.ylim([0,1])
plt.xticks(np.arange(20, 90 + 1, 3))
plt.yticks(np.arange(0, 1 + 0.1, 0.1))
plt.grid(True)
plt.title(f"N:{len(differences[differences > 0])}, Date: {date}")
plt.xlabel("Sun Zenith Angle")
plt.ylabel("Difference from 1st Sun Image")
plt.show()

print("N = ", len(differences))
print("avg:",np.average(differences))
print("med:",np.median(differences))
print("std:",np.std(differences))


