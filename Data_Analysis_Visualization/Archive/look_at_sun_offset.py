# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 11:29:55 2026

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
basepath = 'D:/Data'
differences = []
sun_zeniths = []
test = []


days = ['2025_06_09','2025_06_18','2025_06_23','2025_06_24','2025_06_25',
        '2025_06_30','2025_07_01','2025_07_08','2025_07_09','2025_07_10',
        '2025_07_13','2025_07_17','2025_07_18']

for date in days:
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
            
            if aqnum == 0:
                view_zen = 90 - view_zen 

            I = aq['UV Image Data/I_corrected'][:]
            
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
            
            if aqnum == 1: 
                user_input = input("Save Value? (yes/no): ")

                if user_input.lower() in ["yes", "y", "ye"]:
                    print("Continuing...")
                    differences = np.append(differences,diff)
                    sun_zeniths = np.append(sun_zeniths,sun_zen)
                    # Place the code you want to run next here
                else:
                    print("Skipping...")     
                    continue
