# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import h5py
import cmocean.cm as cmo
import glob
import os
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

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
date = '2025_07_10'
basepath = 'D:/Data'

folderdate = os.path.join(basepath, date)
file = glob.glob(f'{folderdate}/*08_24_54*.h5')

f = h5py.File(file[0], 'r')

# =========================
# Create FIXED figure
# =========================
fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)

# Expand left margin so ylabel is never clipped
ax1 = fig.add_axes([0.10, 0.12, 0.38, 0.78])   # left panel
ax2 = fig.add_axes([0.57, 0.12, 0.38, 0.78])   # right panel

# Prevent autoscaling forever
ax1.set_autoscale_on(False)
ax2.set_autoscale_on(False)

writer = animation.FFMpegWriter(fps=0.5)
video_path = os.path.join(folderdate, f'{date}_polarization_movie_adolp.mp4')

with writer.saving(fig, video_path,dpi=100):

    for aqnum in range(0, 16):

        print("Processing acquisition:", aqnum)

        aq = f[f'Aquistion_{aqnum}']

        view_az = aq['UV Image Data/view_az'][:]
        view_zen = aq['UV Image Data/view_zen'][:]

        I = aq['UV Image Data/I_corrected'][:]
        Q = aq['UV Image Data/Q_corrected'][:]
        U = aq['UV Image Data/U_corrected'][:]

        q = Q / I
        u = U / I

        dolp = np.sqrt(q**2 + u**2) * 100
        aolp = 0.5 * np.arctan2(U, Q)
        aolp = np.mod(np.degrees(aolp), 180)
        
        if aqnum == 0:
            view_zen = 90 - view_zen 

        center_az = view_az[1424,1424]
        center_zen = view_zen[1424,1424]


        im1 = ax1.imshow(
                np.log(dolp),
                cmap='Blues_r',
                vmin=-1,
                vmax=1,
                extent=[view_az.min(), view_az.max(),
                        view_zen.max(), view_zen.min()],
                interpolation='none'
        )

        im2 = ax2.imshow(
                aolp,
                cmap=cmo.phase,
                vmin=0,
                vmax=180,
                extent=[view_az.min(), view_az.max(),
                        view_zen.max(), view_zen.min()],
                interpolation='none'
        )

        # LOCK LIMITS (never change)
        ax1.set_xlim(view_az.min(), view_az.max())
        ax1.set_ylim(view_zen.max(), view_zen.min())
        ax2.set_xlim(view_az.min(), view_az.max())
        ax2.set_ylim(view_zen.max(), view_zen.min())

        # Labels + titles (set once)
        ax1.set_title('log(Degree of Linear Polarization [%])', fontsize=26)
        ax1.set_xlabel('Azimuth [$\circ$]', fontsize=25)
        ax1.set_ylabel('Zenith [$\circ$]', fontsize=25, labelpad=20)

        ax2.set_title('Angle of Linear Polarization [$\circ$]', fontsize=26)
        ax2.set_xlabel('Azimuth [$\circ$]', fontsize=25)

        # Center ticks only
        ax1.set_xticks([center_az])
        ax1.set_yticks([center_zen])
        ax1.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        ax1.set_yticklabels([f'{center_zen:.1f}°'], fontsize=25)

        ax2.set_xticks([center_az])
        ax2.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        ax2.set_yticks([])

        # else:
        #     im1.set_data(np.log(dolp))
        #     im2.set_data(aolp)
            
        #     # LOCK LIMITS (never change)
        #     ax1.set_xlim(view_az.min(), view_az.max())
        #     ax1.set_ylim(view_zen.max(), view_zen.min())
        #     ax2.set_xlim(view_az.min(), view_az.max())
        #     ax2.set_ylim(view_zen.max(), view_zen.min())
            
        #     ax1.set_xticks([center_az])
        #     ax1.set_yticks([center_zen])
        #     ax1.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        #     ax1.set_yticklabels([f'{center_zen:.1f}°'], fontsize=25)

        #     ax2.set_xticks([center_az])
        #     ax2.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        #     ax2.set_yticks([])

        writer.grab_frame()

plt.close(fig)


print("Video saved to:")
print(video_path)

#--------------------------------Stokes-------------------------------------#
# =========================
# Create FIXED figure
# =========================
fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)

# Expand left margin so ylabel is never clipped
ax1 = fig.add_axes([0.10, 0.12, 0.38, 0.78])   # left panel
ax2 = fig.add_axes([0.57, 0.12, 0.38, 0.78])   # right panel

# Prevent autoscaling forever
ax1.set_autoscale_on(False)
ax2.set_autoscale_on(False)

writer = animation.FFMpegWriter(fps=0.5)
video_path = os.path.join(folderdate, f'{date}_polarization_movie_stokes.mp4')

with writer.saving(fig, video_path,dpi=100):

    for aqnum in range(0, 16):

        print("Processing acquisition:", aqnum)

        aq = f[f'Aquistion_{aqnum}']

        view_az = aq['UV Image Data/view_az'][:]
        view_zen = aq['UV Image Data/view_zen'][:]

        I = aq['UV Image Data/I_corrected'][:]
        Q = aq['UV Image Data/Q_corrected'][:]
        U = aq['UV Image Data/U_corrected'][:]

        q = Q / I
        u = U / I

        dolp = np.sqrt(q**2 + u**2) * 100
        aolp = 0.5 * np.arctan2(U, Q)
        aolp = np.mod(np.degrees(aolp), 180)
        
        if aqnum == 0:
            view_zen = 90 - view_zen 

        center_az = view_az[1424,1424]
        center_zen = view_zen[1424,1424]


        im1 = ax1.imshow(
                q,
                cmap=colmap,
                vmin=-0.1,
                vmax=0.1,
                extent=[view_az.min(), view_az.max(),
                        view_zen.max(), view_zen.min()],
                interpolation='none'
        )

        im2 = ax2.imshow(
                u,
                cmap=colmap,
                vmin=-0.1,
                vmax=0.1,
                extent=[view_az.min(), view_az.max(),
                        view_zen.max(), view_zen.min()],
                interpolation='none'
        )

        # LOCK LIMITS (never change)
        ax1.set_xlim(view_az.min(), view_az.max())
        ax1.set_ylim(view_zen.max(), view_zen.min())
        ax2.set_xlim(view_az.min(), view_az.max())
        ax2.set_ylim(view_zen.max(), view_zen.min())

        # Labels + titles (set once)
        ax1.set_title('Q/I', fontsize=26)
        ax1.set_xlabel('Azimuth [$\circ$]', fontsize=25)
        ax1.set_ylabel('Zenith [$\circ$]', fontsize=25, labelpad=20)

        ax2.set_title('U/I', fontsize=26)
        ax2.set_xlabel('Azimuth [$\circ$]', fontsize=25)

        # Center ticks only
        ax1.set_xticks([center_az])
        ax1.set_yticks([center_zen])
        ax1.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        ax1.set_yticklabels([f'{center_zen:.1f}°'], fontsize=25)

        ax2.set_xticks([center_az])
        ax2.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        ax2.set_yticks([])

        # else:
        #     im1.set_data(np.log(dolp))
        #     im2.set_data(aolp)
            
        #     # LOCK LIMITS (never change)
        #     ax1.set_xlim(view_az.min(), view_az.max())
        #     ax1.set_ylim(view_zen.max(), view_zen.min())
        #     ax2.set_xlim(view_az.min(), view_az.max())
        #     ax2.set_ylim(view_zen.max(), view_zen.min())
            
        #     ax1.set_xticks([center_az])
        #     ax1.set_yticks([center_zen])
        #     ax1.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        #     ax1.set_yticklabels([f'{center_zen:.1f}°'], fontsize=25)

        #     ax2.set_xticks([center_az])
        #     ax2.set_xticklabels([f'{center_az:.1f}°'], fontsize=25)
        #     ax2.set_yticks([])

        writer.grab_frame()

plt.close(fig)

#--------------------------------Stokes Avg-------------------------------------#
# =========================
# Create FIXED figure
# =========================
fig = plt.figure(figsize=(18, 8), dpi=100, constrained_layout=False)

# Expand left margin so ylabel is never clipped
ax1 = fig.add_axes([0.10, 0.12, 0.38, 0.78])   # left panel
ax2 = fig.add_axes([0.57, 0.12, 0.38, 0.78])   # right panel

# Prevent autoscaling forever
ax1.set_autoscale_on(False)
ax2.set_autoscale_on(False)

writer = animation.FFMpegWriter(fps=0.5)
video_path = os.path.join(folderdate, f'{date}_polarization_movie_stokesavg.mp4')

with writer.saving(fig, video_path, dpi=100):

    for aqnum in range(0, 16):

        print("Processing acquisition:", aqnum)

        # Clear axes each frame
        ax1.clear()
        ax2.clear()

        aq = f[f'Aquistion_{aqnum}']

        view_az = aq['UV Image Data/view_az'][:]
        view_zen = aq['UV Image Data/view_zen'][:]

        I = aq['UV Image Data/I_corrected'][:]
        Q = aq['UV Image Data/Q_corrected'][:]
        U = aq['UV Image Data/U_corrected'][:]

        q = Q / I
        u = U / I

        # axis 0 = columns, axis 1 = rows
        avgQ = np.flip(np.average(q, axis=1))
        avgU = np.average(u, axis=0)

        dolp = np.sqrt(q**2 + u**2) * 100
        aolp = 0.5 * np.arctan2(U, Q)
        aolp = np.mod(np.degrees(aolp), 180)

        if aqnum == 0:
            view_zen = 90 - view_zen

        center_az = view_az[1424,1424]
        center_zen = view_zen[1424,1424]

        # ---- Plotting ----

        ax1.scatter(avgQ, view_zen[:,0], color='green')
        ax1.axvline(x=0, lw=3, color='red')

        ax2.scatter(avgU, view_az[0,:], color='green')
        ax2.axvline(x=0, lw=3, color='red')

        # Titles
        ax1.set_title('Row Avg Q', fontsize=20)
        ax2.set_title('Column Avg U', fontsize=20)

        # Labels
        ax1.set_xlabel(r'$\bar{r}_{Q}$', fontsize=15)
        ax1.set_ylabel('Zenith [$^\circ$]', fontsize=15)
        ax1.tick_params(axis='both', labelsize=25)


        ax2.set_xlabel(r'$\bar{r}_{U}$', fontsize=15)
        ax2.set_ylabel('Azimuth [$^\circ$]', fontsize=15)
        ax2.tick_params(axis='both', labelsize=25)

        writer.grab_frame()

plt.close(fig)
f.close()

print("Video saved to:")
print(video_path)
