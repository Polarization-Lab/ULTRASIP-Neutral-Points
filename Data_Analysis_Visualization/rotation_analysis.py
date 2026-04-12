import numpy as np
import pandas as pd
import statsmodels.api as sm
import h5py
import os
import glob

# ---------------- Rotation ----------------

def rotate_qu(avgq, avgu, theta):

    t = np.deg2rad(theta)

    q_rot = avgq*np.cos(2*t) + avgu*np.sin(2*t)
    u_rot = -avgq*np.sin(2*t) + avgu*np.cos(2*t)

    return q_rot, u_rot


# ---------------- Data ----------------

date = '2025_07_13'
basepath = 'D:/Data'
folderdate = os.path.join(basepath,date)

files = glob.glob(f'{folderdate}/*.h5')

rotation_angles = np.linspace(-3,3,301)

results = []
rotation_results = []

# ---------------- Main Loop ----------------

for file in files:

    with h5py.File(file,'r') as f:

        if "Neutral Point Estimation" not in f:
            continue

        print("\nProcessing:", os.path.basename(file))

        np_est = f["Neutral Point Estimation"]

        for name, value in np_est.attrs.items():

            var_name = name.replace(" ", "_").replace("-", "_").lower()
            globals()[var_name] = value


        # -------------------------------------
        # PRECOMPUTE acquisition arrays
        # -------------------------------------

        acquisitions = []

        for aqnum in range(len(f.keys())-1):

            if f"Aquistion_{aqnum}" not in f:
                continue

            aq = f[f"Aquistion_{aqnum}"]

            I = aq["UV Image Data/I_corrected"][:]
            Q = aq["UV Image Data/Q_corrected"][:]
            U = aq["UV Image Data/U_corrected"][:]

            q = Q/I
            u = U/I

            vza = aq["UV Image Data/view_zen"][:][:,0]
            vaz = aq["UV Image Data/view_az"][:][0,:]

            sun_az = aq['UV Image Data/sun_az'][()]

            q_start, q_stop = map(int, q_cropped_region.split(':'))
            u_start, u_stop = map(int, u_cropped_region.split(':'))

            avgq = np.flip(np.average(q,axis=1))[q_start:q_stop]
            avgu = np.average(u,axis=0)[u_start:u_stop]

            vza_crop = vza[q_start:q_stop]
            vaz_crop = vaz[u_start:u_stop]

            acquisitions.append({
                "q":q,
                "u":u,
                "avgq":avgq,
                "avgu":avgu,
                "vza":vza_crop,
                "vaz":vaz_crop,
                "sun_az":sun_az,
                "q_start":q_start,
                "u_start":u_start
            })


        # -------------------------------------
        # ROTATION SEARCH
        # -------------------------------------

        best_std = np.inf
        best_rotation = None

        for theta in rotation_angles:

            az_diffs = []

            for acq in acquisitions:
                
                q = acq["q"]
                u = acq["u"]
                avgq = acq["avgq"]
                avgu = acq["avgu"]
                vza_crop = acq["vza"]
                vaz_crop = acq["vaz"]
                sun_az = acq["sun_az"]
                q_start = acq["q_start"]
                u_start = acq["u_start"]

                q_rot,u_rot = rotate_qu(q,u,theta)
                
                q_rot = np.flip(np.average(q_rot,axis=1))[q_start:q_stop]
                u_rot = np.average(u_rot,axis=0)[u_start:u_stop]


                # ----- Q fit -----

                X = sm.add_constant(q_rot)
                weights = np.ones_like(vza_crop)

                qfit = sm.WLS(vza_crop,X,weights=weights).fit()


                # ----- U fit -----

                X = sm.add_constant(u_rot)

                ufit = sm.WLS(vaz_crop,X,weights=weights).fit()

                if ufit.rsquared > 0.9:

                    uint = ufit.params[0]
                    az_diff = sun_az - uint

                    az_diffs.append(az_diff)


            if len(az_diffs) > 1:

                std_val = np.std(az_diffs)

                rotation_results.append({
                    "file":os.path.basename(file),
                    "rotation_deg":theta,
                    "std_abs_az_diff_deg":std_val
                })

                if std_val < best_std:

                    best_std = std_val
                    best_rotation = theta


        print("Best rotation:",best_rotation)
        print("STD:",best_std)



# ---------------- Save ----------------

df = pd.DataFrame(results)

outfile = os.path.join(folderdate,f"{date}_neutral_rotation_corrected.csv")
df.to_csv(outfile,index=False)

rot_df = pd.DataFrame(rotation_results)

outfile2 = os.path.join(folderdate,f"{date}_rotation_scan.csv")
rot_df.to_csv(outfile2,index=False)

print("\nSaved results.")