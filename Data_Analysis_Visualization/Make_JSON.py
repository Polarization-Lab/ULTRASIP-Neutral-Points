
"""
Extract file-level Neutral Point estimates to JSON
Missing attributes are skipped (set to None)
@author: C.M. DeLeon
"""

import os
import glob
import json
import h5py

import numpy as np




def json_safe(x):
    """
    Convert numpy scalars / arrays to JSON-safe Python types.
    """
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# -----------------------------
# User settings
# -----------------------------
basepath = 'D:/Data'
dates = ['2025_05_28','2025_06_04','2025_06_07','2025_06_09','2025_06_10','2025_06_13','2025_06_14',
          '2025_06_18','2025_06_23','2025_06_24','2025_06_25','2025_06_26','2025_06_30','2025_07_01',
          '2025_07_08','2025_07_09','2025_07_10','2025_07_13','2025_07_17','2025_07_18','2025_07_21',
          '2025_10_22','2025_10_23','2025_10_24']


output_dir = 'D:/Analyzed_Data_JSON'
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def get_attr(group, key):
    """Return attribute if it exists, else None."""
    return group.attrs[key] if key in group.attrs else None

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None
    

# -----------------------------
# Main loop
# -----------------------------
for date in dates:
    
    data = {
        "date": date,
        "timestamp": [],
        "acquisition": [],
        "sun_azimuth_deg": [],
        "sun_zenith_deg": [],
        "np_azimuth_deg": [],
        "np_zenith_deg": [],
        "np_az_error_arcsec": [],
        "np_zen_error_arcsec": []
    }


    folderdate = os.path.join(basepath, date)
    files = sorted(glob.glob(os.path.join(folderdate, '*.h5')))

    results = []

    print(f'\nProcessing {date}: {len(files)} files')

    for fname in files:
        print(f'  {os.path.basename(fname)}')

        try:
            with h5py.File(fname, 'r') as f:

                if 'Neutral Point Estimation' not in f:
                    continue

                np_est = f['Neutral Point Estimation']

                # ---- Required datasets ----
                if ('Estimation NP Location (zen,az) [deg]' not in np_est or
                    'Sun Location (zen,az) [deg]' not in np_est):
                    continue

                np_zen, np_az = np_est[
                    'Estimation NP Location (zen,az) [deg]'
                ][()]

                sun_zen, sun_az = np_est[
                    'Sun Location (zen,az) [deg]'
                ][()]
                data["timestamp"].append(
                    json_safe(get_attr(np_est, 'Time Stamp'))
                    )

                data["acquisition"].append(
                    json_safe(get_attr(np_est, 'Aquisition Number'))
                    )

                data["sun_azimuth_deg"].append(json_safe(sun_az))
                data["sun_zenith_deg"].append(json_safe(sun_zen))

                data["np_azimuth_deg"].append(json_safe(np_az))
                data["np_zenith_deg"].append(json_safe(np_zen))

                data["np_az_error_arcsec"].append(
                    json_safe(get_attr(np_est, 'Azimuth Error [arcseconds]'))
                    )

                data["np_zen_error_arcsec"].append(
                    json_safe(get_attr(np_est, 'Zenith Error [arcseconds]'))
                    )

        except Exception as e:
            print(f'    ERROR: {e}')

    # -----------------------------
    # Write JSON
    # -----------------------------
    outname = os.path.join(
        output_dir,
        f'BNP_observations_{date}_v3.json'
    )

    with open(outname, 'w') as f:
        json.dump(data, f, indent=2)

    print(
        f'  Saved {len(data["timestamp"])} NP entries → {outname}'
    )
