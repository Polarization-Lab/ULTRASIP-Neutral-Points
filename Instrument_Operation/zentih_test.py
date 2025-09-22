# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 14:03:05 2025

@author: ULTRASIP_1
"""

#Test for repeating process at specific zenith angles

from datetime import datetime
from suncalc import get_position
import numpy as np
import time

sun_samples = list(range(10, 71, 2))
tolerance = 0.1 # degrees


latitude = 45.66487
longitude = -111.04800


while True:
    dt = datetime.now()
    sun_pos = get_position(dt, longitude, latitude)
    sun_zenith = np.degrees(sun_pos['altitude'])

    for target in sun_samples:
        if abs(sun_zenith - target) < tolerance:
            print(f"Hit target zenith {target}° at {dt}, actual = {sun_zenith:.4f}°")
            time.sleep(360)
    
