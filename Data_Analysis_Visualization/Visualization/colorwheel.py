# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 08:55:21 2024

@author: ULTRASIP_1
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import cmocean.cm as cmo


def radians_to_degrees(x, pos):
    degrees = np.degrees(x)
    return int(degrees)

def colorwheel_imshow(f,data_array,title):
    #f = plt.figure(figsize=(10, 8))
    # ax = f.add_subplot(111)
    quant_steps = 2056
    plt.imshow(data_array,cmap=cmo.phase,vmin=0,vmax=180,interpolation='None')
    plt.title(title,fontsize=20)
    plt.axis('off')
    # Create polar axes
    display_axes = f.add_axes([0.8, 0.7, 0.2, 0.2], projection='polar')

    # Plot the colorbar onto the polar axis
    norm = matplotlib.colors.Normalize(0.0, np.pi)
    cb = matplotlib.colorbar.ColorbarBase(display_axes, cmap=matplotlib.cm.get_cmap(cmo.phase, quant_steps),
                                      norm=norm, orientation='horizontal', format=FuncFormatter(radians_to_degrees))

    # Aesthetics - get rid of border and axis labels
    cb.outline.set_visible(False)
    xL = [0, np.pi/4, np.pi / 2, np.pi*3/4, np.pi]
    cb.set_ticks(xL)
    cb.ax.tick_params(labelsize=20, zorder=10)

    # Set polar axis limits
    display_axes.set_rlim([-1, 1])

    # Adjust tick parameters for better layout
    cb.ax.tick_params(axis='both', which='major', pad=1)

    #plt.show()




