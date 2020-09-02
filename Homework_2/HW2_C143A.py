# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:11:30 2020

@author: Kellen Cheng
"""

import numpy as np
import matplotlib.pyplot as plt

def ptc(y0 , y1 , y2):
    #PTC calculates the tuning curve given average firing rates for certain directions.
    
    # ================================================================ #
    # YOUR CODE HERE:
    #  The function takes three inputs corresponding to the average 
    #  firing rate of a neuron during a reach to 0 degrees (y0), 120 
    #  degrees (y1) and 240 degrees (y2). The outputs, c0, c1, and 
    #  theta0 are the parameters of the tuning curve.
    # ================================================================ #
    
    k0 = (y0 + y1 + y2) / 3 # Calculate k0
    k1 = (np.sqrt(3) * y1 - np.sqrt(3) * y2) / 3 # Calculate k1
    k2 = (2 * y0 - y1 - y2) / 3 # Calculate k2
    
    k0 = 95 / 3
    k1 = 85 / (2 * np.sqrt(3))
    k2 = -35 / 6

    # Calculate c0, c1, theta0 based on our k0, k1, k2 values
    c0 = k0
    c1 = np.sqrt(k1 ** 2 + k2 ** 2)
    theta0 = 180 - (np.arcsin(k1 / c1) * 180 / np.pi)

    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #

    return c0,c1,theta0

c0, c1, theta0=ptc(25,70,10)
print('c0 = ', c0)
print('c1 = ', c1)
print('theta0 = ', theta0)

theta = np.linspace(0, 2*np.pi, num=80)

plt.plot([0,60,120,180,240,300],[25,40,70,30,10,15],'r*',10)
plt.plot(theta * 180 / np.pi,c0 + c1 *np.cos(theta - theta0 * np.pi/180),'b',2)
plt.xlim ([-10 ,370])
plt.ylim ([-10,80])
plt.xlabel(r'$\theta$ (degrees)');
plt.ylabel(r'$f(\theta)$');
