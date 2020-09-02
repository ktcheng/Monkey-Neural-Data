# -*- coding: utf-8 -*-
"""
Created on Sat May 23 17:38:11 2020

@author: Kellen Cheng
"""

import numpy as np

A = np.array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 2, 3, 4]])
B = np.linalg.inv(A)
print(B)