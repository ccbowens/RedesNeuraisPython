# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:47:13 2020

@author: Camila
"""

import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

