#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:38:39 2019

@author: anran
"""

##Wavelet transform
import pywt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

bac = train_close['BAC']

coef = pywt.swt(bac, 'haar', level = 2, trim_approx = True, norm = True)

cA2 = coef[0][0:2750]
cD2 = coef[1][0:2750]
cD1 = coef[2][0:2750]
bac_t = bac[3:2751]

coefnew = np.c_[cA2, cD2, cD1]

##Construct LSTM dataset [sample, timestep, feature]
train_X = np.zeros([len(bac_t)*3, 3])
for i in range(train_X.shape[0]):
    diff = int(np.floor(i/3)*2)
    train_X[i] = coefnew[i-diff]
    
train_X = train_X.reshape((int(train_X.shape[0]/3), 3, train_X.shape[1]))  
train_y = bac_t

##Test set
bac_test = test_close['BAC']
coef_test = pywt.swt(bac_test, 'haar', level = 2, trim_approx = True, norm = True)

cA2_test = coef_test[0][0:680]
cD2_test = coef_test[1][0:680]
cD1_test = coef_test[2][0:680]
bac_test_t = bac_test[3:681]

coef_test_new = np.c_[cA2_test, cD2_test, cD1_test]

test_X = np.zeros([len(bac_test_t)*3, 3])
for i in range(test_X.shape[0]):
    diff = int(np.floor(i/3)*2)
    test_X[i] = coef_test_new[i-diff]
    
test_X = test_X.reshape((int(test_X.shape[0]/3), 3, test_X.shape[1]))  
test_y = bac_test_t

    
    

    













