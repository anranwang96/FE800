#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:37:22 2019

@author: anran
"""
import pandas as pd
import numpy as np

##liquidity test
for i in range(len(trainData_whole)):
    unit_min = min(trainData_whole[i]['Volume'])
    if unit_min > 15600:
        print('True')
    else:
        print('False')

##Outliers test

import matplotlib.pyplot as plt
plt.plot(range(len(train_close['BNDXe'])), train_close['BNDXe']) 

##ADF
from arch.unitroot import ADF
ADF(train_close['AAPL'])

##Difference
aapl_diff = np.diff(train_close['AAPL'])
ADF(aapl_diff)


