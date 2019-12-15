#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:41:48 2019

@author: anran
"""
##Read data from a folder
import pandas as pd
import numpy as np
import glob, os
import pywt
import pandas as pd
import matplotlib
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM



#Training set
path_train = '/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Training'
file_train = glob.glob(os.path.join(path_train, "*.csv"))
print(file_train)

trainData_whole = []

for f in file_train:
    trainData_whole.append(pd.read_csv(f))

#Test set
path_test = '/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Test'
file_test = glob.glob(os.path.join(path_test, "*.csv"))
print(file_test)

testData_whole = []

for f in file_test:
    testData_whole.append(pd.read_csv(f))
    
    
#Training data index
dataset_name = []

for i in range(len(file_train)):
    if file_train[i][-17] == '/':
        dataset_name.append(file_train[i][-16:-14])
    elif file_train[i][-18] == '/':
        dataset_name.append(file_train[i][-17:-14])
    elif file_train[i][-16] == '/':
        dataset_name.append(file_train[i][-15])
    elif file_train[i][-20] == '/':
        dataset_name.append(file_train[i][-19:-14])
    else: 
        dataset_name.append(file_train[i][-18:-14])

#Test data index
dataset_name_test = []

for i in range(len(file_test)):
    if file_test[i][-17] == '/':
        dataset_name_test.append(file_test[i][-16:-14])
    elif file_test[i][-18] == '/':
        dataset_name_test.append(file_test[i][-17:-14])
    elif file_test[i][-16] == '/':
        dataset_name_test.append(file_test[i][-15])
    elif file_test[i][-20] == '/':
        dataset_name_test.append(file_test[i][-19:-14])
    else: 
        dataset_name_test.append(file_test[i][-18:-14])

#Create train dataset consist of close price
train_length = np.asarray([len(trainData_whole[i]) for i in range(len(trainData_whole))])
train_length_max = np.max(train_length)

train_close = pd.DataFrame(np.nan, index = trainData_whole[np.where(train_length == train_length_max)[0][0]]['Date'], columns = dataset_name)

for i in range(len(dataset_name)):
    if len(trainData_whole[i]) < train_length_max:
        train_close.iloc[train_length_max-len(trainData_whole[i]):, i] = list(trainData_whole[i]['Close'])
    else:
        train_close.iloc[:, i] = list(trainData_whole[i]['Close'])
        
    
##Create test dataset consist of 50 stock's close price
test_close = pd.DataFrame(index = testData_whole[0]['Date'], columns = dataset_name_test)

for i in range(test_close.shape[1]):
    test_close.iloc[:, i] = list(testData_whole[i]['Close'])    
    
test_close = test_close[dataset_name]

#Fill null
for i in range(1, (train_close.shape[0]-1)):
    for j in range(train_close.shape[1]):
        if train_close.iloc[i, j] == None and train_close.iloc[i-1, j] != None and train_close.iloc[i+1, j] != None:
            train_close.iloc[i, j] = train_close.iloc[i-1, j]
        
test_close = test_close.fillna(method='pad')

