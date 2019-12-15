#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 10:55:33 2019

@author: anran
"""
import numpy as np
import pandas as pd
import pywt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM

#Training start date
train_startdate = np.array([train_close.iloc[:, j].isna().sum() for j in range(train_close.shape[1])])

#Generate whole models
for j in range(train_close.shape[1]):  
    name = train_close.iloc[:, j].name
    startdate = train_startdate[j]
    
    #get asset train close
    asset = train_close[name][startdate:]
    
    #level = 2 must has a len = multiple of 2**2 = 4
    if len(asset) % 4 != 0:
        mult4 = int(len(asset) - np.floor(len(asset)/4)*4)
        asset = asset[mult4:]      
        
    #wavelet transform
    coef_train = pywt.swt(asset, 'haar', level = 2, trim_approx = True, norm = True)
    
    #smooth and detailed coef
    cA2_train = coef_train[0][:-5]
    cD2_train = coef_train[1][:-5]
    cD1_train = coef_train[2][:-5]
    
    #training output
    train_y = asset[3:-5]
    
    #combine coef
    coef_train_cb = np.c_[cA2_train, cD2_train, cD1_train]

    ##Construct repeated coefficients array
    train_X = np.zeros([len(train_y)*3, 3])
    for i in range(train_X.shape[0]):
        diff = int(np.floor(i/3)*2)
        train_X[i] = coef_train_cb[i-diff]
        
    #Reshape to [sample, timestep, feature]
    train_X = train_X.reshape((int(train_X.shape[0]/3), 3, train_X.shape[1]))  
     
    #Train the model
    model = Sequential()
    model.add(LSTM(20, activation='relu',input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss = 'mse', optimizer = 'adam')
    
    model.fit(train_X, train_y, epochs = 25)
    #Save the model
    model.save('/Users/anran/Desktop/FE/FE800/Works/LSTM_model/'+name+'.h5')
    
    del model

asset = train_close['AAPL'][1:]
coef_train = pywt.swt(asset, 'haar', level = 2, trim_approx = True, norm = True)


#Decompose function
def Stock_wave(data):
    data_close = data['Close']
    data_close_comp = np.array(data_close[3:])
    coef = pywt.swt(data_close, 'haar', level = 2, trim_approx = True, norm = True)
    
    return data_close_comp, coef

#Generate LSTM structure coefficients
def LSTM_structure(coef):
    cA2 = coef[0]
    cD2 = coef[1]
    cD1 = coef[2]
    coef_con = np.c_[cA2, cD2, cD1]
    
    assetlen = (len(cA2)-2) * 3
    
    X = np.zeros([assetlen, 3])
    
    for i in range(X.shape[0]):
        diff = int(np.floor(i/3)*2)
        X[i] = coef_con[i-diff]
    
    X = X.reshape((int(X.shape[0]/3), 3, X.shape[1]))
    
    return X

#Prediction
def Stock_dec(data, model):
    data_close_comp, coef_data = Stock_wave(data)
    data_X = LSTM_structure(coef_data)
    data_pre =  model.predict(data_X)
    
    return data_pre, data_close_comp
    
#Function that return prediction
def Test_prediction(asset_name, test_close):
    #Test set
    asset_test = test_close[asset_name]
    
    coef_test = pywt.swt(asset_test, 'haar', level = 2, trim_approx = True, norm = True)
    
    cA2_test = coef_test[0][:-5]
    cD2_test = coef_test[1][:-5]
    cD1_test = coef_test[2][:-5]
    test_y = asset_test[3:-5]
    
    coef_test_cb = np.c_[cA2_test, cD2_test, cD1_test]
    
    test_X = np.zeros([len(test_y)*3, 3])
    for i in range(test_X.shape[0]):
        diff = int(np.floor(i/3)*2)
        test_X[i] = coef_test_cb[i-diff]
        
    test_X = test_X.reshape((int(test_X.shape[0]/3), 3, test_X.shape[1]))  
    
    #Load the model
    model = load_model('/Users/anran/Desktop/FE/FE800/Works/LSTM_model/'+asset_name+'.h5')

    #Prediction
    test_predict = model.predict(test_X)
    
    prediction = []
    for i in range(test_predict.shape[0]):
        prediction.append(test_predict[i][0])
        
    results = pd.DataFrame({'Prediction': prediction, 
                            'True Value': test_y}, index = test_y.index)
    return results

#Predict whole stocks
Test_predictions = pd.DataFrame(columns = dataset_name)
for name in dataset_name:
    Test_predictions[name] = Test_prediction(name, test_close)['Prediction']
    
#Save predictions
Test_predictions.to_csv("/Users/anran/Desktop/FE/FE800/Works/Test_predictions.csv",index=True,sep=',')


plt.plot(Test_predictions.index[:], Test_predictions['BNDXe'])
       
plt.plot(range(676), Test_predictions['ICVTe'])
plt.plot(range(676), test_close['ICVTe'][3:-5])
plt.show()







