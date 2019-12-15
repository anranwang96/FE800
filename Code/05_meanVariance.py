#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:46:57 2019

@author: anran
"""

import numpy as np
import pandas as pd
from cvxopt import matrix, solvers

#Read predictions of test set
test_predictions = pd.read_csv('/Users/anran/Desktop/FE/FE800/Works/Test_predictions.csv', index_col = 0)

#Log return
def log_return(price):
    returnvalue = np.array([np.log(price[i+1]/price[i]) for i in range(len(price)-1)])
    return pd.Series(returnvalue, index = price.index[1:])

#Daily return
def daily_return(price):
    returnvalue = np.array([(price[i+1]/price[i])-1 for i in range(len(price)-1)])    
    return pd.Series(returnvalue, index = price.index[1:])

#Convert annual to daily
def ann_to_daily(ann_value):
    return np.power(1+ann_value, 1/365)-1

#Test set predicted daily return
test_return = pd.DataFrame()
for name in test_predictions.columns:
    test_return[name] = daily_return(test_predictions[name])

#Real return
real_return = pd.DataFrame()
for name in test_close.columns:
    real_return[name] = daily_return(test_close[name][3:-5])

def day_to_ann(daily):
    return (1 + daily)**365 - 1


#Covariance
##Create a dataframe coverd 60 trade days before 01/09/2017
close_whole = pd.concat([train_close, test_close])

def covforPort(close_whole, date_index):
    return close_whole[date_index-60:date_index].cov()

###    
def meanvariance_portfolio(real_return, predict_return, Epr, cov_data):
    row_len = real_return.shape[0]
    col_len = real_return.shape[1]
    
    start_date = predict_return.index[0]
    
    start_index = np.where(cov_data.index == start_date)[0][0]
    
    weights = pd.DataFrame(index = real_return.index, columns = real_return.columns)
    port_return = pd.DataFrame(index = real_return.index, columns = ['Return'] )
    
    for t in range(start_index, start_index+row_len-1):
        t_small = t - start_index
        
        SIGMA_t = covforPort(cov_data, t)
        
        
        p_structure = []
        for j in range(SIGMA_t.shape[1]):
            p_structure.append(list(SIGMA_t.ix[:, j]))
            
        P = matrix(p_structure)
        
        q = matrix(list(np.repeat(0.0, col_len)))
        
        g1 = np.r_[np.identity(col_len), -1*np.identity(col_len)]
        
        g_structure = []
        for j in range(g1.shape[1]):
            g_structure.append(list(g1[:, j]))
        
        G = matrix(g_structure)  
            
        h_structure = list(np.repeat(0.1, 2*col_len))
        h = matrix(h_structure)
        
        a_structure = []
        for j in range(col_len):
            a_structure.append(list([predict_return.ix[t_small, j], 1.0]))
            A = matrix(a_structure)
            
        b = matrix([ann_to_daily(Epr), 1.0])
        
        result = solvers.qp(P, q, G, h, A, b)
        
        weights_t = np.array(list(result['x']))
        
        port_return_t = np.dot(weights_t, np.asarray(real_return.ix[t_small, :]))
        
        weights.iloc[t_small, :] = weights_t
        port_return.iloc[t_small, :] = port_return_t
    
    return weights, port_return

w, p = meanvariance_portfolio(real_return, test_return, 0.2, close_whole)
w, p = w.iloc[0:-1, ], p.iloc[0:-1, ]       
        






total = np.repeat(0, len(p))
total[0] = 1000000
for i in range(1, len(p)):
    total[i] = total[i-1] * (1 + p.iloc[i-1, ][0])        
plt.plot(range(len(total)), total)

spy_total = np.repeat(0, len(p))
spy_total[0] = 1000000
for i in range(1, len(p)):
    spy_total[i] = spy_total[i-1] * (1 + spy_return[i-1])        
plt.plot(range(len(spy_total)), spy_total)



plt.plot(range(len(spy_total)), spy_total)
plt.plot(range(len(total)), total)
plt.show()






plt.plot(range(len(p)), p)
np.mean(p)

np.mean(p)  
        
spy = pd.read_csv('/Users/anran/Desktop/FE/FE800/Works/SPY.csv')
spy_close = spy['Close']
spy_return = []
for i in range(1, len(spy_close)):
    spy_return.append((spy_close[i]/spy_close[i-1]) - 1)    
spy_return = spy_return[:-1]
spy_return = np.array(spy_return) 
       
plt.plot(range(len(spy)), spy['Close'])       
    
    
    

plt.plot(range(676)[370:400], test_predictions['BA'][370:400])
plt.plot(range(676)[370:400], test_close1['BA'][370:400])
plt.show()




test_close1 = test_close[3:-5]

#error
error = pd.DataFrame(index = test_return.index, columns = test_return.columns)
for j in range(test_return.shape[1]):
    error_j = []
    for i in range(test_return.shape[0]):
        error_j.append(real_return.iloc[i, j] - test_return.iloc[i, j])
    error.iloc[:, j] = error_j

#MSE
mse = pd.DataFrame(index = ['MSE', 'MAE', 'RMSE'], columns = test_predictions.columns)
for j in range(test_return.shape[1]):
    se = []
    ae = []
    for i in range(test_return.shape[0]):
        se.append(error.iloc[i, j]*error.iloc[i, j])
        ae.append(np.abs(error.iloc[i, j]))
    sse = np.sum(se)
    sae = np.sum(ae)
    mse.iloc[0, j] = sse/675
    mse.iloc[1, j] = sae/675
    mse.iloc[2, j] = np.sqrt(sse/675)
        
j = 2    
mse.iloc[:, j] = np.sum(se_j)/len(se_j)

plt.plot(spy_return)
plt.plot(range(674), p)
plt.show()

