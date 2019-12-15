import numpy as np
import gym
from gym import spaces
import pandas as pd
from random import random
import math
import datetime as dt
import tensorflow as tf
import pywt

from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C

MAX_SHARES = 10000
INITIAL_BALANCE = 10000

class Traderenv(gym.Env):
    def __init__(self, df, train, lag=5):
        self.df = df
        self.maxprice = self.df['High'].max()
        self.maxstep = self.df.shape[0]
        self.lag = lag
        self.train = train

        self.features = ['Open', 'High', 'Low', 'Close', 'Adj Close']

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.lag, len(self.features)), dtype=np.float32)

    def _updateState(self):
        self.state = np.array([
            self.df['Open'].loc[self.current_day-self.lag+1:self.current_day]/self.maxprice,
            self.df['High'].loc[self.current_day-self.lag+1:self.current_day]/self.maxprice,
            self.df['Low'].loc[self.current_day-self.lag+1:self.current_day]/self.maxprice,
            self.df['Close'].loc[self.current_day-self.lag+1:self.current_day]/self.maxprice,
            self.df['Adj Close'].loc[self.current_day-self.lag+1:self.current_day]/self.maxprice
        ])
        return self.state

    def reset(self):
        if self.train:
            self.current_day = np.random.randint(self.lag, high=self.df.shape[0]-1)

        else:
            self.current_day = self.lag

        self.history = []
        self.done = False

        self.balance = INITIAL_BALANCE
        self.shares = 0
        self.averageCost = 0
        self.totalSharesSold = 0
        self.totalSalesValue = 0

        return self._updateState()

    def trade_action(self, action):
        if self.train:
            self.nowprice = np.random.uniform(
                self.df['Open'].loc[self.current_day], self.df['Close'].loc[self.current_day])
        else:
            self.nowprice = round(self.df['Close'].loc[self.current_day], 3)

        actionType = np.floor(action[0])
        amount = action[1]

        if actionType == 0:
            # Buy
            maxBuyAmount = np.floor(self.balance / self.nowprice)
            buyAmount = np.floor(maxBuyAmount * amount)
            preAvgCost = self.averageCost * self.shares
            addtionalCost = self.nowprice * buyAmount

            self.balance -= self.nowprice * buyAmount
            self.averageCost = (preAvgCost + addtionalCost) / (self.shares + buyAmount)
            self.shares += buyAmount

            self.action = 'Buy'
            self.amount = buyAmount

        elif actionType == 1:
            sellAmount = np.floor(self.shares * amount)
            self.balance += self.nowprice * sellAmount
            self.shares -= sellAmount
            self.totalSharesSold += sellAmount
            self.totalSalesValue += sellAmount * self.nowprice

            self.action = 'Sell'
            self.amount = sellAmount

        else:
            self.action = 'Hold'
            self.amount = 0

        self.balanceValue = self.balance + self.shares * self.nowprice

    def step(self, action):
        self.trade_action(action)

        self.current_day += 1

        if self.current_day > self.df.shape[0] - 2:
            self.done = True

        delayModifier = self.current_day / self.maxstep

        self.reward = self.balanceValue * delayModifier

        self._updateState()

        self.history.append([self.current_day, self.nowprice,
                             self.action, self.amount,
                             self.shares, self.balanceValue])

        return self.state, self.reward, self.done, {'history': self.history}

    def render(self, mode='human'):
        nowprice = self.closeprice[self.current_day]
        profit = self.profit
        balance = 100 + profit

        print(f'Day: {self.current_day}')
        print(f'Price: {nowprice}')
        print(f'Action: {self.action}')
        print(f'Shares: {self.shares}')
        print(f'Profit: {profit}')
        print(f'Balance: {balance}')

path = '/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Training/AAPL_2006-2016.csv'
df = pd.read_csv(path)
env = Traderenv(df, True)

#Train
policy_kwargs = dict(act_fun=tf.nn.softmax, net_arch=[32, 32])
env = DummyVecEnv([lambda: Traderenv(df, True)])
model = PPO2(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(total_timesteps=30000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, history = env.step(action)

#Test
testpath = '/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Test/AAPL_2017-2019.csv'
testdf = pd.read_csv(testpath)
testenv = DummyVecEnv([lambda: Traderenv(testdf, False)])

obs = testenv.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, history = testenv.step(action)

# Plot
import matplotlib.pyplot as plt
balance = np.array([history[0]['history'][i][5] for i in range(len(history[0]['history']))])
price = np.array([history[0]['history'][i][1] for i in range(len(history[0]['history']))])
plt.plot(balance)

# Wavelet trader
df = pd.read_csv(path)[1:]

class WaveletTraderenv(gym.Env):
    def __init__(self, df, train, lag=5):
        self.df = df
        self.maxprice = self.df['High'].max()
        self.maxstep = self.df.shape[0]
        self.lag = lag
        self.train = train

        self.features = ['Open', 'High', 'Low', 'Close', 'Adj Close']

        self.wavelet = np.array([
            pywt.swt(self.df['Open'].values, 'haar', level=2, trim_approx=True, norm=True),
            pywt.swt(self.df['High'].values, 'haar', level=2, trim_approx=True, norm=True),
            pywt.swt(self.df['Low'].values, 'haar', level=2, trim_approx=True, norm=True),
            pywt.swt(self.df['Close'].values, 'haar', level=2, trim_approx=True, norm=True),
            pywt.swt(self.df['Adj Close'].values, 'haar', level=2, trim_approx=True, norm=True)
        ])

        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.lag, len(self.features), 3), dtype=np.float32)

    def _updateState(self):
        self.state = np.ones([self.lag, len(self.features), 3])
        for i in range(self.lag):
            for j in range(len(self.wavelet)):
                self.state[i, j, ] = (self.wavelet[j][0][self.current_day - self.lag + 1 + i]/self.maxprice,
                                      self.wavelet[j][1][self.current_day - self.lag + 1 + i],
                                      self.wavelet[j][2][self.current_day - self.lag + 1 + i])

         return self.state

    def reset(self):
        if self.train:
            self.current_day = np.random.randint(self.lag, high=len(self.wavelet[0][0])-6)

        else:
            self.current_day = self.lag

        self.history = []
        self.done = False

        self.reward = 0

        self.balance = INITIAL_BALANCE
        self.shares = 0
        self.averageCost = 0
        self.totalSharesSold = 0
        self.totalSalesValue = 0

        return self._updateState()

    def trade_action(self, action):
        if self.train:
            self.nowprice = np.random.uniform(
                self.df['Open'].loc[self.current_day], self.df['Close'].loc[self.current_day])
        else:
            self.nowprice = round(self.df['Close'].loc[self.current_day], 3)

        actionType = np.floor(action[0])
        amount = action[1]

        if actionType == 0:
            # Buy
            maxBuyAmount = np.floor(self.balance / self.nowprice)
            buyAmount = np.floor(maxBuyAmount * amount)
            preAvgCost = self.averageCost * self.shares
            addtionalCost = self.nowprice * buyAmount

            self.balance -= self.nowprice * buyAmount
            self.averageCost = (preAvgCost + addtionalCost) / (self.shares + buyAmount)
            self.shares += buyAmount

            self.action = 'Buy'
            self.amount = buyAmount

        elif actionType == 1:
            sellAmount = np.floor(self.shares * amount)
            self.balance += self.nowprice * sellAmount
            self.shares -= sellAmount
            self.totalSharesSold += sellAmount
            self.totalSalesValue += sellAmount * self.nowprice

            self.action = 'Sell'
            self.amount = sellAmount

        else:
            self.action = 'Hold'
            self.amount = 0

        self.balanceValue = self.balance + self.shares * self.nowprice

    def step(self, action):
        self.trade_action(action)

        self.current_day += 1

        if self.current_day > len(self.wavelet[0][0])-6:
            self.done = True

        delayModifier = self.current_day / self.maxstep

        self.reward = self.balanceValue * delayModifier

        self._updateState()

        self.history.append([self.current_day, self.nowprice,
                             self.action, self.amount,
                             self.shares, self.balanceValue])

        return self.state, self.reward, self.done, {'history': self.history}

env = WaveletTraderenv(df, True)

env = DummyVecEnv([lambda: WaveletTraderenv(df, True)])
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=30000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, history = env.step(action)

#Test
testpath = '/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Test/AAPL_2017-2019.csv'
testdf = pd.read_csv(testpath)
testenv = DummyVecEnv([lambda: WaveletTraderenv(testdf, False)])

obs = testenv.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, history = testenv.step(action)

import matplotlib.pyplot as plt
balance = np.array([history[0]['history'][i][5] for i in range(len(history[0]['history']))])
price = np.array([history[0]['history'][i][1] for i in range(len(history[0]['history']))])
plt.plot(balance)
