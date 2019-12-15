import numpy as np
import gym
from gym import spaces
import pandas as pd
from random import random
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, LSTM
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

import json
import datetime as dt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN

class Traderenv(gym.Env):
    def __init__(self, path, train, lag=3):
        self.df = pd.read_csv(path)
        self.lag = lag
        self.train = train

        self.closeprice = round(self.df['Close'], 2)

        self.actions = [0, 1, 2] #Buy, Sell, Hold

        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lag, 1), dtype=np.float32)

    def _updateState(self):
        self.state = np.array([[self.closeprice[self.current_day-2]],
                               [self.closeprice[self.current_day-1]],
                               [self.closeprice[self.current_day]]])
        return self.state

    def reset(self):
        if self.train:
            self.current_day = np.random.randint(low=self.lag-1, high=len(self.closeprice)-1)

        self.current_day = 2

        self.history = []
        self.profit = 0

        self.shares = 0

        self.done = False

        self._updateState()

        return self.state

    def trade_action(self, action):
        if self.shares == 0:
            if action == 0:
                self.shares = 1
                self.buyprice = self.nowprice
                self.action = 0
            elif action == 1:
                self.shares = -1
                self.sellprice = self.nowprice
                self.action = 1
            else:
                self.shares = 0
                self.action = 2

        elif self.shares == 1:
            if action == 1:
                self.shares = 0
                self.action = 1
                self.profit += self.nowprice - self.buyprice
                self.reward = (self.nowprice - self.buyprice)*10
                self.buyprice = 0
            else:
                self.shares = 1
                self.action = 2

        elif self.shares == -1:
            if action == 0:
                self.shares = 0
                self.action = 0
                self.profit += self.sellprice - self.nowprice
                self.reward = (self.sellprice - self.nowprice)*10
                self.sellprice = 0
            else:
                self.shares = -1
                self.action = 2

    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}

        self.nowprice = self.closeprice[self.current_day]

        self.reward = 0

        self.trade_action(action)

        self.history.append((self.current_day, self.nowprice, self.action, self.shares, self.profit))

        self.current_day += 1

        if self.current_day > len(self.closeprice)-2:
            self.done = True

        self._updateState()

        return self.state, self.reward, self.done, {"history": self.history}

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

path = "/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Training/AAPL_2006-2016.csv"

#Train
env = DummyVecEnv([lambda: Traderenv(path, True)])
model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

#Test
testpath = '/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Test/AAPL_2017-2019.csv'
testenv = DummyVecEnv([lambda: Traderenv(testpath, False)])

obstest = testenv.reset()
for i in range(500):
    action, _states = model.predict(obstest)
    obstest, rewards, done, history = testenv.step(action)

import matplotlib.pyplot as plt
profit = np.array([history[0]['history'][i][4] for i in range(len(history[0]['history']))])
plt.plot(profit)

