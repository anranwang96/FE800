import numpy as np
import gym
from gym import spaces
import pandas as pd
from random import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

INITIAL_BALANCE = 10000

# Train
Rd = Read_data()
train_all, train_name = Rd.read_all_data('/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Training')
train_index = Rd.data_index(train_name)
train_dataset = Rd.dataset(train_all, train_index)
# Test
test_all, test_name = Rd.read_all_data('/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Test')
test_index = Rd.data_index(test_name)
test_dataset = Rd.dataset(test_all, test_index)
test_dataset = test_dataset[train_index]
#
spy = pd.read_csv('/Users/anran/Desktop/FE/FE800/Works/SPY.csv')['Close'][:-1]

# Env
class PortfolioEnv(gym.Env):
    def __init__(self, df, train, lag=5):
        self.df = df
        self.lag = lag
        self.train = train

        self.maxprice = self.df.max()

        self.action_space = spaces.Box(
            low=0, high=1, shape=(50, )
         )

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.lag, self.df.shape[1]), dtype=np.float32
        )

    def _updateState(self):
        self.state = np.ones([5, 50])
        for i in range(5):
            for j in range(50):
                self.state[i][j] = self.df.iloc[self.current_day-self.lag+i, j]/self.maxprice[j]
        return self.state

    def reset(self):
        if self.train:
            self.current_day = np.random.randint(self.lag, self.df.shape[0]-2)
        else:
            self.current_day = self.lag

        self.history = []
        self.done = False
        self.reward = 0

        self.balance = INITIAL_BALANCE

        return self._updateState()

    def step(self, action):
        if self.done:
            return self.state, self.reward, self.done, {}
        
        self.weight = action/sum(action)

        self.returnS = (self.df.iloc[self.current_day, ].values / self.df.iloc[self.current_day-1, ].values) - 1

        self.returnP = sum(self.returnS * self.weight)
        
        self.reward = self.returnP

        self.balance *= (1 + self.returnP)
        
        self.current_day += 1
                        
        if self.current_day == self.df.shape[0]-1:
            self.done = True
        
        self._updateState()

        self.history.append([self.current_day, self.weight, self.returnP, self.balance])

        return self.state, self.reward, self.done, {'history': self.history}
# Train
class Modelresults:
    def __init__(self, train_dataset, test_dataset, bench_dataset):
        self.train = train_dataset
        self.test = test_dataset
        self.bench = bench_dataset

        self.spybalance = (10000 / self.bench[0]) * self.bench
        self.spyreturn = np.array([(self.bench[i + 1] / self.bench[i]) - 1 for i in range(499)])

    def mlpModel(self, policy_kwargs, timesteps, name):
        env = DummyVecEnv([lambda: PortfolioEnv(self.train, True)])

        model = A2C(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

        model.learn(total_timesteps=timesteps)

        model.save('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + name + '.h5')

        del model

    def lstmModel(self, policy_kwargs, timesteps, name):
        env = DummyVecEnv([lambda: PortfolioEnv(self.train, True)])

        model = A2C(MlpLstmPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

        model.learn(total_timesteps=timesteps)

        model.save('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + name + '.h5')

        del model

    def cnnModel(self, policy_kwargs, timesteps, name):
        env = DummyVecEnv([lambda: PortfolioEnv(self.train, True)])

        model = A2C(CnnLstmPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

        model.learn(total_timesteps=timesteps)

        model.save('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + name + '.h5')

        del model

    def testSet(self, name):
        model = A2C.load('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + name + '.h5')

        testenv = DummyVecEnv([lambda: PortfolioEnv(self.test, False)])

        obs = testenv.reset()

        for i in range(500):
            action, _states = model.predict(obs)
            obs, rewards, done, history = testenv.step(action)

        return history

    def getHistory(self, history):
        returnP = np.array([history[0]['history'][i][2] for i in range(len(history[0]['history']))])

        balance = np.array([history[0]['history'][i][3] for i in range(len(history[0]['history']))])

        return returnP, balance


mo = Modelresults(train_dataset, test_dataset, spy)

#####
mo.mlpModel(dict(layers=[128, 128, 128]), 20000, 'mlpModel')
mo.lstmModel(dict(layers=[128, 128, 128]), 20000, 'lstmModel')
mo.cnnModel(dict(layers=[128, 128, 128]), 20000, 'cnnModel')

#####
mlp_history = mo.testSet('mlpModel')
lstm_history = mo.testSet('lstmModel')

#####
mlp_return, mlp_balance = mo.getHistory(mlp_history)
lstm_return, lstm_balance = mo.getHistory(lstm_history)

#####
plt.plot(mlp_balance, linewidth=0.8, label='MLP')
plt.plot(lstm_balance, linewidth=0.8, label='LSTM')
plt.plot(mo.spybalance, linewidth=0.8, label='SPY')
plt.legend(loc='upper left')

#####
plt.plot(mlp_return, linewidth=0.8, label='MLP')
plt.plot(lstm_return, linewidth=0.8, label='LSTM')
plt.plot(mo.spyreturn, linewidth=0.8, label='SPY')
plt.legend(loc='upper left')
