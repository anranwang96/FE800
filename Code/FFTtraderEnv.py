import numpy as np
import gym
from gym import spaces
import pandas as pd
from random import random
import math
import datetime as dt
import tensorflow as tf
import pywt
import scipy as sp

from stable_baselines.common.policies import MlpPolicy, LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN, A2C

MAX_SHARES = 10000
INITIAL_BALANCE = 10000


path =
df = pd.read_csv(path)[1:]
#env = WaveletTraderenv(df, True)

env = DummyVecEnv([lambda: FFTTraderenv(df, True)])
model = A2C(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=30000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, history = env.step(action)

#Test
testpath = '/Users/eternalchallenge/Documents/FE-800-A/Week1 Dataset/Test/AAPL_2017-2019.csv'
testdf = pd.read_csv(testpath)
testenv = DummyVecEnv([lambda: FFTTraderenv(testdf, False)])

obs = testenv.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, history = testenv.step(action)

import matplotlib.pyplot as plt
balance = np.array([history[0]['history'][i][5] for i in range(len(history[0]['history']))])
price = np.array([history[0]['history'][i][1] for i in range(len(history[0]['history']))])
plt.plot(balance)