import numpy as np
import gym
from gym import spaces
import pandas as pd
from random import random
import math
import tensorflow as tf
import matplotlib.pyplot as plt
import pywt


from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C

from readData import Read_data

INITIAL_BALANCE = 10000

# Portfolio Environment
class PortfolioEnv(gym.Env):
    def __init__(self, df, train, wavelet, sharpe, lag=5):
        self.df = df
        self.lag = lag
        self.train = train
        self.wavelet = wavelet
        self.sharpe = sharpe

        self.maxprice = self.df.max()

        if wavelet:
            self.wave = np.array(
                [pywt.swt(self.df.iloc[:, j].values, 'haar', level=2, trim_approx=True, norm=True) for j in range(50)])

        self.action_space = spaces.Box(
            low=0, high=1, shape=(50,)
        )

        if wavelet:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.lag, self.df.shape[1], 3), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=1, shape=(self.lag, self.df.shape[1]), dtype=np.float32
            )

    def _updateState(self):
        if self.wavelet:
            self.state = np.ones([5, 50, 3])
            for x in range(5):
                for y in range(50):
                    for z in range(3):
                        self.state[x, y, z] = self.wave[y][z][self.current_day-self.lag+x]

        else:
            self.state = np.ones([5, 50])
            for i in range(5):
                for j in range(50):
                    self.state[i][j] = self.df.iloc[self.current_day - self.lag + i, j] / self.maxprice[j]

        return self.state

    def reset(self):
        if self.train:
            self.current_day = np.random.randint(self.lag, self.df.shape[0] - 2)
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

        self.weight = action / sum(action)

        self.returnS = (self.df.iloc[self.current_day, ].values / self.df.iloc[self.current_day-1, ].values) - 1

        self.returnP = sum(self.returnS * self.weight)

        if self.sharpe:
            self.cov = np.cov(self.df.iloc[(self.current_day - self.lag):(self.current_day), ].T)
            self.weightT = self.weight.reshape([50, 1])
            self.reward = self.returnP/(np.dot(np.dot(self.weight, self.cov), self.weightT))
        else:
            self.reward = self.returnP

        self.balance *= (1 + self.returnP)

        self.current_day += 1

        if self.current_day == self.df.shape[0] - 1:
            self.done = True

        self._updateState()

        self.history.append([self.current_day, self.weight, self.returnP, self.balance])

        return self.state, self.reward, self.done, {'history': self.history}

# Model training
class Modelresults:
    def __init__(self, train_dataset, test_dataset, bench_dataset):
        self.train = train_dataset
        self.test = test_dataset
        self.bench = bench_dataset

        self.spybalance = (10000 / self.bench[0]) * self.bench
        self.spyreturn = np.array([(self.bench[i + 1] / self.bench[i]) - 1 for i in range(599)])

    def mlpModel(self, policy_kwargs, timesteps, wavelet, sharpe):
        if wavelet:
            if sharpe:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, True, True)])
                rewards = 'Sharpe'
            else:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, True, False)])
                rewards = 'Return'

            obv = 'Wavelet'

        else:
            if sharpe:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, False, True)])
                rewards = 'Sharpe'
            else:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, False, False)])
                rewards = 'Return'

            obv = 'Price'

        model = A2C(MlpPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

        model.learn(total_timesteps=timesteps)

        model.save('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + 'Mlp' + obv + rewards + '.h5')

        del model

    def lstmModel(self, policy_kwargs, timesteps, wavelet, sharpe):
        if wavelet:
            if sharpe:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, True, True)])
                rewards = 'Sharpe'
            else:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, True, False)])
                rewards = 'Return'

            obv = 'Wavelet'

        else:
            if sharpe:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, False, True)])
                rewards = 'Sharpe'
            else:
                env = DummyVecEnv([lambda: PortfolioEnv(self.train, True, False, False)])
                rewards = 'Return'

            obv = 'Price'

        model = A2C(MlpLstmPolicy, env, policy_kwargs=policy_kwargs, verbose=1)

        model.learn(total_timesteps=timesteps)

        model.save('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + 'Lstm' + obv + rewards + '.h5')

        del model

    def testSet(self, neural, obv, rewards):
        model1 = A2C.load('/Users/anran/Desktop/FE/FE800/Works/DRL model/' + neural + obv + rewards + '.h5')

        if obv == 'Wavelet':
            if rewards == 'Sharpe':
                testenv = DummyVecEnv([lambda: PortfolioEnv(self.test, False, True, True)])
            else:
                testenv = DummyVecEnv([lambda: PortfolioEnv(self.test, False, True, False)])

        else:
            if rewards == 'Sharpe':
                testenv = DummyVecEnv([lambda: PortfolioEnv(self.test, False, False, True)])
            else:
                testenv = DummyVecEnv([lambda: PortfolioEnv(self.test, False, False, False)])

        obs = testenv.reset()
        for i in range(600):
            action, _states = model1.predict(obs)
            obs, rewards, done, history = testenv.step(action)

        return history

    def getHistory(self, history):
        returnP = np.array([history[0]['history'][i][2] for i in range(len(history[0]['history']))])

        balance = np.array([history[0]['history'][i][3] for i in range(len(history[0]['history']))])

        return returnP, balance

# Train dataset
Rd = Read_data()
train_all, train_name = Rd.read_all_data('/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Training')
train_index = Rd.data_index(train_name)
train_dataset = Rd.dataset(train_all, train_index).iloc[:-1, ]
del train_all
# Test dataset
test_all, test_name = Rd.read_all_data('/Users/anran/Desktop/FE/FE800/Works/Week1 Dataset/Test')
test_index = Rd.data_index(test_name)
test_dataset = Rd.dataset(test_all, test_index)
test_dataset = test_dataset[train_index]
del test_all
# Spy dataset
spy = pd.read_csv('/Users/anran/Desktop/FE/FE800/Works/SPY.csv')['Close'][:-1]


# Init class
mo = Modelresults(train_dataset, test_dataset, spy)

# Train Sharpe model, Mlp or Lstm, Wavelet or Price.
mo.mlpModel(dict(layers=[512, 256, 128]), 20000, True, True)
mo.mlpModel(dict(layers=[512, 512, 128]), 20000, False, True)

mo.lstmModel(dict(layers=[512, 256, 128]), 20000, True, True)
mo.lstmModel(dict(layers=[512, 256, 128]), 20000, False, True)

# Train Return model, Mlp or Lstm, Wavelet or Price.
mo.mlpModel(dict(layers=[512, 256, 128]), 20000, True, False)
mo.mlpModel(dict(layers=[512, 512, 128]), 20000, False, False)

mo.lstmModel(dict(layers=[512, 256, 128]), 20000, True, False)
mo.lstmModel(dict(layers=[512, 256, 128]), 20000, False, False)


# Get Sharpe model test histories
mlpWaveletSharpe_history = mo.testSet('Mlp', 'Wavelet', 'Sharpe')
mlpPriceSharpe_history = mo.testSet('Mlp', 'Price', 'Sharpe')

lstmWaveletSharpe_history = mo.testSet('Lstm', 'Wavelet', 'Sharpe')
lstmPriceSharpe_history = mo.testSet('Lstm', 'Price', 'Sharpe')

# Get Return model test histories
mlpWaveletReturn_history = mo.testSet('Mlp', 'Wavelet', 'Return')
mlpPriceReturn_history = mo.testSet('Mlp', 'Price', 'Return')

lstmWaveletReturn_history = mo.testSet('Lstm', 'Wavelet', 'Return')
lstmPriceReturn_history = mo.testSet('Lstm', 'Price', 'Return')

# Get Sharpe returns and balances
mlpWaveletSharpe_return, mlpWaveletSharpe_balance = mo.getHistory(mlpWaveletSharpe_history)
mlpPriceSharpe_return, mlpPriceSharpe_balance = mo.getHistory(mlpPriceSharpe_history)

lstmWaveletSharpe_return, lstmWaveletSharpe_balance = mo.getHistory(lstmWaveletSharpe_history)
lstmPriceSharpe_return, lstmPriceSharpe_balance = mo.getHistory(lstmPriceSharpe_history)

# Get Return returns and balances
mlpWaveletReturn_return, mlpWaveletReturn_balance = mo.getHistory(mlpWaveletReturn_history)
mlpPriceReturn_return, mlpPriceReturn_balance = mo.getHistory(mlpPriceReturn_history)

lstmWaveletReturn_return, lstmWaveletReturn_balance = mo.getHistory(lstmWaveletReturn_history)
lstmPriceReturn_return, lstmPriceReturn_balance = mo.getHistory(lstmPriceReturn_history)

# Plot Price balances
plt.plot(mlpPriceReturn_balance, linewidth=0.8, label='MLP Return')
plt.plot(mlpPriceSharpe_balance, linewidth=0.8, label='MLP Sharpe')
plt.plot(lstmPriceReturn_balance, linewidth=0.8, label='LSTM Return')
plt.plot(lstmPriceSharpe_balance, linewidth=0.8, label='LSTM Sharpe')
plt.plot(mo.spybalance, linewidth=1.3, label='SPY')
plt.legend(loc='upper left')

# Plot Wavelets returns
plt.plot(mlpWaveletReturn_balance, linewidth=0.8, label='MLP Return')
plt.plot(mlpWaveletSharpe_balance, linewidth=0.8, label='MLP Sharpe')
plt.plot(lstmWaveletReturn_balance, linewidth=0.8, label='LSTM Return')
plt.plot(lstmWaveletSharpe_balance, linewidth=0.8, label='LSTM Sharpe')
plt.plot(mo.spybalance, linewidth=1.3, label='SPY')
plt.legend(loc='upper left')

# Plot top3
plt.plot(mlpPriceSharpe_balance, linewidth=0.8, label='MLP Price Sharpe')
plt.plot(mlpWaveletSharpe_balance, linewidth=0.8, label='MLP Wavelets Sharpe')
plt.plot(lstmWaveletReturn_balance, linewidth=0.8, label='LSTM Wavelets Return')
plt.plot(mo.spybalance, linewidth=1.4, color='purple', label='SPY')
plt.legend(loc='upper left')


#######
env = DummyVecEnv([lambda: PortfolioEnv(train_dataset, True, False, True)])
model = A2C(MlpPolicy, env, policy_kwargs=dict(layers=[512, 256]), verbose=1)
model.learn(total_timesteps=20000)

testenv = PortfolioEnv(test_dataset, False, False, True)
obs = testenv.reset()
for i in range(500):
    action, _states = model.predict(obs)
    obs, rewards, done, history = testenv.step(action)

balance = np.array([history['history'][i][3] for i in range(len(history['history']))])

#####
a = pd.DataFrame({
    'MlpPriceReturn': np.array([
        np.mean(mlpPriceReturn_return), np.max(mlpPriceReturn_return), np.min(mlpPriceReturn_return), np.std(mlpPriceReturn_return)
    ]),
    'MlpPriceSharpe': np.array([
        np.mean(mlpPriceSharpe_return), np.max(mlpPriceSharpe_return), np.min(mlpPriceSharpe_return), np.std(mlpPriceSharpe_return)
    ]),
    'MlpWaveletReturn': np.array([
        np.mean(mlpWaveletReturn_return), np.max(mlpWaveletReturn_return), np.min(mlpWaveletReturn_return),
        np.std(mlpWaveletReturn_return)
    ]),
    'MlpWaveletSharpe': np.array([
        np.mean(mlpWaveletSharpe_return), np.max(mlpWaveletSharpe_return), np.min(mlpWaveletSharpe_return),
        np.std(mlpWaveletSharpe_return)
    ]),
    'LstmPriceReturn': np.array([
        np.mean(lstmPriceReturn_return), np.max(lstmPriceReturn_return), np.min(lstmPriceReturn_return),
        np.std(lstmPriceReturn_return)
    ]),
    'LstmPriceSharpe': np.array([
        np.mean(lstmPriceSharpe_return), np.max(lstmPriceSharpe_return), np.min(lstmPriceSharpe_return),
        np.std(lstmPriceSharpe_return)
    ]),
    'LstmWaveletReturn': np.array([
        np.mean(lstmWaveletReturn_return), np.max(lstmWaveletReturn_return), np.min(lstmWaveletReturn_return),
        np.std(lstmWaveletReturn_return)
    ]),
    'LstmWaveletSharpe': np.array([
        np.mean(lstmWaveletSharpe_return), np.max(lstmWaveletSharpe_return), np.min(lstmWaveletSharpe_return),
        np.std(lstmWaveletSharpe_return)
    ]),
    'SPY': np.array([
        np.mean(mo.spyreturn), np.max(mo.spyreturn), np.min(mo.spyreturn),
        np.std(mo.spyreturn)
    ])
})
a.index = ['Mean', 'Max', 'Min', 'Std']

spyreturn = mo.spyreturn
percen = mlpWaveletReturn_return[1:]/spyreturn
spy1 = pd.read_csv('/Users/anran/Desktop/FE/FE800/Works/SPY.csv')
diff = mlpWaveletReturn_return[1:] - spyreturn
plt.plot(diff)


spyannul = (1+spyreturn)**252



###
spytotal = pd.read_csv('/Users/anran/Desktop/FE/FE800/SPY.csv')['Close'][:-1]
spyreturn = np.array([(spytotal[i+1]/spytotal[i])-1 for i in range(len(spytotal)-1)])

def tendays(dailyreturn):
    tendaysreturn = []
    tendaysr = 1
    for i in range(len(dailyreturn)):
        tendaysr *= (dailyreturn[i]+1)
        if (i+1) % 10 == 0:
            tendaysreturn.append(tendaysr)
            tendaysr = 1
    return np.array(tendaysreturn)

def semimonth(dailyreturn):
    semireturn = []
    semir = 1
    for i in range(len(dailyreturn)):
        semir *= (dailyreturn[i]+1)
        if (i+1) % 15 == 0:
            semireturn.append(semir)
            semir = 1
    return np.array(semireturn)


spytr = semimonth(spyreturn)
mprtr = semimonth(mlpPriceReturn_return)
mpstr = semimonth(mlpPriceSharpe_return)
mwrtr = semimonth(mlpWaveletReturn_return)
mwstr = semimonth(mlpWaveletSharpe_return)
lprtr = semimonth(lstmPriceReturn_return)
lpstr = semimonth(lstmPriceSharpe_return)
lwrtr = semimonth(lstmWaveletReturn_return)
lwstr = semimonth(lstmWaveletSharpe_return)


plt.plot(np.array([1 for i in range(40)]), linewidth=1.5, label='SPY')
#plt.plot(mprtr/spytr, linewidth=0.5)
plt.plot(mpstr/spytr, linewidth=0.7, label='MLP Price Sharpe')
#plt.plot(mwrtr/spytr, linewidth=0.5)
plt.plot(mwstr/spytr, linewidth=0.7, label='MLP Wavelet Sharpe')
#plt.plot(lprtr/spytr, linewidth=0.5)
#plt.plot(lpstr/spytr, linewidth=0.5)
plt.plot(lwrtr/spytr, linewidth=0.7, label='LSTM Wavelet Return')
#plt.plot(lwstr/spytr, linewidth=0.5)
plt.legend(loc='upper left')



from scipy import stats

stats.ttest_ind(mlpWaveletSharpe_return[400:], spyreturn[400:])

a = (mlpPriceReturn_return+1)**252



plt.plot(mlpPriceReturn_return)
plt.plot(spyreturn)

np.corrcoef(mlpPriceReturn_return, spyreturn)



plt.plot(mo.spybalance*100, color='b')
plt.plot(mlpPriceReturn_balance*100)
