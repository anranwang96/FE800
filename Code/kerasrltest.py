from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, CuDNNLSTM, LSTM
from keras.optimizers import Adam

# keras-rl agent
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

def create_model(shape, nb_actions):
    model = Sequential()
    model.add(LSTM(64, input_shape=shape))
    model.add(Dense(32))
    model.add(Dense(nb_actions, activation='softmax'))


env = PortfolioEnv(train_dataset)
nb_actions = env.action_space.shape[0]

memory = SequentialMemory(limit=50000, window_length=5)
# policy = BoltzmannQPolicy()
policy = EpsGreedyQPolicy()

model = create_model(shape=env.observation_space.shape, nb_actions=nb_actions)
