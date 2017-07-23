import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from gym import wrappers

alpha = .001
alpha_critic=.005
epsilon = 1
decay_rate = .98
h1 = 85
h2 = 85
gamma = .99
it = 800
env = gym.make('CartPole-v0')
nb_actions = env.action_space.n
input_size = env.observation_space.shape[0]
random_transitions = 32
env = wrappers.Monitor(env, 'temporary', force=True)

actor = Sequential()
actor.add(Dense(h1, input_dim=input_size))
actor.add(Activation('elu'))
if h2 > 0:
    actor.add(Dense(h2))
    actor.add(Activation('elu'))
actor.add(Dense(nb_actions))
actor.add(Activation('softmax'))
ada = Adam(lr=alpha, decay=0.0001)
actor.compile(loss="categorical_crossentropy", optimizer=ada)


critic = Sequential()
critic.add(Dense(h1, input_dim=input_size))
critic.add(Activation('elu'))
if h2 > 0:
    critic.add(Dense(h2))
    critic.add(Activation('elu'))
critic.add(Dense(1))
critic.add(Activation('linear'))
ada = Adam(lr=alpha_critic, decay=0.0001)
critic.compile(loss="mse", optimizer=ada)


def reshapeMLPFeature(st):
    return np.reshape(st, (1, input_size))

def train(state, action, reward, state_prime, done):
    target = np.zeros((1, 1)) # one value
    advantage = np.zeros((1, nb_actions))
    value = critic.predict(state)[0]
    if done:
        advantage[0][action] = reward - value
        target[0][0] = reward
    else:
        value_prime = critic.predict(state_prime)[0]
        advantage[0][action] = reward + gamma * (value_prime) - value
        target[0][0] = reward + gamma * value_prime
    actor.fit(state, advantage, verbose=0, epochs=1)
    critic.fit(state, target, verbose=0, epochs=1)

rolling_average = []
x_plots_reward = []
y_plots_reward = []
for e in range(it):
    state = env.reset()
    fin_r = 0
    for t in range(10000):
        value = actor.predict(reshapeMLPFeature(state)).flatten()
        action = np.random.choice(nb_actions, 1, p = value)[0]
        observation, reward, done, _ = env.step(action)
        fin_r += reward
        train(reshapeMLPFeature(state), action, reward, reshapeMLPFeature(observation), done)
        state = observation
        if done:
            break
    epsilon *= decay_rate
    rolling_average.append(fin_r)
    print "reward: " + str(fin_r) + " epsilon: " + str(epsilon) + " iter: " + str(e) + " avg " + str(np.average(rolling_average[20:])) + " " + "cart_pole_ac2"
    x_plots_reward.append(e)
    y_plots_reward.append(fin_r)

env.close()

plt.plot(x_plots_reward, y_plots_reward, label="iters")
plt.ylabel('reward')
plt.xlabel('iteration')
plt.legend(loc='lower right')
plt.title("CartPole rewards")
plt.savefig('cartpole-ac2' + '.png')
