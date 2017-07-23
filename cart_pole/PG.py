import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from gym import wrappers


alpha = .001
epsilon = 1
decay_rate = .98
h1 = 80
h2 = 80
gamma = .99
it = 800
env = gym.make('CartPole-v0')
nb_actions = env.action_space.n
input_size = env.observation_space.shape[0]
env = wrappers.Monitor(env, 'temporary', force=True)

model = Sequential()
model.add(Dense(h1, input_dim=input_size))
model.add(Activation('relu'))
if h2 > 0:
    model.add(Dense(h2))
    model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
ada = Adam(lr=alpha, decay=0.00001)
model.compile(loss="categorical_crossentropy", optimizer=ada)

def reshapeMLPFeature(st):
    return np.reshape(st, (1, input_size))

"""
discount monte carlo over whole sequence
Note on reversal: Think of it this way
first total_reward is 1, so you start with terminal state reward, then discount it's value with rewards later on
so let's say sequence is -1, -1, -1, 1, reversed would be ->
1  + ((eps * 1 = .99) + -1) ... etc.
"""
def discount_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total = 0
    for i in reversed(range(0,len(rewards))):
        total += total * epsilon + rewards[i]
        discounted_rewards[i] = total
    return discounted_rewards

def train(replay):
    sequence_length = len(replay)
    rewards = [x[2] for x in replay]
    discounted_rewards = discount_rewards(rewards)
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)
    states = np.zeros((sequence_length, input_size))
    advantages = np.zeros((sequence_length, nb_actions))
    for i in range(sequence_length):
        states[i] = replay[i][0]
        advantages[i][replay[i][1]] = discounted_rewards[i]
    model.fit(states, advantages, epochs=1, verbose=0)


rolling_average = []
x_plots_reward = []
y_plots_reward = []
for e in range(it):
    state = env.reset()
    fin_r = 0
    temporary_sars = []
    for t in range(10000):
        # Stochastic exploration
        value = model.predict(reshapeMLPFeature(state)).flatten()
        action = np.random.choice(nb_actions, 1, p = value)[0]
        observation, reward, done, _ = env.step(action)
        temporary_sars.append((state, action, reward, None if done else observation))
        fin_r += reward
        state = observation
        if done:
            train(temporary_sars)
            break

    epsilon *= decay_rate
    rolling_average.append(fin_r)
    print "reward: " + str(fin_r) + " epsilon: " + str(epsilon) + " iter: " + str(e) + " avg " + str(np.average(rolling_average[20:])) + " " + "cart_pole_pg"
    x_plots_reward.append(e)
    y_plots_reward.append(fin_r)

env.close()

plt.plot(x_plots_reward, y_plots_reward, label="iters")
plt.ylabel('reward')
plt.xlabel('iteration')
plt.legend(loc='lower right')
plt.title("CartPole rewards")
plt.savefig('cartpole-pg' + '.png')
