import gym
import random
import numpy as np
from gym import wrappers
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

alpha = .01
epsilon = 1
decay_rate = .98
h1 = 80
h2 = 80
gamma = .99
batch_s = 64
train_prob = 1
it = 800
nb_e = 1
replay_memory_length = 650000
env = gym.make('CartPole-v0')
nb_actions = env.action_space.n
input = env.observation_space.shape[0]
output = env.action_space.n
train_count = 1
env = wrappers.Monitor(env, 'temporary', force=True)


def create_simple_model():
    model = Sequential()
    model.add(Dense(h1, input_dim=input))
    model.add(Activation('relu'))
    if h2 > 0:
        model.add(Dense(h2))
        model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))

    ada = Adam(lr=alpha, decay=.00001)
    model.compile(loss="mse", optimizer=ada)
    return model

q1_model = create_simple_model()
q2_model = create_simple_model()

def reshapeMLPFeature(state):
    return np.reshape(state, (1, input))

def backwards(replay):
    for _ in range(train_count):
        random_transitions = random.sample(replay, min(len(replay), batch_s))
        q1_batch_x = []
        q1_batch_y = []
        q2_batch_x = []
        q2_batch_y = []
        for d in random_transitions:
            s = d[0]
            a = d[1]
            r = d[2]
            s_prime = d[3]

            trigger = random.random() > .5
            outs = q1_model.predict(reshapeMLPFeature(s))[0] if trigger else q2_model.predict(reshapeMLPFeature(s))[0]
            act = np.argmax(outs)
            observed_reward = r
            if s_prime is not None:
                value_vec = q2_model.predict(reshapeMLPFeature(s_prime))[0] if trigger else q1_model.predict(reshapeMLPFeature(s_prime))[0]
                max_action = value_vec[act]
                observed_reward = r + gamma * max_action
            outs[a] = observed_reward
            if trigger:
                q1_batch_x.append(s)
                q1_batch_y.append(outs)
            else:
                q2_batch_x.append(s)
                q2_batch_y.append(outs)
        if len(q1_batch_x) > 0:
            q1_model.train_on_batch(np.array(q1_batch_x), np.array(q1_batch_y))
        if len(q2_batch_x) > 0:
            q2_model.train_on_batch(np.array(q2_batch_x), np.array(q2_batch_y))

replay_memory = []
rolling_average = []
x_plots_reward = []
y_plots_reward = []
for e in range(it):
    state = env.reset()
    fin_r = 0
    for t in range(10000):
        action = None
        if epsilon > random.random():
            action = env.action_space.sample()
        else:
            mod_values = q1_model.predict(reshapeMLPFeature(state))[0]
            tm_values = q2_model.predict(reshapeMLPFeature(state))[0]
            vecs = (np.array(mod_values) + np.array(tm_values)) / 2
            action = np.argmax(vecs)
        observation, reward, done, _ = env.step(action)
        replay_memory.append((state, action, reward, None if done else observation))
        fin_r += reward
        if done:
            break
        state = observation
        if random.random() < train_prob:
            backwards(replay_memory)

    if len(replay_memory) > replay_memory_length:
        replay_memory = replay_memory[replay_memory_length:]
    epsilon *= decay_rate
    rolling_average.append(fin_r)
    print "reward: " + str(fin_r) + " epsilon: " + str(epsilon) + " iter: " + str(e) + " avg " + str(np.average(rolling_average[20:])) + " " + "ddqn_cartpole"

    #plots
    x_plots_reward.append(e)
    y_plots_reward.append(fin_r)

env.close()

plt.plot(x_plots_reward, y_plots_reward, label="iters")
plt.ylabel('reward')
plt.xlabel('iteration')
plt.legend(loc='lower right')
plt.title("CartPole rewards")
plt.savefig("cartpole-ddqn" + '.png')

