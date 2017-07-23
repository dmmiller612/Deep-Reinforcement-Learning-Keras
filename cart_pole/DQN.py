import gym
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from gym import wrappers


"""
All parameters are here, which can be reproduced using paper specifications
"""
alpha = .01
epsilon = 1
decay_rate = .97
h1 = 80
h2 = 80
gamma = .99
batch_s = 32
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

"""
Model Parameters are here
"""
model = Sequential()
model.add(Dense(h1, input_dim=input))
model.add(Activation('relu'))
if h2 > 0:
    model.add(Dense(h2))
    model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
ada = Adam(lr=alpha, decay=0.00001)
model.compile(loss="mse", optimizer=ada)

"""
reshapes feature for Keras
"""
def reshapeMLPFeature(st):
    return np.reshape(st, (1, input))

"""
Back propagation layer. Replay is the history of past examinations.
"""
def back_prop(replay):
    for _ in range(train_count): # For batch learning, which kind of stunk :(
        random_transitions = random.sample(replay, min(len(replay), batch_s))
        batch_x = []
        batch_y = []
        for d in random_transitions:
            s = d[0]
            a = d[1]
            r = d[2]
            s_prime = d[3]
            s_outs = model.predict(reshapeMLPFeature(s))[0]
            value_reward = r
            if s_prime is not None: #if none, means it is terminal, we just want the reward
                vec = model.predict(reshapeMLPFeature(s_prime))[0]
                max_action = np.amax(vec) #np.max is just an alias for amax, using it directly.
                value_reward = r + gamma * max_action
            s_outs[a] = value_reward
            batch_x.append(s)
            batch_y.append(s_outs)
        model.train_on_batch(np.array(batch_x), np.array(batch_y))

#parameters for inside of the training
memory = []
rolling_average = []
x_plots_reward = []
y_plots_reward = []

#training starts here
for e in range(it):
    state = env.reset()
    fin_r = 0
    for t in range(10000):
        action = None
        if epsilon > random.random():
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(reshapeMLPFeature(state))[0])
        observation, reward, done, _ = env.step(action)
        memory.append((state, action, reward, None if done else observation))
        fin_r += reward
        state = observation
        if random.random() < train_prob:
            back_prop(memory)
        if done:
            break

    if len(memory) > replay_memory_length:
        memory = memory[replay_memory_length:]
    epsilon *= decay_rate
    rolling_average.append(fin_r)
    print "reward: " + str(fin_r) + " epsilon: " + str(epsilon) + " iter: " + str(e) + " avg " + str(np.average(rolling_average[20:])) + " " + "cart_pole_dqn"

    #plots
    x_plots_reward.append(e)
    y_plots_reward.append(fin_r)


env.close()

plt.plot(x_plots_reward, y_plots_reward, label="iters")
plt.ylabel('reward')
plt.xlabel('iteration')
plt.legend(loc='lower right')
plt.title("CartPole rewards")
plt.savefig('cartpole-dqn' + '.png')

model.save('cartpole-dqn' + '.h5')