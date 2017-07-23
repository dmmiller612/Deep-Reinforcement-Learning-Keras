
"""
WORK IN PROGRESS
"""

import gym
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt
from gym import wrappers


alpha = .001
epsilon = 1
decay_rate = .98
h1 = 120
h2 = 120
gamma = .99
it = 800
env = gym.make('CartPole-v0')
nb_actions = env.action_space.n
input_size = env.observation_space.shape[0]
env = wrappers.Monitor(env, 'temporary', force=True)

"""
Building the model in the explicit manor that doesn't use the cross entropy trick.
Also sharing a hidden layer.
Actor network -> probability of action (policy)
Critic Network -> value function
"""
def build_model():
    state = Input(shape=(input_size,))
    shared = Dense(h1, activation='relu')(state)

    actor_hidden = Dense(h2, activation='relu')(shared)
    actor_action_prob = Dense(nb_actions, activation='softmax')(actor_hidden)

    value_hidden = Dense(h2, activation='relu')(shared)
    state_value = Dense(1, activation='linear')(value_hidden)

    actor = Model(inputs=state, outputs = actor_action_prob)
    critic = Model(inputs=state, outputs=state_value)

    return actor, critic

"""
Optimizer and loss function for the actor network
"""
def build_actor_optimizer(act):
    action = K.placeholder(shape=(None, nb_actions))
    advantages = K.placeholder(shape=(None, ))

    policy = act.output

    #dotprod the actual policy output with the action
    good_prob = K.sum(action * policy, axis = 1)
    #multiply log of probability from above with the advantage. Stop Gradient means don't calculate the gradient in Keras
    eligibility = K.log(good_prob + 1e-10) * K.stop_gradient(advantages)
    loss = -K.sum(eligibility)
    #Entropy is used to improve exploration
    entropy = K.sum(policy * K.log(policy + 1e-10), axis=1)
    actor_loss = loss + .01*entropy

    optimizer = Adam(lr=.001, decay_rate=.00001)
    updates = optimizer.get_updates(act.trainable_weights, [], actor_loss)
    train = K.function([act.input, action, advantages], [], updates=updates)
    return train

"""
optimizer and loss function for critic network (MEAN SQUARED ERROR)
"""
def build_critic_optimizer(crit):
    discounted_reward = K.placeholder(shape=(None, ))
    value = crit.output
    #Mean squared error
    loss = K.mean(K.square(discounted_reward - value))
    optimizer = Adam(lr=.001, decay=.00001)
    updates = optimizer.get_updates(crit.trainable_weights, [], loss)
    train = K.function([crit.input, discounted_reward], [], updates=updates)
    return train

actor, critic = build_model()
critic_optimizer = build_critic_optimizer(critic)
actor_optimizer = build_actor_optimizer(actor)


def reshapeMLPFeature(st):
    return np.reshape(st, (1, input_size))

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
    states = [x[0] for x in replay]
    actions = np.zeros((sequence_length,nb_actions))
    for i in range(sequence_length):
        actions[i][replay[i][1]] = 1
    discounted_rewards = discount_rewards(rewards)

    values = critic.predict(np.array(states))
    values = np.reshape(values, sequence_length)

    advantages = discounted_rewards - values

    actor_optimizer([states, actions, advantages])
    critic_optimizer([states, discounted_rewards])


rolling_average = []
x_plots_reward = []
y_plots_reward = []
for e in range(it):
    state = env.reset()
    fin_r = 0
    temporary_sars = []
    for t in range(10000):
        # Stochastic exploration
        value = actor.predict(reshapeMLPFeature(state)).flatten()
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
