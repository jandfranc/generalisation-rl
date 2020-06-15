import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('1-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('5-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 10

ep_rewards_1 = np.convolve(ep_rewards_1, np.ones((N,))/N, mode='valid')
ep_rewards_5 = np.convolve(ep_rewards_5, np.ones((N,))/N, mode='valid')

plt.figure()
plt.plot(ep_rewards_1)
plt.plot(ep_rewards_5)
plt.legend(["1 Object", "5 Objects"])
plt.xlabel('Evaluation Number')
plt.ylabel('Reward')

plt.savefig('eval.png')

with open('1-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('5-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 100

ep_rewards_1 = np.convolve(ep_rewards_1, np.ones((N,))/N, mode='valid')
ep_rewards_5 = np.convolve(ep_rewards_5, np.ones((N,))/N, mode='valid')

plt.figure()
plt.plot(ep_rewards_1)
plt.plot(ep_rewards_5)
plt.legend(["1 Object", "5 Objects"])
plt.xlabel('Train Number')
plt.ylabel('Reward')

plt.savefig('train.png')
