import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('1-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('2-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open('3-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open('4-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open('5-object-random-location\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 10

ep_rewards_1 = np.convolve(ep_rewards_1, np.ones((N,))/N, mode='valid')
ep_rewards_2 = np.convolve(ep_rewards_2, np.ones((N,))/N, mode='valid')
ep_rewards_3 = np.convolve(ep_rewards_3, np.ones((N,))/N, mode='valid')
ep_rewards_4 = np.convolve(ep_rewards_4, np.ones((N,))/N, mode='valid')
ep_rewards_5 = np.convolve(ep_rewards_5, np.ones((N,))/N, mode='valid')

plt.figure()
plt.plot(ep_rewards_1)
plt.plot(ep_rewards_2)
plt.plot(ep_rewards_3)
plt.plot(ep_rewards_4)
plt.plot(ep_rewards_5)
plt.legend(["1 Object", "2 Objects", "3 Objects", "4 Objects", "5 Objects"])
plt.xlabel('Evaluation Number')
plt.ylabel('Reward')

plt.savefig('eval.png')

with open('1-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('2-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open('3-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open('4-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open('5-object-random-location\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 100

ep_rewards_1 = np.convolve(ep_rewards_1, np.ones((N,))/N, mode='valid')
ep_rewards_2 = np.convolve(ep_rewards_2, np.ones((N,))/N, mode='valid')
ep_rewards_3 = np.convolve(ep_rewards_3, np.ones((N,))/N, mode='valid')
ep_rewards_4 = np.convolve(ep_rewards_4, np.ones((N,))/N, mode='valid')
ep_rewards_5 = np.convolve(ep_rewards_5, np.ones((N,))/N, mode='valid')

plt.figure()
plt.plot(ep_rewards_1)
plt.plot(ep_rewards_2)
plt.plot(ep_rewards_3)
plt.plot(ep_rewards_4)
plt.plot(ep_rewards_5)
plt.legend(["1 Object", "2 Objects", "3 Objects", "4 Objects", "5 Objects"])
plt.xlabel('Train Number')
plt.ylabel('Reward')

plt.savefig('train.png')


with open('episode_rewards_DDQN_test_new_obj_act_1.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_act_2.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_act_3.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_act_4.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_act_5.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 100



plt.figure()
plt.bar(x=["1 Object", "2 Objects", "3 Objects", "4 Objects", "5 Objects"], height=[np.mean(ep_rewards_1),np.mean(ep_rewards_2),np.mean(ep_rewards_3),np.mean(ep_rewards_4),np.mean(ep_rewards_5)])

plt.xlabel('Train Number')
plt.ylabel('Reward')

plt.savefig('test_new.png')
