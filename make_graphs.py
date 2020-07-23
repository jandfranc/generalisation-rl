import pickle
import matplotlib.pyplot as plt
import numpy as np


with open(r'1-object-random-location\models\base_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open(r'2-object-random-location\models\base_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open(r'3-object-random-location\models\base_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open(r'4-object-random-location\models\base_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open(r'5-object-random-location\models\base_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 30

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

plt.figure()
plt.plot(ep_rewards_2)
plt.plot(ep_rewards_3)
plt.plot(ep_rewards_4)
plt.plot(ep_rewards_5)
plt.legend(["2 Objects", "3 Objects", "4 Objects", "5 Objects"])
plt.xlabel('Evaluation Number')
plt.ylabel('Reward')

plt.savefig('eval_without1.png')


print(ep_rewards_1[-1], ep_rewards_2[-1], ep_rewards_3[-1], ep_rewards_4[-1], ep_rewards_5[-1])






with open(r'1-object-random-location\models\base_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open(r'2-object-random-location\models\base_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open(r'3-object-random-location\models\base_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open(r'4-object-random-location\models\base_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open(r'5-object-random-location\models\base_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
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


plt.figure()
plt.plot(ep_rewards_2)
plt.plot(ep_rewards_3)
plt.plot(ep_rewards_4)
plt.plot(ep_rewards_5)
plt.legend(["2 Objects", "3 Objects", "4 Objects", "5 Objects"])
plt.xlabel('Train Number')
plt.ylabel('Reward')

plt.savefig('train_without1.png')












with open('episode_rewards_DDQN_test_new_obj_pretrain_1.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_pretrain_2.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_pretrain_3.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_pretrain_4.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_pretrain_5.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 100

print([np.mean(ep_rewards_1),np.mean(ep_rewards_2),np.mean(ep_rewards_3),np.mean(ep_rewards_4),np.mean(ep_rewards_5)])
print([np.std(ep_rewards_1),np.std(ep_rewards_2),np.std(ep_rewards_3),np.std(ep_rewards_4),np.std(ep_rewards_5)])

plt.figure()
plt.bar(x=["1 Object", "2 Objects", "3 Objects", "4 Objects", "5 Objects"],
        height=[np.mean(ep_rewards_1),np.mean(ep_rewards_2),np.mean(ep_rewards_3),np.mean(ep_rewards_4),np.mean(ep_rewards_5)],
        yerr=[np.std(ep_rewards_1),np.std(ep_rewards_2),np.std(ep_rewards_3),np.std(ep_rewards_4),np.std(ep_rewards_5)],
        capsize=10)

plt.xlabel('Train Number')
plt.ylabel('Reward')
plt.grid(axis='y')

plt.savefig('test_new.png')
















with open(r'1-object-random-location\models\retrain_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open(r'2-object-random-location\models\retrain_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open(r'3-object-random-location\models\retrain_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open(r'4-object-random-location\models\retrain_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open(r'5-object-random-location\models\retrain_model\episode_rewards_DDQN_eval_rewards_p.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 30

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

plt.savefig('eval_newobj.png')

plt.figure()
plt.plot(ep_rewards_2)
plt.plot(ep_rewards_3)
plt.plot(ep_rewards_4)
plt.plot(ep_rewards_5)
plt.legend(["2 Objects", "3 Objects", "4 Objects", "5 Objects"])
plt.xlabel('Evaluation Number')
plt.ylabel('Reward')

plt.savefig('eval_newobj_without1.png')


print(ep_rewards_1[-1], ep_rewards_2[-1], ep_rewards_3[-1], ep_rewards_4[-1], ep_rewards_5[-1])











with open(r'1-object-random-location\models\retrain_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open(r'2-object-random-location\models\retrain_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open(r'3-object-random-location\models\retrain_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open(r'4-object-random-location\models\retrain_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open(r'5-object-random-location\models\retrain_model\episode_rewards_DDQN_rewards_p.pickle', 'rb') as learner:
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

plt.savefig('train_newobj.png')





with open('episode_rewards_DDQN_test_new_obj_post_1.pickle', 'rb') as learner:
    ep_rewards_1 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_post_2.pickle', 'rb') as learner:
    ep_rewards_2 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_post_3.pickle', 'rb') as learner:
    ep_rewards_3 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_post_4.pickle', 'rb') as learner:
    ep_rewards_4 = pickle.load(learner)
    print('o')
with open('episode_rewards_DDQN_test_new_obj_post_5.pickle', 'rb') as learner:
    ep_rewards_5 = pickle.load(learner)
    print('o')
N = 100

print([np.mean(ep_rewards_1),np.mean(ep_rewards_2),np.mean(ep_rewards_3),np.mean(ep_rewards_4),np.mean(ep_rewards_5)])
print([np.std(ep_rewards_1),np.std(ep_rewards_2),np.std(ep_rewards_3),np.std(ep_rewards_4),np.std(ep_rewards_5)])

plt.figure()
plt.bar(x=["1 Object", "2 Objects", "3 Objects", "4 Objects", "5 Objects"],
        height=[np.mean(ep_rewards_1),np.mean(ep_rewards_2),np.mean(ep_rewards_3),np.mean(ep_rewards_4),np.mean(ep_rewards_5)],
        yerr=[np.std(ep_rewards_1),np.std(ep_rewards_2),np.std(ep_rewards_3),np.std(ep_rewards_4),np.std(ep_rewards_5)],
        capsize=10)

plt.xlabel('Train Number')
plt.ylabel('Reward')
plt.grid(axis='y')

plt.savefig('test_new_post.png')
