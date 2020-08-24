import pickle
import pandas as pd

df = pd.DataFrame()

with open('episode_rewards_DDQN_test_old_pre_obj1.pickle', 'rb') as learner:
    df['old_pre_1'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_pre_obj2.pickle', 'rb') as learner:
    df['old_pre_2'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_pre_obj3.pickle', 'rb') as learner:
    df['old_pre_3'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_pre_obj4.pickle', 'rb') as learner:
    df['old_pre_4'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_pre_obj5.pickle', 'rb') as learner:
    df['old_pre_5'] = pickle.load(learner)

with open('episode_rewards_DDQN_test_old_post_1.pickle', 'rb') as learner:
    df['old_post_1'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_post_2.pickle', 'rb') as learner:
    df['old_post_2'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_post_3.pickle', 'rb') as learner:
    df['old_post_3'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_post_4.pickle', 'rb') as learner:
    df['old_post_4'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_old_post_5.pickle', 'rb') as learner:
    df['old_post_5'] = pickle.load(learner)

with open('episode_rewards_DDQN_test_new_obj_pretrain_1.pickle', 'rb') as learner:
    df['new_pre_1'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_pretrain_2.pickle', 'rb') as learner:
    df['new_pre_2'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_pretrain_3.pickle', 'rb') as learner:
    df['new_pre_3'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_pretrain_4.pickle', 'rb') as learner:
    df['new_pre_4'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_pretrain_5.pickle', 'rb') as learner:
    df['new_pre_5'] = pickle.load(learner)

with open('episode_rewards_DDQN_test_new_obj_post_1.pickle', 'rb') as learner:
    df['new_post_1'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_post_2.pickle', 'rb') as learner:
    df['new_post_2'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_post_3.pickle', 'rb') as learner:
    df['new_post_3'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_post_4.pickle', 'rb') as learner:
    df['new_post_4'] = pickle.load(learner)
with open('episode_rewards_DDQN_test_new_obj_post_5.pickle', 'rb') as learner:
    df['new_post_5'] = pickle.load(learner)
