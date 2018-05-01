import gym
import pandas as pd
import numpy as np

# function to unpack observation from gym environment
def unpackObs(obs):
    return  np.round(obs['achieved_goal'], 3).tolist(), \
            np.round(obs['desired_goal'], 3).tolist(), \
            np.round(obs['observation'], 3).tolist()

# Load environment
l_ = []
env = gym.make('FetchReach-v1')
env.reset()
sample_size = 100

# Save to tsv
for i in range(sample_size):
    obs, reward, done, info = env.step(env.action_space.sample())
    achieved_goal, desired_goal, observation = unpackObs(obs)
    l_.append([observation, achieved_goal, desired_goal, reward, done])
df = pd.DataFrame(l_, columns=['Observation', 'AchievedGoal', 'DesiredGoal', 'Reward', 'Done'])
print(df)
df.to_csv('Sample_Data.tsv', sep='\t', index=False)


