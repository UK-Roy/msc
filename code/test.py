import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG

env = gym.make("PandaReachDense-v3")
model = DDPG(policy="MultiInputPolicy", env=env, verbose=1)
model.learn(12_000)

model.save("ddpg_reach")
