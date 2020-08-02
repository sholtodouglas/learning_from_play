# Goal is, make it easy to get play demos for tristan.

import gym
import adept_envs
env = gym.make('kitchen_relax-v1')

o = env.reset()
env.render()

for i in range(0,200):
    env.step(env.action_space.sample())
    env.render()