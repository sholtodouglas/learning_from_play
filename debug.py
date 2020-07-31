# Goal is, make it easy to get play demos for tristan.
import d4rl
import gym

env = gym.make('kitchen-partial-v0')

o = env.reset()
env.render()

for i in range(0,200):
    env.step(env.action_space.sample())
    env.render()