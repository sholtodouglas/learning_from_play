import adept_envs
import gym

env = gym.make('kitchen_relax-v1')
env.render(mode='human')
env.reset()

