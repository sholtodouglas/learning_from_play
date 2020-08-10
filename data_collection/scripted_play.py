# Ok, lets collect data from preprogrammed pick

import gym
import pandaRL
import numpy as np
import os
import shutil
from tqdm import tqdm

env = gym.make('pandaPlay-v0')
env.render(mode='human')
env.reset()

open_gripper = np.array([0.04])
closed_gripper = np.array([0.01])
p = env.panda.bullet_client
which_object = 0

top_down_grip_ori = np.array(env.p.getQuaternionFromEuler([ np.pi, 0, 0]))

def viz_pos(desired_position):
    goal = np.ones(6)
    goal[which_object * 3:(which_object + 1) * 3] = desired_position
    env.panda.reset_goal_pos(goal)
#--These two skills are used both in picking and pushing, use the offset to push by going next to
def go_above(env, obj_number, offset = np.zeros(3)):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos'] + np.array([0, 0.00, 0.1]) + offset
    current_position = env.panda.calc_actor_state()['pos']
    #current_orn = env.panda.calc_actor_state()['orn']
    viz_pos(desired_position)

    
    action = np.concatenate([desired_position , top_down_grip_ori, open_gripper])
    return action

def descend_push(env, obj_number, offset = np.zeros(3)):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos'] + np.array([0, 0,0.0]) + offset
    current_position = env.panda.calc_actor_state()['pos']
    #current_orn = env.panda.calc_actor_state()['orn']

    action = np.concatenate([desired_position , top_down_grip_ori, closed_gripper])
    return action


# Skills only used for picking
def descend(env, obj_number, offset = np.zeros(3)):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos'] + offset
    current_position = env.panda.calc_actor_state()['pos']
    # descend slowly for the sake of the IK
    desired_position[2] = max(desired_position[2], current_position[2] - 0.03)

    #current_orn = p.getEulerFromQuaternion(env.panda.calc_actor_state()['orn'])
    viz_pos(desired_position)
    action = np.concatenate([desired_position , top_down_grip_ori, open_gripper])
    return action

def close(env, obj_number, offset = np.zeros(3)):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos']
    current_position = env.panda.calc_actor_state()['pos']
    #current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position , top_down_grip_ori, closed_gripper])
    return action

def lift(env, obj_number, offset = np.zeros(3)):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos']
    desired_position[2] +=  0.005
    current_position = env.panda.calc_actor_state()['pos']
    viz_pos(desired_position)
    #current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position , top_down_grip_ori, closed_gripper])
    return action

def take_to(env, position, offset = np.zeros(3)):
    desired_position = position
    current_position = env.panda.calc_actor_state()['pos']
    viz_pos(desired_position)
    #current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position , top_down_grip_ori, closed_gripper])*0.5
    return action



def pick_to(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses):
    global which_object
    times = np.array([0.7, 1.2, 1.4, 1.6, 2.0, 2.2]) + t
    #times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) + t
    states = [go_above, descend, close, lift, take_to, go_above]


    take_to_pos = np.random.uniform(env.goal_lower_bound, env.goal_upper_bound)
    goal = env.panda.goal
    goal[which_object*3:(which_object+1)*3] = take_to_pos
    env.panda.reset_goal_pos(goal)
    data = peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=take_to_pos, obj_number=which_object)
    which_object = not which_object # flip which object we are playing with
    return data


def peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=None, offset=np.zeros(3), obj_number=0):
    state_pointer = 0
    while (t < times[state_pointer]):
        if state_pointer == 4:
            action = states[state_pointer](env, goal, offset = np.zeros(3))
        else:
            action = states[state_pointer](env, obj_number=obj_number, offset=offset)
        if not debugging:
            p.saveBullet(example_path + '/env_states/' + str(counter) + ".bullet")
        counter += 1  # little counter for saving the bullet states

        acts.append(action), obs.append(o['observation']), ags.append(
            o['achieved_goal']), \
        cagb.append(o['controllable_achieved_goal']), currentPoses.append(o['joints'])
        o2, r, d, info = env.step(action)
        targetPoses.append(info['target_poses'])
        print(o2['achieved_goal'][14:])
        if d:
            print('Env limits exceeded')
            return {'success':0, 't':t}
        # NOTE! This is different to how it is done in goal conditioned RL, the ag is from
        # the same timestep because thats how we'll use it in LFP (and because its harder to do
        # the rl style step reset in VR teleop.
        o = o2

        t += dt
        if t >= times[state_pointer]:
            state_pointer += 1
            if state_pointer > len(times)-1:
                break

    return {'last_obs': o, 'success': 1, 't':t, 'counter':counter}




debugging = True

dt = 0.04

action_buff = []
observation_buff = []
desired_currentPoses_buff = []
achieved_currentPoses_buff = []
controllable_achieved_goal_buff = []
full_positional_state_buff = []


base_path = 'collected_data/play_demos/'
try:
    os.makedirs(base_path)
except:
    print('Folder already exists')

demo_count = len(list(os.listdir(base_path)))

activities = [pick_to]#, push_directionally]
#activities = [push_directionally]

play_len = 120

for i in tqdm(range(0, 60)):
    o = env.reset()
    t = 0

    acts, obs, currentPoses, ags, cagb, targetPoses = [], [], [], [], [], []
    example_path = base_path + str(demo_count)
    if not debugging:
        os.makedirs(example_path)
        os.makedirs(example_path + '/env_states')
    counter = 0

    #pbar = tqdm(total=play_len)
    while(t < play_len):
        activity_choice = np.random.choice(len(activities))
        result = activities[activity_choice](env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses)
        if not result['success']:
            break
        #pbar.update(result['t'] - t)
        t = result['t']
        counter = result['counter']
        o = result['last_obs']


    if t>(play_len/2): #reasonable length with some play interaction
        if not debugging:

            action_buff.append(acts), observation_buff.append(obs), desired_currentPoses_buff.append(
                currentPoses), achieved_currentPoses_buff.append(ags), \
            controllable_achieved_goal_buff.append(cagb), full_positional_state_buff.append(targetPoses)

            np.savez(base_path + str(demo_count) + '/data', acts=acts, obs=obs,
                     achieved_currentPoses=ags,
                     controllable_achieved_currentPoses=cagb, joint_poses=currentPoses, target_poses=targetPoses)
            demo_count += 1
    else:
        print('Demo failed')
        # delete the folder with all the saved states within it
        if not debugging:
            shutil.rmtree(base_path + str(demo_count))










#
# def push_directionally(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses):
#     times = np.array([0.5, 1.0, 1.4]) + t
#     states = [go_above, descend_push, go_above]
#     # choose a random point in a circle around the block
#     alpha = np.random.random(1)*2*np.pi
#     r = 0.03
#     x,z = r * np.cos(alpha), r * np.sin(alpha)
#     offset = np.array([x,0,z])
#
#
#     return peform_action(env, t, o, counter, acts, obs, currentPoses, ags, cagb, targetPoses, times, states, offset=offset)