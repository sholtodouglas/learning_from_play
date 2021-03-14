# Ok, lets collect data from preprogrammed pick

import gym
import roboticsPlayroomPybullet
import numpy as np
import os
import shutil
from tqdm import tqdm

env = gym.make('pandaPlay1Obj-v0')
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
    desired_position[2] +=  0.1
    current_position = env.panda.calc_actor_state()['pos']
    viz_pos(desired_position)
    #current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([desired_position , top_down_grip_ori, closed_gripper])
    return action

def take_to(env, position, offset = np.zeros(3)):
    desired_position = position
    current_position = env.panda.calc_actor_state()['pos']
    delta = (desired_position - current_position)*0.2
    viz_pos(desired_position)

    #current_orn = env.panda.calc_actor_state()['orn']
    action = np.concatenate([current_position+delta , top_down_grip_ori, closed_gripper])
    return action

def reorient_obj(env, position, offset = np.zeros(3)):
    desired_position = position
    action = np.concatenate([desired_position, env.panda.default_arm_orn, closed_gripper])
    return action

def go_above_reorient(env, obj_number, offset = np.zeros(3)):
    desired_position = env.panda.calc_environment_state()[obj_number]['pos'] + np.array([0, 0.00, 0.1])
    action = np.concatenate([desired_position , env.panda.default_arm_orn, open_gripper])
    return action


def pick_to(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses):
    global which_object
    times = np.array([0.7, 1.2, 1.4, 1.6, 2.5, 2.9]) + t
    #times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) + t
    states = [go_above, descend, close, lift, take_to, go_above]


    take_to_pos = np.random.uniform(env.goal_lower_bound, env.goal_upper_bound)
    goal = env.panda.goal
    goal[which_object*3:(which_object+1)*3] = take_to_pos
    env.panda.reset_goal_pos(goal)
    data = peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=take_to_pos, obj_number=which_object)
    min(env.num_objects-1, not which_object) # flip which object we are playing with
    return data

def pick_reorient(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses):
    global which_object
    times = np.array([0.7, 1.2, 1.4, 1.6, 2.5, 2.9]) + t
    #times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) + t
    states = [go_above, descend, close, lift, reorient_obj, go_above_reorient]


    take_to_pos = env.panda.calc_environment_state()[which_object]['pos'] + np.array([0,0,0.05])
    goal = env.panda.goal
    goal[which_object*3:(which_object+1)*3] = take_to_pos
    env.panda.reset_goal_pos(goal)
    data = peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=take_to_pos, obj_number=which_object)
    which_object = min(env.num_objects-1, not which_object) # flip which object we are playing with
    return data
#################################################### Door script ####################################

door_z = 0.12

def go_up(env, goal):
    desired_position = [0,0,0]
    desired_position[2] = 0.3
    action = np.concatenate([np.array(desired_position), top_down_grip_ori, open_gripper])
    return action

def go_in_front(env, goal):
    door_x =  env.panda.calc_environment_state()[2]['pos'][0]
    desired_position = np.array([door_x, 0.30, door_z])
    action = np.concatenate([desired_position, env.panda.default_arm_orn, open_gripper])
    return action

def close_on_door(env, goal):
    door_x = env.panda.calc_environment_state()[2]['pos'][0]
    desired_position = np.array([door_x, 0.4, door_z])
    action = np.concatenate([desired_position, env.panda.default_arm_orn, closed_gripper])
    return action

def pull_door(env, goal):

    action = np.concatenate([goal, env.panda.default_arm_orn, closed_gripper])
    return action


def toggle_door(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses):
    #times = np.array([0.4, 1.0, 1.4, 1.9, 2.0]) + t
    times = np.array([0.7, 1.0, 1.5, 1.6,2.0]) + t
    #times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) + t
    states = [go_in_front, close_on_door, pull_door, go_in_front, go_up] # go up is optional

    door_x = env.panda.calc_environment_state()[2]['pos'][0]
    if door_x < 0:
        des_x = 0.15
    else:
        des_x = -0.15
    desired_position = np.array([des_x, 0.4, door_z])

    data = peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=desired_position,  obj_number=None)
    return data

################################################### Toggle Drawer #############################################################

drawer_x = -0.15
drawer_handle = 0.25
def go_above_drawer(env, goal):
    drawer_y = -env.panda.calc_environment_state()[3]['pos'][0] -drawer_handle
    desired_position = [drawer_x, drawer_y, -0.00]
    action = np.concatenate([np.array(desired_position), top_down_grip_ori, open_gripper])
    return action

def close_on_drawer(env, goal):
    drawer_y = -env.panda.calc_environment_state()[3]['pos'][0] - drawer_handle
    desired_position = [drawer_x, drawer_y, -0.1]
    action = np.concatenate([np.array(desired_position), top_down_grip_ori, open_gripper])
    return action

def pull_drawer(env, goal):
    desired_position = goal
    action = np.concatenate([np.array(desired_position), top_down_grip_ori, closed_gripper])
    return action

def toggle_drawer(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses):
    #times = np.array([0.4, 1.0, 1.4, 1.9, 2.0]) + t
    times = np.array([0.7, 1.0, 1.5, 1.6]) + t
    #times = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5]) + t
    states = [go_above_drawer, close_on_drawer, pull_drawer, go_up] # go up is optional

    drawer_y = -env.panda.calc_environment_state()[3]['pos'][0]
    if drawer_y < 0:
        des_y = 0.15
    else:
        des_y = -0.15
    desired_position = np.array([drawer_x, -0.2+des_y, -0.1])

    data = peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=desired_position,  obj_number=None)
    return data

def quat_sign_flip(a, idxs):
    for pair in idxs:
        for i in range(1, len(a)):
            quat = a[i, pair[0]:pair[1]]
            last_quat = a[i - 1, pair[0]:pair[1]]
            if (np.sign(quat) == -np.sign(last_quat)).all():  # i.e, it is an equivalent quaternion
                a[i, pair[0]:pair[1]] = - a[i, pair[0]:pair[1]]
    return a

def peform_action(env, t, o, counter, acts,obs,currentPoses,ags,cagb,targetPoses, times, states, goal=None, offset=np.zeros(3), obj_number=0):
    state_pointer = 0
    while (t < times[state_pointer]):
        if obj_number is not None:
            if state_pointer == 4:
                action = states[state_pointer](env, goal, offset = np.zeros(3))
            else:
                action = states[state_pointer](env, obj_number=obj_number, offset=offset)
        else:
            action = states[state_pointer](env, goal)

        if not debugging:
            p.saveBullet(example_path + '/env_states/' + str(counter) + ".bullet")
        counter += 1  # little counter for saving the bullet states

        acts.append(action), obs.append(o['observation']), ags.append(
            o['achieved_goal']), \
        cagb.append(o['controllable_achieved_goal']), currentPoses.append(o['joints'])
        o2, r, d, info = env.step(action)
        targetPoses.append(info['target_poses'])
        #print(o2['achieved_goal'][16:])
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




debugging = False

dt = 0.04

base_path = 'collected_data/scripted_play_demos/'
obs_act_path = base_path + 'obs_act_etc/'
env_state_path = base_path + 'states_and_ims/'


try:
    os.makedirs(obs_act_path)
except:
    pass

try:
    os.makedirs(env_state_path)
except:
    pass



activities = [toggle_drawer, toggle_door, pick_reorient, pick_to] #[pick_to]#, push_directionally]
#activities = [push_directionally]

play_len = 8

for i in tqdm(range(0, 500)): # 60
    o = env.reset()
    t = 0

    acts, obs, currentPoses, ags, cagb, targetPoses = [], [], [], [], [], []

    demo_count = len(list(os.listdir(obs_act_path)))
    example_path = env_state_path + str(demo_count)
    npz_path = obs_act_path + str(demo_count)
    if not debugging:
        os.makedirs(example_path + '/env_states')
        os.makedirs(example_path + '/env_images')
        os.makedirs(npz_path)
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


    if t>6: #reasonable length with some play interaction
        if not debugging:
            acts = quat_sign_flip(np.array(acts), [(3, 7)])
            obs = quat_sign_flip(np.array(obs), [(3, 7), (10, 14)])
            ags = quat_sign_flip(np.array(ags), [(3, 7)])
            np.savez(npz_path+ '/data', acts=acts, obs=obs,
                     achieved_goals =ags,
                     controllable_achieved_goals =cagb, joint_poses=currentPoses, target_poses=targetPoses)
            demo_count += 1
    else:
        print('Demo failed')
        # delete the folder with all the saved states within it
        if not debugging:
            shutil.rmtree(obs_act_path  + str(demo_count))
            shutil.rmtree(env_state_path + str(demo_count))










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