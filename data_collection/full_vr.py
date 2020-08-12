# import os
# os.system(r'"C:\Users\sholt\Desktop\bullet3\bin\App_PhysicsServer_SharedMemory_VR_vs2010_x64_release.exe"')
USE_VR = None
import socket
import pybullet as p
import time
import pybullet_data
import numpy as np
from pickle import dumps
import math
import pandaRL
import gym
import os
# p.connect(p.UDP,"192.168.86.100")
import matplotlib.pyplot as plt


# cid = p.connect(p.SHARED_MEMORY)
#
# if (cid < 0):
#     p.connect(p.GUI)
#     USE_VR = False
# else:
#     print("Connected to shared memory")
#     USE_VR = True  # Use either GUI or VR to set commands.
# p.resetSimulation()

env= gym.make('pandaPlay-v0')
env.vr_activation()
env.reset()
p=env.p

# env= gym.make('pandaPlay-v0')
# env.reset(p)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP , 1)
p.setVRCameraState([0.0, -0.3, -0.9], p.getQuaternionFromEuler([0, 0, 0]))

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)

POS = [1, 1, 1]
ORI = list(p.getQuaternionFromEuler([math.pi / 2, 0, math.pi / 2]))
GRIPPER = 0.0
BUTTON = None

def get_new_command():
    try:

        global POS
        global ORI
        global GRIPPER
        global BUTTON
        events = p.getVREvents()
        e = events[0]
        POS = list(e[POSITION])
        ORI = list(e[ORIENTATION])
        GRIPPER = abs(1 - e[ANALOG]) / 25
        BUTTON = e[BUTTONS][2]
        return 1
        #print(p.getEulerFromQuaternion(ORI))
    except:
        return 0



CONTROLLER = 0
POSITION = 1
ORIENTATION = 2
ANALOG = 3
BUTTONS = 6






base_path = 'collected_data/play_demos/'
obs_act_path = base_path + 'obs_act_etc/'
env_state_path = base_path + 'states_and_ims/'
print(obs_act_path)
try:
    os.makedirs(obs_act_path)
except:
    pass

try:
    os.makedirs(env_state_path)
except:
    pass



demo_count = len(list(os.listdir(obs_act_path)))

debugging = False

example_path = env_state_path + str(demo_count)
npz_path = obs_act_path+str(demo_count)
if not debugging:
    os.makedirs(example_path + '/env_states')
    os.makedirs(example_path + '/env_images')
    os.makedirs(npz_path)
counter = 0
control_frequency = 20 # Hz
t0 = time.time()
next_time = t0 + 1/control_frequency

def do_command(t,t0):
    print(t-t0)
    #print(GRIPPER)
    targetPoses = env.panda.goto(POS, ORI, GRIPPER)
    return targetPoses

acts, obs, ags, cagb, joints, targetJoints = [], [], [], [], [], []


def save_stuff(env):
    # what do we care about, POS, ORI and GRIPPER?
    state = env.panda.calc_state()
    #pos_to_save = list(np.array(POS) - state['observation'][0:3]) # actually, keep it absolute
    action = np.array(POS+ORI+[GRIPPER])
    acts.append(action), obs.append(state['observation']), ags.append(
        state['achieved_goal']), \
    cagb.append(state['controllable_achieved_goal']), joints.append(state['joints'])
    # Saving images to expensive here, regen state! and saveimages there
while not get_new_command():
    pass


def save(npz_path):
    print(npz_path)
    if not debugging:
        np.savez(npz_path + '/data', acts=acts, obs=obs, achieved_goals=ags, controllable_achieved_goals=cagb, joint_poses=joints, target_poses=targetJoints)
    print('Finito')

try:
    while(1):
        get_new_command()
        t = time.time()
        if t >= next_time:
            if not debugging:
                env.p.saveBullet(os.path.dirname(os.path.abspath(__file__)) + '/'+ example_path + '/env_states/' + str(counter) + ".bullet") # ideally this takes roughly the same amount of time
            save_stuff(env)
            target = do_command(t,t0)
            targetJoints.append(target)
            next_time = next_time + 1/control_frequency
            counter += 1

        if BUTTON == 1:
            save(npz_path)
            break


except:
    # finito and save
    save(npz_path)
    
    