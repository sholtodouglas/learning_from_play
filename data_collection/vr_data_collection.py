# import os
# os.system(r'"C:\Users\sholt\Desktop\bullet3\bin\App_PhysicsServer_SharedMemory_VR_vs2010_x64_release.exe"')

debugging = False

import socket
import pybullet as p
import time
import pybullet_data
import numpy as np
from pickle import dumps 
import math
import roboticsPlayroomPybullet
import gym
import os
import shutil
import threading
# p.connect(p.UDP,"192.168.86.100")
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

arm = 'UR5'
if arm == 'UR5':
    print('UR5!')
    env= gym.make('UR5PlayAbsRPY1Obj-v0')
else:
    env= gym.make('PandaPlayAbsRPY1Obj-v0')

env.vr_activation()
env.reset()
p=env.p

# env= gym.make('pandaPlay-v0')
# env.reset(p)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP , 1)
p.setVRCameraState([0.0, -0.3, -1.1], p.getQuaternionFromEuler([0, 0, 0]))

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
        if e[ANALOG] > GRIPPER:
            GRIPPER = min(GRIPPER + 0.25,e[ANALOG])
        else: # i.e, force it to slowly open/close
            GRIPPER = max(GRIPPER - 0.25,e[ANALOG])
        #GRIPPER =  e[ANALOG]
        #print(GRIPPER)
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

if arm == 'UR5':
    base_path = 'collected_data/UR5/'
else:
    base_path = 'collected_data/30Hz_one_obj/'
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






def do_command(t,t0):
    #print(t-t0)
    #print(GRIPPER)
    #print(p.getEulerFromQuaternion(ORI))
    targetPoses = env.panda.goto(POS, ORI, GRIPPER)
    return targetPoses



def save_stuff(env,acts, obs, ags, cagb, joints, acts_rpy, acts_rpy_rel, velocities, obs_rpy, obs_rpy_inc_obj, gripper_proprioception):
    # what do we care about, POS, ORI and GRIPPER?
    state = env.panda.calc_state() 
    #print(p.getEulerFromQuaternion(state['observation'][3:7]))
    
    #pos_to_save = list(np.array(POS) - state['observation'][0:3]) # actually, keep it absolute
    action = np.array(POS+ORI+[GRIPPER]) 
    ori_rpy = p.getEulerFromQuaternion(ORI)
    rel_xyz = np.array(POS)-np.array(state['observation'][0:3])
    rel_rpy = np.array(ori_rpy) - np.array(p.getEulerFromQuaternion(state['observation'][3:7]))
    action_rpy =  np.array(POS+list(ori_rpy)+[GRIPPER])
    action_rpy_rel = np.array(list(rel_xyz)+list(rel_rpy)+[GRIPPER])

    
    acts.append(action), obs.append(state['observation']), ags.append(
        state['achieved_goal']), \
    cagb.append(state['controllable_achieved_goal']), joints.append(state['joints']), acts_rpy.append(action_rpy),
    acts_rpy_rel.append(action_rpy_rel), velocities.append(state['velocity']), obs_rpy.append(state['obs_rpy']), 
    obs_rpy_inc_obj.append(state['obs_rpy_inc_obj']), gripper_proprioception.append(state['gripper_proprioception'])

    

    # Saving images to expensive here, regen state! and saveimages there
while not get_new_command():
    pass

def save_state(env, example_path, counter):
    #env, example_path, counter = args
    env.p.saveBullet(os.path.dirname(os.path.abspath(__file__)) + '/'+ example_path + '/env_states/' + str(counter) + ".bullet") # ideally this takes roughly the same amount of time


def save(npz_path, acts, obs, ags, cagb, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, obs_rpy, obs_rpy_inc_obj, gripper_proprioception):
    print(npz_path)
    if not debugging:
        
        np.savez(npz_path + '/data', acts=acts, obs=obs, achieved_goals=ags, 
        controllable_achieved_goals=cagb, joint_poses=joints, target_poses=targetJoints, acts_rpy=acts_rpy, 
        acts_rpy_rel=acts_rpy_rel, velocities=velocities, obs_rpy=obs_rpy, obs_rpy_inc_obj=obs_rpy_inc_obj, 
        gripper_proprioception=gripper_proprioception)
    print('Finito')

env.p.saveBullet(os.path.dirname(os.path.abspath(__file__)) + '/init_state.bullet') 

while(1):
    time.sleep(1)
    demo_count = len(list(os.listdir(obs_act_path)))
    example_path = env_state_path + str(demo_count)
    npz_path = obs_act_path+str(demo_count)

    if not debugging:
        os.makedirs(example_path + '/env_states')
        os.makedirs(example_path + '/env_images')
        os.makedirs(npz_path)
    counter = 0
    control_frequency = 25 # Hz
    t0 = time.time()
    next_time = t0 + 1/control_frequency
    # reset from init which we created (allows you to press a button on the controller and reset the env)
    env.p.restoreState(fileName = os.path.dirname(os.path.abspath(__file__)) + '/init_state.bullet')
    
    acts, obs, ags, cagb, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, obs_rpy, obs_rpy_inc_obj, gripper_proprioception = [], [], [], [], [], [], [], [],[], [], [], []
    try:
        
        while(1):
            
            
            
            t = time.time()
            if t >= next_time:
                get_new_command()
                
                if counter % 30 == 0:
                    print(1/((1/control_frequency) + (t - next_time))) # prints the current fps
                    if not debugging:
                   
                        thread = threading.Thread(target = save_state, name = str(counter), args = (env, example_path, counter))
                        thread.start()
                    
                save_stuff(env,acts, obs, ags, cagb, joints, acts_rpy, acts_rpy_rel, velocities, obs_rpy, obs_rpy_inc_obj, gripper_proprioception)
                target = do_command(t,t0)
                targetJoints.append(target)
                
                next_time = next_time + 1/control_frequency
                counter += 1

            if BUTTON == 1:
                save(npz_path, acts, obs, ags, cagb, joints, targetJoints, acts_rpy, acts_rpy_rel, velocities, obs_rpy, obs_rpy_inc_obj, gripper_proprioception)
                BUTTON = 0 
                break

    
    except Exception as e:
        print(e)
        if not debugging:
            shutil.rmtree(example_path)
            shutil.rmtree(npz_path)
            print('Ending Data Collection')
            break
        
    