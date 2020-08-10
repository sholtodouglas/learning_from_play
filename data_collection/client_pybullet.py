import socket
import adept_envs
import pickle
import gym 
import numpy as np
import pandaRL
import pybullet as p
import os
import time
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server




def vr_command_to_action(env, last_o, command):
    # The gripper is 0-1. 
    # we want to scale the gripper down to 0-0.04.
    command[7] = abs(1-command[7]) / 25 # and have it so its default open
    # Xyz and Ori are in absolute position, wheras our env is commanded with relative position
    #o[0:7] is the xyzori of the arm
    # current_orn = np.array(p.getEulerFromQuaternion(env.panda.calc_actor_state()['orn']))
    # desired_orn  = np.array(p.getEulerFromQuaternion(command[3:7]))
    # orn_shift = desired_orn - current_orn
    pos_shift = command[0:3] - last_o[0:3]
    command = np.concatenate([pos_shift, command[3:7], [command[7]]])
    
    #print(current_orn)
    #orn = np.array(env.panda.default_arm_orn)-np.array(current_orn)
    
    #print(orn)
    return command


base_path = 'collected_data/play_demos/'
try:
    os.makedirs(base_path)
except:
    print('Folder already exists')

demo_count = len(list(os.listdir(base_path)))
debugging = False


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    example_path = base_path + str(demo_count)
    if not debugging:
        os.makedirs(example_path)
        os.makedirs(example_path + '/env_states')
    counter = 0
    acts, obs, goals, ags, cagb, fpsb, times, joints = [], [], [], [], [], [], [], []

    env = gym.make("pandaPlay-v0")
    env.render(mode='human')
    o = env.reset()
    start_time = time.time()

    while(1):
        # Get the action
        s.sendall(b'R')
        data = s.recv(1024)
        command = pickle.loads(data)
        action = vr_command_to_action(env, o['observation'],command)

        # states for determinsitc reset
        if not debugging:
            p.saveBullet(example_path + '/env_states/' + str(counter) + ".bullet")

        o2, r, d, _ = env.step(action)

        # main buffs - time will be variable but it just gives as an indication of actual frame rate
        # true time is simulation time, we are determinsitically stepping. 
        acts.append(action), obs.append(o['observation']), goals.append(o['desired_goal']), ags.append(
            o2['achieved_goal']), \
        cagb.append(o2['controllable_achieved_goal']), fpsb.append(o2['full_positional_state']), times.append(time.time() -start_time)

        #


        state = env.panda.calc_actor_state()
        print(state['orn'])
        #print([ '%.2f' % elem for elem in state['joints']])
        joints.append(state['joints'])
        # House Keeping
        o = o2
        counter += 1

        if not debugging:
            if counter % 300==0:
                np.savez(base_path + str(demo_count) + '/data', acts=acts, obs=obs,
                             desired_goals=goals,
                             achieved_goals=ags,
                             controllable_achieved_goals=cagb,
                             full_positional_states=fpsb, times = times, joints=joints)




# pos = command[0:3]
# orn = p.getEulerFromQuaternion(command[3:7]) #+ np.array([0,np.pi/2,0])

# env.absolute_command(pos, orn)