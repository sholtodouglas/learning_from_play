
import socket
import adept_envs
import pickle
import gym 
import numpy as np

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432        # The port used by the server

env = gym.make("kitchen_relax-v1")
env.reset()
env.render()

# while(1):
#     joint_vels = np.zeros(9)
#     env.step(np.array(joint_vels))
#     env.render()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    while(1):

        s.sendall(b'R')
        data = s.recv(1024)
        joint_vels = pickle.loads(data)
        #joint_vels = np.zeros(9)
        env.step(np.array(joint_vels))
        env.render()

print('Received', repr(data))