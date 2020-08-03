
USE_VR = False # Use either GUI or VR to set commands. 

import socket
import pybullet as p
import time
import pybullet_data
import numpy as np
from pickle import dumps

#p.connect(p.UDP,"192.168.86.100")
cid = p.connect(p.SHARED_MEMORY)

if (cid<0):
	p.connect(p.GUI)
else:
    print("Connected to shared memory")
p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
p.resetSimulation()
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
# p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP ,0)
# p.setVRCameraState([-0.2,0,-1],p.getQuaternionFromEuler([0,0,-90]))
objects = [p.loadURDF("plane.urdf", 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]
objects = [p.loadURDF("pr2_gripper.urdf", 0.500000,0.300006,0.700000,-0.000000,-0.000000,-0.000031,1.000000)]
pr2_gripper = objects[0]
print (f"pr2_gripper={pr2_gripper}")

jointPositions=[ 0.550569, 0.000000, 0.549657, 0.000000 ]
for jointIndex in range (p.getNumJoints(pr2_gripper)):
	p.resetJointState(pr2_gripper,jointIndex,jointPositions[jointIndex])

pr2_cid = p.createConstraint(pr2_gripper,-1,-1,-1,p.JOINT_FIXED,[0,0,0],[0.2,0,0],[0.500000,0.300006,0.700000])
print (f"pr2_cid={pr2_cid}")

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
p.setRealTimeSimulation(1)

inital_qpos = [1.60000000e-01, -1.76000000e+00,  1.84000000e+00,
        -2.51000000e+00,  3.60000000e-01,  7.90000000e-01,
         1.55000000e+00,  0.00000000e+00,  0.00000000e+00]

if not USE_VR:
    for i, num in enumerate(inital_qpos):
        p.addUserDebugParameter(str(i), -1, 1, 0)


def get_new_command():
    if USE_VR:
        events = p.getVREvents()
    else:
        new_command = [p.readUserDebugParameter(i) for i in range(0,9)]
        #new_command = [p.readUserDebugParameter(0), p.readUserDebugParameter(1), p.readUserDebugParameter(2)]+list(np.ones(6))
        
    return np.array(new_command)

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(1024)
            if not data:
                break
            command = get_new_command()
            #print(command, command.tostring())
            conn.sendall(dumps(command))