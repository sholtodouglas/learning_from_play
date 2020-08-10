#import os
#os.system(r'"C:\Users\sholt\Desktop\bullet3\bin\App_PhysicsServer_SharedMemory_VR_vs2010_x64_release.exe"')
USE_VR = None  
import socket
import pybullet as p
import time
import pybullet_data
import numpy as np
from pickle import dumps
import math 
#p.connect(p.UDP,"192.168.86.100")
cid = p.connect(p.SHARED_MEMORY)

if (cid<0):
    p.connect(p.GUI)
    USE_VR = False
else:
    print("Connected to shared memory")
    USE_VR = True # Use either GUI or VR to set commands. 

p.setAdditionalSearchPath(pybullet_data.getDataPath())
print(pybullet_data.getDataPath())
p.resetSimulation()
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
#p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP , 1)
p.setVRCameraState([0.0, -0.3,-1.5],p.getQuaternionFromEuler([0,0,0]))
objects = [p.loadURDF("plane.urdf", 0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,1.000000)]


p.loadURDF("tray/traybox.urdf", [0, 0.0, -0.1],
                                [0,0,0,1])
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


if not USE_VR:
    for i in range(8):
        p.addUserDebugParameter(str(i), -1, 1, 0)

gripper_max_joint = 0.550569
POS = [1,1,1]
ORI = list(p.getQuaternionFromEuler([math.pi/2, 0, math.pi/2]))
GRIPPER = 0.0
def get_new_command():
    try:
        if USE_VR:
            global POS
            global ORI
            global GRIPPER
            events = p.getVREvents()
            e = events[0]
            POS = list(e[POSITION])
            ORI = list(e[ORIENTATION])
            GRIPPER = e[ANALOG]
            print(p.getEulerFromQuaternion(ORI))
        else:
            POS = [p.readUserDebugParameter(i) for i in range(0,3)]
    except:
        pass


def update_gripper():
    
    ori = p.getQuaternionFromEuler(np.array(p.getEulerFromQuaternion(ORI)) + np.array([-180,0,0]))
    p.changeConstraint(pr2_cid, POS, ori, maxForce=500)
    p.setJointMotorControl2(pr2_gripper, 0, controlMode=p.POSITION_CONTROL,targetPosition=gripper_max_joint - GRIPPER * gripper_max_joint,force=1.0)
    p.setJointMotorControl2(pr2_gripper, 2, controlMode=p.POSITION_CONTROL,targetPosition=gripper_max_joint - GRIPPER * gripper_max_joint,force=1.1)
    

CONTROLLER = 0
POSITION = 1
ORIENTATION = 2
ANALOG=3
BUTTONS=6

# while(1):
#     get_new_command()
#     update_gripper()


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(2)
            if not data:
                break
            get_new_command()
            update_gripper()
            command = np.concatenate([POS,ORI, [GRIPPER]])
            conn.sendall(dumps(command))