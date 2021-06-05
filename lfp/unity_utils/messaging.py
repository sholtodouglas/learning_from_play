try:
    import rospy
    from sensor_msgs.msg import Image as ImageMsg
    from robotics_demo.msg import JointPositions, Observation, AchievedGoal, PositionCommand, RPYProprioState, Velocities, ResetInfo, ToRecord, RPYState, Reengage,  QuaternionProprioState, Goal, TimerBeat
    from robotics_demo.srv import getIK, getIKResponse, getState, getStateResponse, getTime, getTimeResponse
    import time
    from PIL import Image, ImageOps
    import pybullet
except:
    print("Ros/Pybullet not found - no issue if just training")

try:
    from robotics_demo.msg import ResetAngles
except:
    pass
    
import numpy as np

import time
import threading

UNITY_MAX_AG_SIZE = 7+3+1+1+1+7+9
UNITY_MAX_OBS_SIZE = 7+ UNITY_MAX_AG_SIZE
 # obj1, buttons, drawer, door, obj2presence, obj2, buttonxyzs

def rosImg_to_numpy(img): # : ImageMsg
    image_height = img.width
    image_width = img.height
    image = Image.frombytes('RGBA', (image_width,image_height), img.data)
    image = ImageOps.flip(image)
    return np.array(image)


def proprio_quat_to_rpy_vector(o): # : QuaternionProprioState
    x,y,z,q1,q2,q3,q4,gripper = o.pos_x, o.pos_y, o.pos_z, o.q1, o.q2, o.q3, o.q4, o.gripper
    rpy = pybullet.getEulerFromQuaternion([-q2, -q1, q4, q3]) # yay for beautiful conversion between axis
    return np.array([x,y,z,rpy[0], rpy[1], rpy[2], gripper])

def proprio_rpy_to_rpy_vector(o): # : RPYProprioState
    return np.array([o.pos_x, o.pos_y, o.pos_z, o.rot_r, o.rot_p, o.rot_y, o.gripper])

def proprio_rpy_to_ROSmsg(o: np.ndarray):
    return RPYProprioState(o[0],o[1], o[2], o[3], o[4], o[5], o[6])

def ag_to_vector(o): # : AchievedGoal
    return np.array([o.obj1_pos_x, o.obj1_pos_y, o.obj1_pos_z, o.obj1_q1, o.obj1_q2, o.obj1_q3, o.obj1_q4, o.button1, o.button2, o.button3, o.drawer, o.door, 
                    o.obj2_present, o.obj2_pos_x, o.obj2_pos_y, o.obj2_pos_z, o.obj2_q1, o.obj2_q2, o.obj2_q3, o.obj2_q4, o.button1_x, o.button1_y, o.button1_z, o.button2_x,
                    o.button2_y, o.button2_z, o.button3_x, o.button3_y, o.button3_z])

def ag_to_ROSmsg(o: np.ndarray):
    if len(o) < UNITY_MAX_AG_SIZE:
        o = np.concatenate([o, np.zeros(UNITY_MAX_AG_SIZE - len(o))]) # correct for if its a shortened one from an env with one object for example
        
    return AchievedGoal(o[0],o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11], int(o[12]),o[13],o[14],o[15],o[16],o[17],o[18],o[19],o[20],o[21],o[22],o[23],o[24],o[25],o[26],o[27],o[28])


def act_to_jointPositionsROSmsg(j: np.ndarray):
    return JointPositions(j[0], j[1], j[2], j[3], j[4], j[5], j[6], time.time())


def ras_to_vector(o):  #ResetAngles - type check will fail without a ros install, and we don't need it if only training
    return np.array([o.shoulder, o.upper_arm, o.forearm, o.wrist_1, o.wrist_2, o.wrist_3, o.outer_knuckle_left, o.inner_finger_left,\
                     o.inner_knuckle_left, o.outer_knuckle_right, o.inner_finger_right, o.inner_knuckle_right])

def vector_to_ras(o: np.array):
    return ResetAngles(o[0],o[1], o[2], o[3], o[4], o[5], o[6], o[7], o[8], o[9], o[10], o[11])


def unstack(a, axis = 0):
    return [np.take(a, i, axis = axis) for i in range(a.shape[axis])]



    #time.sleep(1/TIME_FREQUENCY)