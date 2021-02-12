import numpy as np
import tensorflow as tf
try:
    import pybullet as p
except:
    print('No pybullet installation found - which is fine if training')
'''
A set of functions which test various elements of the pybullet state space for success
'''

def compare_xyz(g, ag, limits = np.array([0.05, 0.05, 0.05])):
    if (abs(g-ag) > limits).any():
        #print('Failed xyz')
        return False
    else:
        return True
    
    
def compare_RPY(g, ag, limits = np.array([np.pi/4,np.pi/4,np.pi/4])):
    g = np.array(p.getEulerFromQuaternion(g))
    ag = np.array(p.getEulerFromQuaternion(ag))
    if (abs(g-ag) > limits).any():
        #print('Failed rpy')
        return False
    else:
        return True
    
def compare_drawer(g, ag, limit=0.025):
    if abs(g-ag) > limit:
        #print('Failed drawer')
        return False
    else:
        return True
    
def compare_door(g, ag, limit=0.03):
    if abs(g-ag) > 0.04:
        #print('Failed door', g, ag)
        return False
    else:
        return True
    
    
def compare_button(g, ag, limit=0.01):
    if abs(g-ag) >limit: 
        #print('Failed button', g , ag)
        return False
    else:
        return True
    
def compare_dial(g,ag, limit=0.3):
    if abs(g-ag) > limit:
        #print('Failed dial')
        return False
    else:
        return True
    
    
def success_func(g, ag):
    g,ag = np.squeeze(g), np.squeeze(ag)
    if compare_xyz(g[0:3], ag[0:3]) and compare_RPY(g[3:7], ag[3:7]) and compare_drawer(g[7], ag[7]) and compare_door(g[8], ag[8]) and compare_dial(g[10], ag[10]) and compare_button(g[9], ag[9]):
        return True
    else:
        return False


block = [0,3]
qqqq = [3,7]
drawer = 7
door = 8
button = 9
dial = 10

door_positions = {'left': -0.15, 'middle': 0.0, 'right': 0.15}

drawer_positions = {'closed': 0.075, 'middle': 0.035, 'open': -0.05}

button_positions = {'open': 0.029, 'closed': -0.029}

dial_positions = {'one':0, 'default': 0.35, 'two': 0.8}

obj_poses = {'default': [0,0.1,0.0], 'shelf':[0,0.43, 0.27],
             'left':[-0.2, 0.2,0.0], 'right':[0.2,0.2,0.0],
             'closed_drawer': [-0.15, 0.1, -0.07], 
             'open_drawer':[-0.15, -0.1, -0.07], 
             'cupboard_left': [-0.2, 0.45, 0.0], 
             'cupboard_right':[0.2, 0.45, 0.0]}

obj_oris = {'upright': [0, -0.7,0,0.7], 'default':[0.0, 0.0, 0.7071, 0.7071], 
            'lengthways':[0,0,0,1]}

class tester():

    def __init__(self, env):
        self.env = env

    def current_ag(self):
        return self.env.panda.calc_state()['achieved_goal']

    def make_goal(self, obj, pos=None, ori=None):
        '''
        Ori only applies for the block
        '''
        g = self.current_ag()
        if isinstance(obj,list):
            if pos is not None:
                g[block[0]:block[1]] = np.array(pos)
            if ori is not None:
                g[qqqq[0]:qqqq[1]] = np.array(obj_oris[ori])
                if ori == 'upright' and pos is not None:
                    g[2] += 0.025
                
        else:
            g[obj] = pos
            
        return tf.expand_dims(tf.expand_dims(g,0),0)




tasks = {
    'block_left': (block, obj_poses['left'], 'default'),
    'block_right': (block, obj_poses['right'], 'default'),
    'block_shelf': (block, obj_poses['shelf'], 'default'),
    'block_cupboard_right': (block, obj_poses['cupboard_right'], 'default'),
    'block_cupboard_left': (block, obj_poses['cupboard_left'], 'default'),
    'block_drawer':  (block, obj_poses['open_drawer'], 'lengthways'),
    'block_upright': (block,obj_poses['default'], 'upright'),
    'block_lengthways': (block, obj_poses['default'], 'lengthways'),
    'block_lengthways_left': (block, obj_poses['left'], 'lengthways'),
    'block_ori_default': (block, obj_poses['default'], 'default'),
    'button': (button, button_positions['closed']),
    'door_left': (door, door_positions['left']),
    'door_right': (door, door_positions['right']),
    'open_drawer': (drawer, drawer_positions['open']),
    'close_drawer': (drawer, drawer_positions['closed']),
    'dial_on': (dial, dial_positions['one']),
    'dial_off': (dial, dial_positions['two']),
}