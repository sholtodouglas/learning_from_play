import numpy as np
from tensorflow.keras.utils import Progbar

door_positions = {'left': -0.15, 'middle': 0.0, 'right': 0.15}
drawer_positions = {'closed': -0.1, 'middle': 0.0, 'open': 0.1}
button_positions = {'open': 0.029, 'closed': 0.0}
dial_positions = {'default':0}
obj_poses = {'default': [0,0.2,0.0], 'shelf':[0,0.43, 0.27], 'left':[-0.2, 0.2,0.0], 'right':[0.2,0.2,0.0],'closed_drawer': [-0.15, 0.1, -0.09], 'open_drawer':[-0.15, -0.15, -0.09], 'cupboard_left': [-0.2, 0.45, 0.0], 'cupboard_right':[0.2, 0.45, 0.0]}
obj_oris = {'upright': [0,0.7,0,0.7], 'default':[0,0,0,1], 'lengthways':[0.0, 0.0, 0.7071, 0.7071]}


def test_suite_reset(env, obj_ori = 'default', obj_pos = 'default', door = 'middle', drawer = 'middle', button = 'open', dial = 'default'):
    env.reset()
    obj_offset = np.array([0.,0.,0.])
    if  obj_ori == 'upright':
        obj_offset += np.array([0,0.,0.025])
        
    positions = [door_positions[door], drawer_positions[drawer], button_positions[button], dial_positions[dial]]
             
    for idx, j in enumerate(env.panda.joints):
        env.panda.bullet_client.resetJointState(j, 0, positions[idx]) # reset drawer, button etc
        
    env.panda.bullet_client.resetBasePositionAndOrientation(env.panda.objects[0],
                                                            np.array(obj_poses[obj_pos])+obj_offset, obj_oris[obj_ori])
    
    
    
def define_goal(env,obj_ori = 'default', obj_pos = 'default', door = 'middle', drawer = 'middle', button = 'open', dial = 'default'):
    # We know the goal is 
    # objpos (3), obj_ori(4), door, drawer, button, dial
    obj_offset = np.array([0.,0.,0.])
    if  obj_ori == 'upright':
        obj_offset += np.array([0,0.,0.025])
    
    goal = list(np.array(obj_poses[obj_pos])+obj_offset)+list(obj_oris[obj_ori])+[door_positions[door],drawer_positions[drawer], button_positions[button],dial_positions[dial]]
    
    return np.array(goal).astype(np.float32)

def measure_progress(initial_ag, final_ag, end_goal):
    progbar = Progbar(1, verbose=1, interval=0.05)
    initial_distance = abs(end_goal - initial_ag)
    final_distance = abs(end_goal - final_ag)
    progress = 1- (np.mean(final_distance) / np.mean(initial_distance))
    progbar.add(progress, [('L1', np.mean(final_distance))])

def door_left(env):
    test_suite_reset(env, door = 'right')
    return define_goal(env, door = 'left')

def door_right(env):
    test_suite_reset(env, door = 'left')
    return define_goal(env, door = 'right')

def open_drawer(env):
    test_suite_reset(env, drawer='closed')
    return define_goal(env, drawer = 'open')

def close_drawer(env):
    test_suite_reset(env, drawer='open')
    return define_goal(env, drawer = 'closed')

def push_left(env):
    test_suite_reset(env, obj_ori='lengthways')
    return define_goal(env, obj_ori='lengthways', obj_pos='left')

def push_right(env):
    test_suite_reset(env, obj_ori='lengthways')
    return define_goal(env, obj_ori='lengthways', obj_pos='right')

def block_in_cupboard_right(env):
    test_suite_reset(env, door='left')
    return define_goal(env, door = 'left', obj_pos='cupboard_right')

def block_in_cupboard_left(env):
    test_suite_reset(env, door='right')
    return define_goal(env, door = 'right', obj_pos='cupboard_left')

def block_in_cupboard_right_upright(env):
    test_suite_reset(env, door='left', obj_ori='upright')
    return define_goal(env, door = 'left', obj_pos='cupboard_right', obj_ori='upright')

def block_in_cupboard_left_upright(env):
    test_suite_reset(env, door='right', obj_ori='upright')
    return define_goal(env, door = 'right', obj_pos='cupboard_left', obj_ori='upright')

def block_default_to_upright(env):
    test_suite_reset(env, obj_ori='default')
    return define_goal(env, obj_ori='upright')

def block_lengthways_to_upright(env):
    test_suite_reset(env, obj_ori='lengthways')
    return define_goal(env, obj_ori='upright')

def block_default_to_lengthways(env):
    test_suite_reset(env, obj_ori='default')
    return define_goal(env, obj_ori='lengthways')

def block_lengthways_to_default(env):
    test_suite_reset(env, obj_ori='lengthways')
    return define_goal(env, obj_ori='default')

def press_button(env):
    test_suite_reset(env, door='right')
    return define_goal(env, door='right', button='closed')

def block_on_shelf(env):
    test_suite_reset(env, obj_ori='upright')
    return define_goal(env, obj_pos='shelf',obj_ori='upright')

def block_in_open_drawer(env):
    test_suite_reset(env, drawer='open')
    return define_goal(env, drawer='open', obj_pos='open_drawer')

def block_in_open_drawer_lengthways(env):
    test_suite_reset(env, drawer='open', obj_ori='lengthways')
    return define_goal(env, drawer='open', obj_ori='lengthways', obj_pos='open_drawer')

# Multi part objectives
def press_button_with_door_obstacle(env):
    test_suite_reset(env, door='left')
    return define_goal(env, door='right', button='closed')

def block_in_drawer_and_close(env):
    test_suite_reset(env, drawer='open', obj_ori='lengthways')
    return define_goal(env, drawer='closed', obj_ori='lengthways', obj_pos='closed_drawer')

#now, how to measure success?

test_list = [door_left,door_right,open_drawer,close_drawer,push_left,push_right,block_in_cupboard_right,
         block_in_cupboard_left,block_in_cupboard_right_upright, block_in_cupboard_left_upright, block_default_to_upright,
         block_lengthways_to_upright, block_default_to_lengthways, block_lengthways_to_default, press_button,block_on_shelf,
         block_in_open_drawer,block_in_open_drawer_lengthways,
         press_button_with_door_obstacle,block_in_drawer_and_close]


