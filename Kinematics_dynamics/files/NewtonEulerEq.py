#This is for only one timestep at a q,qd,qdd . The outer loop will be adding to get a trajectory and continuous torque.
import numpy as np
import yaml
import os
import sys

fpath = os.path.split(os.path.dirname(__file__))[0]
sys.path.append(fpath)
yaml_file_path = os.path.join(fpath, 'config', 'manipulator_param.yaml')

with open(yaml_file_path) as input:
    param_in = yaml.safe_load(input)

No_joints=2

from utils.dynamics_func import get_Pos_COM_vector 
from utils.dynamics_func import get_Pos_frame_vector 
from utils.dynamics_func import get_joint_vel_vector 
from utils.dynamics_func import get_joint_acc_vector 
from utils.dynamics_func import Outward_itertions 
from utils.dynamics_func import Inward_itertions 


P_c = get_Pos_COM_vector(No_joints, param_in)
P_l = get_Pos_frame_vector(No_joints, param_in)
joint_velocity_vector = get_joint_vel_vector(No_joints, param_in)
joint_acc_vector = get_joint_acc_vector(No_joints, param_in)
F,N = Outward_itertions(No_joints, param_in,joint_velocity_vector,joint_acc_vector,P_l,P_c)
print(F,N)
f,n_t = Inward_itertions(No_joints, param_in,joint_velocity_vector,joint_acc_vector,P_l,P_c,F,N)
print(f,n_t)

