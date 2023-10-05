# for any manipulator with 4 joints i.e 3R joints and 1 end effector
import numpy as np
import yaml
import os
import sys
#print("Current Working Directory:", os.getcwd())

fpath = os.path.split(os.path.dirname(__file__))[0]
sys.path.append(fpath)
#print(fpath)
## Data Read
yaml_file_path = os.path.join(fpath, 'config', 'dh_param.yaml')

with open(yaml_file_path) as input:
    param_in = yaml.safe_load(input)

#num_joints = 4
#O = np.zeros((3,num_joints))   #positions of endpoint of the link
#Z = np.zeros((3,num_joints))   #axis of rotation
#PO = np.zeros((3,num_joints))  #position of COM of link

from utils.dh_matrix import forward_kinematics 
from utils.dh_matrix import transformation_matrix 
from utils.dh_matrix import CalculateJacobians 


dh_params = forward_kinematics(param_in)
print(dh_params)
EE_pos, Z_axis, PO_pos, O_pos  = transformation_matrix(dh_params)
print("End Effector Position:", EE_pos)

Jacobian_Matrix = CalculateJacobians(Z_axis, PO_pos, O_pos)
print(Jacobian_Matrix)







