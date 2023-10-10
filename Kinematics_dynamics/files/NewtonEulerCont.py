#This is for only a trajectory with velocity as a linear function wrt to t . #canvert radian to degrees
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import yaml
import os
import sys

from utils.dynamics_func import get_Pos_COM_vector 
from utils.dynamics_func import get_Pos_frame_vector
from utils.dynamics_func import get_dis_vel_acc
from utils.dynamics_func import Outward_new_itertions 
from utils.dynamics_func import Inward_new_itertions 

fpath = os.path.split(os.path.dirname(__file__))[0]
sys.path.append(fpath)
yaml_file_path = os.path.join(fpath, 'config', 'manipulator_param.yaml')

numsteps = 100
time= np.linspace(0,1,numsteps)
# Define t as a symbol
t = sp.symbols('t')

# Define the velocity function v(t) as an example
v_t_1 = 2 * t**2 + 3 * t + 1  
# Calculate displacement at t_value by integrating the velocity function
s_t_1 = sp.integrate(v_t_1, t)
# Calculate acceleration at t_value by taking the derivative of the velocity function
a_t_1 = sp.diff(v_t_1, t)

# Define the velocity function v(t) as an example
v_t_2 = 3 * t + 1  
# Calculate displacement at t_value by integrating the velocity function
s_t_2 = sp.integrate(v_t_2, t)
# Calculate acceleration at t_value by taking the derivative of the velocity function
a_t_2 = sp.diff(v_t_2, t)



with open(yaml_file_path) as input:
    param_in = yaml.safe_load(input)

No_joints=2

P_c = get_Pos_COM_vector(No_joints, param_in)
P_l = get_Pos_frame_vector(No_joints, param_in)

plt.figure() 

for i in range (numsteps):
  q_1 = s_t_1.subs(t, time[i])
  q_2 = s_t_2.subs(t, time[i])
  q_3 = 0
  qd_1 = v_t_1.subs(t, time[i])
  qd_2 = v_t_2.subs(t, time[i])
  qdd_1 = a_t_1.subs(t, time[i])
  qdd_2 = a_t_2.subs(t, time[i])
  print(q_1,q_2,qd_1,qd_2,qdd_1,qdd_2)

  q_vel,qd_vel,qdd_vel = get_dis_vel_acc(No_joints,q_1,q_2,q_3,qd_1,qd_2,qdd_1,qdd_2)

  F,N = Outward_new_itertions(No_joints, param_in,q_vel,qd_vel,qdd_vel,P_l,P_c)
  #print(F,N)
  f,n_t = Inward_new_itertions(No_joints, param_in,q_vel,qd_vel,qdd_vel,P_l,P_c,F,N)
  #print(f,n_t)
  
  plt.plot(i/100, n_t[2][0], 'bo', label=f'Index {i}')     #plots torque for link 1

plt.xlabel('Time')
plt.ylabel('Torque')
plt.title('Results of Each Iteration')
plt.grid(True)

plt.show()

