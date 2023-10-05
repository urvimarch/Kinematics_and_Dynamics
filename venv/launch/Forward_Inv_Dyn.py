import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import yaml
import os
import sys

from utils.dynamics_func import get_torques 
from utils.dynamics_func import get_acc 


fpath = os.path.split(os.path.dirname(__file__))[0]
sys.path.append(fpath)
yaml_file_path = os.path.join(fpath, 'config', 'manipulator_param.yaml')

with open(yaml_file_path) as input:
    param_in = yaml.safe_load(input)

No_joints=2
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

tau = np.zeros((numsteps, No_joints))
for i in range (numsteps):
    q_1 = s_t_1.subs(t, time[i])
    q_2 = s_t_2.subs(t, time[i])
    qd_1 = v_t_1.subs(t, time[i])
    qd_2 = v_t_2.subs(t, time[i])
    qdd_1 = a_t_1.subs(t, time[i])
    qdd_2 = a_t_2.subs(t, time[i])

    tau_1, tau_2 = get_torques(param_in,q_1,q_2,qd_1,qd_2,qdd_1,qdd_2)
    tau[i,0]=tau_1
    tau[i,1]=tau_2
#print(tau_1, tau_2)

q1_0 = np.zeros((numsteps,No_joints))
qd1_0 = np.zeros((numsteps,No_joints))
#acc_2 = np.zeros((numsteps,No_joints))



for i in range (numsteps-1):
    #print(tau[i,:])
    acc = get_acc(param_in,q1_0[i,0],q1_0[i,1],qd1_0[i,0],qd1_0[i,1],tau[i,:])
    acc_2 = acc.flatten()
    qd1_0[i+1,:] = qd1_0[i,:] + acc_2*(1/numsteps)
    q1_0[i+1,:] = qd1_0[i,:]*(1/numsteps) + 0.5*acc_2*((1/numsteps)**2)
    
print(q1_0,qd1_0)   