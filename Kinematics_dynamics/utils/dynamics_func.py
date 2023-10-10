import numpy as np
import math


def get_Pos_COM_vector(n,mani_param):
    """
    Computes the location of point massses wrt to joint fram eof corresponding link  of a manipulator

    Args:
        Manipulator decription : List of all position of COM of masses
    Returns:
        P_C : location of point massses wrt to joint fram eof corresponding link
    """ 

    Pos_COM_param = [element for sublist in np.array(mani_param['Pos_COM']).T for element in sublist]
    P_c = np.zeros((3,n))       #location of point massses wrt to joint fram eof corresponding link
    for i in range(n):
        P_c[:,i] = [(Pos_COM_param[i]) ,0, 0]

    return P_c

def get_Pos_frame_vector(n,mani_param):
    """
    Computes the location of i+1 link frame wrt to previous i link frame 

    Args:
        Manipulator decription : List of all position of ith frames
    Returns:
        P_l: location of i+1 link frame wrt to previous i link frame .
    """ 

    P_1 = [element for sublist in np.array(mani_param['P_l']) for element in sublist]
    P_0 = [element for sublist in np.zeros((1,n+2)) for element in sublist]  
    p_l = [P_1,P_0,P_0]
    P_L = np.array(p_l)
    return P_L

def get_joint_vel_vector(n,mani_param):
    """
    Gives the joint velocoties in vector form 
    """
    joint_velocity_vector = np.zeros((3,n))
    for j in range(n):
        joint_velocity_vector[0, j] = 0
        joint_velocity_vector[1, j] = 0
        joint_velocity_vector[2, j] = [element for sublist in np.array(mani_param['joint_velocity']).T for element in sublist][j]
    return joint_velocity_vector

def get_joint_acc_vector(n,mani_param):
    joint_acc_vector = np.zeros((3,n))
    for j in range(n):
        joint_acc_vector[0, j] = 0
        joint_acc_vector[1, j] = 0
        joint_acc_vector[2, j] = [element for sublist in np.array(mani_param['joint_acc']).T for element in sublist][j]
    return joint_acc_vector

def rot_matrix(k):
    cos_k = math.cos(k)
    sin_k = math.sin(k)
    rot_mat = np.array([
        [cos_k, -sin_k, 0],
        [sin_k, cos_k, 0],
        [0, 0, 1]
    ])

    return rot_mat

#NewtonEulerfor one timestep
def Outward_itertions(n,mani_param,joint_velocity_vector,joint_acc_vector,P_l,P_c):
    ang_vel = np.zeros((3,n+1))
    ang_acc = np.zeros((3,n+1))
    linear_acc_link_frame = np.zeros((3, n+1))
    linear_acc_link_frame[:,0] = [element for sublist in np.array(mani_param['g']) for element in sublist]
    linear_acc_com_frame = np.zeros((3,n+1))
    F = np.zeros((3,n+1)) # here first column is world link
    N = np.zeros((3,n+1))

    for k in range(n):
        rot_mat = rot_matrix([element for sublist in np.array(mani_param['theta']).T for element in sublist][k])
       
        ang_vel[:,k+1] = np.dot(rot_mat,ang_vel[:,k]) + joint_velocity_vector[:,k]

        ang_acc[:,k+1] = np.dot(rot_mat,ang_acc[:,k]) + np.cross(np.dot(rot_mat,ang_vel[:,k]),joint_velocity_vector[:,k]) + joint_acc_vector[:,k]

        linear_acc_link_frame[:, k + 1] = np.dot(rot_mat, (np.cross(ang_acc[:, k], P_l[:, k]) +
                                                           np.cross(ang_vel[:, k], np.cross(ang_vel[:, k], P_l[:, k])) +
                                                           linear_acc_link_frame[:, k]
                                                           ))

        linear_acc_com_frame[:,k+1] = (
                                        np.cross(ang_acc[:,k+1],P_c[:,k])
                                        + np.cross(ang_vel[:,k+1], np.cross(ang_vel[:,k+1],P_c[:,k]))
                                        + linear_acc_link_frame[:,k+1]
                                        )


        F[:,k+1]= [element for sublist in np.array(mani_param['link_masses']).T for element in sublist][k]*linear_acc_com_frame[:,k+1]

        N[:,k+1]= 0

    return F,N

def Inward_itertions(n,mani_param,joint_velocity_vector,joint_acc_vector,P_l,P_c, F,N):
    f = np.zeros((3, n+1)) # here last column is end effector
    n_t = np.zeros((3, n+1))
    for k in range(n, 0, -1):
        rot_mat = rot_matrix([element for sublist in np.array(mani_param['theta']).T for element in sublist][k-1])

        f[:,k-1] = np.dot(np.linalg.inv(rot_mat),f[:,k])  +  F[:,k-1]      #forces
        n_t[:,k-1] = (
                    np.dot(np.linalg.inv(rot_mat),n_t[:,k])
                    + np.cross(P_c[:,k-1], F[:,k])
                    + np.cross(P_l[:,k+1],np.dot(np.linalg.inv(rot_mat),f[:,k]))
                    )  
    return f,n_t    

#NewtonEuler for continuous vel

def get_dis_vel_acc(n,q1,q2,q3,qd1,qd2,qdd1,qdd2):
    joint_angle_vector = np.zeros((n+1,1))
    joint_velocity_vector = np.zeros((3,n))
    joint_acc_vector = np.zeros((3,n))

    joint_angle_vector[0,0] = q1
    joint_angle_vector[1,0] = q2
    joint_angle_vector[2,0] = q3

    joint_velocity_vector[2,0] = qd1
    joint_velocity_vector[2,1] = qd2
    joint_acc_vector[2,0] = qdd1
    joint_acc_vector[2,1] = qdd2

    return joint_angle_vector,joint_velocity_vector,joint_acc_vector

def Outward_new_itertions(n,mani_param,joint_angle_vector,joint_velocity_vector,joint_acc_vector,P_l,P_c):
    ang_vel = np.zeros((3,n+1))
    ang_acc = np.zeros((3,n+1))
    linear_acc_link_frame = np.zeros((3, n+1))
    linear_acc_link_frame[:,0] = [element for sublist in np.array(mani_param['g']) for element in sublist]
    linear_acc_com_frame = np.zeros((3,n+1))
    F = np.zeros((3,n+1)) # here first column is world link
    N = np.zeros((3,n+1))

    for k in range(n):
        rot_mat = rot_matrix(joint_angle_vector[k,0])
        
        ang_vel[:,k+1] = np.dot(rot_mat,ang_vel[:,k]) + joint_velocity_vector[:,k]

        ang_acc[:,k+1] = np.dot(rot_mat,ang_acc[:,k]) + np.cross(np.dot(rot_mat,ang_vel[:,k]),joint_velocity_vector[:,k]) + joint_acc_vector[:,k]

        linear_acc_link_frame[:, k + 1] = np.dot(rot_mat, (np.cross(ang_acc[:, k], P_l[:, k]) +
                                                           np.cross(ang_vel[:, k], np.cross(ang_vel[:, k], P_l[:, k])) +
                                                           linear_acc_link_frame[:, k]
                                                           ))

        linear_acc_com_frame[:,k+1] = (
                                        np.cross(ang_acc[:,k+1],P_c[:,k])
                                        + np.cross(ang_vel[:,k+1], np.cross(ang_vel[:,k+1],P_c[:,k]))
                                        + linear_acc_link_frame[:,k+1]
                                        )


        F[:,k+1]= ([element for sublist in np.array(mani_param['link_masses']).T for element in sublist][k]*linear_acc_com_frame[:,k+1])

        N[:,k+1]= 0

    return F,N

def Inward_new_itertions(n,mani_param,joint_angle_vector,joint_velocity_vector,joint_acc_vector,P_l,P_c, F,N):
    f = np.zeros((3, n+1)) # here last column is end effector
    n_t = np.zeros((3, n+1))
    for k in range(n, 0, -1):
        rot_mat = rot_matrix(joint_angle_vector[k,0])

        f[:,k-1] = np.dot(np.linalg.inv(rot_mat),f[:,k])  +  F[:,k-1]      #forces
        n_t[:,k-1] = (
                    np.dot(np.linalg.inv(rot_mat),n_t[:,k])
                    + np.cross(P_c[:,k-1], F[:,k])
                    + np.cross(P_l[:,k+1],np.dot(np.linalg.inv(rot_mat),f[:,k]))
                    )  
    return f,n_t    

def get_torques(mani_param,q1,q2,q1_dot,q2_dot,q1_double_dot,q2_double_dot):
    
    m1 = [element for sublist in np.array(mani_param['link_masses']).T for element in sublist][0]
    m2 = [element for sublist in np.array(mani_param['link_masses']).T for element in sublist][1]
    l1 = 1
    l2 = 1
    c1 = math.cos(q1)
    c2 = math.cos(q2)
    s2 = math.sin(q2)
    c12 = math.cos(q1+q2)
    s12 = math.sin(q1+q2)
    g = 9.81

    τ1 =( (m2 * (l2 ** 2) * (q1_double_dot + q2_double_dot))
        + (m2 * l2 * l1*c2 * (2*q1_double_dot + q2_double_dot))
        + ((m1+m2)*(l1**2)*q1_double_dot)
        - (m2 * l1 * l2 * s2 * (q2_dot ** 2))
        - (2 * m2 * l1 * l2 * s2 * q1_dot * q2_dot)
        + (m2 * l2 * c12 * g)
        + ((m1 + m2) * l1 * g * c1))
        
    τ2 = ((m2 * l1 * l2 * c2 * q1_double_dot)
        + (m2 * l2 * g* c12)
        + (m2 * (l2 ** 2) * (q1_double_dot + q2_double_dot))
        + (m2 * l1 * l2 * s2 * (q1_dot ** 2)))
    
    return τ1, τ2

def get_acc(mani_param,q1,q2,q1_dot,q2_dot,torque):
    
    m1 = [element for sublist in np.array(mani_param['link_masses']).T for element in sublist][0]
    m2 = [element for sublist in np.array(mani_param['link_masses']).T for element in sublist][1]
    l1 = 1
    l2 = 1
    c1 = math.cos(q1)
    c2 = math.cos(q2)
    s2 = math.sin(q2)
    c12 = math.cos(q1+q2)
    s12 = math.sin(q1+q2)
    g = 9.81
    M = np.array([
                [m2 * l2**2 + (m1 + m2) * l1**2 + 2 * m2 * l1 * l2 * c2, m2 * l2**2 + m2 * l1 * l2 * c2],
                [m2 * l2**2 + m2 * l1 * l2 * c2, m2 * l2**2 + m2 * l1 * l2 * c2]
                 ])
    

    V = np.array([[-(m2 * l1 * l2 * s2 * (q2_dot ** 2)) - (2 * m2 * l1 * l2 * s2 * q1_dot * q2_dot)],
                  [(m2 * l1 * l2 * s2 * (q1_dot ** 2))]])
    G = np.array([[(m2 * l2 * g* c12)+ ((m1 + m2) * l1 * g * c1)],
                  [(m2 * l2 * g* c12)]])
    
    T = np.array(torque).reshape(-1, 1)
    #print(V,G,T,np.linalg.inv(M))

    acc = np.dot(np.linalg.inv(M),(T-V-G))
    
    #print(acc)

    return acc
