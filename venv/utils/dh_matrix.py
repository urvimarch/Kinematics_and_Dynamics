import numpy as np

def forward_kinematics(params_dict):
    """
    Computes the forward kinematics of a manipulator given the joint angles using DH parameters.
    
    Args:
        joint_angles : List of joint angles in radians.
        
    Returns:
        end_effector_pos: List containing the x, y, z coordinates of the end effector.
    """
    # Define the DH parameters
    # [alpha, a, d, theta]

    link1_param = [element for sublist in np.array(params_dict['link1']) for element in sublist]
    link2_param = [element for sublist in np.array(params_dict['link2']) for element in sublist] 
    link3_param = [element for sublist in np.array(params_dict['link3']) for element in sublist] 
    EE_param = [element for sublist in np.array(params_dict['EE']) for element in sublist] 
    
    dh_parameters = [
        link1_param,
        link2_param,
        link3_param,
        EE_param
    ]
    dh_p = np.array(dh_parameters)
    
    return dh_p

def transformation_matrix(dh_array):
    num_joints = len(dh_array)
    
    # Transformation matrix 
    transformation_matrix = np.eye(4)
    O = np.zeros((3,num_joints))   #positions of endpoint of the link
    PO = np.zeros((3,num_joints))  #position of COM of link one column for each link
    Z = np.zeros((3,num_joints))   #axis of rotation

    for i in range(num_joints):
        # Extract the DH parameters for the current joint
        d,a,alpha,theta = dh_array[i]
        
        # Compute the transformation matrix for the current joint
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # Construct the transformation matrix for the current joint
        current_transformation = np.array([[cos_theta, -sin_theta, 0, a],
                                           [sin_theta*cos_alpha, cos_theta*cos_alpha, -sin_alpha, -d*sin_alpha],
                                           [sin_theta*sin_alpha, cos_theta*sin_alpha, cos_alpha, cos_alpha*d],
                                           [0, 0, 0, 1]])
        
        # Update the overall transformation matrix
        transformation_matrix = np.dot(transformation_matrix, current_transformation)
        
        O[:,i] =  transformation_matrix[:3, 3]
        Z[:,i] = transformation_matrix[:3, 2]       #thir dcolumn of the rotation matrix
        rx1 = 1; ry1 = 0; rz1 =0                    #insert code for this #COM of link wrt the joint 
        roc1 = np.array([[rx1],[ry1],[rz1]])        #insert code for this
        R = transformation_matrix[:3, :3]
        f= np.dot(R, roc1)
        print(f)
        print(O[:,i])
        PO[:,i] = O[:,i] + f.T

    end_effector_pos = transformation_matrix[0:3, 3:4]

    return end_effector_pos,Z,PO,O

def CalculateJacobians(Z_j, PO_j,O_j):
    num_joints = len(PO_j)
    JacobianMatrix = np.zeros((6,num_joints))

    for i in range(num_joints): 
            JacobianMatrix[3:6 , i] = Z_j[:,i]                    #angular velocities
            P =np.array(PO_j[:,num_joints-1]-O_j[:,i])
            JacobianMatrix[0:3 , i] = np.cross(Z_j[:,i].flatten(), P.flatten())       #linear velocities
            #print(i)
            #print(JacobianMatrix)

    return JacobianMatrix