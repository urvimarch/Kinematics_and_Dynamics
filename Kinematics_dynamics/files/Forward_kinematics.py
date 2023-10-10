import numpy as np

# Forward kinematics
# Def1ne the matr1x elements
# dh parameters
c = 0.438*np.pi
theta = [0, 0+c, 0-c, 0]  # Example values for theta
alpha = [0, -np.pi/2, 0, 0]  # Example values for alpha
a = [0, 0, 0.130, 0.124]  # Example values for a
d = [0.777, 0, 0, 0]  # Example values for d

# Create the matrix us1ng NumPy

#link1
T01 = np.array([
    [np.cos(theta[0]), -np.sin(theta[0]), 0, a[0]],
    [np.sin(theta[0])*np.cos(alpha[0]), np.cos(theta[0])*np.cos(alpha[0]), -np.sin(alpha[0]), -np.sin(alpha[0])*d[0]],
    [np.sin(theta[0])*np.sin(alpha[0]), np.cos(theta[0])*np.sin(alpha[0]), np.cos(alpha[0]), np.cos(alpha[0])*d[0]],
    [0, 0, 0, 1]
])
O1 = T01[0:3, 3:4]
R01 = T01[0:3, 0:3]
rx1 = 1; ry1 = 0; rz1 =0 
roc1 = np.array([[rx1],[ry1],[rz1]])
P0c1 = O1 + np.dot(R01, roc1)

#Link 2
T12 = np.array([
    [np.cos(theta[1]), -np.sin(theta[1]), 0, a[1]],
    [np.sin(theta[1])*np.cos(alpha[1]), np.cos(theta[1])*np.cos(alpha[1]), -np.sin(alpha[1]), -np.sin(alpha[1])*d[1]],
    [np.sin(theta[1])*np.sin(alpha[1]), np.cos(theta[1])*np.sin(alpha[1]), np.cos(alpha[1]), np.cos(alpha[1])*d[1]],
    [0, 0, 0, 1]
])
T02 = np.dot(T01,T12)
O2 = T02[0:3, 3:4]
R02 = T02[0:3, 0:3]
rx2 = 1; ry2 = 0; rz2 =0 
roc2 = np.array([[rx2],[ry2],[rz2]])
P0c2 = O2 + np.dot(R02, roc2)

#Link3
T23 = np.array([
    [np.cos(theta[2]), -np.sin(theta[2]), 0, a[2]],
    [np.sin(theta[2])*np.cos(alpha[2]), np.cos(theta[2])*np.cos(alpha[2]), -np.sin(alpha[2]), -np.sin(alpha[2])*d[2]],
    [np.sin(theta[2])*np.sin(alpha[2]), np.cos(theta[2])*np.sin(alpha[2]), np.cos(alpha[2]), np.cos(alpha[2])*d[2]],
    [0, 0, 0, 1]
])
T03 = np.dot(T02,T23)
O3 = T03[0:3, 3:4]
R03 = T03[0:3, 0:3]
rx3 = 1; ry3 = 0; rz3 =0 
roc3 = np.array([[rx3],[ry3],[rz3]])
P0c3 = O3 + np.dot(R03, roc3)

#Link4
T34 = np.array([
    [np.cos(theta[3]), -np.sin(theta[3]), 0, a[3]],
    [np.sin(theta[3])*np.cos(alpha[3]), np.cos(theta[3])*np.cos(alpha[3]), -np.sin(alpha[3]), -np.sin(alpha[3])*d[3]],
    [np.sin(theta[3])*np.sin(alpha[3]), np.cos(theta[3])*np.sin(alpha[3]), np.cos(alpha[3]), np.cos(alpha[3])*d[3]],
    [0, 0, 0, 1]
])
T04 = np.dot(T03,T34)
O4 = T04[0:3, 3:4]
R04 = T04[0:3, 0:3]
rx4 = 1; ry4 = 0; rz4 =0 
roc4 = np.array([[rx4],[ry4],[rz4]])
P0c4 = O4 + np.dot(R04, roc4)

#End Effector
T4EE = np.array([
    [1, 0, 0, 0.126],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

T0EE = np.dot(T04,T4EE)
P0EE = T0EE[0:3, 3:4]    #Coordinates of end effector

# Inverse Kinematics 
# finding jacobians
Z1 = T01[0:3, 2:3]
JO1 = T01[0:3, 2:3]
P11 =np.array(P0c1-O1)
JP1 = np.cross(Z1.flatten(), P11.flatten())

Z2 = T02[0:3, 2:3]
JO2 = T02[0:3, 2:3]
P21=np.array(P0c2-O1)
P22=np.array(P0c2-O2)
JP2 = [[np.cross(Z1.flatten(), P21.flatten())], [np.cross(Z2.flatten(), P22.flatten())]]

Z3 = T03[0:3, 2:3]
JO3 = T03[0:3, 2:3]
P31=np.array(P0c3-O1)
P32=np.array(P0c3-O2)
P33=np.array(P0c3-O3)
JP3= [[np.cross(Z1.flatten(), P31.flatten())], [np.cross(Z2.flatten(), P33.flatten())],[np.cross(Z3.flatten(), P32.flatten())]]

Z4 = T04[0:3, 2:3]
JO4 = T04[0:3, 2:3]
P41=np.array(P0c4-O1)
P42=np.array(P0c4-O2)
P43=np.array(P0c4-O3)
P44=np.array(P0c4-O4)
JP4 = [[np.cross(Z1.flatten(), P41.flatten())], [np.cross(Z2.flatten(), P42.flatten())],[np.cross(Z3.flatten(), P43.flatten())],[np.cross(Z4.flatten(), P44.flatten())]]


PEE1=np.array(P0EE-O1)
PEE2=np.array(P0EE-O2)
PEE3=np.array(P0EE-O3)
PEE4=np.array(P0EE-O4)
JPEE = [[np.cross(Z1.flatten(), PEE1.flatten())], [np.cross(Z2.flatten(), PEE2.flatten())],[np.cross(Z3.flatten(), PEE3.flatten())],[np.cross(Z4.flatten(), PEE4.flatten())]]

print(JPEE)


# for any manipulator with 4 joints
import numpy as np

def forward_kinematics(joint_angles,link_lengths,a,alpha):
    """
    Computes the forward kinematics of a manipulator given the joint angles using DH parameters.
    
    Args:
        joint_angles (list): List of joint angles in radians.
        
    Returns:
        end_effector_pos (list): List containing the x, y, z coordinates of the end effector.
    """
    # Define the DH parameters
    # [alpha, a, d, theta]
    dh_parameters = [
        [alpha[0], a[0], d[0], joint_angles[0]],
        [alpha[1], a[1], d[1], joint_angles[1]],
        [alpha[2], a[2], d[2], joint_angles[2]],
        [alpha[3], a[3], d[3], joint_angles[3]]
    ]
    
    num_joints = len(joint_angles)
    
    # Transformation matrix representing the current pose of the end effector
    transformation_matrix = np.eye(4)
    
    for i in range(num_joints):
        # Extract the DH parameters for the current joint
        alpha, a, d, theta = dh_parameters[i]
        
        # Compute the transformation matrix for the current joint
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # Construct the transformation matrix for the current joint
        current_transformation = np.array([[cos_theta, -sin_theta*cos_alpha, sin_theta*sin_alpha, a*cos_theta],
                                           [sin_theta, cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
                                           [0, sin_alpha, cos_alpha, d],
                                           [0, 0, 0, 1]])
        
        # Update the overall transformation matrix
        transformation_matrix = np.dot(transformation_matrix, current_transformation)
    
    # Extract the end effector position from the transformation matrix
    T4EE = np.array([ [1, 0, 0, 0.126],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    T0EE = np.dot(transformation_matrix,T4EE)
    #Coordinates of end effector
    end_effector_pos = T0EE[0:3, 3:4]
    
    return end_effector_pos

# Example usage
joint_angles = [0.2, 0.4, 0.6, 0.8]  # Angles of the three joints
link_lengths = [1 , 2, 3, 4 ]
a = [0,0,0,0]
alpha = [0,0,0,0]

end_effector_position = forward_kinematics(joint_angles,link_lengths,a,alpha)
print("End Effector Position:", end_effector_position)

#inverse kinematics
end_effector_vel = [1, 1, 1]
def inverse_kinematics(end_effector_vel):

    dh_parameters = [
        [alpha[0], a[0], d[0], joint_angles[0]],
        [alpha[1], a[1], d[1], joint_angles[1]],
        [alpha[2], a[2], d[2], joint_angles[2]],
        [alpha[3], a[3], d[3], joint_angles[3]]
    ]
    q_dot = [0,0,0,0]
    num_joints = len(joint_angles)
    JacobianMatrix = np.zeros(6,num_joints)
    O = np.zeros(3,num_joints)
    Z = np.zeros(3,num_joints)
    R = np.zeros(3,3*num_joints)

    
    for i in range(num_joints):
        # Extract the DH parameters for the current joint
        alpha, a, d, theta = dh_parameters[i]
        
        # Compute the transformation matrix for the current joint
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        # Construct the transformation matrix for the current joint
        current_transformation = np.array([[cos_theta, -sin_theta*cos_alpha, sin_theta*sin_alpha, a*cos_theta],
                                           [sin_theta, cos_theta*cos_alpha, -cos_theta*sin_alpha, a*sin_theta],
                                           [0, sin_alpha, cos_alpha, d],
                                           [0, 0, 0, 1]])
        
        # Update the overall transformation matrix
        transformation_matrix = np.dot(transformation_matrix, current_transformation)
        
        
        O[3,i] =  transformation_matrix[0:3, 3:4]
        R[3,i*3+3] = transformation_matrix[0:3, 0:3]
        Z[3,i] = transformation_matrix[0:3, 2:3]
        rx1 = 1; ry1 = 0; rz1 =0                    #insert code for this
        roc1 = np.array([[rx1],[ry1],[rz1]])        #insert code for this
        P0 = O[3,i] + np.dot(R[3,i*3+3], roc1)

        
        for j in i:
            JacobianMatrix[3:6 , j] = Z
            P =np.array(P0-O[3,j])
            JacobianMatrix[0:3 , j] = np.cross(Z.flatten(), P.flatten()) 
        

        print(JacobianMatrix)


    #returns q_dot
    return(q_dot)


#     end_effector_vel = 0
#    def inverse_kinematics(end_effector_vel):
#    q_dot = [0,0,0,0]
    for i in range(num_joints): 
        JacobianMatrix[3:6 , i] = Z[:,i]
        P =np.array(PO[:,num_joints]-O[:,i])
        JacobianMatrix[0:3 , i] = np.cross(Z[:,i].flatten(), P.flatten())
        print(JacobianMatrix)
    return(q_dot)
 
 