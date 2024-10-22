# Import the needed modules
import math
import numpy as np
import general_robotics_toolbox as rox
import time
from Arm_Lib import Arm_Device
from scipy.spatial.transform import Rotation as R

Arm = Arm_Device()  # Get DOFBOT object
time.sleep(.2)  # Pause for a short time

# Define the Jacobian inverse 
def jacobian_inverse(robot, q0, Rd, Pd, Nmax, alpha, tol):
    # Set up storage space
    n = len(q0)
    q = np.zeros((n, Nmax + 1))
    q[:, 0] = q0
    p0T = np.zeros((3, Nmax + 1))
    RPY0T = np.zeros((3, Nmax + 1))
    iternum = 0

    # Compute the initial forward kinematics
    H = rox.fwdkin(robot, q[:, 0])
    R = H.R
    P = H.p
    P = np.array([[P[0]], [P[1]], [P[2]]])

    # Get the initial error
    dR = np.matmul(R, np.transpose(Rd))
    r = np.array(rox.R2rpy(dR))[None]
    dX = np.concatenate((np.transpose(r), P - Pd))

    # Iterate while any error element is greater than its tolerance
    while (np.absolute(dX) > tol).any():
        # Stop execution if the maximum number of iterations is exceeded
        if iternum < Nmax:
            # Compute the forward kinematics for the current iteration
            H = rox.fwdkin(robot, q[:, iternum])
            R = H.R
            p0T = H.p
            p0T = np.array([[p0T[0]], [p0T[1]], [p0T[2]]])

            # Compute the error
            dR = np.matmul(R, np.transpose(Rd))
            r = np.array(rox.R2rpy(dR))[None]
            dX = np.concatenate((np.transpose(r), p0T - Pd))

            # Calculate the Jacobian matrix
            Jq = rox.robotjacobian(robot, q[:, iternum])

            # Compute the update
            j = np.matmul(np.linalg.pinv(Jq), dX)

            # Update the joint angles
            q[:, iternum + 1] = q[:, iternum] - np.transpose(alpha * j)

            iternum += 1
        else:
            break

    # Return the final estimate of q
    return q[:, iternum]

def moveJoint(jnum, ang, speedtime):
    """
    Move the specified joint to the given position.
    """
    Arm.Arm_serial_servo_write(jnum, ang, speedtime)
    return

# Define the main function
def main():
    # Define inputs for jacobian_inverse function
    Rd = np.array([[0, 0, -1], 
                   [0, -1, 0], 
                   [-1, 0, 0]])
    Pd = np.array([[0], [0], [0.4]])

    # Make sure q0 is in radians
    q0 = np.array([25, 50, 75, 30, 30]) * math.pi / 180

    tol = np.array([0.02, 0.02, 0.02, 0.001, 0.001, 0.001])
    Nmax = 100000
    alpha = 0.1

    # Define all the joint lengths [m]
    l0 = 61 * 10**-3
    l1 = 43.5 * 10**-3
    l2 = 82.85 * 10**-3
    l3 = 82.85 * 10**-3
    l4 = 73.85 * 10**-3
    l5 = 54.57 * 10**-3

    # Define the unit vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    # Define the position vectors from i-1 -> i
    P01 = (l0 + l1) * ez
    P12 = np.zeros(3)
    P23 = l2 * ex
    P34 = -1 * l3 * ez
    P45 = np.zeros(3)
    P5T = -1 * (l4 + l5) * ex

    # Define the class inputs: rotation axes (H), position vectors (P), and joint_type
    H = np.array([ez, -1 * ey, -1 * ey, -1 * ey, -1 * ex]).T
    P = np.array([P01, P12, P23, P34, P45, P5T]).T
    joint_type = [0, 0, 0, 0, 0]

    # Define the Robot class
    robot = rox.Robot(H, P, joint_type)

    # Compute the inverse kinematics
    q = jacobian_inverse(robot, q0, Rd, Pd, Nmax, alpha, tol)

    # Convert solution to degrees
    q = q * 180 / math.pi
    q = q % 360

    # Move the robot joints
    for i in range(1, 6):
        moveJoint(i, q[i - 1], 100)
        time.sleep(0.2)

    # Print the final joint angles
    np.set_printoptions(suppress=True)
    print(" ".join([str(x) for x in np.round(q, 2)]))

# Execute the main function
if __name__ == "__main__":
    main()
