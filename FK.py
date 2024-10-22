import time #import the time module. Used for adding pauses during operation
from Arm_Lib import Arm_Device #import the module associated with the arm
import numpy as np
from scipy.spatial.transform import Rotation as R

Arm = Arm_Device() # Get DOFBOT object
time.sleep(.2) #this pauses execution for the given number of seconds

def main(): #define the main program function
    speedtime = 100 #time in milliseconds to reach desired joint position
    #The print function is used to display helpful information to the console
    print("Input joint number 0 to stop program execution")
    q = readAllActualJointAngles() # read the current position of all joints
    print(q) #NOTE: any indices where q is and indicates the joint is outside its commandable range (<0 or >180)
  
    print("Setting Start Pose")
#     q = np.full((6,1), 90)
#     for i in range(len(ang_array)):
#         print(f"JNUM {i + 1}, ANG: {ang_array[i]}")
#         moveJoint(i + 1,ang_array[i],speedtime) #move the desired joint to the given angle
        
    while True: #keep executing the indented code until jnum=0
        jnum = getJointNumber() #use our defined function to get the joint number
        #if the joint number provided is 0, loop execution ends
        #if the joint number is not 0, we get the angle, move the joint, and read the angle
        if jnum == 0: 
            break
        else:
            ang = getJointAngle(jnum)   #use our defined function to get the joint angle
#             if jnum == 1:
#                 ang_array[0] = int(ang)
#             if jnum == 2:
#                 ang_array[1] = int(ang)
#             if jnum == 3:
#                 ang_array[2] = int(ang)
#             if jnum == 4:
#                 ang_array[3] = int(ang)
#             if jnum == 5:
#                 ang_array[4] = int(ang)
            moveJoint(jnum,ang,speedtime) #move the desired joint to the given angle
            time.sleep(1) #add a pause to allow time for joints to move
            angActual = readActualJointAngle(jnum) #read the actual position of the desired joint
            print("Actual joint angle:",angActual)
    print("Program has been terminated by user") #let the user know the program is no longer executing
    Rot, Pot = fk_Dofbot(q)
    print(Rot.as_matrix())
    print(Pot)


def fk_Dofbot (q):
    # FWDKIN_DOFBOT Computes the end effector position and orientation relative to the base frame for Yahboom's Dofbot manipulator 
    #     using the product of exponentials approach
    # Input :
    # q: 5x1 vector of joint angles in degrees
    #
    # Output :
    # Rot: The 3x3 rotation matrix describing the relative orientation of the end effector frame to the base frame (R_ {0T})
    # Pot: The 3x1 vector describing the position of the end effector relative to the base, 
    #      where the first element is the position along the base frame x-axis,
    #      the second element is the position along the base frame y-axis,
    #      and the third element is the position along the base frame z- axis (P_ {0T})
    
    print(f"Q IN FK FUNCTION: {q}")

    #set up the basis unit vectors
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])
    ez = np.array([0, 0, 1])

    # define the link lengths in meters
    l0 = 0.061 # base to servo 1
    l1 = 0.0435 # servo 1 to servo 2
    l2 = 0.08285 # servo 2 to servo 3
    l3 = 0.08285 # servo 3 to servo 4
    l4 = 0.07385 # servo 4 to servo 5
    l5 = 0.05457 # servo 5 to gripper

    #set up the rotation matrices between subsequent frames
    R01 = rotz(q[0]) # rotation between base frame and 1 frame
    R12 = roty(-q[1]) # rotation between 1 and 2 frames
    R23 = roty(-q[2]) # rotation between 2 and 3 frames
    R34 = roty(-q[3]) # rotation between 3 and 4 frames
    R45 = rotx(-q[4]) # rotation between 4 and 5 frames
    R5T = roty(0) #the tool frame is defined to be the same as frame 5

    #set up the position vectors between subsequent frames
    P01 = (l0+l1)*ez # translation between base frame and 1 frame in base frame
    P12 = np.zeros(3,) # translation between 1 and 2 frame in 1 frame
    P23 = l2*ex # translation between 2 and 3 frame in 2 frame
    P34 = -l3*ez # translation between 3 and 4 frame in 3 frame
    P45 = np.zeros(3,) # translation between 4 and 5 frame in 4 frame
    P5T = -(l4+l5)*ex # translation between 5 and tool frame in 5 frame

    # calculate Rot and Pot
    #Rot is a sequence of rotations
    Rot = R01*R12*R23*R34*R45*R5T
    #Pot is a combination of the position vectors. 
    #    Each vector must be represented in the base frame before addition. 
    #    This is achieved using the rotation matrices.
    Pot = P01 + R01.apply(P12 + R12.apply(P23 + R23.apply(P34 + R34.apply(P45 + R45.apply(P5T)))))

    return Rot, Pot






def rotx(theta):
    # return the principal axis rotation matrix for a rotation about the Xaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rx = R.from_euler('x',theta , degrees = True )
    return Rx

def roty(theta):
    # return the principal axis rotation matrix for a rotation about the Yaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Ry = R.from_euler('y',theta , degrees = True )
    return Ry

def rotz(theta):
    # return the principal axis rotation matrix for a rotation about the Zaxis by theta degrees
    if isinstance(theta,np.ndarray):
        theta = theta[0]
    Rz = R.from_euler('z',theta , degrees = True )
    return Rz

def getJointNumber():
    """
    function used to get the desired joint number using keyboard input
    getJointNumber() requests user input the desired joint number and returns joint number as an integer
    """
    jnum = int(input("Input joint number")) #ask the user to input a joint number. int converts the input to an integer
    print("Joint number: ",jnum) #print out the joint number that was read
    #if the joint number is not valid, keep prompting until a valid number is given
    if jnum<0 or jnum>6:
        while True:
            jnum = int(input("Input valid joint number [1,6]"))
            if jnum>=0 and jnum<=6:
                break
    return jnum #return the read value to the main function

def getJointAngle(jnum):
    """
    function used to get the desired joint angle using keyboard input
    getJointAngle() requests user input the desired joint angle in degrees and returns joint angle as an integer
    function needs to know the target joint (jnum) because joint 5 has a different angle range than the other joints
    """
    ang = int(input("Input angle (degrees)")) #ask the user to input a joint angle in degrees. int converts the input to an integer
    print("Joint angle: ",ang) #print out the joint angle that was read
    #if the joint angle is not valid, keep prompting until a valid number is given   
    if jnum != 5: #range for all joints except 5 is 0 to 180 degrees
        if ang<0 or ang>180:
            while True:
                ang = int(input("Input valid joint angle [0,180]"))
                if ang>=0 and ang<=180:
                    break
    else: #joint 5 range is 0 to 270 degrees
        if ang<0 or ang>270:
            while True:
                ang = int(input("Input valid joint angle [0,270]"))
                if ang>=0 and ang<=270:
                    break
    return ang #return the read value to the main function

def readActualJointAngle(jnum):
    """
    function used to read the position of the specified joint
    readActualJointAngle(jnum) reads the position of joint jnum in degrees
    function returns the joint position in degrees
    """
    # call the function to read the position of joint number jnum
    ang = Arm.Arm_serial_servo_read(jnum)
    return ang

#this cell provides two versions of a function to read all joint angles
import numpy as np #import module numpy, assign new name for module (np) for readability

# function to read and return all joint angles
# returns joint angles as a 1x6 numpy array
def readAllActualJointAngles():
    q = np.array([Arm.Arm_serial_servo_read(1),Arm.Arm_serial_servo_read(2),Arm.Arm_serial_servo_read(3),Arm.Arm_serial_servo_read(4),Arm.Arm_serial_servo_read(5),Arm.Arm_serial_servo_read(6)])
    return q

# second version of function to read and return all joint angles
# returns joint angles as a 6x1 numpy array
def readAllActualJointAngles2():    
    q = np.zeros((6,1)) #set up a 6x1 array placeholder
    for i in range(1,7): #loop through each joint (Note range(1,N) = 1,2,...,N-1)
        #note in Python the array indexing starts at 0 (the reason for i-1 index for q)
        q[i-1] = Arm.Arm_serial_servo_read(i) #store read angle into corresponding index of q
    return q

#execute the main loop unless the stop button is pressed to stop the kernel 
try:
    main()
except KeyboardInterrupt:
    print("Program closed!")
    pass

del Arm # release the arm object
