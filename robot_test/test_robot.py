import URController.URBasic as URBasic
import math
import time
# import URBasic.urScriptExt

def fore_hand(x,z, ur):
    if x >= 0:
        if z < 40:
            z = 40
        else:
            z = z
        target_pose_fore = [x/1000, 300/1000, z/1000, 1.240, 1.168, 1.306]
        ur.movej(pose=target_pose_fore, a=100, v=100)
    else:
        print("X IS LESS THAN 0")

def stand_by(ur):
    target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]
    ur.movej(pose=target_pose_standby, a=2, v=2)

def back_hand(x,z, ur):
    if x < 0 :
        if z < 40:
            z = 40
        else:
            z = z
        target_pose_back = [x/1000, 300/1000,z/1000, 1.206, -1.217, -1.187]
        ur.movej(pose=target_pose_back, a=20, v=20)
    else:
        print('X IS MORE THAN 0')

robot = URBasic.robotModel.RobotModel()

# Replace 'robot_ip' with your robot's IP address
robot_ip = '10.10.0.7'
ur = URBasic.urScriptExt.UrScriptExt(host=robot_ip, robotModel=robot)
# for i in range(10):
#     print('Hello')



# target_pose_fore = [400/1000, 300/1000, 120/1000, 1.240, 1.168, 1.306]
# target_pose_back = [-60/1000, 200/1000, 120/1000, 1.724, -0.866, -1.767]
# target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]

# test_pose = [41/1000, -425/1000, 318/1000, 0.2, -1.587, -0.2]

constant_to_radian = math.pi/180

base = 92.22*constant_to_radian
shoulder = -69.01*constant_to_radian
elbow = 16.31*constant_to_radian
w1 = -39.15*constant_to_radian
w2 = -95.30*constant_to_radian
w3 = 3.7*constant_to_radian

home = [base, shoulder, elbow, w1, w2, w3]

# # ur.movej(pose=target_pose_back, a=0.5, v=0.5)

base= 178*constant_to_radian
shoulder=-100*constant_to_radian
elbow=-139*constant_to_radian
w1=-177*constant_to_radian
w2=-3*constant_to_radian
w3=30*constant_to_radian

# ur.movej(q=home, a=0.5, v=0.5)

# stand_by(ur)
stand_by(ur)
input('enter')
a = time.perf_counter()
fore_hand(300,100, ur)
b = time.perf_counter()
print(b-a)
# back_hand(-620,200, ur)



ur.close()

# # ur.movej(pose=target_pose_standby, a=0.5, v=0.5)
# # ur.movej(pose=target_pose_back, a=0.5, v=0.5)
# ur.movej(pose=target_pose_fore, a=0.5, v=0.5)
                          

# ur.close()