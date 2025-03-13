import URBasic
import math

# import URBasic.urScriptExt

def fore_hand(x, ur):
    target_pose_fore = [x/1000, 300/1000, 120/1000, 1.240, 1.168, 1.306]
    ur.movej(pose=target_pose_fore, a=0.5, v=0.5)

def stand_by(ur):
    target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]
    ur.movej(pose=target_pose_standby, a=0.5, v=0.5)

robot = URBasic.robotModel.RobotModel()

# Replace 'robot_ip' with your robot's IP address
robot_ip = '10.10.0.7'
ur = URBasic.urScriptExt.UrScriptExt(host=robot_ip, robotModel=robot)

# for i in range(10):
#     print('Hello')


stand_by(ur)

ur.close()
# target_pose_fore = [400/1000, 300/1000, 120/1000, 1.240, 1.168, 1.306]
# target_pose_back = [-60/1000, 200/1000, 120/1000, 1.724, -0.866, -1.767]
# target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]

# test_pose = [41/1000, -425/1000, 318/1000, 0.2, -1.587, -0.2]

# constant_to_radian = math.pi/180

# base = 92.22*constant_to_radian
# shoulder = -69.01*constant_to_radian
# elbow = 16.31*constant_to_radian
# w1 = -39.15*constant_to_radian
# w2 = -95.30*constant_to_radian
# w3 = 3.7*constant_to_radian

# home = [base, shoulder, elbow, w1, w2, w3]

# # ur.movej(pose=target_pose_back, a=0.5, v=0.5)

# base= 178*constant_to_radian
# shoulder=-100*constant_to_radian
# elbow=-139*constant_to_radian
# w1=-177*constant_to_radian
# w2=-3*constant_to_radian
# w3=30*constant_to_radian

# # ur.movej(q=home, a=0.5, v=0.5)
# # ur.movej(pose=target_pose_standby, a=0.5, v=0.5)
# # ur.movej(pose=target_pose_back, a=0.5, v=0.5)
# ur.movej(pose=target_pose_fore, a=0.5, v=0.5)
                          

# ur.close()









