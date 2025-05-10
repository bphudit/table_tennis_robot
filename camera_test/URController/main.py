import URBasic
import math
import threading
import time
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

position_standby = [0, 400/1000, 200/1000, 0, 2.186, 2.225]
qnear = [0, -45*math.pi/180, 45*math.pi/180, -34*math.pi/180, -327*math.pi/180, 0*math.pi/180]

home = [0, -90*math.pi/180, 0, -90*math.pi/180, 0, 0]

try:
    print("Home")
    ur.movej(q=home, a=2, v=2)

    print("Standby")

    
    ur.movej(q=ur.get_inverse_kin(x=position_standby, qnear = qnear), a=20, v=20, wait=False)


    # while not all(abs(a - b) < 0.01 for a, b in zip(ur.get_actual_tcp_pose(), position_standby)):
    #     print("Waiting for robot to reach target position...")
    #     time.sleep(0.5)
    time.sleep(0.5)

    print("Home")
    ur.movej(q=home, a=20, v=20)

    # while not all(abs(a - b) < 0.01 for a, b in zip(ur.get_actual_joint_positions(), home)):
    #     print("Waiting for robot to reach target position...")
    #     time.sleep(0.1)

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("Closing connection to robot")
    ur.close()









