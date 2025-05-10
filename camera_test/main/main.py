import time
import json
from camera_manager_instantarray import CameraManager
from serial_control import SerialController
from ball_detection import BallDetector
from visualization import FrameVisualizer
import numpy as np
from ball_triangulation_cv import BallTriangulatorP
# from ball_triangulation import BallTriangulator
from scipy.io import loadmat
from plot_3D import plot_3d
from kalman import init_KF
# from robot_control import movej_command, init_robot, send_ur_script
import copy
import sys
import os
import math
import threading

# Add parent directory to path to find URController
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from URController import URBasic

with open('hsv_values.json', 'r') as f:
    hsv_values = json.load(f)

stereo_params = loadmat('src/stereoParameters.mat')
pose_params = np.load('src/pose_parameters.npz')

def break_end_table(kf, result, pure_prediction_x, pure_prediction_y, pure_prediction_z, bounced, count):
    if result[0] > 2740:
        dt=0.01
        for i in range(1000):
            kf.predict(dt)
            print(kf.x[0])
            print(kf.x[1])
            print(kf.x[2])
            print(kf.x[3])
            print(kf.x[4])
            print(kf.x[5])
            print("--------------------------------")
            if bounced:
                count += 1
                if count >= 10:
                    count = 0
                    bounced = False

            if kf.x[2] <= 20:  
                if not bounced:
                    kf.handle_bounce()
                    bounced = True
                    
            pure_prediction_x.append(kf.x[0])
            pure_prediction_y.append(kf.x[1])
            pure_prediction_z.append(kf.x[2])
        



def main():
    # Initialize components
    serial_controller = SerialController(port='/dev/cu.usbserial-0001', baudrate=115200)
    camera_manager = CameraManager(exposure_time=6000, gain=12, mode=0)
    ball_detector = BallDetector(max_workers=2, 
                                lower_limit=np.array([hsv_values['h_min'], hsv_values['s_min'], hsv_values['v_min']]), 
                                upper_limit=np.array([hsv_values['h_max'], hsv_values['s_max'], hsv_values['v_max']]),
                                downsample_ratio=0.5)  # Use 50% resolution for faster processing
    visualizer = FrameVisualizer(scale_factor=0.5)
    ball_triangulator_cv = BallTriangulatorP(stereo_params, pose_params, coord_frame='table')
    # s = init_robot('10.10.0.7', 30002)

    robot_ip = '10.10.0.7'
    robot = URBasic.robotModel.RobotModel()
    ur = URBasic.urScriptExt.UrScriptExt(host=robot_ip, robotModel=robot)

    qnear = [0, -45, 45, -34, -327, 0]
    for i in range(len(qnear)):
        qnear[i] = qnear[i] * math.pi/180
    # time.sleep(1)
    ur.movej(q=ur.get_inverse_kin(x=[0, 0.4, 0.2, 0, 2.186, 2.225], qnear=qnear), a=2, v=2)


    position_3D = []

    initiated_KF = False
    bounced = False
    count = 0
    prediction_x = []
    prediction_y = []
    prediction_z = []
    pure_prediction_x = []
    pure_prediction_y = []
    pure_prediction_z = []

    robot_premove = False

    try:
        print("Press 'q' to quit")
        serial_controller.start_trigger()
        
        while True:
            # Get synchronized frames and measure processing time
            frame_start_time = time.time()
            img1, img2, bid, tms = camera_manager.get_synchronized_frames()

            # Process frames if available
            if img1 is not None and img2 is not None:
                # Detect balls in both frames
                img1, coords1, img2, coords2 = ball_detector.process_stereo_frames(img1, img2)
            
                # Triangulate ball position
                result = ball_triangulator_cv.process_frames_thread(coords1, coords2, tms)

                if result is not None:
                    position_3D.append(result)
                    print(result)
                    # print(1520//2 - result[1])

                    # if result[0] > 2740:
                    #     dt=0.001
                    #     for i in range(1000):
                    #         kf.predict(dt)
                    #         print(kf.x[0])
                    #         print(kf.x[1])
                    #         print(kf.x[2])
                    #         print(kf.x[3])
                    #         print(kf.x[4])
                    #         print(kf.x[5])
                    #         print("--------------------------------")
                    #         if bounced:
                    #             count += 1
                    #             if count >= 10:
                    #                 count = 0
                    #                 bounced = False

                    #         if kf.x[2] <= 20:  
                    #             if not bounced:
                    #                 kf.handle_bounce()
                    #                 bounced = True
                                    
                    #         pure_prediction_x.append(kf.x[0])
                    #         pure_prediction_y.append(kf.x[1])
                    #         pure_prediction_z.append(kf.x[2])
                    #     break
                        
                    

                    #Kalman Filter 
                    if not initiated_KF:
                        kf = init_KF([[result[1]], [result[0]], [-result[2]], [0], [0], [0]])
                        initiated_KF = True
                        last_tms = tms
                    else:
                        # if kf.x[1] >= 100 and not robot_premove:
                        #     dt = (2740 - kf.x[1])/kf.x[4]
                        #     print(dt, "--------------------------------")
                        #     kf2 = copy.deepcopy(kf)
                        #     kf2.predict(dt[0])
                        #     if kf2.x[0] > 1525//2:
                        #         positions = [-0.3, 0.3, 0.2, 1.206, -1.217, -1.187]
                        #     else:
                        #         positions = [0.3, 0.3, 0.2, 1.240, 1.168, 1.306]

                        #     print("Robot Pre-move")
                        #     ur.movej(ur.get_inverse_kin(x=positions, qnear=qnear),a=2, v=2, wait=True)
                        #     robot_premove = True

                        if kf.x[1] <= 2740//2:
                            dt = (tms - last_tms) * (10**-9)

                            kf.predict(dt)
                            if bounced:
                                count += 1
                                if count >= 10:
                                    count = 0
                                    bounced = False

                            if kf.x[2] <= 20:  
                                if not bounced:
                                    kf.handle_bounce()
                                    bounced = True
                                    print("Bounce")
                            
                            kf.update([[result[1]], [result[0]], [-result[2]]])
                            last_tms = tms

                            prediction_x.append(kf.x[0])
                            prediction_y.append(kf.x[1])
                            prediction_z.append(kf.x[2])
                        else:
                            # a = time.perf_counter()
                            dt=0.01
                            while kf.x[1] < 2740:
                                kf.predict(dt)
                                print(kf.x[0])
                                print(kf.x[1])
                                print(kf.x[2])
                                print(kf.x[3])
                                print(kf.x[4])
                                print(kf.x[5])
                                print("--------------------------------")
                                if bounced:
                                    count += 1
                                    if count >= 10:
                                        count = 0
                                        bounced = False

                                if kf.x[2] <= 20:  
                                    if not bounced:
                                        kf.handle_bounce()
                                        bounced = True
                                        
                                pure_prediction_x.append(kf.x[0])
                                pure_prediction_y.append(kf.x[1])
                                pure_prediction_z.append(kf.x[2])
                            # b = time.perf_counter()
                            # print(f"time: {b-a}")

                            # dt = (2740 - kf.x[1])/kf.x[4]
                            # kf.predict(dt)


                            x = float(1520//2 - kf.x[0])/1000
                            y = 300/1000

                            #Height less than 50mm
                            if kf.x[2] < 50:
                                z = 50/1000
                            elif kf.x[2] > 200:
                                z = 200/1000
                            else:
                                z = float(kf.x[2])/1000


                            print("Move target")
                            print(x, y, z)
                            if x < 0:
                                print("Wrong Side")
                                break
                            else:
                                ur.movej(ur.get_inverse_kin(x=[x, y, z, 1.240, 1.168, 1.306], qnear=qnear), a=100, v=100)
                                break

                            

                print("########################################################")
                
                # Calculate processing time
                frame_processing_time = (time.time() - frame_start_time) * 1000  # ms
                
                # Display frames with stats
                key = visualizer.display_frames(frame1=img1, frame2=img2, coords1=coords1, coords2=coords2, process_time_ms=frame_processing_time)
                
                # Check for quit key
                if key == ord('q'):
                    break
            else:
                # Handle synchronization errors
                print("No synchronized frames available")

    finally:
        # Clean up resources
        serial_controller.stop_trigger()
        camera_manager.close()
        ball_detector.close()
        visualizer.close()
        # ur.movej(q=ur.get_inverse_kin(x=[0, 0.4, 0.2, 0, 2.186, 2.225], qnear=qnear), a=0.5, v=0.5)
        # ur.close()
        print("All resources released")

        # dt=0.01
        # for i in range(1000):
        #     kf.predict(dt)
        #     print(kf.x[0])
        #     print(kf.x[1])
        #     print(kf.x[2])
        #     print(kf.x[3])
        #     print(kf.x[4])
        #     print(kf.x[5])
        #     print("--------------------------------")
        #     if bounced:
        #         count += 1
        #         if count >= 10:
        #             count = 0
        #             bounced = False

        #     if kf.x[2] <= 20:  
        #         if not bounced:
        #             kf.handle_bounce()
        #             bounced = True
                    
        #     pure_prediction_x.append(kf.x[0])
        #     pure_prediction_y.append(kf.x[1])
        #     pure_prediction_z.append(kf.x[2])

        position_3D = np.array(position_3D)
        # np.savez('/Users/phuditthaksinawong/clone_github_table_tennis/table_tennis_robot/camera_test/data/data_paddle.npz', position_3D=position_3D)
        
        # # Calculate and print standard deviation for x, y, z coordinates
        if len(position_3D) > 0:
            x_std = np.std(position_3D[:, 0])
            y_std = np.std(position_3D[:, 1])
            z_std = np.std(position_3D[:, 2])
            
            print(f"Standard deviation of x: {x_std:.6f} mm")
            print(f"Standard deviation of y: {y_std:.6f} mm")
            print(f"Standard deviation of z: {z_std:.6f} mm")
            
            # print(f"Max z val: {np.max(-position_3D[:,2])}")
            # print(f"Min z val: {np.min(-position_3D[:,2])}")
            plot_3d(position_3D[:,1], position_3D[:,0], -position_3D[:,2], prediction_x, prediction_y, prediction_z, pure_prediction_x, pure_prediction_y, pure_prediction_z)
        else:
            print("No position data collected.")

        

if __name__ == "__main__":
    main() 