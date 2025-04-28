import cv2
import numpy as np
import time
from pypylon import pylon
from main.camera_manager_instantarray import CameraManager
from main.serial_control import SerialController


def main():
    serial_controller = SerialController(port='/dev/cu.usbserial-0001', baudrate=115200)
    camera_manager = CameraManager(exposure_time=6000, gain=15, mode=0)
    num = 0

    try:
        serial_controller.start_trigger()
        while True:
            img1, img2, block_id1, block_id2 = camera_manager.get_synchronized_frames()

            # Process frames if available
            if img1 is not None and img2 is not None:
                # Get the current frame size
                height, width, _ = img1.shape

                # print(height, width)

                # Resize the frame
                scale_percent = 70
                new_width = int(width * scale_percent / 100)
                new_height = int(height * scale_percent / 100)
                dim = (new_width, new_height)
                resized_left = cv2.resize(img1, dim, interpolation=cv2.INTER_AREA)
                resized_right = cv2.resize(img2, dim, interpolation=cv2.INTER_AREA)

                # Concatenate images horizontally (side by side)
                combined_img = np.hstack((resized_left, resized_right))
                
                # Add labels for left and right cameras
                cv2.putText(combined_img, "Left Camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(combined_img, "Right Camera", (new_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display the combined image in a single window
                cv2.imshow('Stereo Camera View', combined_img)

                k = cv2.waitKey(5)

                if k == 27:  # ESC key
                    break
                elif k == ord('s'):
                    cv2.imwrite('camera_test/right/img' + str(num) + '.png', img2)
                    cv2.imwrite('camera_test/left/img' + str(num) + '.png', img1)
                    print("image saved!")
                    num += 1
                elif k == ord('q'):
                    break
        
    except KeyboardInterrupt:
        print("\nStopping cameras...")
    finally:
        # Clean up resources
        serial_controller.stop_trigger()
        camera_manager.close()
        print("All resources released")

if __name__ == "__main__":
    main()