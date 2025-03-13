import cv2
from scipy.io import loadmat
import numpy as np

data = loadmat('src/stereoParameters.mat')

# Intrinsic matrices (example values)
K1 = data['K1']

K2 = data['K2']

K1_inv = np.linalg.inv(K1)
K2_inv = np.linalg.inv(K2)

# Initialize a list to store clicked points
points = []
pixel2points = []

# Mouse callback function
def click_event(event, x, y, flags, param):
    global points, img
    # Check if the left mouse button was clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the point to the list
        points.append((x, y))
        
        # Draw a dot at the clicked point
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        
        # Put the coordinates text below the point
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"{x}, {y}"
        pixel2points.append(K2_inv @  np.array([x, y, 1]))
        cv2.putText(img, text, (x, y + 20), font, 0.5, (255, 0, 0), 1)
        
        # Show the updated image
        cv2.imshow("Image", img)
        print(pixel2points)

# Load your image (replace 'your_image.jpg' with your actual image file path)
img = cv2.imread("camera_test/right/img0.png")
# height, width, channels = img.shape

# scale_percent = 200
# new_width = int(width * scale_percent / 100)
# new_height = int(height * scale_percent / 100)
# dim = (new_width, new_height)
# img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

cv2.imshow("Image", img)

# Set the mouse callback function to handle clicks
cv2.setMouseCallback("Image", click_event)

# Wait indefinitely until a key is pressed
cv2.waitKey(0)

# Destroy all OpenCV windows
cv2.destroyAllWindows()

# np.savez("table_pose/right.npz", left_table_point = pixel2points)
np.savez("src/table_pose/right.npz", right_table_point = pixel2points)
