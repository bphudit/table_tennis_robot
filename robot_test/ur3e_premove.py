# Import all functions from the existing file
from ur3e_movej import *

def ping_pong_pre_move(x, y, z, a, v, t):
    """
    Function to handle pre-move for ping pong robot based on early prediction
    
    Args:
        x (float): X coordinate in mm
        y (float): Y coordinate in mm
        z (float): Z coordinate in mm
    """
    # Robot configuration is already defined in the imported module
    # We'll use the existing socket connection if available
    
    # Limit z coordinate to 40mm max
    if z > 40:
        z = 40
    
    # Rotation values
    rx = 1.240
    ry = 1.168
    rz = 1.306
    
    # Determine if forehand or backhand based on x coordinate
    if x < 0:
        # Backhand position (left side)
        print("Detected backhand position (x < 0)")
        position_name = "backhand"
    else:
        # Forehand position (right side)
        print("Detected forehand position (x >= 0)")
        position_name = "forehand"
    
    # Create the final pre-move position
    pre_position = [x, y, z, rx, ry, rz]
    
    # Create movement command
    pre_move_cmd = movej_command(joint_positions = pre_position, a=a, v=v, t=t,)
    
    
    # Connect to the robot (reuse existing connection from ur3e_movej if it exists)
    # The imported module doesn't initialize the connection automatically,
    # we need to create it here
    ROBOT_IP = "10.10.0.7"  # UR3e IP address
    ROBOT_PORT = 30002      # Secondary client interface port
    
    s = init_robot(ROBOT_IP, ROBOT_PORT)
    
    # Send the command
    send_ur_script(s, pre_move_cmd, close_socket=False)
    
    print(f"Pre-move command sent for {position_name} position")
    print(f"Coordinates: x={x}, y={y}, z={z}")
    
    # Note: We're not closing the connection so it can be reused
    # for the final precise movement later
    
    return s  # Return the socket connection for later use

if __name__ == "__main__":
    # Test the function with example coordinates
    test_x = float(input("Enter test x coordinate (mm): "))
    test_y = float(input("Enter test y coordinate (mm): "))
    test_z = float(input("Enter test z coordinate (mm): "))
    
    socket = ping_pong_pre_move(test_x, test_y, test_z)