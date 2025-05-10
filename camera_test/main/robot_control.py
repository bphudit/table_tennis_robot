#!/usr/bin/env python3
import socket
import time
import math

def init_robot(robot_ip, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((robot_ip, port))
    print(f"Connected to {robot_ip}:{port}")

    return s

def send_ur_script(s, script, close_socket=False):
    """
    Send a script to the UR robot
    
    Args:
        s (socket): Socket object for the robot
        script (str): The UR script to send
        close_socket (bool): Whether to close the socket after sending
    """
    try:
        # Send the script
        s.send((script + "\n").encode())
        print("Command sent successfully")
        
        # Close the connection if requested
        if close_socket:
            s.close()
            print("Connection closed")
        
    except Exception as e:
        print(f"Error: {e}")

def close_connection(s):
    """
    Close the socket connection to the robot
    
    Args:
        s (socket): Socket object for the robot
    """
    try:
        s.close()
        print("Connection closed")
    except Exception as e:
        print(f"Error closing connection: {e}")

def movej_command(joint_positions=None, x=None, qnear=None,a=0.5, v=0.5, t=0, r=0, pos='joint'):
    """
    Create a movej command string
    
    Args:
        joint_positions (list): List of 6 joint positions in radians
        a (float): Joint acceleration in rad/s^2
        v (float): Joint speed in rad/s
        t (float): Time in seconds (overrides a and v if non-zero)
        r (float): Blend radius in meters
    
    Returns:
        str: The formatted movej command
    """

    # for i in range(len(joint_positions)):
    #     joint_positions[i] = joint_positions[i] * math.pi/180
    if pos == 'joint':
        for i in range(len(joint_positions)):
            joint_positions[i] = joint_positions[i] * math.pi/180
        if joint_positions is not None:
            return f"movej({joint_positions}, a={a}, v={v}, t={t}, r={r})"
        else:
            print("No joint positions provided")
            return None
    elif pos == 'pose':
        for i in range(len(x)-3):
            if i < 3:
                x[i] = x[i] / 1000
        
        if qnear is not None:
            qnear_copy = qnear.copy()
            for i in range(len(qnear_copy)):
                qnear_copy[i] = qnear_copy[i] * math.pi/180
            return f"movej(get_inverse_kin(x = p{x}, qnear = {qnear_copy}), a={a}, v={v}, t={t}, r={r})"
        else:
            return f"movej(get_inverse_kin(p{x}), a={a}, v={v}, t={t}, r={r})"
    else:
        print("Invalid position type")
        return None






if __name__ == "__main__":
    # Robot configuration
    ROBOT_IP = "10.10.0.7"  # UR3e IP address
    ROBOT_PORT = 30002      # Secondary client interface port

    s = init_robot(ROBOT_IP, ROBOT_PORT)
    
    # Example joint positions (in radians)
    # Format: [base, shoulder, elbow, wrist1, wrist2, wrist3]
    # joint_positions = [-94, -35, 0, 34, -273,328]  
    # command = movej_command(pos='joint', joint_positions=joint_positions)

    commands = []

    qnear = [0, -45, 45, -34, -327, 0]  
    # Send first command - don't close socket yet
    positions_standby = [0, 400, 200, 0, 2.186, 2.225]
    # commands.append(movej_command(pos='pose', x=positions_standby, qnear=qnear))

    # qnear = [0, -45, 45, -34, -327, 0]  
    # Send second command - don't close socket yet
    positions_fore = [200, 300, 200, 1.240, 1.168, 1.306]
    # commands.append(movej_command(pos='pose', x=positions_fore, qnear=qnear))
    
    # qnear = [0, -45, 45, -34, -327, 0]  
    # Send third command - don't close socket yet 
    positions_back = [-300, 300, 100, 1.206, -1.217, -1.187]
    # commands.append(movej_command(pos='pose', x=positions_back, qnear=qnear))



#     script = f"""
# def sequentmove():
#     {commands[1]}
#     sync()
# end
    
# sequentmove()
# """
    script = movej_command(pos='pose', x=positions_standby, qnear=qnear)
    # script = movej_command(pos='pose', x=positions_back, qnear=qnear)
    print(script)
    input('Enter to send command')

    send_ur_script(s, script, close_socket=True)

    print("Program completed") 