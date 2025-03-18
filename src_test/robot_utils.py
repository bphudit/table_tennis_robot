import URController.URBasic as URBasic

class RobotController:
    """
    Universal Robots controller class for table tennis robot.
    Provides methods to move the robot to different positions for hitting the ball.
    """
    def __init__(self, robot_ip='10.10.0.7'):
        """
        Initialize robot controller.
        
        Args:
            robot_ip: IP address of the UR robot
        """
        self.robot = URBasic.robotModel.RobotModel()
        self.ur = URBasic.urScriptExt.UrScriptExt(host=robot_ip, robotModel=self.robot)
        self.is_connected = True
        
    def fore_hand(self, x, z, accel=100, velocity=100):
        """
        Move robot to forehand hitting position.
        
        Args:
            x: X-coordinate in mm
            z: Z-coordinate (height) in mm
            accel: Acceleration parameter
            velocity: Velocity parameter
        
        Returns:
            True if move executed, False otherwise
        """
        if not self.is_connected:
            print("Robot not connected")
            return False
            
        if x >= 0:
            z = max(40, z)  # Ensure minimum height
            # Convert mm to m for robot
            target_pose_fore = [x/1000, 300/1000, z/1000, 1.240, 1.168, 1.306]
            self.ur.movej(pose=target_pose_fore, a=accel, v=velocity)
            return True
        else:
            print("X IS LESS THAN 0 - Cannot execute forehand")
            return False

    def stand_by(self, accel=2, velocity=2):
        """
        Move robot to standby position.
        
        Args:
            accel: Acceleration parameter
            velocity: Velocity parameter
            
        Returns:
            True if move executed, False otherwise
        """
        if not self.is_connected:
            print("Robot not connected")
            return False
            
        target_pose_standby = [91/1000, 625/1000, 418/1000, 0.125, -1.71, -2.639]
        self.ur.movej(pose=target_pose_standby, a=accel, v=velocity)
        return True

    def back_hand(self, x, z, accel=100, velocity=100):
        """
        Move robot to backhand hitting position.
        
        Args:
            x: X-coordinate in mm
            z: Z-coordinate (height) in mm
            accel: Acceleration parameter
            velocity: Velocity parameter
            
        Returns:
            True if move executed, False otherwise
        """
        if not self.is_connected:
            print("Robot not connected")
            return False
            
        if x < 0:
            z = max(40, z)  # Ensure minimum height
            # Convert mm to m for robot
            target_pose_back = [x/1000, 300/1000, z/1000, 1.206, -1.217, -1.187]
            self.ur.movej(pose=target_pose_back, a=accel, v=velocity)
            return True
        else:
            print('X IS MORE THAN 0 - Cannot execute backhand')
            return False
    
    def execute_hit(self, hit_type, x, z, accel=100, velocity=100):
        """
        Execute a hit based on the hit type.
        
        Args:
            hit_type: Type of hit ("forehand", "backhand", or other)
            x: X-coordinate in mm
            z: Z-coordinate (height) in mm
            accel: Acceleration parameter
            velocity: Velocity parameter
            
        Returns:
            True if move executed, False otherwise
        """
        if hit_type.lower() == "forehand":
            return self.fore_hand(x, z, accel, velocity)
        elif hit_type.lower() == "backhand":
            return self.back_hand(x, z, accel, velocity)
        else:
            print(f"Unknown hit type: {hit_type}")
            return False
            
    def move_to_position(self, position, orientation=None, accel=2, velocity=2):
        """
        Move robot to a specific position and orientation.
        
        Args:
            position: [x, y, z] position in meters
            orientation: [rx, ry, rz] orientation or None to use default
            accel: Acceleration parameter
            velocity: Velocity parameter
            
        Returns:
            True if move executed, False otherwise
        """
        if not self.is_connected:
            print("Robot not connected")
            return False
            
        if orientation is None:
            # Default orientation (may need adjustment based on your robot)
            orientation = [1.240, 1.168, 1.306]
            
        pose = list(position) + list(orientation)
        self.ur.movej(pose=pose, a=accel, v=velocity)
        return True
    
    def close(self):
        """
        Close the robot connection.
        """
        if self.is_connected:
            self.ur.close()
            self.is_connected = False
            
    def __del__(self):
        """
        Destructor to ensure robot connection is closed.
        """
        self.close()
        
# Example usage:
"""
robot = RobotController(robot_ip='10.10.0.7')

try:
    # Move to standby position
    robot.stand_by()
    
    # Example hitting positions
    robot.fore_hand(x=200, z=100)  # Forehand hit
    robot.back_hand(x=-200, z=100)  # Backhand hit
    
    # Move back to standby
    robot.stand_by()
    
finally:
    # Always close the connection
    robot.close()
""" 