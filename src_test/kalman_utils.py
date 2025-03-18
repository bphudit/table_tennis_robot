import numpy as np
from kalmanEKF import KalmanFilter

def init_KF(measurements, restitution_coeff=0.8):
    """
    Initialize a Kalman filter for ball tracking.
    
    Args:
        measurements: Initial state measurements [x, y, z, vx, vy, vz]
        restitution_coeff: Coefficient of restitution for ball bounce
        
    Returns:
        Initialized KalmanFilter object
    """
    initial_state = measurements

    # Initial covariance matrix
    initial_cov = np.array([
        [2, 0, 0, 0, 0, 0],  # Variance for position x
        [0, 2, 0, 0, 0, 0],  # Variance for position y
        [0, 0, 5, 0, 0, 0],  # Variance for position z
        [0, 0, 0, 1, 0, 0],  # Variance for velocity x
        [0, 0, 0, 0, 1, 0],  # Variance for velocity y
        [0, 0, 0, 0, 0, 1]   # Variance for velocity z
    ])

    # Process noise covariance matrix (Q)
    # Lower values mean more trust in the model
    Q = np.array([
        [0.1, 0,   0,   0,   0,   0],   # Variance for Px
        [0,   0.1, 0,   0,   0,   0],   # Variance for Py
        [0,   0,   2,   0,   0,   0],   # Variance for Pz (more uncertainty)
        [0,   0,   0,   0.5, 0,   0],   # Variance for Vx
        [0,   0,   0,   0,   0.5, 0],   # Variance for Vy
        [0,   0,   0,   0,   0,   5]    # Variance for Vz (more uncertainty)
    ])

    # Measurement noise covariance matrix (R)
    R = np.array([
        [1.0, 0,   0,   0,   0,   0],   # Variance for measured Px
        [0,   1.0, 0,   0,   0,   0],   # Variance for measured Py
        [0,   0,   0.5, 0,   0,   0],   # Variance for measured Pz
        [0,   0,   0,   5.0, 0,   0],   # Variance for measured Vx
        [0,   0,   0,   0,   5.0, 0],   # Variance for measured Vy
        [0,   0,   0,   0,   0,   2.0]  # Variance for measured Vz
    ])

    kf = KalmanFilter(
        process_noise_std=Q,
        measurement_noise_std=R,
        initial_state=initial_state,
        initial_covariance=initial_cov,
        restitution_coeff=restitution_coeff
    )
    return kf

def calculate_velocity(positions, times):
    """
    Calculate velocities from position and time measurements.
    
    Args:
        positions: List of position measurements [[x1,y1,z1], [x2,y2,z2], ...]
        times: List of timestamps for each position
        
    Returns:
        Array of velocities [vx, vy, vz]
    """
    if len(positions) < 2:
        return np.zeros(3)
    
    dt = times[-1] - times[-2]
    dt = max(dt, 0.00001)  # Avoid division by zero
    
    # Calculate velocities as displacement divided by time
    velocity_x = (positions[-1][0] - positions[-2][0]) / dt
    velocity_y = (positions[-1][1] - positions[-2][1]) / dt
    velocity_z = (positions[-1][2] - positions[-2][2]) / dt
    
    return np.array([velocity_x, velocity_y, velocity_z])

def update_filter_with_measurement(kf, position, velocity, dt=None):
    """
    Update Kalman filter with new measurement.
    
    Args:
        kf: KalmanFilter object
        position: Position measurement [x, y, z]
        velocity: Velocity measurement [vx, vy, vz]
        dt: Time delta for prediction (if None, no prediction is made)
        
    Returns:
        Updated KalmanFilter object
    """
    measurement = np.array([
        [position[0]],
        [position[1]],
        [position[2]],
        [velocity[0]],
        [velocity[1]],
        [velocity[2]]
    ])
    
    kf.update(z=measurement)
    
    if dt is not None:
        kf.predict(dt)
        # Check for bounce and handle it
        if kf.x[2] <= 0:
            kf.handle_bounce()
    
    return kf

def predict_trajectory(kf, dt=0.01, until_y=0, max_steps=1000):
    """
    Predict the future trajectory of the ball.
    
    Args:
        kf: KalmanFilter object
        dt: Time step for prediction
        until_y: Predict until y coordinate reaches this value
        max_steps: Maximum number of prediction steps to avoid infinite loops
        
    Returns:
        predicted_positions: List of predicted positions
        final_state: Final state of the Kalman filter
    """
    # Create a copy of the filter to avoid modifying the original
    prediction_kf = KalmanFilter(
        process_noise_std=kf.Q,
        measurement_noise_std=kf.R,
        initial_state=kf.x,
        initial_covariance=kf.P,
        restitution_coeff=kf.restitution_coeff
    )
    
    predicted_x = []
    predicted_y = []
    predicted_z = []
    
    step = 0
    while prediction_kf.x[1] > until_y and step < max_steps:
        prediction_kf.predict(dt)
        
        if prediction_kf.x[2] <= 0:
            prediction_kf.handle_bounce()
        
        predicted_x.append(prediction_kf.x[0][0])
        predicted_y.append(prediction_kf.x[1][0])
        predicted_z.append(prediction_kf.x[2][0])
        
        step += 1
    
    return (predicted_x, predicted_y, predicted_z), prediction_kf.x

def calculate_hitting_position(kf_state, table_width=1525, offset_x=70, min_z=40):
    """
    Calculate the robot hitting position from the predicted state.
    
    Args:
        kf_state: Kalman filter state (final prediction)
        table_width: Width of the table in mm
        offset_x: X-axis offset for hitting position in mm
        min_z: Minimum z-height for hitting in mm
        
    Returns:
        robot_x: Robot x-axis hitting position
        robot_z: Robot z-axis hitting position
    """
    # Extract x and z from the Kalman filter state
    x = float(kf_state[0][0])
    z = float(kf_state[2][0] - 50)  # Apply offset
    
    # Transform to robot coordinates
    half_table = table_width / 2
    if x < half_table:
        robot_x = (half_table - x) + offset_x
    else:
        robot_x = (half_table - x) - offset_x
    
    # Ensure minimum z height
    robot_z = max(min_z, z)
    
    return robot_x, robot_z

def determine_hit_type(robot_x, robot_z, max_forehand=620, min_backhand=-620, min_z=40, max_z=300):
    """
    Determine the type of hit based on the robot position.
    
    Args:
        robot_x: Robot x position for hitting
        robot_z: Robot z position for hitting
        max_forehand: Maximum x for forehand hits
        min_backhand: Minimum x for backhand hits
        min_z: Minimum z for any hit
        max_z: Maximum z for any hit
        
    Returns:
        hit_type: "forehand", "backhand", or "none"
    """
    if 0 <= robot_x <= max_forehand and min_z <= robot_z <= max_z:
        return "forehand"
    elif min_backhand <= robot_x < 0 and min_z <= robot_z <= max_z:
        return "backhand"
    else:
        return "none" 