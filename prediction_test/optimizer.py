import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
from mpl_toolkits.mplot3d import Axes3D
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the necessary classes from the provided code
# Assuming BounceTrajectoryEvaluator is available from trajectory_evaluator
# and KalmanFilter is available from kalmanEKF
from trajectory_evaluator import *
from src.kalmanEKF import KalmanFilter

def visualize_trajectory_evolution(data_file, table_boundaries=None, step_interval=10, save_plots=False, output_dir=None):
    """
    Visualize how the Kalman filter predictions evolve as more data points are added,
    comparing with the ground truth trajectory at each step.
    
    Args:
        data_file: Path to the .npz file with trajectory data
        table_boundaries: Dictionary with table boundaries {x_min, x_max, y_min, y_max}
        step_interval: How many steps to skip between visualizations (smaller = more plots)
        save_plots: Whether to save plots instead of showing them
        output_dir: Directory to save plots if save_plots is True
        
    Returns:
        Dictionary with results and visualization data
    """
    # Create output directory if saving plots
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = np.load(data_file)
    position_3D = data['position_3D']
    
    print("Step 1: Fitting the ground truth trajectory with the evaluator...")
    # Create the evaluator for ground truth
    evaluator = BounceTrajectoryEvaluator(
        position_3D[:,3],  # Time 
        position_3D[:,1],  # X
        position_3D[:,0],  # Y
        -1 * position_3D[:,2],  # Z (inverted)
        sort_timestamps=True,
        table_boundaries=table_boundaries
    )
    
    # Detect bounces and fit parabolic segments
    evaluator.detect_bounces()
    evaluator.fit_segments()
    
    # Generate the ground truth fitted curves for visualization
    ground_truth_segments = evaluator.generate_fitted_curves(points_per_segment=1000)
    
    # Combine segments into a single ground truth curve
    ground_truth_curve = np.vstack([segment[:, 1:4] for segment in ground_truth_segments])
    ground_truth_times = np.vstack([segment[:, 0:1] for segment in ground_truth_segments])
    
    # Sort ground truth by time
    sort_indices = np.argsort(ground_truth_times.flatten())
    ground_truth_curve = ground_truth_curve[sort_indices]
    ground_truth_times = ground_truth_times[sort_indices]
    
    print("Step 2: Preparing data for Kalman filtering...")
    # Prepare data for Kalman filter
    position_3D_processed = np.array([[pos[1], pos[0], -pos[2], pos[3]] for pos in data['position_3D']])
    
    # Calculate time differences and measurements
    measurements = []
    dt_ls = []
    
    for i in range(1, len(position_3D_processed[:,-1])):
        dt = (position_3D_processed[i][-1] - position_3D_processed[i-1][-1]) * (10**-9)
        dt_ls.append(dt)
        measurements.append((position_3D_processed[i][0], position_3D_processed[i][1], position_3D_processed[i][2]))
    
    measurements = np.array(measurements)
    
    # Initial state for Kalman filter
    initial_state = [[position_3D_processed[0][0]], 
                     [position_3D_processed[0][1]],
                     [position_3D_processed[0][2]],
                     [0],
                     [0],
                     [0]]
    
    # Kalman filter parameters
    initial_cov = np.array([[5, 0, 0, 0, 0, 0],
                            [0, 5, 0, 0, 0, 0],
                            [0, 0, 5, 0, 0, 0],
                            [0, 0, 0, 1000, 0, 0],
                            [0, 0, 0, 0, 1000, 0],
                            [0, 0, 0, 0, 0, 1000]])
    
    # Process Noise
    Q = np.array([
        [1,    0,    0,    0,    0,    0],   # Variance for Px
        [0,    1,    0,    0,    0,    0],   # Variance for Py
        [0,    0,    1,    0,    0,    0],   # Variance for Pz
        [0,    0,    0,    1000,    0,    0],   # Variance for Vx
        [0,    0,    0,    0,    1000,    0],   # Variance for Vy
        [0,    0,    0,    0,    0,    1000]    # Variance for Vz
    ])
    
    # Measurement Noise
    R = np.array([
        [5**2,  0,     0],   # Variance for measured Px
        [0,    5**2,   0],   # Variance for measured Py
        [0,    0,   5**2]    # Variance for measured Pz
    ])
    
    # Function to find the ground truth position at a specific time
    def find_ground_truth_at_time(time_point):
        # Find the segment that contains this time point
        for segment in evaluator.bounce_segments:
            if segment['t_min'] <= time_point <= segment['t_max']:
                # Get parameters for this segment
                ax, bx, cx = segment['x_params']
                ay, by, cy = segment['y_params']
                az, bz, cz = segment['z_params']
                
                # Calculate shifted time
                t_shifted = time_point - segment['t_min']
                
                # Calculate ground truth position using the parabolic model
                x = evaluator.parabolic_motion(t_shifted, ax, bx, cx)
                y = evaluator.parabolic_motion(t_shifted, ay, by, cy)
                z = evaluator.parabolic_motion(t_shifted, az, bz, cz)
                
                return np.array([x, y, z])
        
        # If no segment contains this time, return None
        return None
    
    # Function to make pure predictions from a given state
    def make_pure_predictions(kf_state, dt_ls, start_idx):
        # Initialize Kalman filter with the state at this point
        kf_prediction = KalmanFilter(
            process_noise_std=Q,
            measurement_noise_std=R,
            initial_state=[[kf_state['x'][0][0]], [kf_state['x'][1][0]], [kf_state['x'][2][0]], 
                         [kf_state['x'][3][0]], [kf_state['x'][4][0]], [kf_state['x'][5][0]]],
            initial_covariance=kf_state['P'],
            restitution_coeff=0.9, 
            drag_coeff=0.5
        )
        
        # Pure prediction from this point forward
        predicted_positions = []
        
        bounced = False
        count = 0
        
        # Make predictions for remaining points
        for i in range(start_idx, len(measurements)):
            # Predict next state
            kf_prediction.predict(dt_ls[i-1] if i > 0 else dt_ls[0])
            
            # Handle bounce if needed
            if kf_prediction.x[2] <= 20 and not bounced:
                kf_prediction.handle_bounce()
                bounced = True
                count = 0
            
            if bounced:
                count += 1
                if count >= 10:
                    count = 0
                    bounced = False
            
            # Store prediction
            predicted_pos = np.array([kf_prediction.x[0][0], kf_prediction.x[1][0], kf_prediction.x[2][0]])
            predicted_positions.append(predicted_pos)
        
        return np.array(predicted_positions) if predicted_positions else np.array([])
    
    # Run Kalman filter through all measurements and track evolution
    print("Step 3: Running Kalman filter and tracking evolution...")
    
    # Initialize Kalman filter
    kf = KalmanFilter(
        process_noise_std=Q,
        measurement_noise_std=R,
        initial_state=initial_state, 
        initial_covariance=initial_cov, 
        restitution_coeff=0.9, 
        drag_coeff=0.5
    )
    
    # Store Kalman states at each step
    kalman_states = []
    
    # Initialize with the first point
    kalman_states.append({
        'x': initial_state,
        'P': initial_cov,
        'time': evaluator.T_shifted[0] if len(evaluator.T_shifted) > 0 else 0,
        'measurement': position_3D_processed[0][:3],
        'updated_path': [position_3D_processed[0][:3]],  # Track the path with updates
        'step': 0
    })
    
    # Track MSE at each step
    mse_evolution = []
    
    # Process measurements and visualize at intervals
    bounced = False
    count = 0
    
    print(f"Creating visualizations at every {step_interval} steps...")
    for i in range(len(measurements)):
        # Predict next state
        kf.predict(dt_ls[i])
        
        # Handle bounce if needed
        if kf.x[2] <= 20 and not bounced:
            kf.handle_bounce()
            bounced = True
            count = 0
        
        if bounced:
            count += 1
            if count >= 10:
                count = 0
                bounced = False
        
        # Update with measurement
        kf.update(z=np.array([[measurements[i][0]], 
                              [measurements[i][1]],
                              [measurements[i][2]]]))
        
        # Get the updated path so far
        updated_path = [state['measurement'] for state in kalman_states]
        updated_path.append(measurements[i])
        
        # Store state
        current_time = evaluator.T_shifted[i+1] if i+1 < len(evaluator.T_shifted) else evaluator.T_shifted[-1]
        
        kalman_states.append({
            'x': kf.x.copy(),
            'P': kf.P.copy(),
            'time': current_time,
            'measurement': measurements[i],
            'updated_path': updated_path,
            'step': i+1
        })
        
        # Make pure predictions from this point
        pure_predictions = make_pure_predictions(kalman_states[-1], dt_ls, i+1)
        
        # Calculate MSE between predictions and ground truth
        mse_total = float('inf')
        
        if len(pure_predictions) > 0:
            future_errors = []
            
            for j in range(min(len(pure_predictions), len(measurements) - (i+1))):
                pred_idx = i+1+j
                if pred_idx < len(evaluator.T_shifted):
                    gt_pos = find_ground_truth_at_time(evaluator.T_shifted[pred_idx])
                    
                    if gt_pos is not None and j < len(pure_predictions):
                        error = np.sum((pure_predictions[j] - gt_pos)**2)
                        future_errors.append(error)
            
            if future_errors:
                mse_total = np.mean(future_errors)
        
        mse_evolution.append({
            'step': i+1,
            'mse': mse_total
        })
        
        # Visualize at specified intervals
        if (i+1) % step_interval == 0 or i == len(measurements) - 1:
            print(f"Creating visualization for step {i+1}/{len(measurements)}")
            
            # Create figure
            fig = plt.figure(figsize=(15, 10))
            
            # Plot 3D trajectory
            ax = fig.add_subplot(2, 2, 1, projection='3d')
            
            # Get all points up to current step
            current_points = position_3D_processed[:i+2, :3]  # Include current point
            
            # Plot raw measurement points
            ax.scatter(current_points[:,0], current_points[:,1], current_points[:,2], 
                      c='red', s=30, label='Measured Points')
            
            # Plot Kalman filtered path
            updated_path = np.array(kalman_states[-1]['updated_path'])
            ax.scatter(updated_path[:,0], updated_path[:,1], updated_path[:,2], 
                      c='blue', s=30, label='KF Updated Path')
            
            # Plot pure predictions
            if len(pure_predictions) > 0:
                ax.scatter(pure_predictions[:,0], pure_predictions[:,1], pure_predictions[:,2], 
                         c='orange', s=30, label='KF Pure Prediction')
            
            # Plot ground truth curve
            for segment in ground_truth_segments:
                ax.plot(segment[:,1], segment[:,2], segment[:,3], 'g-', linewidth=2)
            
            # Add a legend point for ground truth
            ax.scatter([], [], [], c='green', s=30, label='Ground Truth')
            
            # Set labels and title
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.set_zlabel('Z [mm]')
            ax.set_title(f'3D Trajectory Evolution (Step {i+1})')
            
            # Add table boundaries if provided
            if table_boundaries:
                x_min, x_max = table_boundaries['x_min'], table_boundaries['x_max']
                y_min, y_max = table_boundaries['y_min'], table_boundaries['y_max']
                z_table = 0
                
                # Table outline
                ax.plot([x_min, x_max, x_max, x_min, x_min],
                        [y_min, y_min, y_max, y_max, y_min],
                        [z_table, z_table, z_table, z_table, z_table],
                        'k-', linewidth=1.0, alpha=0.5)
            
            # Set axis limits based on table boundaries or data
            if table_boundaries:
                ax.set_xlim([0, table_boundaries['x_max']])
                ax.set_ylim([0, table_boundaries['y_max']])
                ax.set_zlim([0, 1500])  # Appropriate height for table tennis
            else:
                # Add some margin around data
                max_vals = np.max(position_3D_processed[:, :3], axis=0)
                min_vals = np.min(position_3D_processed[:, :3], axis=0)
                margin = (max_vals - min_vals) * 0.1
                
                ax.set_xlim([min_vals[0] - margin[0], max_vals[0] + margin[0]])
                ax.set_ylim([min_vals[1] - margin[1], max_vals[1] + margin[1]])
                ax.set_zlim([min_vals[2] - margin[2], max_vals[2] + margin[2]])
            
            ax.legend()
            
            # Plot XY projection
            ax = fig.add_subplot(2, 2, 2)
            
            # Plot measured points
            ax.scatter(current_points[:,0], current_points[:,1], c='red', s=30, label='Measured Points')
            
            # Plot Kalman filtered path
            ax.scatter(updated_path[:,0], updated_path[:,1], c='blue', s=30, label='KF Updated Path')
            
            # Plot pure predictions
            if len(pure_predictions) > 0:
                ax.scatter(pure_predictions[:,0], pure_predictions[:,1], c='orange', s=30, label='KF Pure Prediction')
            
            # Plot ground truth XY projection
            for segment in ground_truth_segments:
                ax.plot(segment[:,1], segment[:,2], 'g-', linewidth=2)
            
            # Set labels and title
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Y [mm]')
            ax.set_title(f'XY Projection (Step {i+1})')
            
            # Add table boundaries
            if table_boundaries:
                ax.plot([x_min, x_max, x_max, x_min, x_min],
                        [y_min, y_min, y_max, y_max, y_min],
                        'k-', linewidth=1.0, alpha=0.5)
                
                ax.set_xlim([0, table_boundaries['x_max']])
                ax.set_ylim([0, table_boundaries['y_max']])
            
            ax.legend()
            ax.grid(True)
            
            # Plot MSE evolution
            ax = fig.add_subplot(2, 2, 3)
            
            # Get MSE values up to current step
            steps = [res['step'] for res in mse_evolution[:i+1]]
            mse_values = [res['mse'] for res in mse_evolution[:i+1]]
            
            # Filter out inf values for plotting
            valid_indices = [j for j, val in enumerate(mse_values) if not np.isinf(val)]
            valid_steps = [steps[j] for j in valid_indices]
            valid_mse = [mse_values[j] for j in valid_indices]
            
            if valid_steps:
                ax.plot(valid_steps, valid_mse, 'b-o')
                ax.axvline(x=i+1, color='r', linestyle='--')
                
                # Find minimum MSE
                min_idx = np.argmin(valid_mse) if valid_mse else 0
                if valid_steps:
                    ax.axvline(x=valid_steps[min_idx], color='g', linestyle='--')
                    
                    # Add annotation for current and min MSE
                    current_mse_val = mse_values[i]
                    # Fixed: Check if current_mse is numeric before using isinf
                    if isinstance(current_mse_val, (int, float)) and not np.isinf(current_mse_val):
                        ax.annotate(f"Current MSE: {current_mse_val:.2f}", 
                                   xy=(i+1, current_mse_val),
                                   xytext=(i+1+5, current_mse_val),
                                   arrowprops=dict(facecolor='red', shrink=0.05))
                    
                    min_mse = valid_mse[min_idx]
                    ax.annotate(f"Min MSE: {min_mse:.2f}", 
                               xy=(valid_steps[min_idx], min_mse),
                               xytext=(valid_steps[min_idx]+5, min_mse),
                               arrowprops=dict(facecolor='green', shrink=0.05))
            
            ax.set_xlabel('Step')
            ax.set_ylabel('MSE')
            ax.set_title('Prediction Error Evolution')
            ax.grid(True)
            
            # Plot XZ projection
            ax = fig.add_subplot(2, 2, 4)
            
            # Plot measured points
            ax.scatter(current_points[:,0], current_points[:,2], c='red', s=30, label='Measured Points')
            
            # Plot Kalman filtered path
            ax.scatter(updated_path[:,0], updated_path[:,2], c='blue', s=30, label='KF Updated Path')
            
            # Plot pure predictions
            if len(pure_predictions) > 0:
                ax.scatter(pure_predictions[:,0], pure_predictions[:,2], c='orange', s=30, label='KF Pure Prediction')
            
            # Plot ground truth XZ projection
            for segment in ground_truth_segments:
                ax.plot(segment[:,1], segment[:,3], 'g-', linewidth=2)
            
            # Set labels and title
            ax.set_xlabel('X [mm]')
            ax.set_ylabel('Z [mm]')
            ax.set_title(f'XZ Projection (Step {i+1})')
            
            # Add table boundaries (just the edge in XZ view)
            if table_boundaries:
                ax.plot([x_min, x_max], [z_table, z_table], 'k-', linewidth=1.0, alpha=0.5)
                
                ax.set_xlim([0, table_boundaries['x_max']])
                ax.set_ylim([0, 1500])  # Appropriate height for table tennis
            
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            
            # Save or show the plot
            if save_plots and output_dir:
                plt.savefig(os.path.join(output_dir, f'trajectory_step_{i+1}.png'), dpi=300)
                plt.close()
            else:
                plt.show()
    
    # Find the optimal stopping point (minimum MSE)
    valid_mse = [res['mse'] for res in mse_evolution if not np.isinf(res['mse'])]
    valid_steps = [res['step'] for res in mse_evolution if not np.isinf(res['mse'])]
    
    if valid_mse:
        best_idx = np.argmin(valid_mse)
        best_step = valid_steps[best_idx]
        best_mse = valid_mse[best_idx]
        
        print(f"\nOptimal stopping point found at step {best_step}")
        print(f"Minimum MSE: {best_mse:.2f}")
        
        # Calculate as percentage of total trajectory
        percentage = (best_step / len(measurements)) * 100
        print(f"This is {percentage:.1f}% of the total trajectory")
        
        # Create final summary visualization
        fig = plt.figure(figsize=(15, 10))
        
        # 3D plot with optimal stopping point
        ax = fig.add_subplot(2, 2, 1, projection='3d')
        
        # All raw points
        ax.scatter(position_3D_processed[:,0], position_3D_processed[:,1], position_3D_processed[:,2], 
                  c='red', s=20, alpha=0.3, label='All Points')
        
        # Mark the optimal stopping point
        optimal_point = position_3D_processed[best_step][:3]
        ax.scatter(optimal_point[0], optimal_point[1], optimal_point[2], 
                  c='black', s=100, marker='*', label='Optimal Stopping Point')
        
        # Plot ground truth
        for segment in ground_truth_segments:
            ax.plot(segment[:,1], segment[:,2], segment[:,3], 'g-', linewidth=2, label='_Ground Truth')
        
        # Add legend entry for ground truth
        ax.scatter([], [], [], c='green', s=30, label='Ground Truth')
        
        # Set title and labels
        ax.set_title(f'Optimal Stopping Point (Step {best_step}, MSE: {best_mse:.2f})')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        
        # Set axis limits based on table boundaries or data
        if table_boundaries:
            ax.set_xlim([0, table_boundaries['x_max']])
            ax.set_ylim([0, table_boundaries['y_max']])
            ax.set_zlim([0, 1500])
            
            # Add table outline
            x_min, x_max = table_boundaries['x_min'], table_boundaries['x_max']
            y_min, y_max = table_boundaries['y_min'], table_boundaries['y_max']
            z_table = 0
            
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    [z_table, z_table, z_table, z_table, z_table],
                    'k-', linewidth=1.0, alpha=0.5)
            
        ax.legend()
        
        # Plot MSE evolution
        ax = fig.add_subplot(2, 2, 2)
        
        if valid_steps:
            ax.plot(valid_steps, valid_mse, 'b-')
            ax.axvline(x=best_step, color='r', linestyle='--')
            
            # Add annotation for best MSE
            ax.annotate(f"Min MSE: {best_mse:.2f} at step {best_step}", 
                       xy=(best_step, best_mse),
                       xytext=(best_step+5, best_mse),
                       arrowprops=dict(facecolor='red', shrink=0.05))
        
        ax.set_xlabel('Step')
        ax.set_ylabel('MSE')
        ax.set_title('MSE Evolution')
        ax.grid(True)
        
        # Add histogram of MSE values
        ax = fig.add_subplot(2, 2, 3)
        ax.hist(valid_mse, bins=20)
        ax.axvline(x=best_mse, color='r', linestyle='--')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Frequency')
        ax.set_title('MSE Distribution')
        ax.grid(True)
        
        # Add percentage visualization
        ax = fig.add_subplot(2, 2, 4)
        
        # Create bar showing percentage of trajectory
        bars = ax.bar(['Before optimal', 'After optimal'], 
                     [percentage, 100-percentage],
                     color=['green', 'lightgray'])
        
        # Add percentage labels on bars
        ax.bar_label(bars, [f"{percentage:.1f}%", f"{100-percentage:.1f}%"], label_type='center')
        
        ax.set_ylim([0, 100])
        ax.set_title('Trajectory Percentage')
        
        plt.tight_layout()
        
        # Save or show the final plot
        if save_plots and output_dir:
            plt.savefig(os.path.join(output_dir, 'optimal_stopping_summary.png'), dpi=300)
            plt.close()
        else:
            plt.show()
        
        # Return results
        return {
            'best_step': best_step,
            'best_mse': best_mse,
            'percentage': percentage,
            'mse_evolution': mse_evolution
        }
    else:
        print("No valid MSE values found. Check your data and model parameters.")
        return None

# Function to analyze multiple trajectories
def analyze_multiple_trajectories(data_files, table_boundaries=None, save_plots=False, output_dir=None):
    """
    Analyze multiple trajectories to find the average optimal stopping point
    
    Args:
        data_files: List of paths to .npz files with trajectory data
        table_boundaries: Dictionary with table boundaries {x_min, x_max, y_min, y_max}
        save_plots: Whether to save plots instead of showing them
        output_dir: Directory to save plots if save_plots is True
        
    Returns:
        Dictionary with analysis results
    """
    results = []
    
    for i, data_file in enumerate(data_files):
        print(f"\nProcessing trajectory {i+1}/{len(data_files)}: {data_file}")
        
        # Create subdirectory for this trajectory if saving plots
        if save_plots and output_dir:
            traj_dir = os.path.join(output_dir, f"trajectory_{i+1}")
            os.makedirs(traj_dir, exist_ok=True)
        else:
            traj_dir = None
        
        # Run visualization with minimal output (only final plot)
        result = visualize_trajectory_evolution(
            data_file,
            table_boundaries=table_boundaries,
            step_interval=999999,  # Only show final visualization
            save_plots=save_plots,
            output_dir=traj_dir
        )
        
        if result:
            results.append({
                'file': data_file,
                'best_step': result['best_step'],
                'best_mse': result['best_mse'],
                'percentage': result['percentage']
            })
    
    # Calculate statistics
    if results:
        step_values = [res['best_step'] for res in results]
        mse_values = [res['best_mse'] for res in results]
        percentage_values = [res['percentage'] for res in results]
        
        avg_step = np.mean(step_values)
        std_step = np.std(step_values)
        
        avg_percentage = np.mean(percentage_values)
        std_percentage = np.std(percentage_values)
        
        print("\nSummary of all trajectories:")
        print(f"Average optimal stopping step: {avg_step:.2f} ± {std_step:.2f}")
        print(f"Average optimal stopping percentage: {avg_percentage:.2f}% ± {std_percentage:.2f}%")
        
        # Create summary visualization
        fig = plt.figure(figsize=(15, 10))
        
        # Plot percentage distribution
        ax = fig.add_subplot(2, 2, 1)
        ax.hist(percentage_values, bins=10)
        ax.axvline(x=avg_percentage, color='r', linestyle='--')
        ax.set_xlabel('Optimal Stopping Percentage')
        ax.set_ylabel('Frequency')
        ax.set_title('Stopping Percentage Distribution')
        ax.grid(True)
        
        # Plot MSE distribution
        ax = fig.add_subplot(2, 2, 2)
        ax.hist(mse_values, bins=10)
        ax.set_xlabel('Minimum MSE')
        ax.set_ylabel('Frequency')
        ax.set_title('MSE Distribution')
        ax.grid(True)
        
        # Plot step distribution
        ax = fig.add_subplot(2, 2, 3)
        ax.hist(step_values, bins=10)
        ax.axvline(x=avg_step, color='r', linestyle='--')
        ax.set_xlabel('Optimal Stopping Step')
        ax.set_ylabel('Frequency')
        ax.set_title('Stopping Step Distribution')
        ax.grid(True)
        
        # Plot scatter of MSE vs percentage
        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(percentage_values, mse_values, c='blue', s=50, alpha=0.7)
        
        # Label each point with trajectory number
        for j, (pct, mse) in enumerate(zip(percentage_values, mse_values)):
            ax.annotate(f"{j+1}", (pct, mse), xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Optimal Stopping Percentage')
        ax.set_ylabel('Minimum MSE')
        ax.set_title('MSE vs Stopping Percentage')
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_plots and output_dir:
            plt.savefig(os.path.join(output_dir, 'trajectory_summary.png'), dpi=300)
        else:
            plt.show()
        
        return {
            'results': results,
            'avg_step': avg_step,
            'std_step': std_step,
            'avg_percentage': avg_percentage,
            'std_percentage': std_percentage,
            'step_values': step_values,
            'percentage_values': percentage_values,
            'mse_values': mse_values
        }
    else:
        print("No valid results found.")
        return None

# Example usage
if __name__ == "__main__":
    # Define table boundaries
    TABLE_BOUNDARIES = {
        'x_min': 0,
        'x_max': 1525,
        'y_min': 0,
        'y_max': 2740
    }
    
    # Get start time
    start_time = time.time()
    
    # Choose analysis mode
    analysis_mode = "single"  # Options: "single", "multiple"
    
    if analysis_mode == "single":
        # Process a single trajectory
        data_file = '/Users/bb/Downloads/data_bounce/data_right11.npz'
        output_dir = 'trajectory_evolution_plots'
        
        # Create visualizations at intervals
        result = visualize_trajectory_evolution(
            data_file, 
            table_boundaries=TABLE_BOUNDARIES,
            step_interval=10,  # Create a visualization every 10 steps
            save_plots=False,  # Set to True to save plots to disk
            output_dir=output_dir
        )
    
    elif analysis_mode == "multiple":
        # Analyze multiple trajectories
        data_files = [f'/Users/bb/Downloads/data_bounce/data_right{i}.npz' for i in range(1, 21)]
        output_dir = 'trajectory_analysis'
        
        results = analyze_multiple_trajectories(
            data_files,
            table_boundaries=TABLE_BOUNDARIES,
            save_plots=False,
            output_dir=output_dir
        )
    
    # Print execution time
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")