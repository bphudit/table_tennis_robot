import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

class BounceTrajectoryEvaluator:
    def __init__(self, T, X, Y, Z):
        self.T = T
        self.X = X
        self.Y = Y
        self.Z = Z
        self.original_data = np.column_stack((X, Y, Z))
        
        # Shift time to start at 0
        self.T_shifted = self.T - self.T.min()
        
        # Physics constants
        self.g = 9.81  # gravity in m/s²
        
        # Store bounce segments
        self.bounce_segments = []
        self.bounce_indices = []

    def parabolic_motion(self, t, a, b, c):
        """General parabolic motion equation: a*t² + b*t + c"""
        return a*t**2 + b*t + c

    def detect_bounces(self, min_bounce_height=10, smoothing_window=3):
        """Detect bounce points in the trajectory using Z-coordinate minima"""
        # Optional: Simple smoothing to reduce noise
        if smoothing_window > 1:
            Z_smooth = np.convolve(self.Z, np.ones(smoothing_window)/smoothing_window, mode='valid')
            T_smooth = self.T_shifted[smoothing_window//2:-(smoothing_window//2) if smoothing_window//2 > 0 else None]
        else:
            Z_smooth = self.Z
            T_smooth = self.T_shifted
            
        # Find local minima in Z (potential bounces)
        # We use -Z because find_peaks finds maxima, not minima
        peaks, _ = find_peaks(-Z_smooth, height=-np.inf, distance=3)
        
        if len(peaks) == 0:
            # No bounces detected, treat entire trajectory as one segment
            self.bounce_indices = [0, len(self.Z)-1]
            return
            
        # Convert peak indices from smoothed array back to original indices
        if smoothing_window > 1:
            peaks = peaks + smoothing_window//2
        
        # Add start and end points to segment the trajectory
        bounce_points = [0] + list(peaks) + [len(self.Z)-1]
        
        # Filter bounce points to ensure they're significant enough
        filtered_bounces = [bounce_points[0]]
        for i in range(1, len(bounce_points)-1):
            # Check if this is a significant bounce
            min_z = self.Z[bounce_points[i]]
            prev_max_z = np.max(self.Z[bounce_points[i-1]:bounce_points[i]+1])
            next_max_z = np.max(self.Z[bounce_points[i]:bounce_points[i+1]+1])
            
            # Only count as bounce if there's sufficient height difference
            if prev_max_z - min_z > min_bounce_height or next_max_z - min_z > min_bounce_height:
                filtered_bounces.append(bounce_points[i])
        
        filtered_bounces.append(bounce_points[-1])
        self.bounce_indices = filtered_bounces
        
        print(f"Detected {len(filtered_bounces)-1} trajectory segments")
        for i in range(len(filtered_bounces)-1):
            print(f"Segment {i+1}: Points {filtered_bounces[i]} to {filtered_bounces[i+1]}")

    def fit_segments(self):
        """Fit separate parabolic models to each trajectory segment"""
        num_segments = len(self.bounce_indices) - 1
        
        for i in range(num_segments):
            start_idx = self.bounce_indices[i]
            end_idx = self.bounce_indices[i+1]
            
            # Extract segment data
            t_seg = self.T_shifted[start_idx:end_idx+1]
            x_seg = self.X[start_idx:end_idx+1]
            y_seg = self.Y[start_idx:end_idx+1]
            z_seg = self.Z[start_idx:end_idx+1]
            
            # Shift time to start at 0 for this segment
            t_seg_shifted = t_seg - t_seg.min()
            
            try:
                # Fit X trajectory with parabola: x(t) = ax*t² + bx*t + cx
                x_params, _ = curve_fit(self.parabolic_motion, t_seg_shifted, x_seg)
                
                # Fit Y trajectory with parabola: y(t) = ay*t² + by*t + cy
                y_params, _ = curve_fit(self.parabolic_motion, t_seg_shifted, y_seg)
                
                # Fit Z trajectory with parabola: z(t) = az*t² + bz*t + cz
                # For physics-based model, we could set az = -0.5*g, but we'll let the data decide
                z_params, _ = curve_fit(self.parabolic_motion, t_seg_shifted, z_seg)
                
                # Store segment parameters
                self.bounce_segments.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    't_min': t_seg.min(),
                    't_max': t_seg.max(),
                    'x_params': x_params,
                    'y_params': y_params,
                    'z_params': z_params
                })
                
                print(f"Segment {i+1} fitted successfully")
            except Exception as e:
                print(f"Failed to fit segment {i+1}, skipping: {e}")
    
    def generate_fitted_curves(self, points_per_segment=100):
        """Generate fitted curves for all segments"""
        fitted_points = []
        
        for segment in self.bounce_segments:
            # Create time points for this segment
            t_dense = np.linspace(segment['t_min'], segment['t_max'], points_per_segment)
            t_dense_shifted = t_dense - segment['t_min']  # Shift to start at 0
            
            # Generate coordinates using fitted parameters
            ax, bx, cx = segment['x_params']
            ay, by, cy = segment['y_params']
            az, bz, cz = segment['z_params']
            
            x_dense = self.parabolic_motion(t_dense_shifted, ax, bx, cx)
            y_dense = self.parabolic_motion(t_dense_shifted, ay, by, cy)
            z_dense = self.parabolic_motion(t_dense_shifted, az, bz, cz)
            
            # Store points
            segment_points = np.column_stack((t_dense, x_dense, y_dense, z_dense))
            fitted_points.append(segment_points)
            
        return fitted_points
    
    def calculate_error_metrics(self):
        """Calculate error metrics for all segments"""
        overall_errors = []
        segment_metrics = []
        
        for i, segment in enumerate(self.bounce_segments):
            start_idx = segment['start_idx']
            end_idx = segment['end_idx']
            
            # Extract segment data
            t_seg = self.T_shifted[start_idx:end_idx+1]
            t_seg_shifted = t_seg - segment['t_min']
            
            x_seg = self.X[start_idx:end_idx+1]
            y_seg = self.Y[start_idx:end_idx+1]
            z_seg = self.Z[start_idx:end_idx+1]
            
            # Generate predictions
            ax, bx, cx = segment['x_params']
            ay, by, cy = segment['y_params']
            az, bz, cz = segment['z_params']
            
            x_pred = self.parabolic_motion(t_seg_shifted, ax, bx, cx)
            y_pred = self.parabolic_motion(t_seg_shifted, ay, by, cy)
            z_pred = self.parabolic_motion(t_seg_shifted, az, bz, cz)
            
            # Calculate errors
            errors = np.sqrt((x_pred - x_seg)**2 + (y_pred - y_seg)**2 + (z_pred - z_seg)**2)
            overall_errors.extend(errors)
            
            # Calculate metrics for this segment
            mse = np.mean(errors**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(errors))
            max_error = np.max(errors)
            
            # Calculate component-wise errors
            x_rmse = np.sqrt(np.mean((x_pred - x_seg)**2))
            y_rmse = np.sqrt(np.mean((y_pred - y_seg)**2))
            z_rmse = np.sqrt(np.mean((z_pred - z_seg)**2))
            
            segment_metrics.append({
                'segment': i+1,
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'Max Error': max_error,
                'X RMSE': x_rmse,
                'Y RMSE': y_rmse,
                'Z RMSE': z_rmse,
                'Points': len(t_seg)
            })
        
        # Calculate overall metrics
        overall_mse = np.mean(np.array(overall_errors)**2)
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = np.mean(np.abs(overall_errors))
        overall_max = np.max(overall_errors)
        
        return {
            'segment_metrics': segment_metrics,
            'overall': {
                'MSE': overall_mse,
                'RMSE': overall_rmse,
                'MAE': overall_mae,
                'Max Error': overall_max
            }
        }
    
    def plot_3d_trajectory(self):
        """Plot 3D trajectory with all segments"""
        # Get fitted curves
        fitted_segments = self.generate_fitted_curves()
        
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot raw data
        ax.scatter(self.X, self.Y, self.Z, c='blue', label='Raw Data', alpha=0.6, s=20)
        
        # Plot bounce points with different marker
        for idx in self.bounce_indices[1:-1]:  # Skip first and last points
            ax.scatter(self.X[idx], self.Y[idx], self.Z[idx], c='black', marker='*', s=100)
        
        # Plot fitted segments with different colors
        colors = plt.cm.jet(np.linspace(0, 1, len(fitted_segments)))
        for i, segment_points in enumerate(fitted_segments):
            ax.plot(segment_points[:,1], segment_points[:,2], segment_points[:,3], 
                    c=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectory with Parabolic Bounce Segments')
        
        # Add legend (with smaller font to accommodate more segments)
        ax.legend(fontsize='small')
        
        plt.tight_layout()
        return fig
    
    def plot_2d_projections(self):
        """Plot 2D projections (XZ and YZ) with color-coded segments"""
        # Get fitted curves
        fitted_segments = self.generate_fitted_curves()
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Generate colors for segments
        colors = plt.cm.jet(np.linspace(0, 1, len(fitted_segments)))
        
        # XZ projection (top plot)
        axes[0].scatter(self.X, self.Z, c='blue', alpha=0.6, s=20, label='Raw Data')
        
        # Mark bounce points
        for idx in self.bounce_indices[1:-1]:  # Skip first and last points
            axes[0].scatter(self.X[idx], self.Z[idx], c='black', marker='*', s=100)
        
        # Plot segments
        for i, segment_points in enumerate(fitted_segments):
            axes[0].plot(segment_points[:,1], segment_points[:,3], 
                     c=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Z')
        axes[0].set_title('XZ Projection (Parabolic Fits)')
        axes[0].grid(True)
        axes[0].legend(fontsize='small')
        
        # YZ projection (bottom plot)
        axes[1].scatter(self.Y, self.Z, c='blue', alpha=0.6, s=20, label='Raw Data')
        
        # Mark bounce points
        for idx in self.bounce_indices[1:-1]:  # Skip first and last points
            axes[1].scatter(self.Y[idx], self.Z[idx], c='black', marker='*', s=100)
        
        # Plot segments
        for i, segment_points in enumerate(fitted_segments):
            axes[1].plot(segment_points[:,2], segment_points[:,3], 
                     c=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        axes[1].set_xlabel('Y')
        axes[1].set_ylabel('Z')
        axes[1].set_title('YZ Projection (Parabolic Fits)')
        axes[1].grid(True)
        axes[1].legend(fontsize='small')
        
        plt.tight_layout()
        return fig
    
    def plot_vs_time(self):
        """Plot X, Y, Z coordinates vs time with fitted segments"""
        # Get fitted curves
        fitted_segments = self.generate_fitted_curves()
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        # Generate colors for segments
        colors = plt.cm.jet(np.linspace(0, 1, len(fitted_segments)))
        
        # X vs Time
        axes[0].scatter(self.T_shifted, self.X, c='blue', alpha=0.6, s=20, label='Raw X')
        
        # Mark bounce points
        for idx in self.bounce_indices[1:-1]:  # Skip first and last points
            axes[0].scatter(self.T_shifted[idx], self.X[idx], c='black', marker='*', s=100)
        
        # Plot segments
        for i, segment_points in enumerate(fitted_segments):
            axes[0].plot(segment_points[:,0], segment_points[:,1], 
                     c=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('X Position')
        axes[0].set_title('X vs Time (Parabolic Fits)')
        axes[0].grid(True)
        axes[0].legend(fontsize='small')
        
        # Y vs Time
        axes[1].scatter(self.T_shifted, self.Y, c='blue', alpha=0.6, s=20, label='Raw Y')
        
        # Mark bounce points
        for idx in self.bounce_indices[1:-1]:  # Skip first and last points
            axes[1].scatter(self.T_shifted[idx], self.Y[idx], c='black', marker='*', s=100)
        
        # Plot segments
        for i, segment_points in enumerate(fitted_segments):
            axes[1].plot(segment_points[:,0], segment_points[:,2], 
                     c=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Y Position')
        axes[1].set_title('Y vs Time (Parabolic Fits)')
        axes[1].grid(True)
        axes[1].legend(fontsize='small')
        
        # Z vs Time
        axes[2].scatter(self.T_shifted, self.Z, c='blue', alpha=0.6, s=20, label='Raw Z')
        
        # Mark bounce points
        for idx in self.bounce_indices[1:-1]:  # Skip first and last points
            axes[2].scatter(self.T_shifted[idx], self.Z[idx], c='black', marker='*', s=100)
        
        # Plot segments
        for i, segment_points in enumerate(fitted_segments):
            axes[2].plot(segment_points[:,0], segment_points[:,3], 
                     c=colors[i], linewidth=2, label=f'Segment {i+1}')
        
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Z Position')
        axes[2].set_title('Z vs Time (Parabolic Fits)')
        axes[2].grid(True)
        axes[2].legend(fontsize='small')
        
        plt.tight_layout()
        return fig
    
    def print_segment_details(self):
        """Print detailed information about each segment"""
        for i, segment in enumerate(self.bounce_segments):
            ax, bx, cx = segment['x_params']
            ay, by, cy = segment['y_params']
            az, bz, cz = segment['z_params']
            
            print(f"\nSegment {i+1} Details:")
            print(f"Time range: {segment['t_min']:.4f} to {segment['t_max']:.4f} seconds")
            print(f"X equation: {ax:.4f}*t² + {bx:.4f}*t + {cx:.4f}")
            print(f"Y equation: {ay:.4f}*t² + {by:.4f}*t + {cy:.4f}")
            print(f"Z equation: {az:.4f}*t² + {bz:.4f}*t + {cz:.4f}")
            
            # For Z, compare with expected gravity coefficient
            print(f"Z acceleration: {2*az:.4f} (Expected from gravity: {-self.g:.4f})")
            
            # Calculate initial velocities (t=0 derivatives)
            vx0 = bx
            vy0 = by
            vz0 = bz
            print(f"Initial velocity: ({vx0:.2f}, {vy0:.2f}, {vz0:.2f}) m/s")
            
            # Calculate horizontal velocity and launch angle
            horizontal_v = np.sqrt(vx0**2 + vy0**2)
            launch_angle = np.degrees(np.arctan2(vz0, horizontal_v))
            print(f"Horizontal velocity: {horizontal_v:.2f} m/s")
            print(f"Launch angle: {launch_angle:.2f} degrees")
    
    def print_error_metrics(self, metrics):
        """Print error metrics for all segments"""
        print("\nError Metrics per Segment:")
        for segment_metric in metrics['segment_metrics']:
            print(f"\nSegment {segment_metric['segment']} ({segment_metric['Points']} points):")
            for key, value in segment_metric.items():
                if key not in ['segment', 'Points']:
                    print(f"  {key}: {value:.4f}")
        
        print("\nOverall Metrics:")
        for key, value in metrics['overall'].items():
            print(f"  {key}: {value:.4f}")
    
    def run(self, min_bounce_height=5, smoothing_window=3):
        """Run the full analysis pipeline"""
        print("Detecting bounces...")
        self.detect_bounces(min_bounce_height, smoothing_window)
        
        print("\nFitting trajectory segments...")
        self.fit_segments()
        
        print("\nGenerating visualizations...")
        fig1 = self.plot_3d_trajectory()
        fig2 = self.plot_2d_projections()
        fig3 = self.plot_vs_time()
        
        print("\nCalculating error metrics...")
        metrics = self.calculate_error_metrics()
        
        self.print_segment_details()
        self.print_error_metrics(metrics)
        
        plt.show()
        
        return metrics


# Load data and run evaluator
data = np.load('/Users/bb/Downloads/data1.npz')['position_3D']
evaluator = BounceTrajectoryEvaluator(data[:,3], data[:,0], data[:,1], data[:,2])
metrics = evaluator.run(min_bounce_height=10, smoothing_window=3)