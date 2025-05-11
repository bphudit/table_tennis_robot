import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import math

class BounceTrajectoryEvaluator:
    def __init__(self, T, X, Y, Z, sort_timestamps=True, table_boundaries=None):
        """
        Initialize the trajectory evaluator
        
        Args:
            T: Time values
            X, Y, Z: Coordinate values
            sort_timestamps: Whether to sort the data by timestamps
            table_boundaries: Dictionary with table boundaries {x_min, x_max, y_min, y_max}
        """
        # Save original data before filtering
        self.original_T = T
        self.original_X = X
        self.original_Y = Y
        self.original_Z = Z
        
        # Apply table boundary filtering if specified
        if table_boundaries is not None:
            # Filter points within table boundaries
            mask = (
                (X >= table_boundaries['x_min']) & 
                (X <= table_boundaries['x_max']) & 
                (Y >= table_boundaries['y_min']) & 
                (Y <= table_boundaries['y_max'])
            )
            
            T = T[mask]
            X = X[mask]
            Y = Y[mask]
            Z = Z[mask]
            
            # Report how many points were filtered out
            points_filtered = len(self.original_X) - len(X)
            percent_filtered = (points_filtered / len(self.original_X)) * 100 if len(self.original_X) > 0 else 0
            print(f"Table boundary filtering: {points_filtered} points removed ({percent_filtered:.1f}%)")
            print(f"Remaining points: {len(X)}")
            
            # If all points were filtered, this is a problem
            if len(X) == 0:
                raise ValueError("All points were outside table boundaries! Check your boundary values.")
        
        # If sort_timestamps is True, sort all data by timestamp
        if sort_timestamps:
            # Create a sorting index based on timestamps
            sorted_indices = np.argsort(T)
            self.T = T[sorted_indices]
            self.X = X[sorted_indices]
            self.Y = Y[sorted_indices]
            self.Z = Z[sorted_indices]
            
            # Check for large gaps in the timestamps
            time_diffs = np.diff(self.T)
            median_diff = np.median(time_diffs)
            large_gaps = np.where(time_diffs > median_diff * 100)[0]
            if len(large_gaps) > 0:
                print(f"Warning: Large time gaps detected at indices {large_gaps}")
                print(f"Time differences at these gaps: {time_diffs[large_gaps]}")
                print(f"Median time difference: {median_diff}")
        else:
            self.T = T
            self.X = X
            self.Y = Y
            self.Z = Z
            
        self.data = np.column_stack((self.X, self.Y, self.Z))
        
        # Normalize timestamp - convert to relative seconds
        # This handles both large microsecond timestamps and regular timestamps
        self.T_shifted = (self.T - self.T.min())
        
        # If timestamps are in microseconds, convert to seconds
        if np.max(self.T_shifted) > 1e10:
            self.T_shifted = self.T_shifted / 1e6
            print("Detected microsecond timestamps, converting to seconds")
        
        # Physics constants
        self.g = 9.81  # gravity in m/s²
        
        # Store bounce segments
        self.bounce_segments = []
        self.bounce_indices = []
        
        # Save table boundaries
        self.table_boundaries = table_boundaries

    def parabolic_motion(self, t, a, b, c):
        """General parabolic motion equation: a*t² + b*t + c"""
        return a*t**2 + b*t + c

    def detect_bounces(self, min_bounce_height=None, smoothing_window=3, verbose=True):
        """Detect bounce points in the trajectory using Z-coordinate minima"""
        # Auto-determine min_bounce_height if not specified
        if min_bounce_height is None:
            z_range = np.max(self.Z) - np.min(self.Z)
            min_bounce_height = z_range * 0.05  # 5% of Z range
            if verbose:
                print(f"Auto-determined min_bounce_height: {min_bounce_height:.2f}")
        
        # Optional: Simple smoothing to reduce noise
        if smoothing_window > 1:
            # Make sure we have enough points for smoothing
            if len(self.Z) < smoothing_window:
                if verbose:
                    print(f"Warning: Not enough points for smoothing window of {smoothing_window}. Using all available points.")
                Z_smooth = self.Z
                T_smooth = self.T_shifted
            else:
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
            if verbose:
                print("No bounces detected. Treating as a single trajectory.")
            return self
            
        # Convert peak indices from smoothed array back to original indices
        if smoothing_window > 1 and len(self.Z) >= smoothing_window:
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
        
        if verbose:
            print(f"Detected {len(filtered_bounces)-1} trajectory segments")
            for i in range(len(filtered_bounces)-1):
                print(f"Segment {i+1}: Points {filtered_bounces[i]} to {filtered_bounces[i+1]}")
        
        return self

    def fit_segments(self, verbose=True):
        """Fit separate parabolic models to each trajectory segment"""
        num_segments = len(self.bounce_indices) - 1
        
        for i in range(num_segments):
            start_idx = self.bounce_indices[i]
            end_idx = self.bounce_indices[i+1]
            
            # Skip segments that are too short (need at least 5 points for stable fitting)
            if end_idx - start_idx < 5:
                if verbose:
                    print(f"Segment {i+1} too short, skipping.")
                continue
            
            # Extract segment data
            t_seg = self.T_shifted[start_idx:end_idx+1]
            x_seg = self.X[start_idx:end_idx+1]
            y_seg = self.Y[start_idx:end_idx+1]
            z_seg = self.Z[start_idx:end_idx+1]
            
            # Check for large time gaps within this segment
            if len(t_seg) > 1:
                t_diffs = np.diff(t_seg)
                median_diff = np.median(t_diffs)
                max_diff = np.max(t_diffs)
                
                # If there's a suspiciously large gap, warn and continue
                if max_diff > median_diff * 100:
                    if verbose:
                        print(f"Warning: Large time gap detected within segment {i+1}. Gap size: {max_diff:.6f}, Median: {median_diff:.6f}")
            
            # Shift time to start at 0 for this segment
            t_seg_shifted = t_seg - t_seg.min()
            
            try:
                # Fit X trajectory with parabola: x(t) = ax*t² + bx*t + cx
                x_params, _ = curve_fit(self.parabolic_motion, t_seg_shifted, x_seg, 
                                       maxfev=10000)  # Increase max iterations
                
                # Fit Y trajectory with parabola: y(t) = ay*t² + by*t + cy
                y_params, _ = curve_fit(self.parabolic_motion, t_seg_shifted, y_seg,
                                       maxfev=10000)
                
                # Fit Z trajectory with parabola: z(t) = az*t² + bz*t + cz
                z_params, _ = curve_fit(self.parabolic_motion, t_seg_shifted, z_seg,
                                       maxfev=10000)
                
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
                
                if verbose:
                    print(f"Segment {i+1} fitted successfully")
            except Exception as e:
                if verbose:
                    print(f"Failed to fit segment {i+1}, skipping: {e}")
        
        return self
    
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
        if len(overall_errors) > 0:
            overall_mse = np.mean(np.array(overall_errors)**2)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = np.mean(np.abs(overall_errors))
            overall_max = np.max(overall_errors)
        else:
            overall_mse = overall_rmse = overall_mae = overall_max = float('nan')
        
        self.metrics = {
            'segment_metrics': segment_metrics,
            'overall': {
                'MSE': overall_mse,
                'RMSE': overall_rmse,
                'MAE': overall_mae,
                'Max Error': overall_max
            }
        }
        
        return self.metrics
    
    def plot_3d_trajectory(self, show=True, limit_axes=True, fig_title=None, show_table_outline=True):
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
        
        # If table boundaries are defined and show_table_outline is True, draw just the table outline
        if self.table_boundaries is not None and show_table_outline:
            # Extract table dimensions
            x_min = self.table_boundaries['x_min']
            x_max = self.table_boundaries['x_max']
            y_min = self.table_boundaries['y_min']
            y_max = self.table_boundaries['y_max']
            
            # Assuming Z=0 is the table height
            z_table = 0
            
            # Plot the table edges only
            ax.plot([x_min, x_max, x_max, x_min, x_min],
                    [y_min, y_min, y_max, y_max, y_min],
                    [z_table, z_table, z_table, z_table, z_table],
                    'k-', linewidth=1.0, alpha=0.5, label='Table')
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = fig_title if fig_title else '3D Trajectory with Parabolic Bounce Segments'
        ax.set_title(title)
        
        # Optionally limit axes to focus on the relevant part of the trajectory
        if limit_axes:
            # Add a small margin around the data
            x_margin = (np.max(self.X) - np.min(self.X)) * 0.1
            y_margin = (np.max(self.Y) - np.min(self.Y)) * 0.1
            z_margin = (np.max(self.Z) - np.min(self.Z)) * 0.1
            
            ax.set_xlim(np.min(self.X) - x_margin, np.max(self.X) + x_margin)
            ax.set_ylim(np.min(self.Y) - y_margin, np.max(self.Y) + y_margin)
            ax.set_zlim(np.min(self.Z) - z_margin, np.max(self.Z) + z_margin)
        
        # Add legend (with smaller font to accommodate more segments)
        ax.legend(fontsize='small')
        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig
    
    def plot_2d_projections(self, show=True, limit_axes=True, fig_title=None, show_table_outline=True):
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
        
        # If table boundaries are defined and show_table_outline is True, draw the table edge
        if self.table_boundaries is not None and show_table_outline:
            # Draw the table top edge in XZ view
            x_min = self.table_boundaries['x_min']
            x_max = self.table_boundaries['x_max']
            z_table = 0
            axes[0].plot([x_min, x_max], [z_table, z_table], 'k-', linewidth=1.0, alpha=0.5)
        
        # Optionally limit axes
        if limit_axes:
            x_margin = (np.max(self.X) - np.min(self.X)) * 0.1
            z_margin = (np.max(self.Z) - np.min(self.Z)) * 0.1
            axes[0].set_xlim(np.min(self.X) - x_margin, np.max(self.X) + x_margin)
            axes[0].set_ylim(np.min(self.Z) - z_margin, np.max(self.Z) + z_margin)
        
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Z')
        title = "XZ Projection (Parabolic Fits)"
        if fig_title:
            title = f"{fig_title} - {title}"
        axes[0].set_title(title)
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
        
        # If table boundaries are defined and show_table_outline is True, draw the table edge
        if self.table_boundaries is not None and show_table_outline:
            # Draw the table top edge in YZ view
            y_min = self.table_boundaries['y_min']
            y_max = self.table_boundaries['y_max']
            z_table = 0
            axes[1].plot([y_min, y_max], [z_table, z_table], 'k-', linewidth=1.0, alpha=0.5)
        
        # Optionally limit axes
        if limit_axes:
            y_margin = (np.max(self.Y) - np.min(self.Y)) * 0.1
            z_margin = (np.max(self.Z) - np.min(self.Z)) * 0.1
            axes[1].set_xlim(np.min(self.Y) - y_margin, np.max(self.Y) + y_margin)
            axes[1].set_ylim(np.min(self.Z) - z_margin, np.max(self.Z) + z_margin)
        
        axes[1].set_xlabel('Y')
        axes[1].set_ylabel('Z')
        title = "YZ Projection (Parabolic Fits)"
        if fig_title:
            title = f"{fig_title} - {title}"
        axes[1].set_title(title)
        axes[1].grid(True)
        axes[1].legend(fontsize='small')
        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig
    
    def plot_vs_time(self, show=True, fig_title=None):
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
        
        # If table boundaries are defined, draw horizontal lines for table boundaries
        if self.table_boundaries is not None:
            x_min = self.table_boundaries['x_min']
            x_max = self.table_boundaries['x_max']
            axes[0].axhline(y=x_min, color='k', linestyle='--', alpha=0.3)
            axes[0].axhline(y=x_max, color='k', linestyle='--', alpha=0.3)
        
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('X Position')
        title = "X vs Time (Parabolic Fits)"
        if fig_title:
            title = f"{fig_title} - {title}"
        axes[0].set_title(title)
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
        
        # If table boundaries are defined, draw horizontal lines for table boundaries
        if self.table_boundaries is not None:
            y_min = self.table_boundaries['y_min']
            y_max = self.table_boundaries['y_max']
            axes[1].axhline(y=y_min, color='k', linestyle='--', alpha=0.3)
            axes[1].axhline(y=y_max, color='k', linestyle='--', alpha=0.3)
        
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Y Position')
        title = "Y vs Time (Parabolic Fits)"
        if fig_title:
            title = f"{fig_title} - {title}"
        axes[1].set_title(title)
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
        
        # If showing table outline, add a horizontal line at Z=0 (table surface)
        if self.table_boundaries is not None:
            axes[2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Z Position')
        title = "Z vs Time (Parabolic Fits)"
        if fig_title:
            title = f"{fig_title} - {title}"
        axes[2].set_title(title)
        axes[2].grid(True)
        axes[2].legend(fontsize='small')
        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return fig
    
    def print_segment_details(self):
        """Print detailed information about each segment"""
        results = []
        
        for i, segment in enumerate(self.bounce_segments):
            ax, bx, cx = segment['x_params']
            ay, by, cy = segment['y_params']
            az, bz, cz = segment['z_params']
            
            segment_info = {
                'segment': i+1,
                'time_range': (segment['t_min'], segment['t_max']),
                'x_equation': (ax, bx, cx),
                'y_equation': (ay, by, cy),
                'z_equation': (az, bz, cz),
                'z_acceleration': 2*az,
                'initial_velocity': (bx, by, bz)
            }
            
            # Calculate horizontal velocity and launch angle
            vx0, vy0, vz0 = bx, by, bz
            horizontal_v = np.sqrt(vx0**2 + vy0**2)
            launch_angle = np.degrees(np.arctan2(vz0, horizontal_v))
            
            segment_info['horizontal_velocity'] = horizontal_v
            segment_info['launch_angle'] = launch_angle
            
            results.append(segment_info)
            
            print(f"\nSegment {i+1} Details:")
            print(f"Time range: {segment['t_min']:.6f} to {segment['t_max']:.6f} seconds")
            print(f"X equation: {ax:.6f}*t² + {bx:.6f}*t + {cx:.4f}")
            print(f"Y equation: {ay:.6f}*t² + {by:.6f}*t + {cy:.4f}")
            print(f"Z equation: {az:.6f}*t² + {bz:.6f}*t + {cz:.4f}")
            
            # For Z, compare with expected gravity coefficient
            print(f"Z acceleration: {2*az:.6f} (Expected from gravity: {-self.g:.4f})")
            
            print(f"Initial velocity: ({vx0:.2f}, {vy0:.2f}, {vz0:.2f}) m/s")
            print(f"Horizontal velocity: {horizontal_v:.2f} m/s")
            print(f"Launch angle: {launch_angle:.2f} degrees")
        
        return results
    
    def print_error_metrics(self):
        """Print error metrics for all segments"""
        if self.metrics is None:
            self.calculate_error_metrics()
            
        print("\nError Metrics per Segment:")
        for segment_metric in self.metrics['segment_metrics']:
            print(f"\nSegment {segment_metric['segment']} ({segment_metric['Points']} points):")
            for key, value in segment_metric.items():
                if key not in ['segment', 'Points']:
                    print(f"  {key}: {value:.4f}")
        
        print("\nOverall Metrics:")
        for key, value in self.metrics['overall'].items():
            print(f"  {key}: {value:.4f}")
        
        return self.metrics
    
    def handle_split_trajectory(self, time_gap_threshold=100, verbose=True):
        """
        Analyzes the data to identify distinct trajectories separated by large time gaps
        Returns a list of trajectory segment index ranges
        
        Args:
            time_gap_threshold: Factor relative to median time difference to consider a gap
            verbose: Whether to print information about detected trajectories
        
        Returns:
            List of tuples (start_idx, end_idx) for each distinct trajectory
        """
        # Calculate time differences
        if len(self.T) <= 1:
            return [(0, 0)]
            
        time_diffs = np.diff(self.T)
        median_diff = np.median(time_diffs)
        
        # Find large gaps (e.g., gaps > threshold * the median difference)
        large_gaps = np.where(time_diffs > median_diff * time_gap_threshold)[0]
        
        if len(large_gaps) == 0:
            # No large gaps, single trajectory
            trajectories = [(0, len(self.T)-1)]
            if verbose:
                print("Single continuous trajectory detected")
        else:
            # Multiple trajectories
            trajectories = []
            
            # Start with the first segment
            start_idx = 0
            
            # Add each segment separated by large gaps
            for gap_idx in large_gaps:
                trajectories.append((start_idx, gap_idx))
                start_idx = gap_idx + 1
            
            # Add the final segment
            trajectories.append((start_idx, len(self.T)-1))
            
            if verbose:
                print(f"{len(trajectories)} separate trajectories detected")
                for i, (start, end) in enumerate(trajectories):
                    print(f"Trajectory {i+1}: Points {start} to {end} ({end-start+1} points)")
        
        return trajectories
    
    def run(self, min_bounce_height=None, smoothing_window=3, limit_axes=True, time_gap_threshold=100, 
            show_plots=True, fig_title=None, show_table_outline=True):
        """Run the full analysis pipeline"""
        print("Detecting bounces...")
        self.detect_bounces(min_bounce_height, smoothing_window)
        
        print("\nFitting trajectory segments...")
        self.fit_segments()
        
        print("\nGenerating visualizations...")
        if show_plots:
            self.plot_3d_trajectory(show=False, limit_axes=limit_axes, fig_title=fig_title, 
                                   show_table_outline=show_table_outline)
            self.plot_2d_projections(show=False, limit_axes=limit_axes, fig_title=fig_title,
                                    show_table_outline=show_table_outline)
            self.plot_vs_time(show=False, fig_title=fig_title)
        
        print("\nCalculating error metrics...")
        self.calculate_error_metrics()
        
        self.print_segment_details()
        self.print_error_metrics()
        
        if show_plots:
            plt.show()  # Display all figures at once
        
        return self.metrics


def analyze_multiple_trajectories(data, min_bounce_height=None, smoothing_window=3, 
                                time_gap_threshold=100, table_boundaries=None,
                                show_table_outline=True):
    """
    Process a dataset with multiple trajectories, correctly handling large time gaps
    
    Args:
        data: Numpy array with columns [X, Y, Z, T] or path to npz file
        min_bounce_height: Minimum height for bounce detection (None for auto)
        smoothing_window: Window size for smoothing data
        time_gap_threshold: Threshold factor for detecting time gaps
        table_boundaries: Dictionary with table boundaries {x_min, x_max, y_min, y_max}
        show_table_outline: Whether to show table outline in plots
        
    Returns:
        Dictionary with metrics and results for each trajectory
    """
    # Check if data is a file path or numpy array
    if isinstance(data, str):
        # Load from file
        data = np.load(data)['position_3D']
    
    # Create evaluator with timestamp sorting enabled
    main_evaluator = BounceTrajectoryEvaluator(
        data[:,3],  # Time 
        data[:,1],  # X
        data[:,0],  # Y
        -1 * data[:,2],  # Z (inverted)
        sort_timestamps=True,  # Sort data by timestamps
        table_boundaries=table_boundaries  # Apply table boundary filtering
    )
    
    # First detect separate trajectories from large time gaps
    trajectories = main_evaluator.handle_split_trajectory(time_gap_threshold=time_gap_threshold, verbose=True)
    
    all_results = []
    
    # Process each trajectory separately
    for i, (start_idx, end_idx) in enumerate(trajectories):
        print(f"\nProcessing trajectory {i+1} (points {start_idx} to {end_idx})")
        
        # Create a new evaluator with just this trajectory segment
        traj_evaluator = BounceTrajectoryEvaluator(
            main_evaluator.T[start_idx:end_idx+1],
            main_evaluator.X[start_idx:end_idx+1],
            main_evaluator.Y[start_idx:end_idx+1],
            main_evaluator.Z[start_idx:end_idx+1],
            sort_timestamps=False,  # Already sorted
            table_boundaries=table_boundaries  # Keep table boundaries
        )
        
        # Run analysis on this trajectory segment
        fig_title = f"Trajectory {i+1}"
        metrics = traj_evaluator.run(
            min_bounce_height=min_bounce_height,
            smoothing_window=smoothing_window,
            limit_axes=True,
            fig_title=fig_title,
            show_table_outline=show_table_outline
        )
        
        # Store results for this trajectory
        all_results.append({
            'trajectory_id': i+1,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'metrics': metrics,
            'evaluator': traj_evaluator
        })
    
    print("\nAll trajectories processed successfully!")
    return all_results


# Example usage showing how to analyze trajectories with table boundaries
if __name__ == "__main__":
    import numpy as np
    
    # Define table boundaries
    TABLE_BOUNDARIES = {
        'x_min': 0,
        'x_max': 1525,
        'y_min': 0,
        'y_max': 2740
    }
    
    # Load data
    data_file = '/Users/bb/Downloads/data_bounce/data_right11.npz'
    
    # Run analysis with table boundaries
    results = analyze_multiple_trajectories(
        data_file,
        min_bounce_height=None,  # Auto-determine
        smoothing_window=3,
        time_gap_threshold=100,
        table_boundaries=TABLE_BOUNDARIES,
        show_table_outline=True  # Just show table outline, not the green surface
    )