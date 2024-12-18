import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_all_camera_poses(all_camera_poses_history, binocular_poses=None):
    """
    Visualize all camera poses throughout optimization iterations
    all_camera_poses_history: dict of camera_idx -> list of (R, t) tuples for each iteration
    binocular_poses: dict of camera_idx -> (R, t) tuple for binocular calibration results
    """
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Different colors for different cameras (brighter colors for dark background)
    colors = ['#ff4444', '#44ff44', '#4444ff', '#ff44ff', '#ffff44', '#44ffff']
    
    # Find the range of positions to set appropriate axis limits
    all_positions = []
    for cam_poses in all_camera_poses_history.values():
        for _, t in cam_poses:
            all_positions.append(t.flatten())
    if binocular_poses is not None:
        for _, t in binocular_poses.values():
            all_positions.append(t.flatten())
    all_positions = np.array(all_positions)
    
    # Calculate axis limits with padding
    min_pos = all_positions.min(axis=0)
    max_pos = all_positions.max(axis=0)
    range_pos = max_pos - min_pos
    padding = range_pos * 0.2  # 20% padding
    
    # Store camera data for bundle visualization
    final_cameras = {'multi': {}, 'binocular': {}}
    bundle_lines = []
    show_bundles = False
    
    def draw_bundles():
        # Clear previous bundle lines
        for line in bundle_lines:
            line.remove()
        bundle_lines.clear()
        
        if show_bundles:
            # Draw bundle for reference camera
            bundle_length = np.mean(range_pos) * 2.0  # Length of bundle line
            ref_dir = ref_R[:, 2]  # Z-axis direction
            bundle_end = ref_pos + ref_dir * bundle_length
            line = ax.plot([ref_pos[0], bundle_end[0]],
                         [ref_pos[1], bundle_end[1]],
                         [ref_pos[2], bundle_end[2]],
                         'w--', alpha=0.5, linewidth=1)[0]
            bundle_lines.append(line)
            
            # Draw bundles for multi cameras
            for cam_idx, (R, t) in final_cameras['multi'].items():
                pos = t.flatten()
                R_mat = R if len(R.shape) == 2 else R.reshape(3, 3)
                dir_vec = R_mat[:, 2]  # Z-axis direction
                bundle_end = pos + dir_vec * bundle_length
                line = ax.plot([pos[0], bundle_end[0]],
                             [pos[1], bundle_end[1]],
                             [pos[2], bundle_end[2]],
                             ':', color=colors[cam_idx % len(colors)],
                             alpha=0.5, linewidth=1)[0]
                bundle_lines.append(line)
            
            # Draw bundles for binocular cameras
            if binocular_poses is not None:
                for cam_idx, (R, t) in final_cameras['binocular'].items():
                    pos = t.flatten()
                    R_mat = R if len(R.shape) == 2 else R.reshape(3, 3)
                    dir_vec = R_mat[:, 2]  # Z-axis direction
                    bundle_end = pos + dir_vec * bundle_length
                    line = ax.plot([pos[0], bundle_end[0]],
                                 [pos[1], bundle_end[1]],
                                 [pos[2], bundle_end[2]],
                                 '--', color=colors[cam_idx % len(colors)],
                                 alpha=0.5, linewidth=1)[0]
                    bundle_lines.append(line)
        
        fig.canvas.draw_idle()
    
    # Plot reference camera (camera 1) at origin
    ref_pos = np.zeros(3)
    ref_R = np.eye(3)
    # Plot reference camera position
    ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c='white', s=200, alpha=1.0, marker='*')
    # Plot reference camera orientation
    axis_length = np.mean(range_pos) * 0.15  # Make reference camera axes slightly larger
    for axis_idx, (axis, axis_color) in enumerate(zip(ref_R.T, ['r', 'g', 'b'])):
        ax.quiver(ref_pos[0], ref_pos[1], ref_pos[2],
                axis[0], axis[1], axis[2],
                length=axis_length, color=axis_color, alpha=1.0, linewidth=2)
    ax.text(ref_pos[0], ref_pos[1], ref_pos[2], 'Ref Cam (1)', 
            fontsize=12, weight='bold', color='white')
    
    # Plot other cameras
    for cam_idx, camera_poses_history in all_camera_poses_history.items():
        color = colors[cam_idx % len(colors)]
        
        # Plot camera positions for this camera
        for i, (R, t) in enumerate(camera_poses_history):
            pos = t.flatten()
            
            # Store final camera pose for bundle visualization
            if i == len(camera_poses_history)-1:
                final_cameras['multi'][cam_idx] = (R, t)
            
            # Plot camera position
            size = 100 if i == len(camera_poses_history)-1 else 30
            alpha = 1.0 if i == len(camera_poses_history)-1 else 0.5
            marker = '^' if i == len(camera_poses_history)-1 else 'o'
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=size, alpha=alpha, marker=marker)
            
            # Plot camera orientation (principal axis)
            axis_length = np.mean(range_pos) * 0.1  # Scale arrows relative to scene size
            R_mat = R if len(R.shape) == 2 else R.reshape(3, 3)
            for axis_idx, (axis, axis_color) in enumerate(zip(R_mat.T, ['r', 'g', 'b'])):
                if i == len(camera_poses_history)-1:  # Only show axes for final position
                    ax.quiver(pos[0], pos[1], pos[2],
                            axis[0], axis[1], axis[2],
                            length=axis_length, color=axis_color, alpha=0.7)
            
            # Add labels with white color for dark background
            if i == 0:
                ax.text(pos[0], pos[1], pos[2], f'Cam{cam_idx} Start', 
                       fontsize=8, color='white')
            elif i == len(camera_poses_history)-1:
                ax.text(pos[0], pos[1], pos[2], f'Cam{cam_idx} Multi', 
                       fontsize=10, color='white')
    
    # Plot binocular calibration results if provided
    if binocular_poses is not None:
        for cam_idx, (R, t) in binocular_poses.items():
            color = colors[cam_idx % len(colors)]
            pos = t.flatten()
            
            # Store for bundle visualization
            final_cameras['binocular'][cam_idx] = (R, t)
            
            # Plot camera position
            ax.scatter(pos[0], pos[1], pos[2], c=color, s=100, alpha=1.0, marker='o')
            
            # Plot camera orientation
            R_mat = R if len(R.shape) == 2 else R.reshape(3, 3)
            for axis_idx, (axis, axis_color) in enumerate(zip(R_mat.T, ['r', 'g', 'b'])):
                ax.quiver(pos[0], pos[1], pos[2],
                        axis[0], axis[1], axis[2],
                        length=axis_length, color=axis_color, alpha=0.7)
            
            ax.text(pos[0], pos[1], pos[2], f'Cam{cam_idx} Bin', 
                   fontsize=10, color='white')
    
    # Set axis limits with padding
    max_range = max(abs(min_pos.min()), abs(max_pos.max()))
    ax.set_xlim([-max_range - padding[0], max_range + padding[0]])
    ax.set_ylim([-max_range - padding[1], max_range + padding[1]])
    ax.set_zlim([-max_range - padding[2], max_range + padding[2]])
    
    # Set labels and title with white color
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    ax.set_title('Camera Calibration Results\n' +
                'Circle: Binocular, Triangle: Multi-Camera\n' +
                'Press B to toggle bundles (-- Binocular, : Multi)', 
                color='white', pad=20)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    
    # Make background completely black
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Remove all grid lines and panes
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Set the color of the axis panes to none
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    
    # Remove grid lines completely
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Keep only the axis labels
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    
    # Add mouse interaction
    def on_scroll(event):
        curr_xlim = ax.get_xlim()
        curr_ylim = ax.get_ylim()
        curr_zlim = ax.get_zlim()
        
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None:
            return
        
        base_scale = 0.8
        if event.button == 'up':
            scale_factor = 1/base_scale
        else:
            scale_factor = base_scale
        
        ax.set_xlim([xdata - (xdata - curr_xlim[0]) * scale_factor,
                     xdata + (curr_xlim[1] - xdata) * scale_factor])
        ax.set_ylim([ydata - (ydata - curr_ylim[0]) * scale_factor,
                     ydata + (curr_ylim[1] - ydata) * scale_factor])
        ax.set_zlim([curr_zlim[0] * scale_factor, curr_zlim[1] * scale_factor])
        
        fig.canvas.draw_idle()
    
    def on_mouse_move(event):
        if event.button == 3:  # Right mouse button
            dx = event.xdata - on_mouse_move.last_x if hasattr(on_mouse_move, 'last_x') else 0
            dy = event.ydata - on_mouse_move.last_y if hasattr(on_mouse_move, 'last_y') else 0
            
            curr_xlim = ax.get_xlim()
            curr_ylim = ax.get_ylim()
            
            ax.set_xlim([curr_xlim[0] - dx, curr_xlim[1] - dx])
            ax.set_ylim([curr_ylim[0] - dy, curr_ylim[1] - dy])
            
            fig.canvas.draw_idle()
        
        on_mouse_move.last_x = event.xdata
        on_mouse_move.last_y = event.ydata
    
    def on_key(event):
        nonlocal show_bundles
        if event.key == 'b':
            show_bundles = not show_bundles
            draw_bundles()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
