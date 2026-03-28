import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from CloudCTUtils import *
from CloudCT_NoiseUtils import *
import glob
import re
import argparse
from multiprocessing import Pool
from itertools import repeat
import scipy.io as sio
from mayavi import mlab
from skimage import filters, morphology
import sys
from dataclasses import dataclass
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Assumes this path exists in your environment
sys.path.append('./CloudCT_utils/')
try:
    from CloudCTUtils import *
except ImportError:
    print("Warning: CloudCTUtils not found. Ensure the path is correct.")
    
# Assumes this path exists in your environment
sys.path.append('./vadim_generate_data/')
try:
    from vadim_generate_data import *
except ImportError:
    print("Warning: CloudCTUtils not found. Ensure the path is correct.")

    
    
class InteractivePklViewer:
    def __init__(self, directory_path, target_key='images_noise'):
        """
        Initializes the interactive viewer with a new button for 3D mask projection.
        """
        self.directory_path = directory_path
        self.target_key = target_key
        
        # 1. Find and sort all .pkl files in the directory
        self.filepaths = sorted(glob.glob(os.path.join(directory_path, "*.pkl")))
        self.current_idx = 0
        
        if not self.filepaths:
            print(f"No .pkl files found in {directory_path}")
            return
            
        print(f"Found {len(self.filepaths)} .pkl files. Opening viewer...")

        # 2. Setup the main figure and subplots
        self.fig, self.axes = plt.subplots(2, 5, figsize=(15, 8))
        self.axes_flat = self.axes.flatten()
        
        # Make room at the bottom for the buttons
        self.fig.subplots_adjust(bottom=0.2)
        
        # 3. Create the buttons
        # [left, bottom, width, height]
        axprev = self.fig.add_axes([0.30, 0.05, 0.1, 0.075])
        axnext = self.fig.add_axes([0.45, 0.05, 0.1, 0.075])
        axmask = self.fig.add_axes([0.65, 0.05, 0.15, 0.075]) # The New Button!
        
        self.bprev = Button(axprev, '◄ Previous')
        self.bnext = Button(axnext, 'Next ►')
        self.bmask = Button(axmask, 'Project 3D Masks', color='lightblue')
        
        # Link buttons to their functions
        self.bprev.on_clicked(self.prev_file)
        self.bnext.on_clicked(self.next_file)
        self.bmask.on_clicked(self.show_masks_overlay) # Link new function

        # 4. Draw the first file
        self.update_plot()
        plt.show()

    def load_data(self, filepath):
        """Loads the pkl file and extracts the specific image array."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if self.target_key not in data:
            return None, data
            
        images = np.array(data[self.target_key])
        
        # Apply your specific slicing logic
        if len(images.shape) == 5:
            images = images[0]          # Remove first dimension
            images = images[:, 0, :, :] # Take first channel
            
        return images, data

    def update_plot(self):
        """Clears the axes and draws the base images for the current file."""
        filepath = self.filepaths[self.current_idx]
        filename = os.path.basename(filepath)
        images, _ = self.load_data(filepath)

        for ax in self.axes_flat:
            ax.clear()
            ax.axis('off')

        if images is not None:
            vmin, vmax = np.min(images), np.max(images)
            num_images_to_show = min(10, len(images))
            
            for i in range(num_images_to_show):
                ax = self.axes_flat[i]
                img = images[i]
                
                # Plot the image (Add .T here if your specific arrays need transposing)
                ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
                ax.set_title(f'View {i+1}')
                
            self.fig.suptitle(f"File: {filename}\n({self.current_idx + 1} of {len(self.filepaths)})", fontsize=14)
        else:
            self.fig.suptitle(f"File: {filename}\n[Key '{self.target_key}' not found]", color='red')

        self.fig.canvas.draw()

    def next_file(self, event):
        self.current_idx = (self.current_idx + 1) % len(self.filepaths)
        self.update_plot()

    def prev_file(self, event):
        self.current_idx = (self.current_idx - 1) % len(self.filepaths)
        self.update_plot()

    # ==========================================
    # THE NEW PROJECTION & OVERLAY LOGIC
    # ==========================================
    def get_dense_points(self, mask_3d, dx, dy, dz, points_per_voxel=10):
        """Helper to convert a boolean 3D mask into a dense point cloud."""
        points_3d_indices = np.argwhere(mask_3d)
        if len(points_3d_indices) == 0:
            return np.empty((0, 3))
            
        voxel_sizes = np.array([dx, dy, dz])
        points_3d = points_3d_indices * voxel_sizes
        
        base_points = np.repeat(points_3d, points_per_voxel, axis=0)
        random_offsets = np.random.rand(base_points.shape[0], 3) * voxel_sizes
        return base_points + random_offsets

    def show_masks_overlay(self, event):
        """Extracts 3D masks, projects them to 2D, and plots them in a new window."""
        filepath = self.filepaths[self.current_idx]
        images, data = self.load_data(filepath)
        
        if images is None:
            print("Cannot project masks: Base images missing.")
            return

        print(f"Projecting masks for {os.path.basename(filepath)}...")
        
        # 1. Extract necessary data
        grid = data['grid']
        cameras_P = data['cameras_P']
        ext_mask = data['ext'] > 0      # Ground Truth
        pred_mask = data['mask'] > 0    # Reconstructed/Loaded Mask
        
        img_width, img_height = images[0].shape
        
        # 2. Extract grid dimensions exactly as in your script
        xgrid = np.float32(grid[0])
        dx = xgrid[1] - xgrid[0]
        ygrid = np.float32(grid[1])
        dy = ygrid[1] - ygrid[0]
        zgrid = np.float32(grid[2][:-2])
        dz = zgrid[1] - zgrid[0]

        # 3. Generate dense point clouds
        gt_points = self.get_dense_points(ext_mask, dx, dy, dz)
        pred_points = self.get_dense_points(pred_mask, dx, dy, dz)
        
        # 4. Project to 2D (Using your existing external function)
        gt_2d_masks = project_3d_mask_to_2d(gt_points, cameras_P, img_width, img_height)
        pred_2d_masks = project_3d_mask_to_2d(pred_points, cameras_P, img_width, img_height)

        # 5. Plotting in a NEW figure so we don't destroy the navigation window
        fig_mask, axes_mask = plt.subplots(2, 5, figsize=(16, 8))
        fig_mask.suptitle(f"Mask Projections: {os.path.basename(filepath)}\nRed = GT ('ext') | Green = Loaded ('mask')", fontsize=14, fontweight='bold')
        
        vmin, vmax = np.min(images), np.max(images)
        overlay_opacity = 0.4
        
        for i, ax in enumerate(axes_mask.flat):
            if i >= len(images): break
            
            # Plot Base Image
            ax.imshow(images[i], cmap='gray', vmin=vmin, vmax=vmax)
            
            # Create a blank RGBA overlay canvas
            overlay = np.zeros((img_height, img_width, 4), dtype=np.float32)
            
            # Extract 2D masks for this view (assuming project_3d_mask_to_2d returns shape (10, H, W, 1) or similar)
            gt_m = gt_2d_masks[i].squeeze() if gt_2d_masks[i].ndim > 2 else gt_2d_masks[i]
            pred_m = pred_2d_masks[i].squeeze() if pred_2d_masks[i].ndim > 2 else pred_2d_masks[i]
            
            # Map GT to RED channel
            overlay[..., 0] = 1.0 * gt_m
            # Map Pred to GREEN channel
            overlay[..., 1] = 1.0 * pred_m
            
            # Set Alpha where either mask exists
            combined_mask = np.logical_or(gt_m > 0, pred_m > 0)
            overlay[..., 3] = overlay_opacity * combined_mask
            
            ax.imshow(overlay)
            ax.set_title(f"View {i+1}")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()


def project_3d_mask_to_2d(points_3d, cameras_P, img_width, img_height):
    """
    Projects a set of 3D points onto multiple camera views to create 2D masks.
    
    Args:
        points_3d: (N, 3) numpy array of 3D coordinates where your mask is 1.
        cameras_P: List of 10 projection matrices (3x4), where P = K @ Extrinsic.
        img_width: Width of the original images.
        img_height: Height of the original images.
        
    Returns:
        List of (img_height, img_width) 2D binary masks.
        
    Notes:
    You hit the nail on the head! Because your intrinsic matrix (k) is normalized to a sensor size of [-1, 1],
    the projection math is mapping your 3D points into a Normalized Device Coordinate (NDC) space, not directly into pixel coordinates.

    In your current setup, the center of the image is (0, 0). However, when we create
    a 2D NumPy array for an image, the top-left corner must be (0, 0) and the bottom-right corner must be (img_width, img_height).

    To fix this, we just need to add a translation and scaling step to convert those [-1, 1]
    values into actual pixel row/column indices right after dividing by the depth.

    """
    # 1. Convert 3D points to homogeneous coordinates (x, y, z, 1)
    num_points = points_3d.shape[0]
    points_homog = np.hstack((points_3d, np.ones((num_points, 1))))
    
    projected_masks = []
    
    for P in cameras_P:
        # 1. Project points
        camera_coords = P @ points_homog.T
        depths = camera_coords[2, :]
    
        # 2. Divide by depth to get Normalized Coordinates [-1, 1]
        u_norm = camera_coords[0, :] / (depths + 1e-8)
        v_norm = camera_coords[1, :] / (depths + 1e-8)
    
        # 3. Convert Normalized Coordinates to Pixel Coordinates
        # Scale from [-1, 1] to [0, 1], then multiply by image dimensions
        u = (u_norm + 1.0) / 2.0 * img_width
    
        # IMPORTANT Y-AXIS NOTE: 
        # Standard normalized coordinates usually have Y pointing UP.
        # Image arrays have Y (rows) pointing DOWN. 
        # If your clouds render upside down, change the line below to:
        # v = (1.0 - v_norm) / 2.0 * img_height
        v = (v_norm + 1.0) / 2.0 * img_height 
    
        # 4. Round to nearest pixel integer
        u = np.round(u).astype(int)
        v = np.round(v).astype(int)
    
        # 5. Filter out points outside the bounds or behind the camera
        valid_indices = (
                (u >= 0) & (u < img_width) & 
                (v >= 0) & (v < img_height) & 
                (depths > 0)
            )
    
        valid_u = u[valid_indices]
        valid_v = v[valid_indices]
    
       
        mask_2d = np.zeros((img_height, img_width), dtype=np.float32)
    
        mask_2d[valid_u, valid_v] = True  # Red channel
    
        projected_masks.append(mask_2d)
        
    return projected_masks
        


# ==========================================
# Execution
# ==========================================
if __name__ == '__main__':
    PKL_DIRECTORY = "/wdata_visl/tamarsd/NN_Data/vadim_runs/up_sop_data_rando/train/ocean_brdf/"
    viewer = InteractivePklViewer(directory_path=PKL_DIRECTORY, target_key='images_noise')
