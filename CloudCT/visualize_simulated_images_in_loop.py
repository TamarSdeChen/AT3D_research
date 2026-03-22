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


class InteractivePklViewer:
    def __init__(self, directory_path, target_key='images_noise'):
        """
        Initializes the interactive viewer.
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
        self.fig, self.axes = plt.subplots(2, 5, figsize=(12, 8))
        self.axes_flat = self.axes.flatten()
        
        # Make room at the bottom for the buttons
        self.fig.subplots_adjust(bottom=0.2)
        
        # 3. Create the buttons
        # [left, bottom, width, height]
        axprev = self.fig.add_axes([0.35, 0.05, 0.1, 0.075])
        axnext = self.fig.add_axes([0.55, 0.05, 0.1, 0.075])
        
        self.bprev = Button(axprev, '◄ Previous')
        self.bnext = Button(axnext, 'Next ►')
        
        # Link buttons to their functions
        self.bprev.on_clicked(self.prev_file)
        self.bnext.on_clicked(self.next_file)

        # 4. Draw the first file
        self.update_plot()
        plt.show()

    def load_data(self, filepath):
        """Loads the pkl file and extracts the specific image array."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        if self.target_key not in data:
            return None
            
        images = np.array(data[self.target_key])
        
        # Apply your specific slicing logic
        # Handle shape (1, 10, 3, 116, 116) -> (10, 116, 116)
        if len(images.shape) == 5:
            images = images[0]          # Remove first dimension
            images = images[:, 0, :, :] # Take first channel
            
        return images

    def update_plot(self):
        """Clears the axes and draws the images for the current file."""
        filepath = self.filepaths[self.current_idx]
        filename = os.path.basename(filepath)
        images = self.load_data(filepath)

        # Clear all previous images
        for ax in self.axes_flat:
            ax.clear()
            ax.axis('off')

        if images is not None:
            num_images_to_show = min(10, len(images))
            
            for i in range(num_images_to_show):
                ax = self.axes_flat[i]
                img = images[i].T
                
                # Plot the image
                ax.imshow(img, cmap='gray')
                ax.set_title(f'Image {i+1}')
                ax.axis('off')
                
            self.fig.suptitle(f"File: {filename}\n({self.current_idx + 1} of {len(self.filepaths)})", fontsize=14)
        else:
            self.fig.suptitle(f"File: {filename}\n[Key '{self.target_key}' not found]", color='red')

        # Force the canvas to redraw
        self.fig.canvas.draw()

    def next_file(self, event):
        """Moves to the next file, looping back to 0 if at the end."""
        self.current_idx = (self.current_idx + 1) % len(self.filepaths)
        self.update_plot()

    def prev_file(self, event):
        """Moves to the previous file, looping to the end if at 0."""
        self.current_idx = (self.current_idx - 1) % len(self.filepaths)
        self.update_plot()

# ==========================================
# How to run the viewer:
# ==========================================
if __name__ == '__main__':
    
    if (1):
        
        # Define the directory containing your multiple .pkl files
        #PKL_DIRECTORY = "/wdata/tamarsd/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/const_sun_random_rotation/train"
        PKL_DIRECTORY = "/wdata_visl/tamar_nadav_generated_clouds/2026/Vadim_tune_AT3D_research/up_sop_data_rando/train/ocean_brdf/"
        # You can change 'images' to 'images_scatter', 'images_clean', etc.
        viewer = InteractivePklViewer(directory_path=PKL_DIRECTORY, target_key='images_noise')
