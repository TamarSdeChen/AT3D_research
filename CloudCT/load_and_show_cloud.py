"""
Script to load and visualize cloud data from pickle files.
Usage: python load_and_show_cloud.py <path_to_pkl_file>
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
from datetime import datetime


def load_cloud_data(pkl_path):
    """Load cloud data from pickle file."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")
    
    print(f"Loading cloud data from: {pkl_path}")
    with open(pkl_path, 'rb') as infile:
        cloud = pickle.load(infile)
    
    return cloud


def print_cloud_info(cloud):
    """Print information about the loaded cloud data."""
    print("\n" + "="*60)
    print("CLOUD DATA INFORMATION")
    print("="*60)
    
    print("\nKeys in cloud dictionary:")
    for key in cloud.keys():
        if isinstance(cloud[key], np.ndarray):
            print(f"  {key}: shape {cloud[key].shape}, dtype {cloud[key].dtype}")
        else:
            print(f"  {key}: {type(cloud[key])} = {cloud[key]}")
    
    # Print metadata if available
    print("\nMetadata:")
    if 'cloud_path' in cloud:
        print(f"  Cloud path: {cloud['cloud_path']}")
    if 'sun_zenith' in cloud:
        print(f"  Sun zenith: {cloud['sun_zenith']}")
    if 'sun_azimuth' in cloud:
        print(f"  Sun azimuth: {cloud['sun_azimuth']}")
    if 'wind_speed' in cloud:
        print(f"  Wind speed: {cloud['wind_speed']}")
    if 'rotation_angle_deg' in cloud:
        print(f"  Rotation angles (deg): {cloud['rotation_angle_deg']}")
    
    print("="*60 + "\n")


def plot_images(images, title_prefix="Images", save_dir=None):
    """Plot I, Q, U components of images."""
    if images is None or len(images) == 0:
        print(f"No {title_prefix} to plot")
        return
    
    images = np.array(images)
    print(f"Plotting {title_prefix}, shape: {images.shape}")
    
    # Handle different possible shapes
    if len(images.shape) == 5:  # (num_rotations, num_images, 3, H, W)
        num_rotations, num_images = images.shape[0], images.shape[1]
        images = images.reshape(-1, *images.shape[2:])  # Flatten rotations
        print(f"  Flattened from rotations: new shape {images.shape}")
    elif len(images.shape) == 4:  # (num_images, 3, H, W)
        num_images = images.shape[0]
    elif len(images.shape) == 3:  # (num_images, H, W) - grayscale images
        num_images = images.shape[0]
        # Add channel dimension for compatibility with plotting code
        images = images[:, np.newaxis, :, :]  # (num_images, 1, H, W)
        print(f"  Added channel dimension: new shape {images.shape}")
    else:
        print(f"Unexpected image shape: {images.shape}")
        return
    
    # Limit to 9 images for display
    num_images_to_show = min(9, num_images)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine number of channels (1 for grayscale, 3 for I/Q/U)
    num_channels = images.shape[1] if len(images.shape) >= 2 else 1
    stokes_components = ['I'] if num_channels == 1 else ['I', 'Q', 'U']
    
    # Plot I, Q, U components (matching the original plot_cloud_images pattern)
    for stokes_idx, stokes_name in enumerate(stokes_components):
        fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
        # fig.subplots_adjust(hspace=0.2, wspace=0.2)
        axarr = axarr.flatten()
        
        for i in range(num_images_to_show):
            ax = axarr[i]
            # Match original pattern: squeeze and copy
            image = np.squeeze(images[i].copy())
            
            # Handle image shape - should be (3, H, W) or (1, H, W) or (H, W) after squeeze
            if len(image.shape) == 3:  # (3, H, W) or (1, H, W)
                if image.shape[0] == 1:  # Single channel grayscale
                    im_data = image[0, ...]  # Extract the single channel
                else:  # Multi-channel (I, Q, U)
                    im_data = image[stokes_idx, ...]
            elif len(image.shape) == 2:  # (H, W) - single channel after squeeze
                im_data = image
            else:
                print(f"Unexpected image shape at index {i}: {image.shape}")
                ax.axis('off')
                continue
            
            im = ax.imshow(im_data, cmap='gray')
            ax.set_title(f'Image {i+1}')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.01)
            plt.colorbar(im, cax=cax)
        
        # Hide unused subplots
        for i in range(num_images_to_show, 9):
            axarr[i].axis('off')
        
        fig.suptitle(f'{title_prefix} - {stokes_name}', size=16, y=0.95)
        
        if save_dir:
            filename = f"{title_prefix.lower().replace(' ', '_')}_{stokes_name}_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        else:
            plt.show()
        
        plt.close(fig)


def plot_masks(mask, mask_morph=None, save_dir=None):
    """Plot 3D masks as slices."""
    if mask is None:
        print("No mask to plot")
        return
    
    mask = np.array(mask)
    print(f"Plotting mask, shape: {mask.shape}")
    
    # Handle different mask shapes
    if len(mask.shape) == 4:  # (num_rotations, D, H, W)
        mask = mask[0]  # Take first rotation
    elif len(mask.shape) == 3:  # (D, H, W)
        pass
    else:
        print(f"Unexpected mask shape: {mask.shape}")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot slices through the mask
    depth, height, width = mask.shape
    slice_indices = [depth//4, depth//2, 3*depth//4]
    
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(15, 5))
    if len(slice_indices) == 1:
        axes = [axes]
    
    for ax, idx in zip(axes, slice_indices):
        im = ax.imshow(mask[idx, ...], cmap='gray')
        ax.set_title(f'Mask slice {idx}/{depth}')
        plt.colorbar(im, ax=ax)
    
    fig.suptitle('Mask (3D slices)', size=16)
    
    if save_dir:
        filename = f"mask_{timestamp}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
    else:
        plt.show()
    
    plt.close(fig)
    
    # Plot morphological mask if available
    if mask_morph is not None:
        mask_morph = np.array(mask_morph)
        if len(mask_morph.shape) == 4:
            mask_morph = mask_morph[0]
        
        fig, axes = plt.subplots(1, len(slice_indices), figsize=(15, 5))
        if len(slice_indices) == 1:
            axes = [axes]
        
        for ax, idx in zip(axes, slice_indices):
            im = ax.imshow(mask_morph[idx, ...], cmap='gray')
            ax.set_title(f'Mask Morph slice {idx}/{depth}')
            plt.colorbar(im, ax=ax)
        
        fig.suptitle('Mask Morphological (3D slices)', size=16)
        
        if save_dir:
            filename = f"mask_morph_{timestamp}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        else:
            plt.show()
        
        plt.close(fig)


def show_cloud_data(pkl_path, save_dir=None, show_images=True, show_masks=True):
    """Main function to load and display cloud data."""
    # Load data
    cloud = load_cloud_data(pkl_path)
    
    # Print information
    print_cloud_info(cloud)
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Plot images
    if show_images:
        if 'images' in cloud:
            plot_images(cloud['images'], title_prefix="Images", save_dir=save_dir)
        
        if 'images_scatter' in cloud:
            plot_images(cloud['images_scatter'], title_prefix="Images Scatter", save_dir=save_dir)
        
        if 'images_clean' in cloud:
            plot_images(cloud['images_clean'], title_prefix="Images Clean", save_dir=save_dir)
        
        if 'images_clean_scatter' in cloud:
            plot_images(cloud['images_clean_scatter'], title_prefix="Images Clean Scatter", save_dir=save_dir)
    
    # Plot masks
    if show_masks:
        mask = cloud.get('mask', None)
        mask_morph = cloud.get('mask_morph', None)
        plot_masks(mask, mask_morph, save_dir=save_dir)
    
    print("\nVisualization complete!")


def main():
    """Command line interface."""
    if len(sys.argv) < 2:
        print("Usage: python load_and_show_cloud.py <path_to_pkl_file> [save_dir]")
        print("\nExample:")
        print("  python load_and_show_cloud.py cloud_results_649.pkl")
        print("  python load_and_show_cloud.py cloud_results_649.pkl ./output")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    show_cloud_data(pkl_path, save_dir=save_dir)


if __name__ == "__main__":
    pkl_path = "/wdata/roironen/Data/BOMEX_128x128x100_5000CCN_50m_micro_256/10cameras_20m/test/cloud_results_0.pkl"
    #"/wdata_visl/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/const_env_params/train/cloud_results_0.pkl"
    #"/wdata/tamarsd/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D_ONLY_I_no_rotation/const_sun_random_rotation/cloud_results_0.pkl"
    #"/wdata/inbalkom/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/train/cloud_results_0.pkl"
    
    save_dir = "/wdata/tamarsd/AT3D_research/CloudCT/results_roi_shdom/cloud_results_0_plots"
    # "/wdata/tamarsd/AT3D_research/CloudCT/results_TAMAR/cloud_results_0_plots"
    show_cloud_data(pkl_path, save_dir=save_dir)
    # main()

