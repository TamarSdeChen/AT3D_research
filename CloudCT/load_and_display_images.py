"""
Simple script to load pickle file and display images.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_display_images():
    """
    Load pickle file from hardcoded path and display first 5 images.
    """
    # Hardcoded pickle file path
    pkl_path = "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/train/cloud_results_0.pkl"
    
    # Load pickle file
    print(f"Loading pickle file from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get images from dictionary
    images = data['images']
    images = np.array(images)
    
    print(f"Images shape: {images.shape}")
    
    # Plot first 5 images
    num_images_to_show = 5
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(num_images_to_show):
        ax = axes[i]
        img = images[i]  # Grayscale image: 116x116
        im = ax.imshow(img, cmap='gray')
        ax.set_title(f'Image {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.suptitle(f'First {num_images_to_show} Images', fontsize=16)
    plt.tight_layout()
    
    # Save to PNG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "saved_images")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'first_5_images.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def load_at3d_pkl():
    """
    Load AT3D pickle file from hardcoded path and display first 5 images from all image keys.
    """
    # Hardcoded AT3D pickle file path
    pkl_path = "/wdata/tamarsd/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/const_sun_random_rotation/train/cloud_results_0.pkl"
    
    # Load pickle file
    print(f"Loading AT3D pickle file from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Setup output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "saved_images")
    os.makedirs(output_dir, exist_ok=True)
    
    # List of all image keys to plot
    image_keys = ['images', 'images_scatter', 'images_clean', 'images_clean_scatter']
    
    # Plot each image key
    for key in image_keys:
        if key not in data:
            print(f"\n'{key}' key not found in data, skipping...")
            continue
        
        print(f"\nProcessing '{key}'...")
        images = data[key]
        images = np.array(images)
        
        print(f"Original {key} shape: {images.shape}")
        
        # Handle shape (1, 10, 3, 116, 116)
        # Remove first dimension and take first channel (index 0)
        images = images[0]  # Remove first dimension -> (10, 3, 116, 116)
        images = images[:, 0, :, :]  # Take first channel -> (10, 116, 116)
        
        print(f"Processed {key} shape: {images.shape}")
        
        # Plot first 5 images
        num_images_to_show = 5
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i in range(num_images_to_show):
            ax = axes[i]
            img = images[i]  # Grayscale image: 116x116
            im = ax.imshow(img, cmap='gray')
            ax.set_title(f'Image {i+1}')
            ax.axis('off')
            plt.colorbar(im, ax=ax)
        
        # Hide unused subplot
        axes[5].axis('off')
        
        plt.suptitle(f'AT3D - First {num_images_to_show} {key.replace("_", " ").title()}', fontsize=16)
        plt.tight_layout()
        
        # Save to PNG
        output_filename = f'at3d_first_5_{key}.png'
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    print(f"\nAll images saved to '{output_dir}' directory")


def compare_roironen_vs_at3d(cloud_number, seven=False, image_type='clean'):
    """
    Load and compare images from roironen and AT3D datasets side by side.
    Creates a 2x4 subplot with:
    - First row: 4 images from roironen dataset
    - Second row: 4 images from AT3D dataset
    
    Args:
        cloud_number: The cloud result number to load (e.g., 0, 1, 2, ...)
        seven: Whether to use the subset of seven clouds dataset
        image_type: Type of AT3D images to load - 'clean' or 'noise'
    """
    # Define paths
    if seven:
        roironen_path = f"/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/test/cloud_results_{cloud_number}.pkl"
        at3d_path = f"/wdata/tamarsd/NN_Data/subset_of_seven_clouds/CloudCT_SIMULATIONS_AT3D_no_rotation_50_wavelength_try/const_sun_random_rotation/cloud_results_{cloud_number}.pkl"
    else:
        roironen_path = f"/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/10cameras_20m/train/cloud_results_{cloud_number}.pkl"
        at3d_path = f"/wdata/tamarsd/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/const_sun_random_rotation/train/cloud_results_{cloud_number}.pkl"
    
    # Load roironen pickle file
    print(f"Loading roironen pickle file from: {roironen_path}")
    with open(roironen_path, 'rb') as f:
        roironen_data = pickle.load(f)
    
    # Load AT3D pickle file
    print(f"Loading AT3D pickle file from: {at3d_path}")
    with open(at3d_path, 'rb') as f:
        at3d_data = pickle.load(f)
    
    # Get roironen images
    roironen_images = np.array(roironen_data['images'])
    print(f"Roironen images shape: {roironen_images.shape}")
    
    # Get AT3D images - handle shape (1, 10, 3, 116, 116)
    image_key = f'images_{image_type}'
    at3d_images = np.array(at3d_data[image_key][0,...])
    print(f"AT3D images ({image_type}) original shape: {at3d_images.shape}")
    
    # Create 2x4 subplot
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Plot first row: roironen images (4 images)
    for i in range(4):
        ax = axes[0, i]
        img = roironen_images[i]
        im = ax.imshow(img, cmap='gray')
        ax.set_title(f'Roironen - Cam {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Plot second row: AT3D images (4 images)
    for i in range(4):
        ax = axes[1, i]
        img = at3d_images[i]
        im = ax.imshow(img, cmap='gray')
        ax.set_title(f'AT3D - Cam {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Roi ronen shdom vs AT3D ({image_type}) Comparison - Cloud {cloud_number}', fontsize=16)
    plt.tight_layout()
    
    # Setup output directory and save
    output_dir = f"/wdata/tamarsd/AT3D_research/CloudCT/Figures/subset_of_7_clouds_no_rotation_50_wavelenght_{image_type}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'subset_of_7_clouds_{cloud_number}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()
    plt.close()


if __name__ == '__main__':
    # load_and_display_images()
    # load_at3d_pkl()
    compare_roironen_vs_at3d(6066, seven=True, image_type='clean')  # Options: 'clean' or 'noise'

    # seven clouds are: test: 6004,6037,6066 train: 6010,6502, 6473, 6293
