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


def visualize_cloud_isosurface(points_3d_physical, vol_data, xgrid, ygrid, zgrid, threshold_factor=0.1, color=(1, 1, 1)):
    
    vol_data = np.nan_to_num(vol_data).astype(float)
    max_val = np.max(vol_data)
    
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
    dx = xgrid[1] - xgrid[0]
    dy = ygrid[1] - ygrid[0]
    dz = zgrid[1] - zgrid[0]
    
    xp = points_3d_physical[:, 0]
    yp = points_3d_physical[:, 1]
    zp = points_3d_physical[:, 2]    

    # 2. Define the Isosurface Threshold
    # In your work, this corresponds to the learned threshold tau [cite: 1264]
    iso_val = threshold_factor * max_val
    print(f"Visualizing Isosurface at threshold: {iso_val:.4f}")

    mlab.figure(size=(1000, 800), bgcolor=(0.1, 0.1, 0.1), fgcolor=(1, 1, 1))
    voxel_size = 0.05
    mlab.points3d(
        xp, yp, zp,
        mode='cube',
        color=(0.1, 0.5, 0.8),    # A nice cloud-blue color (RGB, 0.0 to 1.0)
        scale_factor=voxel_size, 
        scale_mode='none',
        opacity=0.8               # Slight transparency helps when viewing dense clouds
    )
    # 3. Create the Pipeline
    src = mlab.pipeline.scalar_field(X, Y, Z, vol_data)
    src.spacing = [dx, dy, dz]

    # 4. Generate the Isosurface (The 3D Envelope)
    # This renders the boundary of the non-opaque object [cite: 54]
    iso = mlab.pipeline.iso_surface(src, 
                                    contours=[iso_val], 
                                    color=color, 
                                    opacity=0.7)
    
    # 5. Add mesh-lines to see the 3D structure better
    iso.actor.property.representation = 'surface'
    iso.actor.property.backface_culling = True
    mlab.outline()
    
    # 6. Final Annotations
    # ==========================================
    # Draw Absolute Origin and Reference Axes
    # ==========================================
    # Set the length of the axes (in km). Adjust this if your cloud is larger/smaller.
    axis_length = 1.0 
    
    # Draw a small white sphere exactly at the origin (0, 0, 0)
    mlab.points3d(0, 0, 0, color=(1, 1, 1), scale_factor=0.05, mode='sphere')
    
    # 1. X-Axis (Red)
    # quiver3d(x, y, z, u, v, w) -> starts at (x,y,z) and points in direction (u,v,w)
    mlab.quiver3d(0, 0, 0, axis_length, 0, 0, 
                  color=(1, 0, 0), mode='arrow', scale_factor=1)
    mlab.text3d(axis_length + 0.05, 0, 0, 'X', color=(1, 0, 0), scale=0.1)
    
    # 2. Y-Axis (Green)
    mlab.quiver3d(0, 0, 0, 0, axis_length, 0, 
                  color=(0, 1, 0), mode='arrow', scale_factor=1)
    mlab.text3d(0, axis_length + 0.05, 0, 'Y', color=(0, 1, 0), scale=0.1)
    
    # 3. Z-Axis (Blue)
    mlab.quiver3d(0, 0, 0, 0, 0, axis_length, 
                  color=(0, 0, 1), mode='arrow', scale_factor=1)
    mlab.text3d(0, 0, axis_length + 0.05, 'Z', color=(0, 0, 1), scale=0.1)
    # ==========================================
    # ==========================================
    
    mlab.orientation_axes()
    mlab.view(azimuth=145, elevation=65, distance='auto')
    
    mlab.show()


# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------
# ----------------------------------------------

def extract_cloud_mask(image, max_cloud_fraction=0.4, min_contrast_ratio=1.5):
    """
    Dynamically finds a cloud mask and raises a warning if glint dominates.
    """
    # 1. Find dynamic threshold using Otsu's method
    # It looks at the image histogram and finds the best valley to split classes
    threshold = filters.threshold_otsu(image)
    
    # 2. Create the raw binary mask
    raw_mask = image > threshold
    
    # 3. Clean up the mask (remove stray bright glint pixels)
    # min_size depends on your grid. 50 pixels is a safe start for a 100x100 grid.
    clean_mask = morphology.remove_small_objects(raw_mask, min_size=50)
    
    # --- WARNING LOGIC ---
    
    # Check 1: Did the mask grab too much of the image? (Glint blowout)
    cloud_fraction = np.sum(clean_mask) / clean_mask.size
    
    # Check 2: Is the contrast high enough?
    cloud_mean_brightness = np.mean(image[clean_mask])
    background_mean_brightness = np.mean(image[~clean_mask])
    
    # Avoid division by zero
    if background_mean_brightness < 1e-6:
        background_mean_brightness = 1e-6 
        
    contrast_ratio = cloud_mean_brightness / background_mean_brightness
    
    is_valid_cloud = True
    warning_msg = ""
    
    if cloud_fraction > max_cloud_fraction:
        is_valid_cloud = False
        warning_msg = f"WARNING: Mask covers {cloud_fraction*100:.1f}% of image. Glint likely dominates."
    elif contrast_ratio < min_contrast_ratio:
        is_valid_cloud = False
        warning_msg = f"WARNING: Low contrast (Ratio: {contrast_ratio:.2f}). Cloud brightness is below/near glint reflection."

    if not is_valid_cloud:
        print(warning_msg)
        # You can handle the "bad" mask here. 
        # For now, we return a blank mask and the False flag.
        return np.zeros_like(image, dtype=bool), False

    return clean_mask, True


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
    
    red_overlays = []
    
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
    
        # Create an RGBA overlay: (Height, Width, 4 channels)
        # Default is all zeros (completely transparent black)
        overlay = np.zeros((img_height, img_width, 4), dtype=np.float32)
    
        # Set the valid pixels to Red (R=1.0) and fully opaque (Alpha=1.0)
        overlay[valid_u, valid_v, 0] = 1.0  # Red channel
        overlay[valid_u, valid_v, 3] = 1.0  # Alpha channel
    
        red_overlays.append(overlay)
        
    return red_overlays
        
    
    
def plot_10_image_with_mask_comparison(images_set1, images_set2, masks, 
                             title1="Part 1: Current 10 Images", 
                             title2="Part 2: Comparison Images", 
                             cmap1='gray', cmap2='gray',
                             vmin=None, vmax=None):
    """
    Plots two sets of 10 images in a 2-part figure (each part is a 2x5 grid).
    
    Parameters:
    - images_set1: List or array of 10 images for the top section.
    - images_set2: List or array of 10 images for the bottom section (or masks for overlay).
    - title1, title2: Titles for the top and bottom subfigures.
    - cmap1, cmap2: Colormaps for the respective sets.
    - vmin, vmax: Intensity limits for the colorbar. Auto-calculated if None.
    """
    
    # Auto-calculate consistent color scaling based on the first set if not provided
    if vmin is None:
        vmin = np.min([np.min(img) for img in images_set1])
    if vmax is None:
        vmax = np.max([np.max(img) for img in images_set1])

    # Create the main figure and split it into top and bottom subfigures
    fig = plt.figure(figsize=(16, 10))
    subfigs = fig.subfigures(2, 1, hspace=0.1)
    img_height, img_width = images_set1[0].shape
    # ---------------------------------------------------------
    # TOP SECTION (Set 1)
    # ---------------------------------------------------------
    subfig1 = subfigs[0]
    subfig1.suptitle(title1, fontsize=16, fontweight='bold')
    axs1 = subfig1.subplots(2, 5)

    for i, ax in enumerate(axs1.flat):
        if i < len(images_set1):
            im1 = ax.imshow(images_set1[i], cmap=cmap1, vmin=vmin, vmax=vmax)
            ax.set_title(f"Image {i+1}")
        ax.axis('off')

    # Add colorbar for the top set
    subfig1.colorbar(im1, ax=axs1, shrink=0.8, label='Value')

    # ---------------------------------------------------------
    # BOTTOM SECTION (Set 2)
    # ---------------------------------------------------------
    subfig2 = subfigs[1]
    subfig2.suptitle(title2, fontsize=16, fontweight='bold')
    axs2 = subfig2.subplots(2, 5)

    for i, ax in enumerate(axs2.flat):
        if i < len(images_set2):
            # Just plot the second set of images independently
            im2 = ax.imshow(images_set2[i], cmap=cmap2, vmin=vmin, vmax=vmax)
            # cloud mask:
            overlay = np.zeros((img_height, img_width, 4), dtype=np.float32)
            mask = masks[i]
            # Set the valid pixels to Red (R=1.0) and fully opaque (Alpha=1.0)
            overlay[..., 1] = 1.0 * mask #   # Red channel
            overlay[..., 3] = 1.0 * mask
            ax.imshow(overlay, alpha=0.4)            
            ax.set_title(f"View {i+1} with static mask")
        ax.axis('off')

    # Add colorbar for the bottom set
    subfig2.colorbar(im2, ax=axs2, shrink=0.8, label='Value')

    return fig



# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------
# --------------------------------------------------



# ==========================================
# How to run the viewer:
# ==========================================
if __name__ == '__main__':
    
    if (1):
        
        # Define the directory containing your multiple .pkl files
        #PKL_DIRECTORY = "/wdata/tamarsd/NN_Data/BOMEX_256x256x100_5000CCN_50m_micro_256/CloudCT_SIMULATIONS_AT3D/const_sun_random_rotation/train"
        #PKL_DIRECTORY = "/wdata_visl/tamar_nadav_generated_clouds/2026/Vadim_tune_AT3D_research/up_sop_data_rando/train/ocean_brdf/"
        PKL_DIRECTORY = "/wdata_visl/tamarsd/NN_Data/vadim_runs/up_sop_data_rando/train/ocean_brdf/"
        # You can change 'images' to 'images_scatter', 'images_clean', etc.
        
        # space carving debug:
        
        # Paths
        atmosphere = xr.open_dataset('../data/ancillary/AFGL_summer_mid_lat.nc')
        reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 5.0]}) #ASK YOAV
        # get optical property generator #####
        config_path = "configs/params_cloudct.yaml"  # Default to CloudCT config            
        run_params = load_run_params(params_path=config_path)        
        
        wavelength_bands = run_params['wavelengths']
        mean_wavelengths = [np.mean(wavelength_band) for wavelength_band in wavelength_bands]
        
        path_mie: str = '../mie_tables'
        
        cloud_name = 1023 # 1023
        filepath = os.path.join(PKL_DIRECTORY, 'cloud_results_' + str(cloud_name) + '.pkl')
        filepath_nc = filepath.replace('.pkl', '.nc')
        
        print("\n>>> Loading cloud files (pkl,nc) I/O...")        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        loaded_sensor_list = load_sensor_list(filepath_nc)
        
        # --------------------------------------------------
        # --------------------------------------------------        # --------------------------------------------------
        # --------------------------------------------------        
        # --------------------------------------------------
        # --------------------------------------------------
        func_mask, func_stats = selective_space_carving(
              data,
              loaded_sensor_list, # the new saved sensor lits only used for dymanic space carving.
              run_params,
              threshold_scalar=0.6, # Scale the treshold loaded from the ran params file
              active_treshold_views_number = 5, # less than this number, the space carving considered as failure
              atmosphere_path='../data/ancillary/AFGL_summer_mid_lat.nc',
              user_excluded_views=None, 
              drop_glint_views=True,
              use_tamar_2d_masks=False, # True - if Tamar prefer to use the original (saved) masks, and so the space carving.
              n_jobs=40
        )
        
        # --------------------------------------------------
        # --------------------------------------------------        # --------------------------------------------------
        # --------------------------------------------------        # --------------------------------------------------
        # --------------------------------------------------        # --------------------------------------------------
        # --------------------------------------------------
        
        images_rte_simulated = np.array(data['images_noise'])
        
        cameras_pos_km = data['cameras_pos'] # shape 10,3
        cameras_P = data['cameras_P'] # data['cameras_P'].shape (10, 3, 4)
        # 'images_noise', 'images', 'mask', 'mask_morph', 'cloud_path', 
        # 'sun_zenith', 'sun_azimuth', 'cameras_pos', 'cameras_P', 'grid', 'ext'
        sun_zenith = data['sun_zenith']
        sun_azimuth = data['sun_azimuth']
        grid = data['grid']
        ext = data['ext']
        static_mask = data['mask']
        
        # extract the grid:
        xgrid = np.float32(grid[0])
        dx = xgrid[1] - xgrid[0]
        ygrid = np.float32(grid[1])
        dy =ygrid[1] - ygrid[0]
        zgrid = np.float32(grid[2][:-2])
        dz = zgrid[1] - zgrid[0]
        
        ext_mask = ext > 0
        #  3D points
        points_3d_indices = np.argwhere(ext_mask == 1)        
        # To convert your voxel indices into physical 3D coordinates, you just need to multiply 
        # the points_3d array by your voxel dimensions (dx, dy, dz) before you pass it to the projection function.
        # Because points_3d is an $(N, 3)$ numpy array, you
        # can scale all of the points simultaneously using vector multiplication.
        voxel_sizes = np.array([dx, dy, dz])
        # Scale to physical coordinates
        # This multiplies column 0 by dx, column 1 by dy, and column 2 by dz
        points_3d = (points_3d_indices * voxel_sizes) # + (voxel_sizes / 2.0)
        # Generate the Dense Sub-Voxel Points:
        # in order to fill in discrete grid gaps to create a solid-looking projection.
        N = 10  # Number of random points per voxel
        # If base_points has M points, this creates (M * N) points
        base_points = np.repeat(points_3d, N, axis=0)
        # Generate random offsets uniformly between 0 and the voxel size
        # np.random.rand generates values in [0, 1), which we scale by the voxel dimensions
        random_offsets = np.random.rand(base_points.shape[0], 3) * voxel_sizes
        # 3. Add the offsets to get the final dense point cloud
        dense_points_3d = base_points + random_offsets
        print(f"Original points: {len(points_3d)}, Dense points: {len(dense_points_3d)}")
        if 0:
            visualize_cloud_isosurface(points_3d, ext_mask, xgrid, ygrid, zgrid, threshold_factor=0.1, color=(1, 0, 0))
        
        # define the property grid - which is equivalent to the base RTE grid
        rte_grid = at3d.grid.make_grid(dx, xgrid.size,
                                       dy, ygrid.size,
                                       zgrid)
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        
        # work on loaded images and sensors from the loaded sensor list:
        images_from_loaded_sensor_list = [] # used for validation
        masks_from_loaded_sensor_list = [] # 
        
        for i in range(images_rte_simulated.shape[0]):
            """
            When AT3D saves the raw sensor data to a NetCDF file, it flattens
            the 2D image grid into a single 1D array of pixels.
            So each loaded_sensor_list[i]['I'].data is 1D flaten image.
            To reconstruct the image, you simply need to pull the 1D array of Stokes values
            (like 'I') and use NumPy to reshape it back into a 2D grid using those resolution attributes.
            """
            # 2. Extract the grid dimensions from the attributes
            nx = int(loaded_sensor_list[i].attrs['x_resolution'])
            ny = int(loaded_sensor_list[i].attrs['y_resolution'])
            
            # 3. Reshape it back into a 2D array
            
            cloud_mask_2d = loaded_sensor_list[i]['cloud_mask'].data.reshape((nx, ny), order='F')            
            image_2d = loaded_sensor_list[i]['I'].data.reshape((nx, ny), order='F')
            
            masks_from_loaded_sensor_list.append(cloud_mask_2d)
            images_from_loaded_sensor_list.append(image_2d)
            
        images_from_loaded_sensor_list = np.array(images_from_loaded_sensor_list)
        # Show to compare:
        fig = plot_10_image_with_mask_comparison(
            images_set1=images_rte_simulated, 
            images_set2=images_from_loaded_sensor_list,
            masks=masks_from_loaded_sensor_list, 
            title1="Part 1: Images loaded from pkl",
            title2="Part 2: Masks images loaded from sensor list"
        )        
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # --------------------------------------------------------------
        # generate rayleigh + ocean reference background
        
        # one function to generate rayleigh scattering.
        rayleigh_scattering = at3d.rayleigh.to_grid(mean_wavelengths, atmosphere, rte_grid)
        solvers_dict = at3d.containers.SolversDict()
        # note we could set solver dependent surfaces / sources / numerical_config here
        # just as we have got solver dependent optical properties.
        
        sun_azimuth= run_params['const_sun_azimuth']
        sun_zenith = run_params['const_sun_zenith']
        
        for wavelength in mean_wavelengths:
            medium = {
                'rayleigh': rayleigh_scattering[wavelength]
            }
            config = at3d.configuration.get_config()
    
            config['num_mu_bins'] = 8
            config['num_phi_bins'] = 16
            config['split_accuracies'] = 0.1
            config['max_total_mb'] = 100000
            
            config['spherical_harmonics_accuracy'] = 0.01
            config['num_sh_term_factor'] = 5
            config['high_order_radiance'] = True
            config['max_total_mb'] = 100000
            ocean_wind_speed = run_params['ocean_wind_speed'] # m/s
            pigmentation = 0.1
            surface_model = at3d.surface.ocean_unpolarized(ocean_wind_speed,pigmentation)
            
            solvers_dict.add_solver(
                wavelength,
                at3d.solver.RTE(
                    numerical_params=config,
                    surface=surface_model,
                    source=at3d.source.solar(wavelength, np.cos(sun_zenith * np.pi / 180), sun_azimuth),
                    medium=medium,
                    num_stokes=1
                )
    
            )
    
            solvers_dict.solve(n_jobs=40, maxiter=run_params['maxiter'])
    
            ##### define sensors #####
            GSD = run_params['GSD']  # km
            SATS_NUMBER_SETUP = run_params['SATS_NUMBER']
            sensor_dict = at3d.containers.SensorsDict()
            # Converts a Python list of AT3D xarray datasets back into a formal SensorsDict.
            # Loop through your list and add them one by one
            sensor_name='CloudCT dymanic masks'
            for image_dataset in loaded_sensor_list:
                sensor_dict.add_sensor(sensor_name, image_dataset)
            # render reference images
            sensor_dict.get_measurements(solvers_dict, n_jobs=40, verbose=True) # RENDERING
            ocean_reference_images = sensor_dict.get_images(sensor_name)
        # ---------------------------------------------------------
        # 1.  first 10 images
        # ---------------------------------------------------------
        images_part1 = images_rte_simulated # 10,nx,ny
        img_width, img_height = images_part1[0].shape
        # Execute GT maks  projection
        GT_MASK_red_overlays = project_3d_mask_to_2d(dense_points_3d, cameras_P, img_width, img_height)
        images_part2 =  GT_MASK_red_overlays
        
        # Determine global min and max for the common colorbar
        vmin = np.min(images_part1)
        vmax = np.max(images_part1)
        
        # ---------------------------------------------------------
        # 2. Set up the Main Figure and Subfigures
        # ---------------------------------------------------------
        fig = plt.figure(figsize=(16, 10))
        
        # Split the main figure into 2 vertical sections (Top and Bottom)
        # hspace adds a visual gap between the two parts
        subfigs = fig.subfigures(2, 1, hspace=0.1)
        
        # ---------------------------------------------------------
        # 3. PART 1: Plot the currently available 10 images (Top)
        # ---------------------------------------------------------
        subfig1 = subfigs[0]
        subfig1.suptitle("Part 1: Current 10 Images", fontsize=16, fontweight='bold')
        
        # Create a 2x5 grid inside the top subfigure
        axs1 = subfig1.subplots(2, 5)
        
        for i, ax in enumerate(axs1.flat):
            im = ax.imshow(images_part1[i], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f"Image {i+1}")
            ax.axis('off')
        
        # Add a common colorbar for Part 1
        # 'ax=axs1' tells matplotlib to steal space from the whole 2x5 grid for the colorbar
        subfig1.colorbar(im, ax=axs1, shrink=0.8, label='Value')
        
        
        # ---------------------------------------------------------
        # 4. PART 2: Leave space for the next 10 images (Bottom)
        # ---------------------------------------------------------
        subfig2 = subfigs[1]
        subfig2.suptitle("Part 2: Images with Projected GT Cloud Masks", fontsize=16, fontweight='bold')
        # Create a 2x5 grid inside the bottom subfigure
        axs2 = subfig2.subplots(2, 5)
        
        for i, ax in enumerate(axs2.flat):
            # Plot the base grayscale image (use the same ones from Part 1)
            
            
            ocean_reference_image = ocean_reference_images[i]['I'].data
            corrected_image =  images_part1[i] - 0.95 * ocean_reference_image
            
            # Cloud Shadows make negative values so I want to clip.
            # Clip negative values to 0 (This erases the cloud shadows)
            # np.clip(array, min_value, max_value). 'None' means no upper limit.
            # corrected_image = np.clip(corrected_image, 0, None)
            # im = ax.imshow(corrected_image, cmap='gray', vmin=vmin, vmax=vmax)
            im = ax.imshow(images_part1[i], cmap='gray', vmin=vmin, vmax=vmax)
            # Plot the transparent red overlay directly on top
            ax.imshow(GT_MASK_red_overlays[i])
            ax.set_title(f"View {i+1} Overlay")
            ax.axis('off')
           
            
        subfig2.colorbar(im, ax=axs2, shrink=0.8, label='Value')
    
        # Show the complete layout
        
        
        # ---------------------------------------------------------
        # Separate Figure: Semi-Transparent Overlap
        # ---------------------------------------------------------
        # Create a brand new figure (2 rows, 5 columns)
        fig_overlap, axes_overlap = plt.subplots(2, 5, figsize=(16, 8))
        fig_overlap.suptitle("Original images  + GT (red) masks vs. extracted (green) dynamic Mask", fontsize=16, fontweight='bold')
        
        # Define how transparent you want the red overlay to be
        # 0.0 = completely invisible, 1.0 = solid red
        overlay_opacity = 0.4 
        
        space_carving_agreement = 0
        in_glint_views = 0
        
        for i, ax in enumerate(axes_overlap.flat):
            # 1. Plot the base grayscale image first
            # Using the vmin/vmax from your earlier contrast calculations
            ax.imshow(images_part1[i], cmap='gray', vmin=vmin, vmax=vmax)
            
            # 2. Plot the red RGBA overlay on top, forcing the transparency
            ax.imshow(GT_MASK_red_overlays[i], alpha=overlay_opacity)
            
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            # ---------------------------------------------------------
            
            # cloud mask:
            ocean_reference_image = ocean_reference_images[i]['I'].data
            
            # Apply your 0.95 scaling rule to prevent over-subtraction
            scaled_reference = 0.95 * ocean_reference_image
            """
            The addition of the 0.95 scaling factor before subtraction is a great empirical trickit helps
            prevent "over-subtracting" the background, which can happen if the physical cloud absorbs some
            of the atmospheric path radiance.
            """            
            # Grab the specific threshold for camera 'i' from your dictionary/list
            current_threshold = 0.6 * run_params['radiance_thresholds'][i]
            
            # Call the new function
            mask, corrected_image, is_valid = extract_mask_from_reference(
                cloudy_image=images_part1[i], 
                reference_image=scaled_reference,
                absolute_threshold=current_threshold,  # <--- Passed here!
                max_cloud_fraction=0.85,
                min_contrast_ratio=1.5
            )
            
            if is_valid:
                print(f"View {i+1}: Valid mask generated.")
                if np.mean(ocean_reference_image) > 0.02:
                    print(f"The view {i} is in gling, would we use it for the scape carving?")
                    in_glint_views += 1 # only the valide views are counted here. should I chnge it?
                else:
                    space_carving_agreement += 1
                
            else:
                print(f"View {i+1}: Mask generation failed or flagged as heavy glint or very small cloud.")
                
            if is_valid:
                
                overlay = np.zeros((img_height, img_width, 4), dtype=np.float32)
                # updat sensor list masks for the space carving:
                sensor_dict[sensor_name]['sensor_list'][i]['cloud_mask'] = ('nrays', mask.flatten(order='F'))            
                # ----------------------
                # ----------------------
                
                # Set the valid pixels to Red (R=1.0) and fully opaque (Alpha=1.0)
                overlay[..., 1] = 1.0 * mask #   # Red channel
                overlay[..., 3] = 1.0 * mask
                ax.imshow(overlay, alpha=overlay_opacity)
                
            

            
            # ----------------------------------------------
            # ----------------------------------------------
            # ----------------------------------------------
            
            ax.set_title(f"View {i+1} Overlap")
            ax.axis('off')
        
        # Adjust spacing so the titles don't overlap
        plt.tight_layout()
        
        # The Agreement Test (Happens AFTER the 10-camera loop finishes)
        if space_carving_agreement <= 5:
            print(f"WARNING: Only {space_carving_agreement}/10 camera views are valid.\n Glint interference is too high or the cloud is too small.")
            print("Skipping 3D space carving for this scene...")
            
            # This will skip the rest of the code below and jump to the next iteration of your outer loop
            #continue
            plt.show()
            
            if func_mask is None:
                print("Carving aborted.")
                print("Srats.")
                print(func_status)
            sys.exit(0)
        
        else:
            print('getting CloudCT''s space carving')
            space_carver = at3d.space_carve.SpaceCarver(rte_grid, bcflag=3)
            if in_glint_views > 4:
                agreement = max((space_carving_agreement - 4 - 1), 3)/ 10
            else:
                agreement = max((space_carving_agreement), 3)/ 10
    
            carved_volume = space_carver.carve(loaded_sensor_list, agreement=(0.0, agreement), linear_mode=False)
            carved_mask = carved_volume.mask.data
            npad = ((1, 1), (1, 1), (1, 1))
    
            mask_data_padded = np.pad(carved_mask.copy(),
                                      pad_width=npad, mode='constant', constant_values=0)
    
            carved_mask = carved_mask > 0  # convert from int to bool
    
            struct = ndimage.generate_binary_structure(3, 2)
            mask_morph = ndimage.binary_closing(mask_data_padded, struct)
            mask_morph = mask_morph[1:-1, 1:-1, 1:-1]
            
        # ----------------------------------------
        # ----------------------------------------
        # ----------------------------------------
        # Debug dynamic masks:
        # ----------------------------------------
        # ----------------------------------------
        # ----------------------------------------
        #  3D points
        points_3d_indices = np.argwhere(carved_mask == 1)        
        voxel_sizes = np.array([dx, dy, dz])
        points_3d = (points_3d_indices * voxel_sizes) # + (voxel_sizes / 2.0)
        base_points = np.repeat(points_3d, N, axis=0)
        random_offsets = np.random.rand(base_points.shape[0], 3) * voxel_sizes
        dense_points_3d = base_points + random_offsets
        
        #  3D points func
        func_points_3d_indices = np.argwhere(func_mask == 1)        
        voxel_sizes = np.array([dx, dy, dz])
        func_points_3d = (func_points_3d_indices * voxel_sizes) # + (voxel_sizes / 2.0)
        func_base_points = np.repeat(func_points_3d, N, axis=0)
        func_random_offsets = np.random.rand(func_base_points.shape[0], 3) * voxel_sizes
        func_dense_points_3d = func_base_points + func_random_offsets        
        if 1:
            visualize_cloud_isosurface(points_3d, ext_mask, xgrid, ygrid, zgrid, threshold_factor=0.1, color=(1, 0, 0))
        
        # Execute the projection
        red_overlays = project_3d_mask_to_2d(dense_points_3d, cameras_P, img_width, img_height)        
        func_red_overlays = project_3d_mask_to_2d(func_dense_points_3d, cameras_P, img_width, img_height)        
        
        # ---------------------------------------------------------
        # Separate Figure: Semi-Transparent Overlap
        # ---------------------------------------------------------
        # Create a brand new figure (2 rows, 5 columns)
        fig_overlap, axes_overlap = plt.subplots(2, 5, figsize=(16, 8))
        fig_overlap.suptitle("Original images vs. Projected (green) dynamic Mask", fontsize=16, fontweight='bold')
        
        # Define how transparent you want the red overlay to be
        # 0.0 = completely invisible, 1.0 = solid red
        overlay_opacity = 0.4 
        
        for i, ax in enumerate(axes_overlap.flat):
            # 1. Plot the base grayscale image first
            # Using the vmin/vmax from your earlier contrast calculations
            ax.imshow(images_part1[i], cmap='gray', vmin=vmin, vmax=vmax)
            
            # cloud mask:
            overlay = np.zeros((img_height, img_width, 4), dtype=np.float32)
            mask = red_overlays[i]
            # Set the valid pixels to Red (R=1.0) and fully opaque (Alpha=1.0)
            overlay[..., 1] = 1.0 * mask[..., 0] #   # Red channel
            overlay[..., 3] = 1.0 * mask[..., 0]
            overlay[..., 0] = 1.0 * func_red_overlays[i][..., 0] # test my function
            ax.imshow(overlay, alpha=overlay_opacity)
            
            ax.set_title(f"View {i+1} Overlap")
            ax.axis('off')
        
        # Adjust spacing so the titles don't overlap
        plt.tight_layout()
        
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        print(f"SUCCESS: {space_carving_agreement} valid views obtained, while agreement set to {agreement}...")
        
        plt.show()
                
        
