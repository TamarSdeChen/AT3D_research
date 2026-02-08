import scipy.io as sio
import scipy.special as ssp
import matplotlib.pyplot as plt
import numpy as np
import at3d
import matplotlib.pyplot as plt
# import mayavi.mlab as mlab
import os
import logging
from collections import OrderedDict
import xarray as xr
# import transformations as transf
import pickle
import pandas as pd
import warnings
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
import copy
import yaml
import CloudCT_Imager
import random
import matplotlib
# matplotlib.use('TkAgg')
import itertools

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
r_earth = 6371.0  # km
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x, 3)


def safe_mkdirs(path):
    """Safely create path, warn in case of race."""

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            import errno
            if e.errno == errno.EEXIST:
                warnings.warn(
                    "Failed creating path: {path}, probably a race".format(path=path)
                )


def save_to_csv(cloud_scatterer, file_name, comment_line='', OLDPYSHDOM=False):
    """
    
    A utility function to save a microphysical medium.
    After implementation put as a function in util.py under the name 
    save_to_csv.
    
    Format:
    
    
    Parameters
    ----------
    path: str
        Path to file.
    comment_line: str, optional
        A comment line describing the file.
    OLDPYSHDOM: boll, if it is True, save the txt in old version of pyshdom.
    
    Notes
    -----
    CSV format is as follows:
    # comment line (description)
    nx,ny,nz # nx,ny,nz
    dx,dy # dx,dy [km, km]
    z_levels[0]     z_levels[1] ...  z_levels[nz-1] 
    x,y,z,lwc,reff,veff
    ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]
    .
    .
    .
    ix,iy,iz,lwc[ix, iy, iz],reff[ix, iy, iz],veff[ix, iy, iz]
    
    
    
    """
    xgrid = cloud_scatterer.x
    ygrid = cloud_scatterer.y
    zgrid = cloud_scatterer.z

    dx = cloud_scatterer.delx.item()
    dy = cloud_scatterer.dely.item()
    dz = round(np.diff(zgrid)[0], 5)

    REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.lwc)
    REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
    REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)

    y, x, z = np.meshgrid(range(cloud_scatterer.sizes.get('y')), \
                          range(cloud_scatterer.sizes.get('x')), \
                          range(cloud_scatterer.sizes.get('z')))

    if not OLDPYSHDOM:

        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{}, {}, {} '.format(int(cloud_scatterer.sizes.get('x')), \
                                         int(cloud_scatterer.sizes.get('y')), \
                                         int(cloud_scatterer.sizes.get('z')), \
                                         ) + "# nx,ny,nz\n")
            # dx,dy # dx,dy [km, km]
            f.write('{:2.3f}, {:2.3f} '.format(dx, dy) + "# dx,dy [km, km]\n")

            # z_levels[0]     z_levels[1] ...  z_levels[nz-1] 

            np.savetxt(f, \
                       X=np.array(zgrid).reshape(1, -1), \
                       fmt='%2.3f', delimiter=', ', newline='')
            f.write(" # altitude levels [km]\n")
            f.write("x,y,z,lwc,reff,veff\n")

            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), \
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask, ...]
            np.savetxt(f, X=data, fmt='%d ,%d ,%d ,%.5f ,%.3f ,%.5f')

    else:
        # save in the old version:
        with open(file_name, 'w') as f:
            f.write(comment_line + "\n")
            # nx,ny,nz # nx,ny,nz
            f.write('{} {} {} '.format(int(cloud_scatterer.sizes.get('x')), \
                                       int(cloud_scatterer.sizes.get('y')), \
                                       int(cloud_scatterer.sizes.get('z')), \
                                       ) + "\n")

            # dx,dy ,z
            np.savetxt(f, X=np.concatenate((np.array([dx, dy]), zgrid)).reshape(1, -1), fmt='%2.3f')
            # z_levels[0]     z_levels[1] ...  z_levels[nz-1] 

            data = np.vstack((x.ravel(), y.ravel(), z.ravel(), \
                              REGULAR_LWC_DATA.ravel(), REGULAR_REFF_DATA.ravel(), REGULAR_VEFF_DATA.ravel())).T
            # Delete unnecessary rows e.g. zeros in lwc
            mask = REGULAR_LWC_DATA.ravel() > 0
            data = data[mask, ...]
            np.savetxt(f, X=data, fmt='%d %d %d %.5f %.3f %.5f')


def load_from_csv_shdom(path, density=None, origin=(0.0, 0.0)):
    df = pd.read_csv(path, comment='#', skiprows=3, delimiter=' ')
    nx, ny, nz = np.genfromtxt(path, skip_header=1, max_rows=1, dtype=int, delimiter=' ')
    dx, dy = np.genfromtxt(path, max_rows=1, usecols=(0, 1), dtype=float, skip_header=2)
    z_grid = np.genfromtxt(path, max_rows=1, usecols=range(2, 2 + nz), dtype=float, skip_header=2)
    z = xr.DataArray(z_grid, coords=[range(nz)], dims=['z'])

    dset = at3d.grid.make_grid(dx, nx, dy, ny, z)

    for index, name in zip([3, 4, 5], ['lwc', 'reff', 'veff']):
        # initialize with np.nans so that empty data is np.nan
        variable_data = np.zeros((dset.sizes['x'], dset.sizes['y'], dset.sizes['z']))
        i = df.values[:, 0].astype(int)
        j = df.values[:, 1].astype(int)
        k = df.values[:, 2].astype(int)

        variable_data[i, j, k] = df.values[:, index]
        dset[name] = (['x', 'y', 'z'], variable_data)

    if density is not None:
        assert density in dset.data_vars, \
            "density variable: '{}' must be in the file".format(density)

        dset = dset.rename_vars({density: 'density'})
        dset.attrs['density_name'] = density

    dset.attrs['file_name'] = path

    return dset


def load_from_airmspi_mat(microphysics_path, mask_path, density=None):
    microphysics = sio.loadmat(microphysics_path)
    mask = sio.loadmat(mask_path)['mask']

    reff_data = microphysics['cloud_reff']
    veff_data = microphysics['cloud_veff']
    lwc_data = microphysics['cloud_lwc']
    dx = dy = 0.05  # km
    dz = 0.04
    nx, ny, nz = lwc_data.shape
    z_grid = np.linspace(0.,nz*dz-dz,nz)
    z = xr.DataArray(z_grid, coords=[range(nz)], dims=['z'])

    dset = at3d.grid.make_grid(dx, nx, dy, ny, z)

    dset['lwc'] = (['x', 'y', 'z'], lwc_data)
    dset['reff'] = (['x', 'y', 'z'], reff_data)
    dset['veff'] = (['x', 'y', 'z'], veff_data)

    if density is not None:
        assert density in dset.data_vars, \
            "density variable: '{}' must be in the file".format(density)

        dset = dset.rename_vars({density: 'density'})
        dset.attrs['density_name'] = density

    dset.attrs['file_name'] = microphysics_path

    return dset, mask

def load_params(params_path, param_type='run_params'):
    """
    TODO
    Args:
        params_path ():

    Returns:

    """
    logger = logging.getLogger(__name__)

    # Load run parameters
    params_file_path = params_path
    logger.debug(f"loading params from {params_file_path}")

    with open(params_file_path, 'r') as f:
        params = yaml.full_load(f)

    logger.debug(f"running with params:{params}")
    # TODO: add schemas.
    # if param_type == 'run_params':
    #     run_params_schema.validate(params)
    # elif param_type == 'imager_params':
    #     imager_params_schema.validate(params)
    # elif param_type == 'clouds':
    #     logger.debug('Currently no schema validation for clouds')
    # else:
    #     raise NotImplementedError


    return params


# def show_scatterer(cloud_scatterer):
#
#     """
#     Show the scatterer in 3D with Mayavi.
#     """
#
#     ShowVolumeBox = True
#
#     xgrid = cloud_scatterer.x
#     ygrid = cloud_scatterer.y
#     zgrid = cloud_scatterer.z
#
#     dx = cloud_scatterer.delx.item()
#     dy = cloud_scatterer.dely.item()
#     dz = round(np.diff(zgrid)[0],5)
#
#     REGULAR_LWC_DATA = np.nan_to_num(cloud_scatterer.density)
#     REGULAR_REFF_DATA = np.nan_to_num(cloud_scatterer.reff)
#     REGULAR_VEFF_DATA = np.nan_to_num(cloud_scatterer.veff)
#
#
#     show_field = REGULAR_LWC_DATA
#     data_type = 'LWC [g/m^3]'
#
#     mlab.figure(size=(600, 600))
#     X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')
#     figh = mlab.gcf()
#     src = mlab.pipeline.scalar_field(X, Y, Z, show_field, figure=figh)
#
#     src.spacing = [dx, dy, dz]
#     src.update_image_data = True
#
#     isosurface = mlab.pipeline.iso_surface(src, contours=[0.1*show_field.max(),\
#                                                           0.2*show_field.max(),\
#                                                           0.3*show_field.max(),\
#                                                           0.4*show_field.max(),\
#                                                           0.5*show_field.max(),\
#                                                           0.6*show_field.max(),\
#                                                           0.7*show_field.max(),\
#                                                           0.8*show_field.max(),\
#                                                           0.9*show_field.max(),\
#                                                           ],opacity=0.9,figure=figh)
#
#     mlab.outline(figure=figh,color = (1, 1, 1))  # box around data axes
#     mlab.orientation_axes(figure=figh)
#     mlab.axes(figure=figh, xlabel="x (km)", ylabel="y (km)", zlabel="z (km)")
#     color_bar = mlab.colorbar(title=data_type, orientation='vertical', nb_labels=5)
#
#     if(ShowVolumeBox):
#         # The _max is one d_ after the last point of the xgrid (|_|_|_|_|_|_|_->|).
#         x_min = xgrid[0]
#         x_max = round(xgrid[-1].item() + dx,5)
#
#         y_min = ygrid[0]
#         y_max = round(ygrid[-1].item() + dy,5)
#
#         z_min = zgrid[0]
#         z_max = round(zgrid[-1].item() + dz,5)
#
#         xm = [x_min, x_max, x_max, x_min, x_max, x_max, x_min, x_min ]
#         ym = [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max ]
#         zm = [z_min, z_min, z_max, z_max, z_min, z_max, z_max, z_min ]
#         # Medium cube
#         triangles = [[0,1,2],[0,3,2],[1,2,5],[1,4,5],[2,5,6],[2,3,6],[4,7,6],[4,5,6],[0,3,6],[0,7,6],[0,1,4],[0,7,4]];
#         obj = mlab.triangular_mesh( xm, ym, zm, triangles,color = (0.0, 0.17, 0.72),opacity=0.3,figure=figh)
#
#
#     #mlab.show()

# ---------------------------------------------------

def StringOfPearls(SATS_NUMBER=10, orbit_altitude=500, widest_view=False, move_nadir_x=0, move_nadir_y=0):
    """
    Set orbit parmeters:
         input:
         SATS_NUMBER - int, how many satellite to put?
         move_nadir_x/y - move in x/y to fit perfect nadir view.

         WIDEST_VIEW - bool, If WIDEST_VIEW False, the setup is the original with 100km distance between satellites.
         If it is True the distance become 200km.

         returns sat_positions: np.array of shape (SATS_NUMBER,3).
         The satellites setup alwas looks like \ \ | / /.
    """
    Rsat = orbit_altitude  # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R

    if (widest_view):
        Darc = 200
    else:
        Darc = 100  # km # distance between adjecent satellites (on arc).

    Dtheta = Darc / R  # from the center of the earth.

    # where to set the satelites?
    theta_config = np.arange(-0.5 * SATS_NUMBER, 0.5 * SATS_NUMBER) * Dtheta  # double for wide angles

    theta_config = theta_config[::-1]  # put sat1 to be the rigthest
    # print('Satellites angles relative to center of earth:')
    # for i,a in enumerate(theta_config):
    # print("{}: {}".format(i,a))

    theta_max, theta_min = max(theta_config), min(theta_config)

    X_config = r_orbit * np.sin(theta_config) + move_nadir_x
    Z_config = r_orbit * np.cos(theta_config) - r_earth
    Y_config = np.zeros_like(X_config) + move_nadir_y

    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.

    Satellites_angles = np.rad2deg(np.arctan(X_config / Z_config))
    print('Satellites angles are:')
    print(Satellites_angles)
    print("max angle {}\nmin angle {}\n".format(theta_max, theta_min))

    # find near nadir view:
    # since in this setup y=0:
    near_nadir_view_index = np.argmin(np.abs(X_config))

    return sat_positions.T, near_nadir_view_index, theta_max, theta_min


def visualize_satellite_positions(sat_positions_before, sat_positions_after, title_suffix="", save_path=None):
    """
    Visualize satellite positions before and after perturbation.
    
    Parameters:
    -----------
    sat_positions_before : np.ndarray
        Satellite positions before perturbation, shape (SATS_NUMBER, 3) or (N, SATS_NUMBER, 3)
    sat_positions_after : np.ndarray
        Satellite positions after perturbation, shape (SATS_NUMBER, 3) or (N, SATS_NUMBER, 3)
    title_suffix : str, optional
        Additional text to add to the plot title
    save_path : str, optional
        Path to save the figure. If None, saves as 'satellite_positions_before_after.png' in current directory
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Handle different input shapes
    if len(sat_positions_before.shape) == 3:
        # If shape is (N, SATS_NUMBER, 3), use first augmentation
        sat_pos_before = sat_positions_before[0, :, :]
        sat_pos_after = sat_positions_after[0, :, :]
    else:
        sat_pos_before = sat_positions_before
        sat_pos_after = sat_positions_after
    
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Calculate data ranges to set equal aspect ratios
    all_positions = np.vstack([sat_pos_before, sat_pos_after])
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
    max_range = max(x_range, y_range, z_range)
    
    # Calculate centers for each axis
    x_center = (all_positions[:, 0].max() + all_positions[:, 0].min()) / 2
    y_center = (all_positions[:, 1].max() + all_positions[:, 1].min()) / 2
    z_center = (all_positions[:, 2].max() + all_positions[:, 2].min()) / 2
    
    # Set equal limits for both plots
    x_lim = [x_center - max_range/2, x_center + max_range/2]
    y_lim = [y_center - max_range/2, y_center + max_range/2]
    z_lim = [z_center - max_range/2, z_center + max_range/2]
    
    # Arrow spans the full Z-axis range
    arrow_start_z = z_lim[0]
    arrow_end_z = z_lim[1]
    arrow_length = arrow_end_z - arrow_start_z
    
    # Plot before perturbation
    ax1.scatter(sat_pos_before[:, 0], sat_pos_before[:, 1], sat_pos_before[:, 2], 
                c='blue', marker='o', s=100, label='Before perturbation', alpha=0.7)
    # Draw lines connecting satellites
    ax1.plot(sat_pos_before[:, 0], sat_pos_before[:, 1], sat_pos_before[:, 2], 
             'b--', alpha=0.3, linewidth=1)
    # Add Z-axis arrow (up direction) - spans full Z range
    ax1.quiver(x_center, y_center, arrow_start_z, 0, 0, arrow_length,
               color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
    ax1.text(x_center, y_center, arrow_end_z + arrow_length*0.05, 'Z (up)', 
             color='green', fontsize=10, fontweight='bold')
    ax1.set_xlabel('X (km)', fontsize=10)
    ax1.set_ylabel('Y (km)', fontsize=10)
    ax1.set_zlabel('Z (km)', fontsize=10)
    ax1.set_title(f'Satellite Positions BEFORE Perturbation{title_suffix}', fontsize=12)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax1.legend()
    ax1.grid(True)
    
    # Plot after perturbation
    ax2.scatter(sat_pos_after[:, 0], sat_pos_after[:, 1], sat_pos_after[:, 2], 
                c='red', marker='^', s=100, label='After perturbation', alpha=0.7)
    # Draw lines connecting satellites
    ax2.plot(sat_pos_after[:, 0], sat_pos_after[:, 1], sat_pos_after[:, 2], 
             'r--', alpha=0.3, linewidth=1)
    # Add Z-axis arrow (up direction) - spans full Z range
    ax2.quiver(x_center, y_center, arrow_start_z, 0, 0, arrow_length,
               color='green', arrow_length_ratio=0.3, linewidth=2, alpha=0.8)
    ax2.text(x_center, y_center, arrow_end_z + arrow_length*0.05, 'Z (up)', 
             color='green', fontsize=10, fontweight='bold')
    ax2.set_xlabel('X (km)', fontsize=10)
    ax2.set_ylabel('Y (km)', fontsize=10)
    ax2.set_zlabel('Z (km)', fontsize=10)
    ax2.set_title(f'Satellite Positions AFTER Perturbation{title_suffix}', fontsize=12)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)
    ax2.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure instead of showing
    from datetime import datetime
    
    if save_path is None:
        save_path = 'satellite_positions_before_after.png'
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_parts = os.path.splitext(save_path)
    save_path_with_timestamp = f"{path_parts[0]}_{timestamp}{path_parts[1]}"
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path_with_timestamp)
    if save_dir and not os.path.exists(save_dir):
        safe_mkdirs(save_dir)
    
    plt.savefig(save_path_with_timestamp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {save_path_with_timestamp}")


def CreateVaryingStringOfPearls(SATS_NUMBER=10, ORBIT_ALTITUDE=500, move_nadir_x=0, move_nadir_y=0, DX=0, DY=0, DZ=0, N=1):
    """
    Create the Multiview Setup on orbit.
    Perturb the Satellites positions within the given limits and direct them with LOOKAT vector
    Set the Imagers at thier Locations + Orientations.
    The output here will be a list of Imagers.
    Each Imager will be updated here with respect to the defined geomtric considerations.


    Parameters:
    input:
        SATS_NUMBER - the number of satellites in the setup, int.
        ORBIT_ALTITUDE - in km  , float.
        DX - perturbation limit in X axis (perturbation allowed between +-DX from sat location)
        DY - perturbation limit in Y axis (perturbation allowed between +-DY from sat location)
        DZ - perturbation limit in Z axis (perturbation allowed between +-DZ from sat location)
        N - number of perturbation augmentations, int
    output:
        sat_positions - position of each perturbed satellite, ndarray of size (N, SATS_NUMBER, 3)
        near_nadir_view_indices - index of the near nadir satellite for each perturbation, ndarray of size (N)
        theta_max - maximum theta angle for each perturbation, ndarray of size (N)
        theta_min - minimum theta angle for each perturbation, ndarray of size (N)
    """

    sat_positions, _, _, _ = StringOfPearls(SATS_NUMBER=SATS_NUMBER, orbit_altitude=ORBIT_ALTITUDE,
                                    move_nadir_x=move_nadir_x,
                                    move_nadir_y=move_nadir_y)
    sat_positions = np.stack([sat_positions]*N)
    dx = np.random.uniform(low=-DX, high=DX, size=(N,SATS_NUMBER))
    dy = np.random.uniform(low=-DY, high=DY, size=(N,SATS_NUMBER))
    dz = np.random.uniform(low=-DZ, high=DZ, size=(N,SATS_NUMBER))
    
    # Store positions before perturbation for visualization
    sat_positions_before = sat_positions.copy()
    
    sat_positions[:, :, 0] += dx
    sat_positions[:, :, 1] += dy
    sat_positions[:, :, 2] += dz
    
    # Visualize satellite positions before and after perturbation
    # visualize_satellite_positions(sat_positions_before, sat_positions, save_path=f"/wdata/tamarsd/AT3D_research/CloudCT/figures/noise_vs_clean_pos/satellite_positions_before_after.png")
    
    X_config = np.squeeze(sat_positions[:, :, 0])
    Y_config = np.squeeze(sat_positions[:, :, 1])
    Z_config = np.squeeze(sat_positions[:, :, 2])

    # find near nadir view:
    # since in this setup (sat_y-lookat_y)=0:
    near_nadir_view_indices = np.argmin(np.abs(X_config))

    # find theta angles for cloudbow
    satellites_theta_angles = np.rad2deg(np.arctan(X_config / (Z_config+r_earth)))
    theta_max, theta_min = np.max(satellites_theta_angles), np.min(satellites_theta_angles)

    return sat_positions, near_nadir_view_indices, theta_max, theta_min


#---------------------------------------------------
#---------------------------------------------------
#---------------------------------------------------
def AddCloudBowScan2VaryingStringOfPearls(sat_positions=np.array([0, 0, 500]), lookat=np.array([0, 0, 0]), cloudbow_additional_scan=6,
                                          cloudbow_range=[135, 155], theta_max=60, theta_min=-60, sun_zenith=180, sun_azimuth=0):
    """
    TODO

    input:
        ORBIT_ALTITUDE - in km  , float.
        cloudbow_range - list of two elements - the cloudbow range in degrees.
        cloudbow_additional_scan - integer - how manny samples to add in the cloudbow range (input param cloudbow_range).
        theta_max, theta_min - floats - setup extrims off-nadir view angles in radian.

    Output:
        interpreted_sat_positions - np.array with the additional scan positions, shape (scanes,3).
        sat_index - integer, it is the sat index that perform the cloudbow scan.
        result_phis - list of the cloudbow angular samples 
        not_cloudbow_startind - index of a satellite from which we add manual cloudbow samples 
    """
    assert cloudbow_range[0] < cloudbow_range[1], "bad order of the cloudbow_range input"

    dtheta = 0.01
    sat_thetas = np.arange(theta_min, theta_max, dtheta)
    sat_thetas = sat_thetas[::-1]  # put sat1 to be the rigthest
    theta_indexes = np.arange(len(sat_thetas))

    r_ref = np.array([0, 0, r_earth])

    sat_directions = lookat - sat_positions
    sat_directions = -sat_directions / np.linalg.norm(sat_directions, axis=1, keepdims=True)

    #### Compute sun_direction ####
    SUN_THETA = np.deg2rad(sun_zenith)
    SUN_PHI = np.deg2rad(sun_azimuth)
    # calc sun direction from lookat to sun:
    sun_x = np.sin(SUN_THETA) * np.cos(SUN_PHI)
    sun_y = np.sin(SUN_THETA) * np.sin(SUN_PHI)
    sun_z = np.cos(SUN_THETA)
    sun_direction = np.array([sun_x, sun_y, sun_z])

    # Phi - scattering angle, THETA angles WRT Earth center.
    desired_phis = np.linspace(start=cloudbow_range[0], stop=cloudbow_range[1], num=cloudbow_additional_scan)        
    # scattering angles of current setup:
    curr_phis = np.rad2deg(np.arccos(np.dot(sun_direction, sat_directions.T)))
    # find one sample of curr_phis that is within the cloudbow_range
    cond = (curr_phis<cloudbow_range[1]) * (curr_phis>cloudbow_range[0])
    desired_thetas = np.zeros_like(desired_phis)
    # phis for angles relative to satellite and lookat.
    # thetas for angles relative to satellite and center for earth. 

    if np.any(cond):
        # there is angle in the range, use it:
        sat_index = np.argwhere(cond)[0].item()
    else:
        # there is no angle in the range, so find the closest:
        sat_index = np.argmin(np.abs(curr_phis - cloudbow_range[0]))

    found_sat =  sat_positions[sat_index, :]
    found_sat_relative_to_EC =  r_ref + found_sat            
    found_phi = curr_phis[sat_index]
    print("found angles to check")
    print(curr_phis)
    print("satellite index {} is in the cloudbow range, with angle {}.".format(sat_index,found_phi))                     

    X_config = (found_sat_relative_to_EC[2]) * np.sin(np.deg2rad(sat_thetas)) + found_sat_relative_to_EC[0]
    Z_config = (found_sat_relative_to_EC[2]) * np.cos(np.deg2rad(sat_thetas)) - r_earth
    Y_config = found_sat_relative_to_EC[1]*np.ones_like(X_config)
    sample_sat_positions = np.vstack([X_config, Y_config, Z_config])
    # sample_sat_positions.shape (3,#)

    X_config_relative_to_EC = X_config
    Y_config_relative_to_EC = Y_config
    Z_config_relative_to_EC = Z_config + r_earth
    sample_sat_positions_relative_to_EC = np.vstack([X_config_relative_to_EC,\
                                                     Y_config_relative_to_EC,\
                                                         Z_config_relative_to_EC])

    sample_sat_direction = lookat[:, np.newaxis] - sample_sat_positions
    sample_sat_direction = -sample_sat_direction / np.linalg.norm(sample_sat_direction, axis=0, keepdims=True)
    sat_sun_angles = np.rad2deg(np.arccos(np.dot(sun_direction, sample_sat_direction)))
    # filter relavent range:
    cond = (sat_sun_angles<=cloudbow_range[1]) * (sat_sun_angles>=cloudbow_range[0])
    filter_indexes = np.argwhere(cond)
    
    #if len(filter_indexes) == 0:
        #print("In this geometry, there is no cloudbow scan")
        #return None, None, None, None
        
    assert len(filter_indexes) > 0, "In this geometry, there is no cloudbow scan"

    filter_indexes = list(itertools.chain(*filter_indexes))
    sat_sun_angles = sat_sun_angles[filter_indexes]
    sample_sat_positions  = sample_sat_positions[:,filter_indexes]
    sample_sat_positions_relative_to_EC = sample_sat_positions_relative_to_EC[:,filter_indexes]
    sat_thetas = sat_thetas[filter_indexes]

    """
    When debug, use visualization:

    value = 1 * np.ones_like(X_config) 
    mlab.figure()

    mlab.points3d(X_config_relative_to_EC,\
    Y_config_relative_to_EC,\
    Z_config_relative_to_EC, value, scale_factor=1,
    color=(0, 1, 0))  



    mlab.points3d(found_sat_relative_to_EC[0], found_sat_relative_to_EC[1], \
    found_sat_relative_to_EC[2], 5, scale_factor=1,
    color=(1, 0, 0)) 

    at the end, visualize with:
    X_config_relative_to_EC = test_X_config
    Y_config_relative_to_EC = test_Y_config
    Z_config_relative_to_EC = test_Z_config + r_earth
    sample_sat_positions_relative_to_EC = np.vstack([X_config_relative_to_EC,\
                                              Y_config_relative_to_EC,\
                                              Z_config_relative_to_EC])

    value = 2 * np.ones_like(sample_sat_positions_relative_to_EC[0,:]) 
    mlab.points3d(sample_sat_positions_relative_to_EC[0,:],\
                  sample_sat_positions_relative_to_EC[1,:],\
                  sample_sat_positions_relative_to_EC[2,:], value, scale_factor=1,
                              color=(0, 0, 1))     

    """

    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    #-------------------------------------------------
    j1 = np.argmin(np.abs(desired_phis[0] - sat_sun_angles))
    first_theta = sat_thetas[j1]
    d = 0.5  # degrees # TODO - find this treshold as the maximum alowed
    low_bound = first_theta - d
    up_bound = first_theta + d  # degrees        
    for i, phi in enumerate(desired_phis):
        while (True):

            j = np.argmin(np.abs(phi - sat_sun_angles))
            candidat = sat_thetas[j]
            if ((low_bound <= candidat) and (candidat <= up_bound)):
                desired_thetas[i] = candidat
                low_bound = candidat - d
                up_bound = candidat + d  # degrees
                sat_sun_angles[j] = -200  # give invalid value

                test_X_config = (found_sat_relative_to_EC[2]) * np.sin(np.deg2rad(candidat)) + found_sat_relative_to_EC[0]
                test_Z_config = (found_sat_relative_to_EC[2]) * np.cos(np.deg2rad(candidat)) - r_earth
                test_Y_config = found_sat_relative_to_EC[1]*np.ones_like(test_X_config)

                test_sat_positions = np.vstack([test_X_config, test_Y_config , test_Z_config]) # path.shape = (3,#sats) in km.
                test_sat_direction = lookat[:,np.newaxis] - test_sat_positions
                test_sat_direction = -test_sat_direction/np.linalg.norm(test_sat_direction, axis=0, keepdims=True)
                test_phis = np.rad2deg(np.arccos(np.dot(sun_direction, test_sat_direction))) 
                print("Found phi {}".format(test_phis))
                break
            else:
                sat_sun_angles[j] = -200  # give invalid value


    #-------------------------------------------------
    #-------------------------------------------------
    # result_phis should be close to desired_phis, check it here:
    test_X_config = (found_sat_relative_to_EC[2]) * np.sin(np.deg2rad(desired_thetas)) + found_sat_relative_to_EC[0]
    test_Z_config = (found_sat_relative_to_EC[2]) * np.cos(np.deg2rad(desired_thetas)) - r_earth
    test_Y_config = found_sat_relative_to_EC[1]*np.ones_like(test_X_config)

    test_sat_positions = np.vstack([test_X_config, test_Y_config, test_Z_config])  # path.shape = (3,#sats) in km.
    test_sat_direction = lookat[:, np.newaxis] - test_sat_positions
    test_sat_direction = -test_sat_direction / np.linalg.norm(test_sat_direction, axis=0, keepdims=True)
    result_phis = np.rad2deg(np.arccos(np.dot(sun_direction, test_sat_direction)))  

    desired_thetas = np.deg2rad(desired_thetas)  # convert to radian

    # if we can't get all the desired cloudbow angles, just continue scanning with the same dtheta between scans:
    dtheta = np.diff(desired_thetas)
    new_theta_inds = np.argwhere(np.abs(dtheta) < 0.5e-3)
    not_cloudbow_startind = None
    if new_theta_inds.size != 0 and np.array_equal(new_theta_inds.ravel(), np.arange(new_theta_inds[0], len(dtheta))):
        not_cloudbow_startind = int(np.argwhere(np.abs(dtheta) < 0.5e-3)[0])
        rest_of_dthetas = dtheta[not_cloudbow_startind - 1]
        num_of_new_thetas = len(desired_thetas) - (not_cloudbow_startind + 1)
        desired_thetas[not_cloudbow_startind + 1:] = (desired_thetas[not_cloudbow_startind] +
                                                      np.arange(1, num_of_new_thetas + 1) * rest_of_dthetas)
    elif (new_theta_inds.size != 0) or (new_theta_inds.size == 0 and np.any(np.abs(desired_phis - result_phis) >= 2)):
        raise Exception("Something went wrong in the cloudbow scanning calculations.")

    # result_phis should be close to desired_phis, check it here:
    test_X_config = (found_sat_relative_to_EC[2]) * np.sin((desired_thetas)) + found_sat_relative_to_EC[0]
    test_Z_config = (found_sat_relative_to_EC[2]) * np.cos((desired_thetas)) - r_earth
    test_Y_config = found_sat_relative_to_EC[1]*np.ones_like(test_X_config)

    test_sat_positions = np.vstack([test_X_config, test_Y_config, test_Z_config])  # path.shape = (3,#sats) in km.
    test_sat_direction = lookat[:, np.newaxis] - test_sat_positions
    test_sat_direction = -test_sat_direction / np.linalg.norm(test_sat_direction, axis=0, keepdims=True)
    result_phis = np.rad2deg(np.arccos(np.dot(sun_direction, test_sat_direction)))  

    # interpreted_sat_positions
    interpreted_sat_positions = test_sat_positions.T 
    return interpreted_sat_positions, sat_index, result_phis, not_cloudbow_startind


def StringOfPearlsCloudBowScan(orbit_altitude=500, lookat=np.array([0, 0, 0]), cloudbow_additional_scan=6,
                               cloudbow_range=[135, 155], theta_max=60, theta_min=-60, sun_zenith=180, sun_azimuth=0,
                               move_nadir_x=0, move_nadir_y=0):
    """
    TODO

    input:
        ORBIT_ALTITUDE - in km  , float.
        cloudbow_range - list of two elements - the cloudbow range in degrees.
        cloudbow_additional_scan - integer - how manny samples to add in the cloudbow range (input param cloudbow_range).
        theta_max, theta_min - floats - setup extrims off-nadir view angles in radian.

    Output:
        interpreted_sat_positions - np.array with the additional scan positions, shape (scanes,3).
        result_phis - list of the cloudbow angular samples 
        not_cloudbow_startind - index of a satellite from which we add manual cloudbow samples 
    """
    assert cloudbow_range[0] < cloudbow_range[1], "bad order of the cloudbow_range input"

    Rsat = orbit_altitude  # km orbit altitude
    R = r_earth + Rsat
    r_orbit = R

    sat_thetas = np.arange(theta_min, theta_max, 0.0001)
    sat_thetas = sat_thetas[::-1]  # put sat1 to be the rigthest

    X_config = r_orbit * np.sin(sat_thetas) + move_nadir_x
    Z_config = r_orbit * np.cos(sat_thetas) - r_earth
    Y_config = np.zeros_like(X_config) + move_nadir_y
    sat_positions = np.vstack([X_config, Y_config, Z_config])  # path.shape = (3,#sats) in km.
    sat_direction = lookat[:, np.newaxis] - sat_positions
    sat_direction = -sat_direction / np.linalg.norm(sat_direction, axis=0, keepdims=True)

    #### Compute sun_direction ####
    SUN_THETA = np.deg2rad(sun_zenith)
    SUN_PHI = np.deg2rad(sun_azimuth)
    # calc sun direction from lookat to sun:
    sun_x = np.sin(SUN_THETA) * np.cos(SUN_PHI)
    sun_y = np.sin(SUN_THETA) * np.sin(SUN_PHI)
    sun_z = np.cos(SUN_THETA)
    sun_direction = np.array([sun_x, sun_y, sun_z])

    # virtual_sun = lookat - 600*sun_direction
    # mlab.quiver3d(virtual_sun[0], virtual_sun[1], virtual_sun[2],
    # sun_direction[0],sun_direction[1],sun_direction[2],
    # line_width=3.0,color = (1,1,0),opacity=1,mode='2ddash',scale_factor=1)

    # Phi - scattering angle
    sat_sun_angles = np.rad2deg(np.arccos(np.dot(sun_direction, sat_direction)))
    sat_thetas = np.rad2deg(sat_thetas)  # convert to degrees
    
    # filter relavent range:
    cond = (sat_sun_angles<=cloudbow_range[1]) * (sat_sun_angles>=cloudbow_range[0])
    filter_indexes = np.argwhere(cond)     
    assert len(filter_indexes) > 0, "In this geometry, there is no cloudbow scan"
    
    # interpreted_thetas based on desired_phis
    desired_phis = np.linspace(start=cloudbow_range[0], stop=cloudbow_range[1], num=cloudbow_additional_scan)
    desired_thetas = np.zeros_like(desired_phis)
    # phis for angles relative to satellite and lookat.
    # thetas for angles relative to satellite and center for earth.
    j1 = np.argmin(np.abs(desired_phis[0] - sat_sun_angles))
    first_theta = sat_thetas[j1]
    d = 0.5  # degrees # TODO - find this treshold as the maximum alowed
    low_bound = first_theta - d
    up_bound = first_theta + d  # degrees
    for i, phi in enumerate(desired_phis):
        while (True):

            j = np.argmin(np.abs(phi - sat_sun_angles))
            candidat = sat_thetas[j]
            if ((low_bound <= candidat) and (candidat <= up_bound)):
                desired_thetas[i] = candidat
                low_bound = candidat - d
                up_bound = candidat + d  # degrees
                sat_sun_angles[j] = -200  # give invalid value
                break
            else:
                sat_sun_angles[j] = -200  # give invalid value

    # result_phis should be close to desired_phis, check it here:
    test_X_config = r_orbit * np.sin(np.deg2rad(desired_thetas)) + move_nadir_x
    test_Z_config = r_orbit * np.cos(np.deg2rad(desired_thetas)) - r_earth
    test_Y_config = np.zeros_like(test_X_config) + move_nadir_y
    test_sat_positions = np.vstack([test_X_config, test_Y_config, test_Z_config])  # path.shape = (3,#sats) in km.
    test_sat_direction = lookat[:, np.newaxis] - test_sat_positions
    test_sat_direction = -test_sat_direction / np.linalg.norm(test_sat_direction, axis=0, keepdims=True)
    result_phis = np.rad2deg(np.arccos(np.dot(sun_direction, test_sat_direction)))

    # assert np.all(np.abs(
    #     desired_phis - result_phis) < 2), "Something went wrong in the cloudbow scanning calculations."  # 2 degree margin
    # plt.plot(sat_thetas,sat_sun_angles)
    # plt.plot(desired_thetas,result_phis,'.')
    # plt.show()
    desired_thetas = np.deg2rad(desired_thetas)  # convert to radian
    """
    Linear interpulation is bad option here since the sat_sun_angles shape is ~parabolic.
    interpreted_thetas = np.interp(desired_phis, sat_sun_angles, sat_thetas)
    interpreted_thetas = np.deg2rad(interpreted_thetas)# convert to redian
    """

    # if we can't get all the desired cloudbow angles, just continue scanning with the same dtheta between scans:
    dtheta = np.diff(desired_thetas)
    new_theta_inds = np.argwhere(np.abs(dtheta) < 0.5e-3)
    not_cloudbow_startind = None    
    if new_theta_inds.size != 0 and np.array_equal(new_theta_inds.ravel(), np.arange(new_theta_inds[0], len(dtheta))):
        not_cloudbow_startind = int(np.argwhere(np.abs(dtheta) < 0.5e-3)[0])
        rest_of_dthetas = dtheta[not_cloudbow_startind - 1]
        num_of_new_thetas = len(desired_thetas) - (not_cloudbow_startind + 1)
        desired_thetas[not_cloudbow_startind + 1:] = (desired_thetas[not_cloudbow_startind] +
                                                      np.arange(1, num_of_new_thetas + 1) * rest_of_dthetas)
    elif (new_theta_inds.size != 0) or (new_theta_inds.size == 0 and np.any(np.abs(desired_phis - result_phis) >= 2)):
        raise Exception("Something went wrong in the cloudbow scanning calculations.")
    # interpreted_sat_positions
    interp_X_config = r_orbit * np.sin(desired_thetas) + move_nadir_x
    interp_Z_config = r_orbit * np.cos(desired_thetas) - r_earth
    interp_Y_config = np.zeros_like(desired_thetas) + move_nadir_y
    interpreted_sat_positions = np.vstack([interp_X_config, interp_Y_config, interp_Z_config])

    print(interpreted_sat_positions)
    print("angular gap (resolution)")
    print("gaps: ", np.diff(result_phis))
    print("resolution: ", np.mean(np.diff(result_phis)))

    adjecent_distances = np.diff(interpreted_sat_positions, axis=-1)
    adjecent_distances = np.linalg.norm(adjecent_distances, axis=0)  # in km
    time_gaps = adjecent_distances / 7.612  # satellite velocity at 500km orbit is assumed to be 7.612 km/sec.
    # time_gaps in sec.
    print("time gaps between adjecent cloudbow scans are:")
    print(time_gaps)
    print("total time of cloudbow scan is:")
    print(time_gaps.sum())

    Satellites_angles = np.rad2deg(np.arctan(interp_X_config / interp_Z_config))
    print('Cloudbow satellites angles are:')
    print(Satellites_angles)

    # for each sat-center angle (theta) compute sat-sun angle (phi)

    # 1d interpolate phi --> theta

    # Arange N phis in range 135-165
    # for each phi --> theta --> x,y,z sat
    return interpreted_sat_positions.T, result_phis, not_cloudbow_startind





def show_results(sensor_dict):
    # see images:
    from datetime import datetime
    
    # Create output directory
    output_dir = '/wdata/tamarsd/AT3D_research/CloudCT/figures/results_clouds'
    safe_mkdirs(output_dir)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for instrument in sensor_dict:
        sensor_images = sensor_dict.get_images(instrument)

        PNCHANNELS = 1  # polarized channels
        pol_channels = ['I']
        if 'Q' in list(sensor_images[0].keys()) and 'U' in list(sensor_images[0].keys()):
            PNCHANNELS = 3
            pol_channels = ['I', 'Q', 'U']

        nrows = 2
        LN = len(sensor_images)
        if LN % nrows == 0:
            ncols = int(LN / nrows)
        else:
            if ((LN / nrows) > int(LN / nrows)):
                ncols = int(LN / nrows) + 1
            else:
                ncols = int(LN / nrows)

                # ------------------------------
        fontsize = 16
        for pol_channel in pol_channels:

            fig = plt.figure(figsize=(20, 10))
            fontsize = 16
            fig.subplots_adjust(hspace=0.4, wspace=0.4)
            max_ = 0
            min_ = 1e7

            for index, sensor in enumerate(sensor_images):
                img = sensor[pol_channel].T.data
                max_ = max(max_, img.max())
                min_ = min(min_, img.min())

            cmap = 'jet'

            for index, sensor in enumerate(sensor_images):
                img = sensor[pol_channel].T.data
                if pol_channel == 'I':
                    min_ = 0
                    cmap = 'gray'

                ii = index + 1
                ax = fig.add_subplot(nrows, ncols, ii)
                im = ax.imshow(img, cmap=cmap, vmin=min_, vmax=max_)
                title = "{}".format(index)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

            fig.suptitle("{}: channel {}".format(instrument, pol_channel), size=16, y=0.95)
            
            # Save figure with timestamp
            filename = f"{instrument}_{pol_channel}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            #plt.savefig(filepath, dpi=150, bbox_inches='tight')
            #plt.close(fig)
            #print(f"Figure saved to: {filepath}")

    print('done')


def draw_scatter_plot(images_set1, images_set2, title_list):
    pixel_precent = 0.5
    assert images_set1.shape == images_set2.shape, "Can't compare images of different sizes."
    assert len(title_list) == images_set1.shape[1], "wrong number of titles."
    for index, title in enumerate(title_list):
        curr_images_set1 = np.squeeze(images_set1.copy()[:,index, :, :])
        curr_images_set2 = np.squeeze(images_set2.copy()[:, index, :, :])
        fig, axarr = plt.subplots(int(images_set1.shape[0]/5), 5, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.2, wspace=0.2)
        axarr = axarr.flatten()
        for ax, image1, image2 in zip(axarr, curr_images_set1, curr_images_set2):
            image1 = image1.ravel()
            image2 = image2.ravel()
            maxval = np.max([image1.max(), image2.max()])
            minval = np.min([image1.min(), image2.min()])
            rand_ind = np.random.choice(np.arange(len(image1)), size=int(pixel_precent * len(image1)), replace=False)
            ax.plot(image1[rand_ind],image2[rand_ind],'.')
            ax.plot([minval, maxval], [minval, maxval], '-r')
            ax.set_xlabel("before noise")
            ax.set_ylabel("after noise")
        fig.suptitle(title, size=16, y=0.95)
        # plt.savefig('/home/inbalkom/Downloads/'+title+'_scatterplot.png')

    plt.show()

def show_scatter_plot_colorbar(gt_param, est_param, param_name='parameter', pixel_precent = 0.1,
                      colorbar_param = None, colorbar_name = None):
    gt_param = gt_param.ravel()
    est_param = est_param.ravel()
    max_val = max(gt_param.max(), est_param.max())
    min_val = min(gt_param.min(), est_param.min())
    rand_ind = np.random.choice(np.arange(len(gt_param)), size=int(pixel_precent * len(gt_param)), replace=False)
    fig, ax = plt.subplots()
    if colorbar_param is None:
        ax.scatter(gt_param[rand_ind], est_param[rand_ind], facecolors='none', edgecolors='b')
        ax.set_title('Estimated vs. True Values of the ' + param_name)
    else:
        colorbar_param = colorbar_param.ravel()
        scat = ax.scatter(gt_param[rand_ind], est_param[rand_ind], c=colorbar_param, cmap=plt.cm.jet)
        plt.colorbar(scat)
        ax.set_title('Estimated vs. True Values of the ' + param_name + f'\ncolor-coded by' + colorbar_name)

    ax.set_xlim([0.9 * min_val, 1.1 * max_val])
    ax.set_ylim([0.9 * min_val, 1.1 * max_val])
    ax.plot(ax.get_xlim(), ax.get_ylim(), c='r', ls='--')
    ax.set_ylabel('Estimated', fontsize=14)
    ax.set_xlabel('True', fontsize=14)
    ax.set_aspect('equal')

    plt.show()

def generate_random_surface_wind_speed(wind_mean, wind_std):
    """
    Get random surface wind speed out of 2-parameters Weibull distribution.
    The two parameters are approximated using the wind speed mean and STD,
    as described in https://journals.ametsoc.org/view/journals/clim/19/4/jcli3640.1.xml
    :param wind_mean - the wind speed mean (m/s) scalar
    :param wind_std - the wind speed std (m/s) scalar
    :return wind_speed - random wind speed in (m/s) scalar
    """
    a = (wind_mean/wind_std)**1.086
    gamma = wind_mean/ssp.gamma(1+(1/a))
    wind_speed = gamma * np.random.default_rng().weibull(a)
    return wind_speed

def generate_random_sun_angles_for_lat(Lat):
    day_num = np.random.default_rng().integers(1, high=365, endpoint=True)
    LST = np.random.default_rng().integers(0, high=23, endpoint=True)  # local solar time

    delta = 23.45 * np.sin(np.deg2rad((360 / 365) * (284 + day_num)))  # Declination
    h = (LST - 12) * 15  # local hour angle
    sun_alt = np.rad2deg(
        np.arcsin(np.sin(np.deg2rad(Lat)) * np.sin(np.deg2rad(delta)) +
                  np.cos(np.deg2rad(Lat)) * np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(h))))
    sun_azimuth = np.rad2deg(np.arcsin(np.cos(np.deg2rad(delta)) * np.sin(np.deg2rad(h)) / np.cos(np.deg2rad(sun_alt))))
    while (not np.isreal(sun_alt)) or (not np.isreal(sun_azimuth)) or (sun_alt < 0):
        day_num = np.random.default_rng().integers(1, high=365, endpoint=True)
        LST = np.random.default_rng().integers(0, high=23, endpoint=True)

        delta = 23.45 * np.sin(np.deg2rad((360 / 365) * (284 + day_num)))
        h = (LST - 12) * 15
        sun_alt = np.rad2deg(
            np.arcsin(np.sin(np.deg2rad(Lat)) * np.sin(np.deg2rad(delta)) +
                      np.cos(np.deg2rad(Lat)) * np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(h))))
        sun_azimuth = np.rad2deg(
            np.arcsin(np.cos(np.deg2rad(delta)) * np.sin(np.deg2rad(h)) / np.cos(np.deg2rad(sun_alt))))
    sun_zenith = sun_alt + 90
    return sun_azimuth, sun_zenith


def generate_random_sun_angles_from_sunsync_orbit(sunsync_file_path, zenith_thr):
    with open(sunsync_file_path, 'rb') as f:
        data = pickle.load(f)
    azimuths = np.array([data_line["sun_azimuth"] for data_line in data])
    zeniths = np.array([data_line["sun_elevation"]+90 for data_line in data])
    latitudes = np.array([data_line["latitude"] for data_line in data])
    longitudes = np.array([data_line["longitude"] for data_line in data])
    utc_times = np.array([data_line["utc_time"] for data_line in data])
    sat_dirs = np.array([data_line["motion_direction"] for data_line in data])
    azimuths = azimuths[zeniths > zenith_thr]
    latitudes = latitudes[zeniths > zenith_thr]
    longitudes = longitudes[zeniths > zenith_thr]
    utc_times = utc_times[zeniths > zenith_thr]
    sat_dirs = sat_dirs[zeniths > zenith_thr]
    zeniths = zeniths[zeniths > zenith_thr]

    sun_idx = np.random.default_rng().integers(0, high=len(zeniths), endpoint=False)

    sun_azimuth = azimuths[sun_idx]
    sun_zenith = zeniths[sun_idx]
    lat = latitudes[sun_idx]
    long = longitudes[sun_idx]
    utc_time = utc_times[sun_idx]
    sat_dir_angle = sat_dirs[sun_idx]
    return sun_azimuth, sun_zenith, utc_time, lat, long, sat_dir_angle


def calc_image_in_scattering_plane(sensor, sensor_name, sun_azimuth, sun_zenith, theta_dir, path_stamp):
    theta_filename = os.path.join(theta_dir, path_stamp,
                                  sensor_name + '_sa' + str(sun_azimuth) + '_sz' + str(sun_zenith) + '.pkl')
    if os.path.exists(theta_filename):
        with open(theta_filename, 'rb') as f:
            theta_rad_mat = pickle.load(f)
            print("Theta matrix file of {} was read for projection {}.".format(sensor_name, path_stamp))
    else:
        print('Converting {} for projection {}'.format(sensor_name, path_stamp))
        zenith_dir = np.array([0, 0, 1])
        PHI = sensor.ray_phi.data
        THETA = np.arccos(sensor.ray_mu.data)  # mu is defined as -z !!!
        resolution = sensor.image_dims.data
        PHI = PHI.reshape(resolution, order='F')
        THETA = THETA.reshape(resolution, order='F')
        MU = np.cos(THETA)
        RAY_Z = -MU
        RAY_X = np.sin(np.arccos(MU)) * np.cos(PHI)
        RAY_Y = np.sin(np.arccos(MU)) * np.sin(PHI)

        theta_rad_mat = np.zeros_like(RAY_X)

        alpha = (180 - sun_zenith) * np.pi / 180
        beta = sun_azimuth * np.pi / 180
        sun_dir = np.array([np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)])

        for index, (d_x, d_y, d_z, phi_dir) in enumerate(zip(RAY_X.flatten(),
                                                             RAY_Y.flatten(),
                                                             RAY_Z.flatten(),
                                                             PHI.flatten())):
            ray_dir = np.array([d_x, d_y, d_z])

            persp_cam_vec = np.cross(zenith_dir, ray_dir)
            persp_cam_vec = persp_cam_vec / np.linalg.norm(persp_cam_vec)
            persp_cam_phi = (np.arctan2(persp_cam_vec[1], persp_cam_vec[0]) + np.pi).astype(
                np.float64)  # phi_dir

            scat_vec = np.cross(sun_dir, ray_dir)
            scat_vec = scat_vec / np.linalg.norm(scat_vec)
            scat_phi = (np.arctan2(scat_vec[1], scat_vec[0]) + np.pi).astype(np.float64)

            cos = np.dot(persp_cam_vec, scat_vec)
            theta_rad = np.arccos(cos)

            d = persp_cam_phi - scat_phi
            if (d > 2 * np.pi):
                d = d - 2 * np.pi

            if ((d <= np.pi) and (d >= 0)):
                dphi = 1.0

            elif ((d <= -np.pi) and (d >= -2 * np.pi)):
                dphi = 1.0

            else:
                dphi = -1.0

            theta_rad *= dphi

            xpix_ind, ypix_ind = np.unravel_index(index, (resolution[0], resolution[1]))
            theta_rad_mat[xpix_ind, ypix_ind] = theta_rad
        # save theta_rad_mat for future runs
        if not os.path.exists(theta_dir):
            # Create a new directory because it does not exist
            os.makedirs(theta_dir)
            print("The directory for saving theta matrices was created.")
        if not os.path.exists(os.path.join(theta_dir, path_stamp)):
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(theta_dir, path_stamp))
            print("The directory for saving theta matrices for projection {} was created.".format(path_stamp))
        with open(filename, 'wb') as outfile:
            pickle.dump(theta_rad_mat, outfile, protocol=pickle.HIGHEST_PROTOCOL)
            print("Theta matrix file was saved.")

            theta_filename = os.path.join(theta_dir, path_stamp, sensor_name + '.pkl')
            if os.path.exists(theta_filename):
                with open(theta_filename, 'rb') as f:
                    theta_rad_mat = pickle.load(f)
                    print("Theta matrix file of {} was read for projection {}.".format(sensor_name, path_stamp))

    image = images[proj_ind]
    scatter_image = np.zeros_like(image)
    stokes_V = 0
    for index, (stokes_I, stokes_Q, stokes_U) in enumerate(zip(image[0].flatten(),
                                                               image[1].flatten(),
                                                               image[2].flatten())):
        theta_rad = theta_rad_mat.flatten()[index]
        R_theta = np.array([[1, 0, 0, 0], [0, np.cos(2 * theta_rad), np.sin(2 * theta_rad), 0],
                            [0, -np.sin(2 * theta_rad), np.cos(2 * theta_rad), 0], [0, 0, 0, 1]])

        xpix_ind, ypix_ind = np.unravel_index(index, (resolution[0], resolution[1]))
        # S # this pixel stokes vector:

        S = np.vstack([stokes_I,
                       stokes_Q,
                       stokes_U,
                       stokes_V])  # stokes vector

        S_at_pixel = S
        # now S has 4 elements.

        Sconvertaed_at_pixel = np.dot(R_theta, S_at_pixel)
        S_3 = Sconvertaed_at_pixel[0:3]
        scatter_image[:, xpix_ind, ypix_ind] = np.squeeze(S_3)
    return scatter_image


def calc_image_in_scattering_plane_vectorbase(sensor, sensor_image, sensor_name, sun_azimuth, sun_zenith):
    if sensor_image.shape[0]==3:
        stokes = np.transpose(sensor_image,(1,2,0))
    elif sensor_image.shape[2]==3:
        stokes = sensor_image
    else:
        raise ValueError("sensor_image must have dimension 3")
    print('Converting {}'.format(sensor_name))
    zenith_dir = np.array([0, 0, 1])
    PHI = sensor.ray_phi.data
    THETA = np.arccos(sensor.ray_mu.data)  # mu is defined as -z !!!
    resolution = sensor.image_shape.data
    # PHI = PHI.reshape(resolution, order='F')
    # THETA = THETA.reshape(resolution, order='F')
    MU = np.cos(THETA)
    RAY_Z = -MU
    RAY_X = np.sin(np.arccos(MU)) * np.cos(PHI)
    RAY_Y = np.sin(np.arccos(MU)) * np.sin(PHI)

    theta_rad_mat = np.zeros_like(RAY_X)

    alpha = (180 - sun_zenith) * np.pi / 180
    beta = sun_azimuth * np.pi / 180
    sun_dir = np.array([np.sin(alpha) * np.cos(beta), np.sin(alpha) * np.sin(beta), np.cos(alpha)])

    ray_dirs = np.vstack([RAY_X, RAY_Y, RAY_Z])

    persp_cam_vecs = np.cross(zenith_dir, ray_dirs, axis=0)
    persp_cam_vecs = persp_cam_vecs / np.linalg.norm(persp_cam_vecs, axis=0)
    persp_cam_phis = (np.arctan2(persp_cam_vecs[1], persp_cam_vecs[0]) + np.pi).astype(np.float64)

    scat_vecs = np.cross(sun_dir, ray_dirs, axis=0)
    scat_vecs = scat_vecs / np.linalg.norm(scat_vecs, axis=0)
    scat_phis = (np.arctan2(scat_vecs[1], scat_vecs[0]) + np.pi).astype(np.float64)

    # Find theta
    # fast way of dot product from - https://stackoverflow.com/questions/37670658/python-dot-product-of-each-vector-in-two-lists-of-vectors
    cos = np.einsum('ji, ji->i', persp_cam_vecs, scat_vecs)
    cos = np.clip(cos, -1, 1)
    theta_rad = np.arccos(cos)  # polarizer_dir for 0[deg] is lo direction.

    # Find dphi -  to dicide if the angle is theta_rad or -theta_rad.
    d = persp_cam_phis - scat_phis
    dphi = -1 * np.ones_like(d)
    d[d > 2 * np.pi] = d[d > 2 * np.pi] - 2 * np.pi
    dphi[(d <= np.pi) * (d >= 0)] = 1.0
    dphi[(d <= -np.pi) * (d >= -2 * np.pi)] = 1.0

    theta_rad *= dphi  # it is very important to give here the sign of theta.
    theta_rad_mat = theta_rad
    theta_rad_mat = theta_rad_mat.reshape(resolution, order='F')

    cos2theta = np.cos(2 * theta_rad_mat).flatten()
    sin2theta = np.sin(2 * theta_rad_mat).flatten()
    zeros = np.zeros_like(cos2theta)
    ones = np.ones_like(cos2theta)
    row0 = np.vstack([ones, zeros, zeros])[:, np.newaxis, :]
    row1 = np.vstack([zeros, cos2theta, sin2theta])[:, np.newaxis, :]
    row2 = np.vstack([zeros, -sin2theta, cos2theta])[:, np.newaxis, :]
    row0 = row0.transpose([1, 0, 2])
    row1 = row1.transpose([1, 0, 2])
    row2 = row2.transpose([1, 0, 2])
    ROT_MAT = np.vstack([row0, row1, row2])
    # reminder:
    # shape of stokes (cnx, cny, 3)
    # shape of ROT_MAT (3,3,cnx*cny)
    vector_stokes = np.reshape(stokes, [-1, 3])
    vector_stokes = vector_stokes.T

    npix = resolution[0]*resolution[1]
    scatter_image = np.zeros([3, npix])
    for index in range(npix):
        Sconvertaed_at_pixel = np.dot(ROT_MAT[...,index], vector_stokes[...,index])
        scatter_image[:,index] = Sconvertaed_at_pixel

    scatter_image = scatter_image.reshape([3]+list(resolution), order='C')
    assert np.allclose(scatter_image[0], stokes[:,:,0]), "Bad calculation of scattering plane."
    return scatter_image


def setup_rotation_matrices(theta_deg, lookat):
    """
    Setup rotation matrices for Z-axis rotation around a pivot point.
    
    Parameters:
    -----------
    theta_deg : float
        Rotation angle in degrees
    lookat : array-like, shape (3,)
        Pivot point [x, y, z] around which to rotate
        
    Returns:
    --------
    ROT_TOTAL : numpy.ndarray, shape (4, 4)
        Combined transformation matrix (translate-rotate-translate back)
    ROT_Z : numpy.ndarray, shape (4, 4)
        Rotation matrix around Z-axis
    """
    theta_rad = np.deg2rad(theta_deg)
    
    # 1. Translation to origin
    T_inv = np.array([
        [1, 0, 0, -lookat[0]], 
        [0, 1, 0, -lookat[1]], 
        [0, 0, 1, -lookat[2]], 
        [0, 0, 0, 1]
    ])
    
    # 2. Rotation matrix around Z
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    ROT_Z = np.array([
        [cos_t, -sin_t, 0, 0], 
        [sin_t, cos_t, 0, 0], 
        [0, 0, 1, 0], 
        [0, 0, 0, 1]
    ])
    
    # 3. Translation back to cloud center
    T = np.array([
        [1, 0, 0, lookat[0]], 
        [0, 1, 0, lookat[1]], 
        [0, 0, 1, lookat[2]], 
        [0, 0, 0, 1]
    ])
    
    ROT_TOTAL = T @ ROT_Z @ T_inv
    
    return ROT_TOTAL, ROT_Z


def apply_rotation_to_sensor_positions(sat_positions, ROT_TOTAL, ROT_Z, up_list):
    """
    Apply rotation transformation to satellite positions and up vectors.
    
    Parameters:
    -----------
    sat_positions : numpy.ndarray, shape (N, 3) or (1, N, 3)
        Original satellite positions
    ROT_TOTAL : numpy.ndarray, shape (4, 4)
        Combined transformation matrix
    ROT_Z : numpy.ndarray, shape (4, 4)
        Rotation matrix around Z-axis
    up_list : numpy.ndarray, shape (N, 3)
        Original up vectors
        
    Returns:
    --------
    transformed_positions : numpy.ndarray, shape (N, 3)
        Rotated satellite positions
    transformed_up_vectors : numpy.ndarray, shape (N, 3)
        Rotated up vectors
    """
    # Handle both (N, 3) and (1, N, 3) shapes
    if sat_positions.ndim == 3:
        positions = sat_positions[0]
    else:
        positions = sat_positions
    
    transformed_positions = []
    transformed_up_vectors = []
    
    for position, up_vector in zip(positions, up_list):
        # Transform position
        new_position = np.dot(ROT_TOTAL, np.append(position, 1))
        transformed_positions.append(new_position[0:3])
        
        # Transform up vector
        new_up_vector = np.dot(ROT_Z[0:3, 0:3], up_vector)
        transformed_up_vectors.append(new_up_vector)
    
    return np.array(transformed_positions), np.array(transformed_up_vectors)
    
def rotate_sun_angles(zenith_deg, azimuth_deg, rotation_theta_deg):
    # 1. Convert degrees to radians
    zen_rad = np.deg2rad(zenith_deg)
    azi_rad = np.deg2rad(azimuth_deg)
    rot_rad = np.deg2rad(rotation_theta_deg)
    
    # 2. Create Sun Unit Vector (Direction toward the Sun)
    # Using the standard: Z is up, Azimuth 0 is +X
    sun_vec = np.array([
        np.sin(zen_rad) * np.cos(azi_rad),
        np.sin(zen_rad) * np.sin(azi_rad),
        np.cos(zen_rad)
    ])
    
    # 3. Create the 3x3 Rotation Matrix around Z
    cos_t, sin_t = np.cos(rot_rad), np.sin(rot_rad)
    R_z = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ])
    
    # 4. Rotate the vector
    new_sun_vec = R_z @ sun_vec
    
    # 5. Convert back to Zenith and Azimuth
    new_zenith = np.rad2deg(np.arccos(new_sun_vec[2]))
    new_azimuth = np.rad2deg(np.arctan2(new_sun_vec[1], new_sun_vec[0]))
    
    return new_zenith, new_azimuth % 360

def visualize_rotation(sat_positions_before, sat_positions_after, lookat, rotation_angle_deg, 
                       title_suffix="", save_path=None):
    """
    Visualize satellite positions before and after rotation around Z-axis.
    
    Parameters:
    -----------
    sat_positions_before : np.ndarray
        Satellite positions before rotation, shape (SATS_NUMBER, 3) or (1, SATS_NUMBER, 3)
    sat_positions_after : np.ndarray
        Satellite positions after rotation, shape (SATS_NUMBER, 3) or (1, SATS_NUMBER, 3)
    lookat : array-like, shape (3,)
        Pivot point [x, y, z] around which rotation occurs
    rotation_angle_deg : float
        Rotation angle in degrees
    title_suffix : str, optional
        Additional text to add to the plot title
    save_path : str, optional
        Path to save the figure. If None, saves as 'satellite_positions_rotation.png' in current directory
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Handle different input shapes
    if len(sat_positions_before.shape) == 3:
        # If shape is (N, SATS_NUMBER, 3), use first augmentation
        sat_pos_before = sat_positions_before[0, :, :]
        sat_pos_after = sat_positions_after[0, :, :]
    else:
        sat_pos_before = sat_positions_before
        sat_pos_after = sat_positions_after
    
    lookat = np.array(lookat)
    
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Calculate data ranges to set equal aspect ratios
    all_positions = np.vstack([sat_pos_before, sat_pos_after, lookat.reshape(1, -1)])
    x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
    y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
    z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
    max_range = max(x_range, y_range, z_range)
    
    # Calculate centers for each axis
    x_center = (all_positions[:, 0].max() + all_positions[:, 0].min()) / 2
    y_center = (all_positions[:, 1].max() + all_positions[:, 1].min()) / 2
    z_center = (all_positions[:, 2].max() + all_positions[:, 2].min()) / 2
    
    # Set equal limits for both plots
    x_lim = [x_center - max_range/2, x_center + max_range/2]
    y_lim = [y_center - max_range/2, y_center + max_range/2]
    z_lim = [z_center - max_range/2, z_center + max_range/2]
    
    # Plot before rotation
    ax1.scatter(sat_pos_before[:, 0], sat_pos_before[:, 1], sat_pos_before[:, 2], 
                c='red', marker='o', s=100, label='Original', alpha=0.7)
    # Draw lines connecting satellites
    ax1.plot(sat_pos_before[:, 0], sat_pos_before[:, 1], sat_pos_before[:, 2], 
             'r--', alpha=0.3, linewidth=1)
    # Plot lookat point (pivot)
    ax1.scatter(lookat[0], lookat[1], lookat[2], 
                c='green', marker='*', s=200, label='Cloud Center (Pivot)', alpha=0.9)
    # Draw lines from satellites to lookat point
    for i in range(len(sat_pos_before)):
        ax1.plot([sat_pos_before[i, 0], lookat[0]], 
                 [sat_pos_before[i, 1], lookat[1]], 
                 [sat_pos_before[i, 2], lookat[2]], 
                 'r--', alpha=0.2, linewidth=0.5)
    ax1.set_xlabel('X [km]', fontsize=10)
    ax1.set_ylabel('Y [km]', fontsize=10)
    ax1.set_zlabel('Z [km]', fontsize=10)
    ax1.set_title(f'Original Positions{title_suffix}', fontsize=12)
    ax1.set_xlim(x_lim)
    ax1.set_ylim(y_lim)
    ax1.set_zlim(z_lim)
    ax1.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax1.legend()
    ax1.grid(True)
    
    # Plot after rotation
    ax2.scatter(sat_pos_after[:, 0], sat_pos_after[:, 1], sat_pos_after[:, 2], 
                c='blue', marker='^', s=100, label=f'Rotated {rotation_angle_deg}°', alpha=0.7)
    # Draw lines connecting satellites
    ax2.plot(sat_pos_after[:, 0], sat_pos_after[:, 1], sat_pos_after[:, 2], 
             'b--', alpha=0.3, linewidth=1)
    # Plot lookat point (pivot)
    ax2.scatter(lookat[0], lookat[1], lookat[2], 
                c='green', marker='*', s=200, label='Cloud Center (Pivot)', alpha=0.9)
    # Draw lines from satellites to lookat point
    for i in range(len(sat_pos_after)):
        ax2.plot([sat_pos_after[i, 0], lookat[0]], 
                 [sat_pos_after[i, 1], lookat[1]], 
                 [sat_pos_after[i, 2], lookat[2]], 
                 'b--', alpha=0.2, linewidth=0.5)
    ax2.set_xlabel('X [km]', fontsize=10)
    ax2.set_ylabel('Y [km]', fontsize=10)
    ax2.set_zlabel('Z [km]', fontsize=10)
    ax2.set_title(f'Rotated Positions ({rotation_angle_deg}° around Z-axis){title_suffix}', fontsize=12)
    ax2.set_xlim(x_lim)
    ax2.set_ylim(y_lim)
    ax2.set_zlim(z_lim)
    ax2.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure instead of showing
    from datetime import datetime
    
    if save_path is None:
        save_path = 'satellite_positions_rotation.png'
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_parts = os.path.splitext(save_path)
    save_path_with_timestamp = f"{path_parts[0]}_{timestamp}{path_parts[1]}"
    
    # Create directory if it doesn't exist
    save_dir = os.path.dirname(save_path_with_timestamp)
    if save_dir and not os.path.exists(save_dir):
        safe_mkdirs(save_dir)
    
    plt.savefig(save_path_with_timestamp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Rotation visualization saved to: {save_path_with_timestamp}")


def main():

    print('done')


if __name__ == '__main__':
    main()