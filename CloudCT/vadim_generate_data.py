import cv2
import at3d
from datetime import datetime
import numpy as np
import xarray as xr
from collections import OrderedDict
import pylab as py
import matplotlib.pyplot as plt
import pickle
import os
import scipy.io as sio
import netCDF4
import re
import csv
import glob
from scipy import ndimage
import pandas as pd
from mpl_toolkits.axes_grid1 import AxesGrid, make_axes_locatable
from multiprocessing import Pool
from itertools import repeat
from CloudCTUtils import *
from CloudCT_NoiseUtils import *
import matplotlib
import yaml
# matplotlib.use('TkAgg')

# constants
r_earth = 6371.0  # km
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]



def main(clouds_path, config_path="configs/params_cloudct.yaml"):
    run_params = load_run_params(params_path=config_path)
    cloud_ids = [i.split('/')[-1].split('cloud')[1].split('.txt')[0] for i in
                 glob.glob(clouds_path)]

    all_cloud_paths = ['/'.join(clouds_path.split('/')[:-1]) + '/cloud' + str(cloud_id) + '.txt' for cloud_id in cloud_ids]
    clouds_params = [dict([('path', cloud_path), ('init_lwc', 0.1), ('init_reff', 10)]) for cloud_path in all_cloud_paths]
    clouds = [(str(cloud_id), cloud_params) for cloud_id, cloud_params in zip(cloud_ids, clouds_params)]


    with Pool(processes=run_params['max_simultaneous_simulations']) as p:
        p.map(run_simulation, zip(repeat(run_params), clouds))

    print('finished successfully.')

def simple_main(run_params, clouds_path):
    cloud_ids = [i.split('/')[-1].split('cloud')[1].split('.txt')[0] for i in
                 glob.glob(clouds_path)]
    n = int(len(cloud_ids)/3)
    for cloud_id in cloud_ids[:n]:
        cloud_name = str(cloud_id)
        cloud_path = '/'.join(clouds_path.split('/')[:-1])+'/cloud'+cloud_name+'.txt'
        cloud_params = dict([('path', cloud_path), ('init_lwc', 0.1), ('init_reff', 10)])
        cloud = (cloud_name, cloud_params)
        run_simulation((run_params, cloud))

    print('done')

def run_simulation(args):
    run_params, (cloud_name, cloud_params) = args
    print(f"Simulation of cloud {cloud_name} is running.")
   
    user =  run_params['USER']
    if user == 'Vadim':
        run_params['images_path_for_nn'] = '/wdata_visl/tamar_nadav_generated_clouds/2026/Vadim_tune_AT3D_research/up_circular_data_rando/'        
    else:
        pass
    
    path_stamp = 'train'
    # Define the surface mode of the ocean:
    if run_params['use_ocean_brdf']:
        print("Configuring surface: Ocean Unpolarized (Cox-Munk)")
        ocean_wind_speed = run_params['ocean_wind_speed'] # m/s
        pigmentation = 0.1
        """
        In the context of the 6S ocean model, "pigmentation" refers to the concentration of phytoplankton
        (specifically Chlorophyll-a) in the water. It dictates the "ocean color"how much light
        penetrates the water and scatters back up to your sensor, versus how much is absorbed.
        It is measured in milligrams per cubic meter (mg/m³).
        We assume that in open ocean, pigmentation is 0.1 mg/m³.
        """
        surface_model = at3d.surface.ocean_unpolarized(ocean_wind_speed,pigmentation)
        path_stamp = os.path.join(path_stamp, 'ocean_brdf')
        
    else:
        print("Configuring surface: Lambertian")
        surface_albedo = 0.05 # Typical dark ocean diffuse albedo
        surface_model = at3d.surface.lambertian(surface_albedo)
        path_stamp = os.path.join(path_stamp, 'lambertian')
    
    
    
    filename = os.path.join(run_params['images_path_for_nn'],path_stamp,
                            'cloud_results_' + cloud_name + '.pkl')

    if os.path.exists(filename):
        print(f'skipping cloud in {filename}')
        return
    
    if run_params['IF_NEW_TXT']:
        cloud_scatterer = at3d.util.load_from_csv(cloud_params['path'], density='lwc', origin=(0.0, 0.0))
    else:
        cloud_scatterer_not_padded, dx, nx, dy, ny, z = load_from_csv_shdom(cloud_params['path'], density='lwc', origin=(0.0,0.0))
        cloud_scatterer = pad_cloud_scatterer(cloud_scatterer_not_padded, dx, dy, pad_side=0, pad_bottom=0, pad_top=2)

    # make sure all values will exist in the mie tables
    cloud_scatterer.veff.data[cloud_scatterer.veff.data <= 0.02] = 0.0201
    cloud_scatterer.veff.data[cloud_scatterer.veff.data >= 0.55] = 0.55
    cloud_scatterer.reff.data[cloud_scatterer.reff.data <= 0.01] = 0.0101
    cloud_scatterer.reff.data[cloud_scatterer.reff.data >= 35] = 35-1.1e-3
    cloud_scatterer_not_padded.veff.data[cloud_scatterer_not_padded.veff.data <= 0.02] = 0.0201
    cloud_scatterer_not_padded.veff.data[cloud_scatterer_not_padded.veff.data >= 0.55] = 0.55
    cloud_scatterer_not_padded.reff.data[cloud_scatterer_not_padded.reff.data <= 0.01] = 0.0101
    cloud_scatterer_not_padded.reff.data[cloud_scatterer_not_padded.reff.data >= 35] = 35-1.1e-3
   
    # load atmosphere
    if user == 'Vadim':
        atmosphere = xr.open_dataset('../data/ancillary/AFGL_summer_mid_lat.nc')    
    else:
        atmosphere = xr.open_dataset('../AT3D_research/data/ancillary/AFGL_summer_mid_lat.nc')
    # subset the atmosphere, choose only the bottom twenty km.
    reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 5.0]}) #ASK YOAV
    # merge the atmosphere and cloud z coordinates
    merged_z_coordinate = at3d.grid.combine_z_coordinates([reduced_atmosphere, cloud_scatterer]) # we need to combine! (in shdom it is doing it by itself and in at3d not)

    # define the property grid - which is equivalent to the base RTE grid
    rte_grid = at3d.grid.make_grid(cloud_scatterer.x.diff('x')[0], cloud_scatterer.x.data.size,
                                   cloud_scatterer.y.diff('y')[0], cloud_scatterer.y.data.size,
                                   merged_z_coordinate)
 

    cloud_scatterer_on_rte_grid = at3d.grid.resample_onto_grid(rte_grid, cloud_scatterer)

    size_distribution_function = at3d.size_distribution.gamma

    ##### get optical property generator #####
    wavelength_bands = run_params['wavelengths']
    mean_wavelengths = [np.mean(wavelength_band) for wavelength_band in wavelength_bands]

    mie_mono_tables = OrderedDict()
    for mean_wavelength, wavelength_band in zip(mean_wavelengths, wavelength_bands):
        wavelength_band_tuple = (wavelength_band[0], wavelength_band[1])

        wavelen1, wavelen2 = wavelength_band_tuple

        if wavelen1 == wavelen2:
            wavelength_averaging = False
            formatstr = 'mie_mono_Water_{}nm.nc'.format(int(1e3 * wavelen1))
        else:
            wavelength_averaging = True
            formatstr = 'mie_mono_averaged_Water_{}-{}nm.nc'.format(int(1e3 * wavelength_band[0]),
                                                                    int(1e3 * wavelength_band[1]))
        mie_mono_table = at3d.mie.get_mono_table(
            'Water', wavelength_band_tuple,
            wavelength_averaging=wavelength_averaging,
            max_integration_radius=65.0,
            minimum_effective_radius=0.1,
            relative_dir='/wdata/inbalkom/AT3D_CloudCT_shared_files/mie_tables/',
            verbose=False
        )
        mie_mono_tables[mean_wavelength] = mie_mono_table
        print('added mie with wavelength of {}'.format(mean_wavelength))
        # mie_mono_table.to_netcdf(mono_path)



    optical_property_generator = at3d.medium.OpticalPropertyGenerator(
        'cloud',
        mie_mono_tables,
        size_distribution_function,
        reff=np.linspace(0.01, 35.0, 30),
        veff=np.linspace(0.02, 0.56, 10),
    )
    
    only_cloud_optical_properties = optical_property_generator(cloud_scatterer_not_padded)    ##ASK VADIM
    optical_properties = optical_property_generator(cloud_scatterer_on_rte_grid)
    
    # Calculate extinction from optical properties
    only_coud_extinction = np.array(only_cloud_optical_properties[mean_wavelengths[0]].extinction)
    # Calculate extinction from optical properties

    n_nonzero = np.count_nonzero(only_coud_extinction)
    if n_nonzero < 10:
        print(f'Skipping cloud {cloud_name}: only {n_nonzero} non-zero extinction voxels (< 200).')
        return

    # if run_params.get('plot_mip', False):
    #     mip_dir = run_params.get('plot_mip_path')
    #     mip_axis = run_params.get('plot_mip_axis', 'z')
    #     if mip_dir is not None:
    #         axis_name = str(mip_axis).lower() if isinstance(mip_axis, str) else ('x', 'y', 'z')[int(mip_axis)]
    #         mip_path = os.path.join(mip_dir, f'cloud_{cloud_name}', f'mip_{axis_name}.png')
    #         plot_mip(only_coud_extinction, save_path=mip_path, title=f'Cloud {cloud_name} MIP ({axis_name})', axis=mip_axis)

    # one function to generate rayleigh scattering.
    rayleigh_scattering = at3d.rayleigh.to_grid(mean_wavelengths, atmosphere, rte_grid)

    solvers_dict = at3d.containers.SolversDict()
    # note we could set solver dependent surfaces / sources / numerical_config here
    # just as we have got solver dependent optical properties.
    
    sun_azimuth= run_params['const_sun_azimuth']
    sun_zenith = run_params['const_sun_zenith']
    

    cloud = {'images_noise': [],
             'images': [],
             'mask': [],
             'mask_morph': [],
             'cloud_path': cloud_params['path'],
             'sun_zenith': sun_zenith,
             'sun_azimuth': sun_azimuth,
             'cameras_pos': [],
             'cameras_P': [],
             'grid': [],
             'ext': only_coud_extinction
             }

    
    for wavelength in mean_wavelengths:
        medium = {
            'cloud': optical_properties[wavelength],
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

        solvers_dict.solve(n_jobs=run_params['n_jobs'], maxiter=run_params['maxiter'])

        ##### define sensors #####
        GSD = run_params['GSD']  # km
        SATS_NUMBER_SETUP = run_params['SATS_NUMBER']
        sensor_dict = at3d.containers.SensorsDict()

        xgrid = np.float32(cloud_scatterer.x.data)
        ygrid = np.float32(cloud_scatterer.y.data)
        zgrid = np.float32(cloud_scatterer.z.data)
        grid = np.array([xgrid, ygrid, zgrid], dtype=object)

        dx = cloud_scatterer.delx.item() # res of each grid cell in x direction
        dy = cloud_scatterer.dely.item() # res of each grid cell in y direction
        dz = round(np.diff(zgrid)[0], 5) # res of each grid cell in z direction
        nx, ny, nz = cloud_scatterer.dims['x'], cloud_scatterer.dims['y'], cloud_scatterer.dims['z']

        PIXEL_FOOTPRINT = GSD  # km
        L = max(xgrid.max() - xgrid.min(), ygrid.max() - ygrid.min())

        fov = 2 * np.rad2deg(np.arctan(0.5 * L / (run_params['R_sat'])))
        cny = int(np.floor(L / PIXEL_FOOTPRINT))
        cnx = int(np.floor(L / PIXEL_FOOTPRINT))

        CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]
        # Somtimes it is more convinient to use wide fov to see the whole cloud
        # from all the view points. so the FOV is also tuned:
        IFTUNE_CAM = True
        # --- TUNE FOV, CNY,CNX:
        if (IFTUNE_CAM):
            L *= run_params['tune_scalar']
            fov = 2 * np.rad2deg(np.arctan(0.5 * L / (run_params['R_sat'])))
            cny = int(np.floor(L / PIXEL_FOOTPRINT))
            cnx = int(np.floor(L / PIXEL_FOOTPRINT))

            # not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
        # tuning is applied by the variavle LOOKAT.
        LOOKAT = CENTER_OF_MEDIUM_BOTTOM
        if (IFTUNE_CAM):
            LOOKAT[2] = 0.68 * nx * dz  # tuning. if IFTUNE_CAM = False, just lookat the bottom

        SAT_LOOKATS = np.array(SATS_NUMBER_SETUP * LOOKAT).reshape(-1,3)  # currently, all satellites lookat the same point.

        print(20 * "-")
 
        print("CAMERA intrinsics summary")
        print("fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov, cnx, cny))

        print(20 * "-")

       
        d_omega = calculate_delta_omega(run_params['R_max'], run_params['R_earth'], run_params['R_sat'])
        print(f"Delta Omega: {d_omega:.4f} rad ({np.degrees(d_omega):.2f} degrees)")
        sat_positions = sample_camera_locations_randomized(run_params['SATS_NUMBER'], run_params['R_sat'], run_params['R_earth'], d_omega)
        up_list = np.array(sat_positions.shape[1] * [0, 1, 0]).reshape(-1, 3)  # default up vector per camera.
        
 
        names = ["sat" + str(i + 1) for i in range(sat_positions.shape[1])]

        # Zenith angle (from vertical) per camera
        coords_km = np.asarray(sat_positions[0])
        zenith_angles_deg = calculate_zenith_angles(coords_km, run_params['R_earth'])
        print("Zenith angles (deg from vertical) per camera:")
        for name, angle_deg in zip(names, zenith_angles_deg):
            print(f"  {name}: {angle_deg:.2f} deg")
        
        
        
        for mean_wavelength in mean_wavelengths:
            for position_vector, lookat_vector, up_vector in zip(sat_positions[0],
                                                                       SAT_LOOKATS, up_list):
                loop_sensor = at3d.sensor.perspective_projection(wavelength=mean_wavelength, fov=fov,
                                                                 x_resolution=cnx, y_resolution=cny,
                                                                 position_vector=position_vector,
                                                                 lookat_vector=lookat_vector,
                                                                 up_vector=up_vector, stokes=run_params['stokes'])

                sensor_dict.add_sensor('CloudCT'+str(int(mean_wavelength*1e3)), loop_sensor)
        print('Done defining CloudCT''s sensors')
        print('getting CloudCT''s measurments')
        
        # Next part will be the rendering, when the RTE solver is prepared (below).
        # get the measurements
        sensor_dict.get_measurements(solvers_dict, n_jobs=run_params['n_jobs'], verbose=True) # RTE + RENDERING
        show_results(sensor_dict)
        print('Done getting CloudCT''s measurments')

        if not run_params['cancel_noise']:
            sensor_dict_clean = copy.deepcopy(sensor_dict)

            images_clean = []
            for instrument_ind, (instrument, sensor_group) in enumerate(sensor_dict_clean.items()):
                sensor_images = sensor_dict_clean.get_images(instrument)
                sensor_group_list = sensor_dict_clean[instrument]['sensor_list']
                assert len(names) == len(sensor_group_list), "len(names) does not match len(sensor_group_list)"
                for sensor_ind, (sensor, sensor_name) in enumerate(zip(sensor_group_list, names)):
                    # add image I to 'images_clean' in order to save in file
                    curr_image = np.array(sensor_images[sensor_ind].I.data)
                    images_clean.append(curr_image*np.cos(np.deg2rad(180-sun_zenith)))  # Multiply the images by cos(sun-zenith-angle)
                    # copied = sensor.copy(deep=True) 
            # images clean is a lise with clean images, each image with size 1,116,116               
            # The sensor_dict_clean is no not polarized and we don't need to convert the Stokes vector between camera and scattering frames
            images_noise = add_noise_to_images(run_params, sensor_dict_clean, sun_zenith, names, cnx, cny) # return 10,116,116,1 
        else:
            images_clean=[]

        if run_params.get('plot_simulation_images', False):
            images_noise_for_plot = None
            if not run_params['cancel_noise']:
                images_noise_for_plot = np.array(images_noise)[..., 0]
            save_dir = run_params.get('plot_simulation_images_path')
            if save_dir is not None:
                save_dir = os.path.join(save_dir, f'cloud_{cloud_name}')
            plot_simulation_images(images_clean, images_noise_for_plot, show=True, save_dir=save_dir)

            # Optional: visualize satellite positions in a circular configuration,
            # including the common look-at point and lines from each camera to it.
            sat_vis_path = None
            if save_dir is not None:
                sat_vis_path = os.path.join(save_dir, 'circular_camera_distribution.png')
                proj_vis_path = os.path.join(save_dir, 'camera_positions_projections.png')
            else:
                sat_vis_path = None
                proj_vis_path = None
            plot_circular_camera_distribution(sat_positions, run_params, save_path=sat_vis_path, lookat=LOOKAT)
            plot_camera_positions_projections(sat_positions, run_params, save_path=proj_vis_path, lookat=LOOKAT)

        # ----------------------------------------------------

        sensor_list = []
        # images = []
        ray_mu_list = []
        ray_phi_list = []
        projection_matrices = []
        # images_scatter = []
        for instrument_ind, (instrument, sensor_group) in enumerate(sensor_dict.items()):
            sensor_images = sensor_dict.get_images(instrument)
            sensor_group_list = sensor_dict[instrument]['sensor_list']
            assert len(names) == len(sensor_group_list), "len(names) does not match len(sensor_group_list)"
            for sensor_ind, (sensor, sensor_name) in enumerate(zip(sensor_group_list, names)):
                copied = sensor.copy(deep=True)

                # add ray_mu and ray_phi to lists for future scattering plane calculations
                ray_mu_list.append(copied.ray_mu.data)
                ray_phi_list.append(copied.ray_phi.data)

                # create 'sensor_list' for space carving - without cloudbow!
                if (len(names[sensor_ind].split('_')) == 1) or (len(names[sensor_ind].split('_'))==2 and names[sensor_ind][-2:] == 's0'):
                    ray_mask_pixel = np.zeros(copied.npixels.size, dtype=int)
                    ray_mask_pixel[np.where(copied.I.data > run_params['radiance_thresholds'][sensor_ind])] = 1
                    copied['weights'] = ('nrays', copied.I.data)
                    copied['cloud_mask'] = ('nrays', ray_mask_pixel[copied.pixel_index.data])
                    sensor_list.append(copied)

                # add projection_matrix to 'projection_matrices' in order to save in file
                projection_matrices.append(np.reshape(copied.attrs['projection_matrix'], (3, 4)))

                # # add image to 'images' in order to save in file
                # curr_image = np.array([sensor_images[sensor_ind].I.data.T])
                # images.append(curr_image)
                # images_scatter.append(calc_image_in_scattering_plane_vectorbase(copied, curr_image, sensor_name, sun_azimuth,
                #                                                                       sun_zenith))


            print('getting CloudCT''s space carving')
            space_carver = at3d.space_carve.SpaceCarver(rte_grid, bcflag=3)
            agreement = 0.8
            carved_volume = space_carver.carve(sensor_list, agreement=(0.0, agreement), linear_mode=False)
            mask4file = carved_volume.mask.data[:, :, :cloud_scatterer.z.data.size]
            npad = ((1, 1), (1, 1), (1, 1))

            mask_data_padded = np.pad(mask4file.copy(),
                                      pad_width=npad, mode='constant', constant_values=0)

            mask4file = mask4file > 0  # convert from int to bool

            struct = ndimage.generate_binary_structure(3, 2)
            mask_morph = ndimage.binary_closing(mask_data_padded, struct)
            mask_morph = mask_morph[1:-1, 1:-1, 1:-1]

            # remove cloud mask values at outer boundaries to prevent interaction with open boundary conditions.
            # carved_volume.mask[0] = carved_volume.mask[-1] = carved_volume.mask[:, 0] = carved_volume.mask[:, -1] = 0.0

            cloud['images_noise'] = (np.array(images_noise)[..., 0])#list of 10 each item is 116,116,1
            cloud['images'] = (np.array(images_clean)) 
            cloud['mask'] = (mask4file)
            cloud['mask_morph'] = (mask_morph)
            cloud['cameras_pos'] = (sat_positions)
            cloud['cameras_P'] = (np.array(projection_matrices))
            cloud['grid'] = (grid)
            
    


    if not os.path.exists(os.path.join(run_params['images_path_for_nn'], path_stamp)):
        # Create a new directory because it does not exist
        safe_mkdirs(os.path.join(run_params['images_path_for_nn'], path_stamp))
        print("The directory for saving cloud results for option {} was created.".format(path_stamp))

    print(f'saving cloud in {filename}')
    with open(filename, 'wb') as outfile:
        pickle.dump(cloud, outfile, protocol=pickle.HIGHEST_PROTOCOL)

    print("--------------")


def process_Rois_projections(projections, mean_x, mean_y, stokes, wavelengths, fill_ray_variables=True):
    sensor_roi_projections = at3d.containers.SensorsDict()
    image_shape = np.array([350,350])
    proj_names = []
    for wavelength in wavelengths:
        for i, (key, projection) in enumerate(projections.items()):
            mask = projection['mask']

            center_x, center_y = ndimage.center_of_mass(mask)

            height, width = mask.shape[:2]
            t_height = int(height / 2 - center_x)
            t_width = int(width / 2 - center_y)
            T = np.float32([[1, 0, t_width], [0, 1, t_height]])


            x_s = int(height / 2) - int(image_shape[0] / 2)
            x_e = int(height / 2) + int(image_shape[0] / 2)
            y_s = int(width / 2) - int(image_shape[1] / 2)
            y_e = int(width / 2) + int(image_shape[1] / 2)

            # assert np.all(mask[x_s:x_e,y_s:y_e])
            x = np.full(mask.shape, np.nan)
            x[mask] = projection['x']
            x = cv2.warpAffine(x, T, (width, height), borderValue=np.nan)
            x = x[x_s:x_e, y_s:y_e] + mean_x
            x = x.flatten()

            y = np.full(mask.shape, np.nan)
            y[mask] = projection['y']
            y = cv2.warpAffine(y, T, (width, height), borderValue=np.nan)
            y = y[x_s:x_e, y_s:y_e] + mean_y
            y = y.flatten()

            z = np.full(mask.shape, np.nan)
            z[mask] = projection['z']
            z = cv2.warpAffine(z, T, (width, height), borderValue=np.nan)
            z = z[x_s:x_e, y_s:y_e]
            z = z.flatten()

            mu = np.full(mask.shape, np.nan)
            mu[mask] = projection['mu']
            mu = cv2.warpAffine(mu, T, (width, height), borderValue=np.nan)
            mu = mu[x_s:x_e, y_s:y_e]
            mu = mu.flatten()

            phi = np.full(mask.shape, np.nan)
            phi[mask] = projection['phi']
            phi = cv2.warpAffine(phi, T, (width, height), borderValue=np.nan)
            phi = phi[x_s:x_e, y_s:y_e]
            phi = phi.flatten()

            sensor = at3d.sensor.make_sensor_dataset(
                x, y, z, mu, phi, stokes, wavelength, fill_ray_variables=fill_ray_variables)

            sensor['image_shape'] = xr.DataArray(image_shape, coords={'image_dims': ['nx', 'ny']}, dims='image_dims')
            instrument_name = str(int(wavelength*1000))
            sensor_roi_projections.add_sensor(instrument_name, sensor)
            proj_names.append(key)

    return sensor_roi_projections, proj_names


def plot_cloud_images(images):
   
    
    # Create output directory
    output_dir = '/wdata/tamarsd/AT3D_research/CloudCT/figures/results_clouds'
    safe_mkdirs(output_dir)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ------------------
    # I:
    fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axarr = axarr.flatten()
    for ax, image in zip(axarr, images):
        image = np.squeeze(image.copy())
        im = ax.imshow(image[0, ...], cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax)
    fig.suptitle('I', size=16, y=0.95)
    
    # Save I figure
    filename = f"cloud_images_I_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {filepath}")

  
    
    

    # --------------------
    print('done plotting')

def load_run_params(params_path):
    # Load run parameters
    params_file_path = params_path
    with open(params_file_path, 'r') as f:
        run_params = yaml.full_load(f)

    return run_params


if __name__ == '__main__':
    # Load configuration from YAML file
    # Change this path to use either params_cloudct.yaml or params_airmspi.yaml
    #config_path = "/wdata/tamarsd/AT3D_research/CloudCT/configs/params_cloudct.yaml"
    config_path = "configs/params_cloudct.yaml"  # Default to CloudCT config    
    run_params = load_run_params(params_path=config_path)
    
    clouds_path = "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"
    # "/wdata/yaelsc/Data/CASS_50m_256x256x139_600CCN/64_64_32_cloud_fields/cloud*.txt" - CASS 
   
    # "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"
    
    #"/wdata/tamarsd/DATA_7_CLOUDS_TEXT/fast/cloud*.txt"
    #"/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"
    #
    #"/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"
    #"/wdata/roironen/Data/subset_of_seven_clouds/clouds/cloud*.txt"
    #"/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"

    #main(clouds_path, config_path)
    simple_main(run_params, clouds_path)