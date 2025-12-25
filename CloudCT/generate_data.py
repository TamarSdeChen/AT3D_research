import cv2
import at3d
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
    # cloud_ids = [6004]
    # cloud_ids = [350]
    for cloud_id in cloud_ids:
        cloud_name = str(cloud_id)
        cloud_path = '/'.join(clouds_path.split('/')[:-1])+'/cloud'+cloud_name+'.txt'
        cloud_params = dict([('path', cloud_path), ('init_lwc', 0.1), ('init_reff', 10)])
        cloud = (cloud_name, cloud_params)
        run_simulation((run_params, cloud))

    print('done')

def run_simulation(args):
    run_params, (cloud_name, cloud_params) = args
    print(f"Simulation of cloud {cloud_name} is running.")

    
    if not run_params['IF_AIRMSPI']:
        if run_params['IS_SUN_CONST'] == 0:
            path_stamp = 'varying_sun_lambertian_surface'
        else:
            path_stamp = 'const_sun_lambertian_surface'
        filename = os.path.join(run_params['images_path_for_nn'],path_stamp,
                                'cloud_results_' + cloud_name + '.pkl')

        if os.path.exists(filename):
            print(f'skipping cloud in {filename}')
            return
    
    if run_params['IF_AIRMSPI']:
        Inbals_projections_path = '/wdata/inbalkom/Data/AirMSPI/Projections/train'
        # output_base_path = run_params['satellites_images_path']

        format_ = '*'  # load
        paths = sorted(glob.glob(Inbals_projections_path + '/' + format_))
        n_files = len(paths)
        num_exist=0
        for path in paths:
            path_stamp = path.split('/')[-1][:-4]
            filename = os.path.join(run_params['images_path_for_nn'], 'SIMULATED_AIRMSPI_TRAIN_' + path_stamp,
                                    'cloud_results_' + cloud_name + '.pkl')
            if os.path.exists(filename):
                num_exist += 1
        if num_exist == n_files:
            print(f'skipping cloud in {cloud_name}')
            return

    if run_params['IF_NEW_TXT']:
        cloud_scatterer = at3d.util.load_from_csv(cloud_params['path'], density='lwc', origin=(0.0, 0.0))
    else:
        cloud_scatterer = load_from_csv_shdom(cloud_params['path'], density='lwc', origin=(0.0,0.0))

    # make sure all values will exist in the mie tables
    cloud_scatterer.veff.data[cloud_scatterer.veff.data <= 0.02] = 0.0201
    cloud_scatterer.veff.data[cloud_scatterer.veff.data >= 0.55] = 0.55
    cloud_scatterer.reff.data[cloud_scatterer.reff.data <= 0.01] = 0.0101
    cloud_scatterer.reff.data[cloud_scatterer.reff.data >= 35] = 35-1.1e-3

    if run_params['IF_AIRMSPI']:
        cloud_scatterer.density.data = cloud_scatterer.density.data/10

    # load atmosphere
    atmosphere = xr.open_dataset('../AT3D_research/data/ancillary/AFGL_summer_mid_lat.nc')
    # subset the atmosphere, choose only the bottom twenty km.
    reduced_atmosphere = atmosphere.sel({'z': atmosphere.coords['z'].data[atmosphere.coords['z'].data <= 20.0]})
    # merge the atmosphere and cloud z coordinates
    merged_z_coordinate = at3d.grid.combine_z_coordinates([reduced_atmosphere, cloud_scatterer])

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
        mono_path = os.path.join('/wdata/inbalkom/AT3D_CloudCT_shared_files/mie_tables', formatstr)
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
    optical_properties = optical_property_generator(cloud_scatterer_on_rte_grid)

    # one function to generate rayleigh scattering.
    rayleigh_scattering = at3d.rayleigh.to_grid(mean_wavelengths, atmosphere, rte_grid)

    solvers_dict = at3d.containers.SolversDict()
    # note we could set solver dependent surfaces / sources / numerical_config here
    # just as we have got solver dependent optical properties.
    if run_params['IS_SUN_CONST']:
        sun_location_num = 1
        sun_azimuth_list = [run_params['const_sun_azimuth']]
        sun_zenith_list = [run_params['const_sun_zenith']]
        lat_list = [-999]
        long_list = [-999]
        utc_time_list = [0]
        print('set const sun_azimuth as {}deg and const sun_zenith as {}deg'.format(run_params['const_sun_azimuth'], run_params['const_sun_zenith']))
    else:
        sun_location_num = run_params['num_of_sun_locations']
        sun_azimuth_list = []
        sun_zenith_list = []
        lat_list = []
        long_list = []
        utc_time_list = []
        for _ in range(sun_location_num):
            if run_params['use_sunsync_file']:
                assert (90 < run_params['zenith_thr'] < 180)
                sun_azimuth, sun_zenith, utc_time, lat, long, sat_dir_angle = (
                    generate_random_sun_angles_from_sunsync_orbit(run_params['sunsync_file_path'], run_params['zenith_thr']))
                # sun_azimuth = sun_azimuth - sat_dir_angle  # azimuth relatively to the direction of motion
                # if sun_azimuth < -180:
                #     sun_azimuth = sun_azimuth + 360
                if sun_azimuth > 180:
                    sun_azimuth = sun_azimuth - 360
            else:
                assert (-90 < run_params['Lat_for_sun_angles'] < 90)
                sun_azimuth, sun_zenith = generate_random_sun_angles_for_lat(run_params['Lat_for_sun_angles'])
                lat = -999
                long = -999
                utc_time = 0
            sun_azimuth_list.append(sun_azimuth)
            sun_zenith_list.append(sun_zenith)
            lat_list.append(lat)
            long_list.append(long)
            utc_time_list.append(utc_time)
        # sun_azimuth_list = [97, 160]
        # sun_zenith_list = [148, 102]
        print('set varying sun_azimuth and sun zenith with augmentations.')

    surface = at3d.surface.lambertian(0.05)

    cloud = {'images': [],
             'images_scatter': [],
             'images_clean': [],
             'images_clean_scatter': [],
             'mask': [],
             'mask_morph': [],
             'cloud_path': cloud_params['path'],
             'sun_zenith': np.array(sun_zenith_list),
             'sun_azimuth': np.array(sun_azimuth_list),
             'ray_mu': [],
             'ray_phi': [],
             'cameras_pos': [],
             'cameras_P': [],
             'grid': [],
             'not_cloudbow_startind': [],
             'cloudbow_sample_angles': []
             }

    for location_idx, sun_zenith, sun_azimuth in zip(range(sun_location_num), sun_zenith_list, sun_azimuth_list):
        print('beginning solving RTE for sun aug #{}/{}'.format(location_idx+1,sun_location_num))
        for wavelength in mean_wavelengths:
            medium = {
                'cloud': optical_properties[wavelength],
                'rayleigh': rayleigh_scattering[wavelength]
            }
            config = at3d.configuration.get_config()

            solvers_dict.add_solver(
                wavelength,
                at3d.solver.RTE(
                    numerical_params=config,
                    surface=surface,
                    source=at3d.source.solar(wavelength, np.cos(sun_zenith * np.pi / 180), sun_azimuth),
                    # np.cos((180-cam_zenith)*np.pi/180), cam_azimuth),
                    medium=medium,
                    num_stokes=3
                )

            )

        solvers_dict.solve(n_jobs=run_params['n_jobs'], maxiter=run_params['maxiter'])

        ##### define sensors #####

        if run_params['IF_AIRMSPI']:
            # TODO - fix for sun augment!!
            PAD_SIDES = 0
            # center of domain
            mean_x = cloud_scatterer.x.diff('x')[0] * 0.5 * (cloud_scatterer.x.data.size + 2 * PAD_SIDES)
            mean_y = cloud_scatterer.y.diff('y')[0] * 0.5 * (cloud_scatterer.y.data.size + 2 * PAD_SIDES)

            Inbals_projections_path = '/wdata/inbalkom/Data/AirMSPI/Projections/train'
            # output_base_path = run_params['satellites_images_path']

            format_ = '*'  # load
            paths = sorted(glob.glob(Inbals_projections_path + '/' + format_))
            n_files = len(paths)
            for path in paths:
                path_stamp = path.split('/')[-1][:-4]
                print('defining AIRMSPI''s {} sensors'.format(path_stamp))
                # Output_path = os.path.join(output_base_path, 'SIMULATED_AIRMSPI_TRAIN_' + path_stamp)
                # if not os.path.exists(Output_path):
                #     os.mkdir(Output_path)

                # ---------------------------------------
                # ---------OPEN PROJECTIONS--------------
                # ---------------------------------------
                with open(path, 'rb') as f:
                    projections = pickle.load(f)

                sensor_dict, names = process_Rois_projections(projections, mean_x.data, mean_y.data, stokes=run_params['stokes'],
                                                       wavelengths=mean_wavelengths, fill_ray_variables=True)

                print('Done defining AIRMSPI''s {} sensors'.format(path_stamp))
                print('getting AIRMSPI''s {} measurments'.format(path_stamp))
                # Next part will be the rendering, when the RTE solver is prepared (below).
                # get the measurements
                sensor_dict.get_measurements(solvers_dict, n_jobs=run_params['n_jobs'], verbose=True)
                print('Done getting AIRMSPI''s {} measurments'.format(path_stamp))

                # add AirMSPI noise to rendered images:
                sensor_dict, _ = add_airmspi_noise(sensor_dict, run_params['stokes'], names)

                # Perform some cloud masking using a single fixed threshold based on the observation that
                # everywhere else will be very dark.
                sensor_list = []
                images = []
                ray_mu_list = []
                ray_phi_list = []
                for instrument_ind, (instrument, sensor_group) in enumerate(sensor_dict.items()):
                    sensor_images = sensor_dict.get_images(instrument)
                    sensor_group_list = sensor_dict[instrument]['sensor_list']
                    assert len(names) == len(sensor_group_list), "len(names) does not match len(sensor_group_list)"
                    for sensor_ind, sensor in enumerate(sensor_group_list):
                        if (run_params['stokes'] == ['I']) or (run_params['stokes'] == 'I'):
                            curr_image = np.array([sensor_images[sensor_ind].I.data])
                        else:
                            curr_image = np.stack(
                                [sensor_images[sensor_ind][pol_channel].data for pol_channel in run_params['stokes']])
                        images.append(curr_image)
                        copied = sensor.copy(deep=True)

                        # add ray_mu and ray_phi to lists for future scattering plane calculations
                        ray_mu_list.append(copied.ray_mu.data)
                        ray_phi_list.append(copied.ray_phi.data)

                        # create 'sensor_list' for space carving
                        ray_mask_pixel = np.zeros(copied.npixels.size, dtype=int)
                        ray_mask_pixel[np.where(copied.I.data > run_params['radiance_thresholds'][sensor_ind])] = 1
                        copied['weights'] = ('nrays', copied.I.data)
                        copied['cloud_mask'] = ('nrays', ray_mask_pixel[copied.pixel_index.data])
                        sensor_list.append(copied)

                print('getting AIRMSPI''s {} space carving'.format(path_stamp))
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

                if 0:
                    plot_cloud_images(images)
                    a=5


                cloud = {'images': np.array(images),
                         'mask': mask4file,
                         'mask_morph': mask_morph,
                         'cloud_path': cloud_params['path'],
                         'sun_zenith': sun_zenith,
                         'sun_azimuth': sun_azimuth,
                         'wind_speed': surface_wind_speed,
                         'ray_mu': np.array(ray_mu_list),
                         'ray_phi': np.array(ray_phi_list)}


                filename = os.path.join(run_params['images_path_for_nn'],'SIMULATED_AIRMSPI_TRAIN_'+path_stamp,
                                        'cloud_results_' + cloud_name + '.pkl')
                print(f'saving cloud in {filename}')

                if not os.path.exists(os.path.join(run_params['images_path_for_nn'],'SIMULATED_AIRMSPI_TRAIN_'+path_stamp)):
                    # Create a new directory because it does not exist
                    safe_mkdirs(os.path.join(run_params['images_path_for_nn'],'SIMULATED_AIRMSPI_TRAIN_'+path_stamp))
                    print("The directory for saving cloud results for projection {} was created.".format(path_stamp))

                with open(filename, 'wb') as outfile:
                    pickle.dump(cloud, outfile, protocol=pickle.HIGHEST_PROTOCOL)

                print("--------------")
        else:
            # if not airmspi - string of pearls
            print('defining String of Pearls perspective sensors')
            GSD = run_params['GSD']  # km
            Rsat = run_params['Rsat']  # km
            SATS_NUMBER_SETUP = run_params['SATS_NUMBER']
            cloudbow_additional_scan = run_params['cloudbow_additional_scan']
            sensor_dict = at3d.containers.SensorsDict()

            xgrid = np.float32(cloud_scatterer.x.data)
            ygrid = np.float32(cloud_scatterer.y.data)
            zgrid = np.float32(cloud_scatterer.z.data)
            grid = np.array([xgrid, ygrid, zgrid], dtype=object)

            dx = cloud_scatterer.delx.item()
            dy = cloud_scatterer.dely.item()
            dz = round(np.diff(zgrid)[0], 5)
            nx, ny, nz = cloud_scatterer.dims['x'], cloud_scatterer.dims['y'], cloud_scatterer.dims['z']

            PIXEL_FOOTPRINT = GSD  # km
            L = max(xgrid.max() - xgrid.min(), ygrid.max() - ygrid.min())

            fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
            cny = int(np.floor(L / PIXEL_FOOTPRINT))
            cnx = int(np.floor(L / PIXEL_FOOTPRINT))

            CENTER_OF_MEDIUM_BOTTOM = [0.5 * nx * dx, 0.5 * ny * dy, 0]
            # Somtimes it is more convinient to use wide fov to see the whole cloud
            # from all the view points. so the FOV is also tuned:
            IFTUNE_CAM = True
            # --- TUNE FOV, CNY,CNX:
            if (IFTUNE_CAM):
                L *= run_params['tune_scalar']
                fov = 2 * np.rad2deg(np.arctan(0.5 * L / (Rsat)))
                cny = int(np.floor(L / PIXEL_FOOTPRINT))
                cnx = int(np.floor(L / PIXEL_FOOTPRINT))

                # not for all the mediums the CENTER_OF_MEDIUM_BOTTOM is a good place to lookat.
            # tuning is applied by the variavle LOOKAT.
            LOOKAT = CENTER_OF_MEDIUM_BOTTOM
            if (IFTUNE_CAM):
                LOOKAT[2] = 0.68 * nx * dz  # tuning. if IFTUNE_CAM = False, just lookat the bottom

            SAT_LOOKATS = np.array(SATS_NUMBER_SETUP * LOOKAT).reshape(-1,
                                                                       3)  # currently, all satellites lookat the same point.

            print(20 * "-")
            print(20 * "-")
            print(20 * "-")

            print("CAMERA intrinsics summary")
            print("fov = {}[deg], cnx = {}[pixels],cny ={}[pixels]".format(fov, cnx, cny))

            print(20 * "-")
            print(20 * "-")
            print(20 * "-")

            sat_positions, near_nadir_view_index, theta_max, theta_min = \
                StringOfPearls(SATS_NUMBER=SATS_NUMBER_SETUP,
                               orbit_altitude=Rsat,
                               move_nadir_x=CENTER_OF_MEDIUM_BOTTOM[0],
                               move_nadir_y=CENTER_OF_MEDIUM_BOTTOM[1])

            names = ["sat" + str(i + 1) for i in range(len(sat_positions))]

            if cloudbow_additional_scan>0:
                print(f"CloudCT has {cloudbow_additional_scan} samples in the cloudbow range.")

                if np.all(np.all(SAT_LOOKATS, axis=0)):
                    cloudbow_lookat = SAT_LOOKATS[0, :]
                else:
                    # if the lookats are different, just calculate the mean lookat.
                    cloudbow_lookat = np.mean(SAT_LOOKATS, axis=0)

                """
                INBAL TODO - monitor cloudbow_sample_angles, not_cloudbow_startind
                """
                try:
                    cloud_bow_sat_positions, cloudbow_sample_angles, not_cloudbow_startind = \
                        StringOfPearlsCloudBowScan(Rsat,
                                                   cloudbow_lookat,
                                                   cloudbow_additional_scan,
                                                   run_params['cloudbow_range'],
                                                   theta_max, theta_min,
                                                   sun_zenith, sun_azimuth,
                                                   move_nadir_x=CENTER_OF_MEDIUM_BOTTOM[0],
                                                   move_nadir_y=CENTER_OF_MEDIUM_BOTTOM[1]
                                                   )

                    # what is the scan_imager_index?

                    distances = sat_positions - cloud_bow_sat_positions[0]
                    distances = np.linalg.norm(distances, axis=1)
                    scan_imager_index = np.argmin(distances)
                    # update positions, lookats and names:
                    sat_positions = np.append(sat_positions, cloud_bow_sat_positions, axis=0)
                    cloudbow_lookat = np.tile(cloudbow_lookat, (cloudbow_additional_scan, 1))
                    SAT_LOOKATS = np.append(SAT_LOOKATS, cloudbow_lookat, axis=0)

                    for i in range(cloudbow_additional_scan):
                        names.append(names[scan_imager_index] + '_s{}'.format(i + 1))
                    names[scan_imager_index] = names[scan_imager_index] + '_s{}'.format(0)

                    assert sat_positions.shape[1] == 3, "Problem in satellites positions."
                    assert SAT_LOOKATS.shape[1] == 3, "Problem in satellites pointing."
                    assert len(names) == (cloudbow_additional_scan + SATS_NUMBER_SETUP), \
                        "Problem in satellites counting."

                except Exception as e:
                    print(f'FAILED TO SIMULATE {cloud_name}, sun aug #{location_idx}, {e}')
                    return
            else:  #no cloudbow scan
                cloudbow_sample_angles = None
                not_cloudbow_startind = None


            # we intentionally, work with projections lists.
            up_list = np.array(len(sat_positions) * [0, 1, 0]).reshape(-1, 3)  # default up vector per camera.
            for mean_wavelength in mean_wavelengths:
                for position_vector, lookat_vector, up_vector in zip(sat_positions,
                                                                           SAT_LOOKATS, up_list):
                    loop_sensor = at3d.sensor.perspective_projection(wavelength=mean_wavelength, fov=fov,
                                                                     x_resolution=cnx, y_resolution=cny,
                                                                     position_vector=position_vector,
                                                                     lookat_vector=lookat_vector,
                                                                     up_vector=up_vector, stokes=run_params['stokes'],
                                                                     sub_pixel_ray_args={'method': at3d.sensor.stochastic,
                                                                                         'nrays': 1})

                    sensor_dict.add_sensor('CloudCT'+str(int(mean_wavelength*1e3)), loop_sensor)
            print('Done defining CloudCT''s sensors')
            print('getting CloudCT''s measurments')
            # Next part will be the rendering, when the RTE solver is prepared (below).
            # get the measurements
            sensor_dict.get_measurements(solvers_dict, n_jobs=run_params['n_jobs'], verbose=True)
            show_results(sensor_dict)
            print('Done getting CloudCT''s measurments')



            # ----------------------------------------------------

            # if 1:
            #     plot_cloud_images(images0)
            #     a = 5

            if not run_params['cancel_noise']:
                sensor_dict_clean = copy.deepcopy(sensor_dict)

                images_clean = []
                images_clean_scatter = []
                for instrument_ind, (instrument, sensor_group) in enumerate(sensor_dict_clean.items()):
                    sensor_images = sensor_dict_clean.get_images(instrument)
                    sensor_group_list = sensor_dict_clean[instrument]['sensor_list']
                    assert len(names) == len(sensor_group_list), "len(names) does not match len(sensor_group_list)"
                    for sensor_ind, (sensor, sensor_name) in enumerate(zip(sensor_group_list, names)):
                        # add image to 'images_clean' in order to save in file
                        if (run_params['stokes'] == ['I']) or (run_params['stokes'] == 'I'):
                            curr_image = np.array([sensor_images[sensor_ind].I.data.T])
                        elif run_params['stokes'] == ['I', 'Q']:
                            curr_image = np.array(
                                [sensor_images[sensor_ind].I.data.T, sensor_images[sensor_ind].Q.data.T])
                        elif run_params['stokes'] == ['I', 'Q', 'U']:
                            curr_image = np.array(
                                [sensor_images[sensor_ind].I.data.T, sensor_images[sensor_ind].Q.data.T,
                                 sensor_images[sensor_ind].U.data.T])
                        elif run_params['stokes'] == ['I', 'Q', 'U', 'V']:
                            curr_image = np.array(
                                [sensor_images[sensor_ind].I.data.T, sensor_images[sensor_ind].Q.data.T,
                                 sensor_images[sensor_ind].U.data.T, sensor_images[sensor_ind].V.data.T])
                        images_clean.append(curr_image*np.cos(np.deg2rad(180-sun_zenith)))  # Multiply the images by cos(sun-zenith-angle)
                        copied = sensor.copy(deep=True)
                        images_clean_scatter.append(calc_image_in_scattering_plane_vectorbase(copied, curr_image*np.cos(np.deg2rad(180-sun_zenith)), sensor_name, sun_azimuth,
                                                                      sun_zenith))
                sensor_dict_clean = update_images_in_sensor_dict(images_clean, sensor_dict_clean)
                sensor_dict = add_noise_to_images_in_camera_plane(run_params, sensor_dict_clean, sun_zenith, names, cnx, cny)
            else:
                images_clean=[]

            # ----------------------------------------------------

            sensor_list = []
            images = []
            ray_mu_list = []
            ray_phi_list = []
            projection_matrices = []
            images_scatter = []
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

                    # add image to 'images' in order to save in file

                    if (run_params['stokes'] == ['I']) or (run_params['stokes'] == 'I'):
                        curr_image = np.array([sensor_images[sensor_ind].I.data.T])
                    elif run_params['stokes'] == ['I', 'Q']:
                        curr_image = np.array([sensor_images[sensor_ind].I.data.T, sensor_images[sensor_ind].Q.data.T])
                    elif run_params['stokes'] == ['I', 'Q', 'U']:
                        curr_image = np.array([sensor_images[sensor_ind].I.data.T, sensor_images[sensor_ind].Q.data.T,
                                               sensor_images[sensor_ind].U.data.T])
                    elif run_params['stokes'] == ['I', 'Q', 'U', 'V']:
                        curr_image = np.array([sensor_images[sensor_ind].I.data.T, sensor_images[sensor_ind].Q.data.T,
                                               sensor_images[sensor_ind].U.data.T, sensor_images[sensor_ind].V.data.T])
                    images.append(curr_image)
                    images_scatter.append(calc_image_in_scattering_plane_vectorbase(copied, curr_image, sensor_name, sun_azimuth,
                                                                                          sun_zenith))

            if 0:
                plot_cloud_images(images)
                a = 5

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

            cloud['images'].append(np.array(images))
            cloud['images_scatter'].append(np.array(images_scatter))
            cloud['images_clean'].append(np.array(images_clean))
            cloud['images_clean_scatter'].append(np.array(images_clean_scatter))
            cloud['mask'].append(mask4file)
            cloud['mask_morph'].append(mask_morph)
            cloud['ray_mu'].append(np.array(ray_mu_list))
            cloud['ray_phi'].append(np.array(ray_phi_list))
            cloud['cameras_pos'].append(sat_positions)
            cloud['cameras_P'].append(np.array(projection_matrices))
            cloud['grid'].append(grid)
            cloud['not_cloudbow_startind'].append(not_cloudbow_startind)
            cloud['cloudbow_sample_angles'].append(cloudbow_sample_angles)

    cloud['images'] = np.array(cloud['images'])
    cloud['images_scatter'] = np.array(cloud['images_scatter'])
    cloud['images_clean'] = np.array(cloud['images_clean'])
    cloud['images_clean_scatter'] = np.array(cloud['images_clean_scatter'])
    cloud['mask'] = np.array(cloud['mask'])
    cloud['mask_morph'] = np.array(cloud['mask_morph'])
    cloud['ray_mu'] = np.array(cloud['ray_mu'])
    cloud['ray_phi'] = np.array(cloud['ray_phi'])
    cloud['cameras_pos'] = np.array(cloud['cameras_pos'])
    cloud['cameras_P'] = np.array(cloud['cameras_P'])
    cloud['grid'] = np.array(cloud['grid'])
    cloud['not_cloudbow_startind'] = np.array(cloud['not_cloudbow_startind'])
    cloud['cloudbow_sample_angles'] = np.array(cloud['cloudbow_sample_angles'])


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

    # ------------------
    # Q:
    fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axarr = axarr.flatten()
    for ax, image in zip(axarr, images):
        image = np.squeeze(image.copy())
        im = ax.imshow(image[1, ...], cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax)
    fig.suptitle('Q', size=16, y=0.95)

    # ------------------
    # U:
    fig, axarr = plt.subplots(3, 3, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    axarr = axarr.flatten()
    for ax, image in zip(axarr, images):
        image = np.squeeze(image.copy())
        im = ax.imshow(image[2, ...], cmap='gray')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.01)
        plt.colorbar(im, cax=cax)
    fig.suptitle('U', size=16, y=0.95)

    # --------------------
    plt.show()
    # mlab.show()
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
    config_path = "/wdata/tamarsd/AT3D_research/CloudCT/configs/params_cloudct.yaml"  # Default to CloudCT config
    run_params = load_run_params(params_path=config_path)
    clouds_path = "/wdata/roironen/Data/BOMEX_256x256x100_5000CCN_50m_micro_256/clouds/cloud*.txt"

    
    #main(clouds_path, config_path)
    simple_main(run_params, clouds_path)