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
from CloudCTUtils import *
import random
import matplotlib
# matplotlib.use('TkAgg')
import itertools
from scipy.special import j0, jv
from CloudCTUtils import *

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
r_earth = 6371.0  # km
origin, xaxis, yaxis, zaxis = [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]


# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def get_uncertainties(params):
    gain_std_percents = 0  # default
    global_bias_std_percents = 0  # default
    forward_dir_addition = ''
    if params['use_cal_uncertainty']:
        forward_dir_addition = f'_cal_uncertainty_{random.randint(1, 1000)}'
        if params['use_bias']:
            global_bias_std_percents = params['use_bias']
            forward_dir_addition += f"_use_bias_{params['max_bias']}"
        if params['use_gain']:
            gain_std_percents = params['use_gain']
            forward_dir_addition +=  f"_use_gain_{params['max_gain']}"

    return gain_std_percents, global_bias_std_percents, forward_dir_addition


def setup_imager(imager_options):
    """
    TODO

    Returns:

    """
    Imager_params_Path = imager_options['Imager_params_Path']
    Imager_params = load_params(Imager_params_Path, param_type='imager_params')
    sensor_params = Imager_params['sensor']
    lens_params = Imager_params['lens']
    imager_channels = Imager_params['imager_channels']
    imager_bands = Imager_params['imager_bands']
    # csv files to use:
    DARK_NOISE_table = os.path.join(Imager_params['CSV_BASE_PATH'], sensor_params['DARK_NOISE_CSV_FILE'])
    TRANSMISSION_CSV_FILE = os.path.join(Imager_params['CSV_BASE_PATH'], lens_params['TRANSMISSION_CSV_FILE'])

    TRANSMISSION_CHANNEL_TYPE = lens_params['TRANSMISSION_CHANNEL_TYPE']

    # Define imager:

    # 1. Define sensor:
    sensor = CloudCT_Imager.SensorFPA(PIXEL_SIZE=sensor_params['PIXEL_SIZE'], FULLWELL=sensor_params['FULLWELL'],
                             CHeight=sensor_params['CHeight'],
                             CWidth=sensor_params['CWidth'], READOUT_NOISE=sensor_params['READOUT_NOISE'],
                             TEMP=sensor_params['TEMP'], BitDepth=sensor_params['BitDepth'], TYPE=sensor_params['TYPE'])

    qe = CloudCT_Imager.EFFICIENCY()  # define QE object
    for channel in imager_channels:
        QE_CSV_FILE = os.path.join(Imager_params['CSV_BASE_PATH'], channel + '.csv')
        qe.Load_EFFICIENCY_table(csv_table_path=QE_CSV_FILE, channel=channel)

    # set sensor efficiency:
    sensor.set_QE(qe)
    sensor.Load_DARK_NOISE_table(DARK_NOISE_table)

    # 2. Define lens:
    lens = CloudCT_Imager.LensSimple(FOCAL_LENGTH=lens_params['FOCAL_LENGTH'], DIAMETER=lens_params['DIAMETER'])
    # shdom.LensSimple means that the lens model is simple and without MTF considerations but still with focal and diameter.
    lens_transmission = CloudCT_Imager.EFFICIENCY()
    lens_transmission.Load_EFFICIENCY_table(csv_table_path=TRANSMISSION_CSV_FILE, channel=TRANSMISSION_CHANNEL_TYPE)
    lens.set_TRANSMISSION(lens_transmission)

    # 3. create imager:
    # set spectrum:
    scene_spectrum = CloudCT_Imager.SPECTRUM(channels=imager_channels, bands=imager_bands)
    if_valid, imager_type_to_return = scene_spectrum.is_valid_spectrum_for_imager()
    assert if_valid, "Invalide spectrum for imager usage."

    # merge quantum effciency with the defined spectrum:
    imager_type = Imager_params['imager_type']
    # TODO delete from code:
    # if imager_type == 'Polarized_sensor' or imager_type == 'Polarized_filter':
    # assert Imager_params['NUM_STOKES']<=3, "Inconsistency in imager definations."
    # else:
    ## means it is Radiance_sensor.
    # assert Imager_params['NUM_STOKES'] == 1, "Inconsistency in imager definations."
    # STOKES_WEIGHTS = Imager_params['STOKES_WEIGHTS']
    # stokes_weights = [int(i) for i in STOKES_WEIGHTS.split()]
    # assert sum(stokes_weights[1::]) == 0, "Inconsistency in imager definations."

    imager = CloudCT_Imager.Imager(sensor=sensor, lens=lens, scene_spectrum=scene_spectrum, TYPE=imager_type)

    return imager, imager_type_to_return


def setup_imagers(params, sun_zenith):
    # Imagers_nember_per_setup in the run params described how manny imagers of diffrent type in this simulation:
    # For instance, SWIR and VIS or just POLARIZED.
    Imagers_number_per_setup = len(params['Imagers'].keys())
    imagers = {}  # Evrey imager has imager id

    # If we do not use simple imager we probably use imager with a band/s. If wavelength_averaging = False - we use simple imager and the medium iprots the coresponded mie tables.
    wavelength_averaging = False
    USE_STOKES = False  # if there will be at least one imager with polarization, it will be True.
    # STOKES_WEIGHTS = '1 1 1 0' # Meanwhile it is fixed here.
    STOKES_WEIGHTS = '1 0 0 0'

    number_of_real_imagers = 0
    for imager_id, imager_options in params['Imagers'].items():
        imager, imager_type = setup_imager(imager_options)
        imager_true_indices = imager_options['true_indices']
        if len(imager_true_indices) == 0:
            # skip this imager id, it is inactive.
            Imagers_number_per_setup -= 1
            continue

        if imager_type == 'real':
            # if the imager is simple it has monochromatic chanels, than the Mie tables will be consistent.
            wavelength_averaging = True
            number_of_real_imagers += 1

        # setup other parameters from the run params:
        imager.set_solar_beam_zenith_angle(
            SZA=sun_zenith)  # update the imager with solar irradince with the solar zenith angle.
        # TODO -  use here pysat or pyEpham package to predict the orbital position of the nadir view sattelite and the SZA.

        imager.change_temperature(params['temperature']-273.15)  # convert Kelvin to Celcius
        imager.set_Imager_altitude(H=params['Rsat'])  # in km
        imager.IS_VALIDE_LENS_DIAMETER()

        if imager.IS_HAS_POLARIZATION():
            USE_STOKES = True
            STOKES_WEIGHTS = '1 1 1 0'

        imagers[imager_id] = imager

    # check if all imagers are real or simple (Must be all of the same type):
    assert (number_of_real_imagers == 0 or number_of_real_imagers == Imagers_number_per_setup), \
        "The imagers MUST all be either simple or real not a mixture"

    return imagers, USE_STOKES, STOKES_WEIGHTS, wavelength_averaging


def convertStocks_vectorbase(sensor_dict, r_sat, GSD, method = 'meridian2camera'):
    """
    TODO
    """
    
    sensor_dict_out = copy.deepcopy(sensor_dict)
    
    
    VISUALIZE = True    
    
    
    assert len(sensor_dict) == 1, "Currently doesn't soppurt more than 1 instrument"
    FIRST_SENSOR = sensor_dict.get_image(list(sensor_dict.keys())[0], 0)
    cnx, cny = FIRST_SENSOR.dims['imgdim0'],  FIRST_SENSOR.dims['imgdim1']# TODO what if the camera multiband?
    npix = cnx*cny
    nviews = len(sensor_dict.get_images(list(sensor_dict.keys())[0]))
    sensor_list = sensor_dict[list(sensor_dict.keys())[0]]['sensor_list']
    # list of <xarray.Dataset>
    one_sensor_example = sensor_list[0]
    

    stokes_converted = np.zeros([nviews, 3, npix])
    # stokes_converted size is [view,stokes,nx,ny]
    
    for instrument in sensor_dict:
        sensor_images = sensor_dict.get_images(instrument)
        sensor_list = sensor_dict[instrument]['sensor_list']
        
        
        PNCHANNELS = 1 # polarized channels
        pol_channels = ['I']
        if 'Q' in list(sensor_images[0].keys()) and 'U' in list(sensor_images[0].keys()):
            print("The images are polarized")
            PNCHANNELS = 3
            pol_channels = ['I','Q','U']
        else:
            raise Exception("Stokes convertion can't work on unpolarized images.")

        
        
                    
        for sensor_index,sensor_image in enumerate(sensor_images):
            stokes = np.dstack([sensor_image[pol_channel].data for pol_channel in pol_channels])
            sensor = sensor_list[sensor_index]
            
            #--------------------------------------------------
            #--------------------------------------------------
            # get ray directions:
            
            ray_mus  = sensor.ray_mu
            ray_phis = sensor.ray_phi
            ray_xs   = sensor.ray_x
            ray_ys   = sensor.ray_y
            ray_zs   = sensor.ray_z
        
        
            # angles to rays convertion
            PHI = ray_phis
            RAY_Z = ray_mus
            RAY_X = np.sin(np.arccos(ray_mus))*np.cos(ray_phis) 
            RAY_Y = np.sin(np.arccos(ray_mus))*np.sin(ray_phis)         
            theta_rad_mat = np.zeros_like(RAY_Z)
            #other view parameteres:
            lookat = sensor.attrs['lookat']
            position = sensor.attrs['position']
            optical_axis_direction = lookat - position
            # calc the principle ray direction
            prd = optical_axis_direction / np.linalg.norm(optical_axis_direction)   
            
            rotation_matrix = sensor.attrs['rotation_matrix'].reshape(3,3)
            
            cam_dir_x =  np.dot(rotation_matrix,xaxis)
            cam_dir_y =  np.dot(rotation_matrix,yaxis) 
            cam_dir_z =  np.dot(rotation_matrix,zaxis) 
            
            
            left_of_polarizer = cam_dir_y.copy()  #
            left_of_polarizer = left_of_polarizer / np.linalg.norm(left_of_polarizer)
            
            # calc polaizer direction at the principle ray direction (prd)
            polarizer_dir_at_prd = np.cross(left_of_polarizer, prd)  # this makes a perfect alinment of the polarizer with cameras x-axis
        
            
            # rel_ang is the angle of rotation that the scocks vector in meridian frame should be
            # transformed to get the stocks vector in the camera frame.

            zenith_dir = np.array([0, 0, 1])
            theta_rad_mat = np.zeros([cnx,cny])
            
            
            #--------------------------------------------------
            # vector calculations:
            scale_for_lookats = 0.1*ray_zs.data+1 # km
            scale_for_lookats = scale_for_lookats[np.newaxis,:]
            ray_origins = np.vstack([ray_xs.data,ray_xs.data,ray_xs.data])
            # ray_origins.shape is (3, cnx*cny)
            ray_dirs = np.vstack([RAY_X.data,RAY_Y.data,RAY_Z.data])
            # ray_dirs.shape is (3, cnx*cny)
            
            lookats_pix = ray_origins + scale_for_lookats*ray_dirs # we have a freadom to play with it i think.
            # pick 3 point on the plane:
            p1 = lookats_pix
            p2 = ray_origins
            p3 = lookats_pix + (zenith_dir*scale_for_lookats.T).T

            # These two vectors are in the plane
            v1 = p3 - p1
            v2 = p2 - p1
            # the cross product is a vector normal to the plane
            r = np.cross(v1, v2,axis=0)
            r_norm = np.linalg.norm(r, axis=0)
            
            # update polarizer_dir as the polarizer is spherical and any ray is perpendicular to its surface
            polarizer_dir = np.cross(left_of_polarizer,ray_dirs,axis=0)
            polarizer_phi = (np.arctan2(polarizer_dir[1], polarizer_dir[0]) + np.pi).astype(np.float64)
            
            # continue to meridian coordinates:
            z = ray_dirs
            r = r/np.linalg.norm(r, axis=0)
            l = np.cross(z,r, axis=0)            
            assert np.allclose(z , np.cross(r, l,axis=0)), "Bad calculation of vectors directions."
            
            # Find theta
            # fast way of dot product from - https://stackoverflow.com/questions/37670658/python-dot-product-of-each-vector-in-two-lists-of-vectors
            cos = np.einsum('ji, ji->i', polarizer_dir, l)
            cos = np.clip(cos, -1, 1)
            theta_rad = np.arccos(cos) # polarizer_dir for 0[deg] is lo direction.
            
            # Find dphi -  to dicide if the angle is theta_rad or -theta_rad.
            d = PHI.data - polarizer_phi
            dphi = -1*np.ones_like(d)
            d[d>2*np.pi] = d[d>2*np.pi] - 2*np.pi
            dphi[(d<=np.pi) * (d>=0)] = 1.0
            dphi[(d<=-np.pi) * (d>=-2*np.pi)] = 1.0
            
            theta_rad *= dphi  # it is very important to give here the sign of theta.
            theta_rad_mat = theta_rad
            theta_rad_mat[r_norm==0] = 0
            
            theta_rad_mat = theta_rad_mat.reshape([cnx,cny], order='F')
            #print(theta_rad_mat[50,50])
            # out of the "if matrix exists"
            
            cos2theta = np.cos(2*theta_rad_mat).flatten()
            sin2theta = np.sin(2*theta_rad_mat).flatten()
            zeros = np.zeros_like(cos2theta)
            ones = np.ones_like(cos2theta)
            row0 = np.vstack([ones,zeros,zeros])[:,np.newaxis,:]
            row1 = np.vstack([zeros,cos2theta,sin2theta])[:,np.newaxis,:]
            row2 = np.vstack([zeros,-sin2theta,cos2theta])[:,np.newaxis,:]
            row0 = row0.transpose([1,0,2])
            row1 = row1.transpose([1,0,2])
            row2 = row2.transpose([1,0,2])
            ROT_MAT = np.vstack([row0,row1,row2])
            # reminder:
            # shape of stokes (cnx, cny, 3)
            # shape of ROT_MAT (3,3,cnx*cny)
            vector_stokes = np.reshape(stokes,[-1,3])
            vector_stokes = vector_stokes.T
            #vector_stokes = vector_stokes[:,np.newaxis,:]
            # shape of vector_stokes (3,1,cnx*cny)
            #vector_stokes_sq = np.squeeze(vector_stokes)
            
            if(method == 'meridian2camera'):
                for index in range(npix):
                    Sconvertaed_at_pixel = np.dot(ROT_MAT[...,index] , vector_stokes[...,index])
                    # update the big matrix stock_converted:
                    # stokes_converted size is [view,stokes,nx,ny]
                    stokes_converted[sensor_index,:,index] = Sconvertaed_at_pixel
                    
            elif(method == 'camera2meridian'):
                for index in range(npix):
                    Sconvertaed_at_pixel = np.dot(np.linalg.inv(ROT_MAT[...,index]) , vector_stokes[...,index])
                    stokes_converted[sensor_index,:,index] = Sconvertaed_at_pixel

            else:
                raise Exception("Unknown method of stokes convertion.")
                
            if np.any(np.isnan(stokes_converted[sensor_index,...])):
                raise Exception("Bad rotation of the meridian/camera plane")
            
            
                
                
            # Here we want to update the images of the sensor_dict, spesificaly: I, Q, U.
            # here we still in the sensor_inxed loop.
    
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['Q'].data = stokes_converted[sensor_index,1,...].reshape([cnx,cny], order='C').flatten(order='F')
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['U'].data = stokes_converted[sensor_index,2,...].reshape([cnx,cny], order='C').flatten(order='F')

        stokes_converted = stokes_converted.reshape([nviews, 3, cnx, cny], order='C')
            
    return stokes_converted, sensor_dict_out


def create_images_list(sensor_dict,stokes_list,names):
    images = []
    for instrument_ind, (instrument, sensor_group) in enumerate(sensor_dict.items()):
        sensor_images = sensor_dict.get_images(instrument)
        sensor_group_list = sensor_dict[instrument]['sensor_list']
        assert len(names) == len(sensor_group_list), "len(names) does not match len(sensor_group_list)"
        for sensor_ind, sensor in enumerate(sensor_group_list):
            if (stokes_list == ['I']) or (stokes_list == 'I'):
                curr_image = np.array([sensor_images[sensor_ind].I.data])
            else:
                curr_image = np.stack([sensor_images[sensor_ind][pol_channel].data for pol_channel in stokes_list])
            images.append(curr_image)
    return images


def convertStocks(sensor_dict, r_sat, GSD, method='meridian2camera'):
    """
    TODO
    returns sensor_dict_out = copy.deepcopy(sensor_dict)

    """

    sensor_dict_out = copy.deepcopy(sensor_dict)

    VISUALIZE = False
    assert len(sensor_dict) == 1, "Currently doesn't soppurt more than 1 instrument"
    FIRST_SENSOR = sensor_dict.get_image(list(sensor_dict.keys())[0], 0)
    cnx, cny = FIRST_SENSOR.dims['imgdim0'], FIRST_SENSOR.dims['imgdim1']  # TODO what if the camera multiband?
    npix = cnx * cny
    nviews = len(sensor_dict.get_images(list(sensor_dict.keys())[0]))
    sensor_list = sensor_dict[list(sensor_dict.keys())[0]]['sensor_list']
    # list of <xarray.Dataset>
    one_sensor_example = sensor_list[0]

    stokes_converted = np.zeros([nviews, 3, cnx, cny])
    # stokes_converted size is [view,stokes,nx,ny]

    for instrument in sensor_dict:
        sensor_images = sensor_dict.get_images(instrument)
        sensor_list = sensor_dict[instrument]['sensor_list']

        PNCHANNELS = 1  # polarized channels
        pol_channels = ['I']
        if 'Q' in list(sensor_images[0].keys()) and 'U' in list(sensor_images[0].keys()):
            print(" The images are polarized")
            PNCHANNELS = 3
            pol_channels = ['I', 'Q', 'U']
        else:
            raise Exception("Stokes convertion can't work on unpolarized images.")

        for sensor_index, sensor_image in enumerate(sensor_images):
            stokes = np.dstack([sensor_image[pol_channel].data for pol_channel in pol_channels])
            sensor = sensor_list[sensor_index]

            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------

            # get theta filename:
            current_lookat = sensor.lookat
            current_origin = sensor.position
            current_fov = float_round(one_sensor_example.fov_deg)
            current_gsd = GSD

            lookat_str = '_'.join(str(int(np.round(x * 1000))) for x in current_lookat)
            origin_str = '_'.join(str(np.round(x, 3)) for x in current_origin)
            dir_path = os.path.join('/wdata/inbalkom/AT3D_CloudCT_shared_files', 'polar_convert_angles')
            filename = os.path.join(dir_path,
                                    'theta_pixx_' + str(cnx) + '_pixy_' + str(cny) + '_gsd_' + str(
                                        current_gsd) + '_lookat_' + lookat_str + '_origin_' + origin_str + '.pkl')

            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------
            # --------------------------------------------------

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------

            # get ray directions:

            ray_mus = sensor.ray_mu
            ray_phis = sensor.ray_phi
            ray_xs = sensor.ray_x
            ray_ys = sensor.ray_y
            ray_zs = sensor.ray_z

            # angles to rays convertion
            PHI = ray_phis
            RAY_Z = ray_mus
            RAY_X = np.sin(np.arccos(ray_mus)) * np.cos(ray_phis)
            RAY_Y = np.sin(np.arccos(ray_mus)) * np.sin(ray_phis)

            # scale_factor=2*scale
            if (VISUALIZE):
                inte_length = 200
                mlab.quiver3d(ray_xs[100], ray_ys[100], ray_zs[100], inte_length * RAY_X[100],
                              inte_length * RAY_Y[100], inte_length * RAY_Z[100], \
                              line_width=0.001, color=(0.6, 1.0, 0), opacity=0.2,
                              scale_factor=1)  # ,scale_factor=2*scale

                # at3d.sensor.show_sensors(sensor_list, scale = 50, axisWidth = 1.0, axisLenght=30, Show_Rays =  True, FullCone = True)
                mlab.orientation_axes()

            # other view parameteres:
            lookat = sensor.attrs['lookat']
            position = sensor.attrs['position']
            optical_axis_direction = lookat - position
            # calc the principle ray direction
            prd = optical_axis_direction / np.linalg.norm(optical_axis_direction)

            rotation_matrix = sensor.attrs['rotation_matrix'].reshape(3, 3)

            cam_dir_x = np.dot(rotation_matrix, xaxis)
            cam_dir_y = np.dot(rotation_matrix, yaxis)
            cam_dir_z = np.dot(rotation_matrix, zaxis)

            left_of_polarizer = cam_dir_y.copy()  #
            left_of_polarizer = left_of_polarizer / np.linalg.norm(left_of_polarizer)

            # calc polaizer direction at the principle ray direction (prd)
            polarizer_dir_at_prd = np.cross(left_of_polarizer,
                                            prd)  # this makes a perfect alinment of the polarizer with cameras x-axis

            # rel_ang is the angle of rotation that the scocks vector in meridian frame should be
            # transformed to get the stocks vector in the camera frame.

            zenith_dir = np.array([0, 0, 1])

            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    theta_rad_mat = pickle.load(f)
                    print("Theta matrix file of {} was read.".format(sensor_index))
            else:
                print('Converting {}'.format(sensor_index))
                """
                that how I want to setup the polarizer, to aligned 
                with camera xaxis at principle point direction.
                It is where polarizer_angle = 0 # ged
                """
                # the inverse of flatten - I.flatten().reshape(I.shape, order='C')
                theta_rad_mat = np.zeros([cnx, cny])
                for index, (d_x, d_y, d_z, origin_x, origin_y, origin_z, phi_dir) in enumerate(zip(RAY_X.data,
                                                                                                   RAY_Y.data,
                                                                                                   RAY_Z.data,
                                                                                                   ray_xs.data,
                                                                                                   ray_ys.data,
                                                                                                   ray_zs.data,
                                                                                                   PHI.data)):
                    scale_for_lookat = 0.1 * origin_z + 1  # km
                    ray_origin = np.array([origin_x, origin_y, origin_z])
                    ray_dir = np.array([d_x, d_y, d_z])

                    lookat_pix = ray_origin + scale_for_lookat * ray_dir  # we have a freadom to play with it i think.
                    # pick 3 point on the plane:
                    p1 = lookat_pix
                    p2 = ray_origin
                    p3 = lookat_pix + zenith_dir * scale_for_lookat

                    # These two vectors are in the plane
                    v1 = p3 - p1
                    v2 = p2 - p1
                    # the cross product is a vector normal to the plane
                    r = np.cross(v1, v2)
                    if (np.all(r == 0)):
                        # V1 parallel to V2 -> meridian aligned with camera reference.
                        theta_rad = 0
                        # update polarizer_dir as the polarizer is spherical and any ray is perpendicular to its surface
                        polarizer_dir = np.cross(left_of_polarizer, ray_dir)
                        polarizer_phi = (np.arctan2(polarizer_dir[1], polarizer_dir[0]) + np.pi).astype(np.float64)

                    else:
                        # continue to meridian coordinates:
                        z = ray_dir
                        r = r / np.linalg.norm(r)
                        l = np.cross(z, r)
                        assert np.allclose(z, np.cross(r, l)), "Bad calculation of vectors directions."
                        # find theta
                        # update polarizer_dir as the polarizer is spherical and any ray is perpendicular to its surface
                        polarizer_dir = np.cross(left_of_polarizer, ray_dir)
                        polarizer_phi = (np.arctan2(polarizer_dir[1], polarizer_dir[0]) + np.pi).astype(np.float64)

                        cos = np.dot(polarizer_dir, l)
                        cos = np.clip(cos, -1, 1)
                        # I had a problem here, the cos can be something like -1.000000000000002 and this is not valid number for acos
                        theta_rad = np.arccos(cos)  # polarizer_dir for 0[deg] is lo direction.

                    """
                            See page 51 in the book Scattering, Absorption, and Emission of Light by Small Particles.
                            https://onlinelibrary.wiley.com/doi/pdf/10.1002/9783527618156.

                            The rotation angle is theta_rad.

                    """
                    # Find dphi -  to dicide if the angle is theta_rad or -theta_rad.
                    d = phi_dir - polarizer_phi
                    if (d > 2 * np.pi):
                        d = d - 2 * np.pi

                    if ((d <= np.pi) and (d >= 0)):
                        dphi = 1.0

                    elif ((d <= -np.pi) and (d >= -2 * np.pi)):
                        dphi = 1.0

                    else:
                        dphi = -1.0

                    theta_rad *= dphi  # it is very important to give here the sign of theta.

                    xpix_ind, ypix_ind = np.unravel_index(index, (cnx, cny), order='F')

                    theta_rad_mat[xpix_ind, ypix_ind] = theta_rad

                # save theta_rad_mat for future runs
                if not os.path.exists(dir_path):
                    # Create a new directory because it does not exist
                    os.makedirs(dir_path)
                    print("The directory 'polar_convert_angles' was created.")
                with open(filename, 'wb') as outfile:
                    pickle.dump(theta_rad_mat, outfile, protocol=pickle.HIGHEST_PROTOCOL)
                    print("Theta matrix file was saved.")

            # out of the "if matrix exists"
            for index, (d_x, d_y, d_z, origin_x, origin_y, origin_z, phi_dir, theta_rad) in enumerate(zip(RAY_X.data,
                                                                                                          RAY_Y.data,
                                                                                                          RAY_Z.data,
                                                                                                          ray_xs.data,
                                                                                                          ray_ys.data,
                                                                                                          ray_zs.data,
                                                                                                          PHI.data,
                                                                                                          theta_rad_mat.flatten())):
                R_theta = np.array([[1, 0, 0, 0], [0, np.cos(2 * theta_rad), np.sin(2 * theta_rad), 0],
                                    [0, -np.sin(2 * theta_rad), np.cos(2 * theta_rad), 0], [0, 0, 0, 1]])

                # find 2d indexes:
                xpix_ind, ypix_ind = np.unravel_index(index, (cnx, cny))
                # S # this pixel stokes vector:
                stokes_I = stokes[xpix_ind, ypix_ind, 0]
                stokes_Q = stokes[xpix_ind, ypix_ind, 1]
                stokes_U = stokes[xpix_ind, ypix_ind, 2]

                S = np.vstack([stokes_I,
                               stokes_Q,
                               stokes_U,
                               0])
                # stocks vector

                S_at_pixel = S
                # now S has 3 elements.

                if (method == 'meridian2camera'):

                    Sconvertaed_at_pixel = np.dot(R_theta,
                                                  S_at_pixel)  # transfer the reference frame from meridian plane to camera plane

                elif (method == 'camera2meridian'):
                    Sconvertaed_at_pixel = np.dot(np.linalg.inv(R_theta),
                                                  S_at_pixel)  # inverse transfer the reference frame from meridian plane to camera plane

                else:
                    raise Exception("Unknown method of stokes convertion.")

                Sconvertaed_at_pixel = Sconvertaed_at_pixel[:-1]  # get rid of V=0

                if (any(np.isnan(elem) for elem in Sconvertaed_at_pixel)):
                    print('-------------------------------------')
                    print(ray_dir)
                    print(theta_rad)
                    print(R_theta)
                    print('.....')
                    print(Sconvertaed_at_pixel)
                    raise Exception("Bad rotation of the meridian/camera plane")

                # update the big matrix stock_converted:
                # stokes_converted size is [view,stokes,nx,ny]
                stokes_converted[sensor_index, :, xpix_ind, ypix_ind] = \
                    np.squeeze(Sconvertaed_at_pixel)

            # Here we want to update the images of the sensor_dict, spesificaly: I, Q, U.
            # here we still in the sensor_inxed loop.

            sensor_dict_out[instrument]['sensor_list'][sensor_index]['Q'].data = stokes_converted[
                sensor_index, 1, ...].flatten(order='F')
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['U'].data = stokes_converted[
                sensor_index, 2, ...].flatten(order='F')

    return stokes_converted, sensor_dict_out


def imitate_measurements_with_polarizer_at(S=None, polarizer_orientation_deg=0):
    """
    Imitate measurements at polarizer at different linear polarizer orientations relative to 0[deg] which is the cameas x axis.
    The Stock representation MUST be in the camera frame.
    S - list or image of Stokes representation.
    polarizer_orientation_deg - float of list of floats that has the polarization angle with respect to polarizer at zero which is aligned
    with cameras X-axis.
    """

    # stocks_meridian can be np.array of size (num_stokes, cnx, cny, channels) or (num_stokes, cnx, cny)
    # or it may be a list of above entity.
    if isinstance(S, list):
        num_stokes = S[0].shape[0]

        if (S[0].ndim == 4):
            N_channels = S[0].shape[3]
        else:
            N_channels = 1

        N_views = len(S)
        S = np.array(S)


    else:
        num_stokes = S.shape[0]
        if (S.ndim == 4):
            N_channels = S.shape[3]
        else:
            N_channels = 1

        N_views = 1
        S = S[np.newaxis, ...]

    if (N_channels == 1):
        S = S[..., np.newaxis]

    cnx, cny = S.shape[2:4]

    # which polarization angle to use?
    if (isinstance(polarizer_orientation_deg, list)):
        N_polar_angles = len(polarizer_orientation_deg)
        measurements = np.zeros([N_polar_angles, N_views, cnx, cny, N_channels])
    else:
        polarizer_orientation_deg = [polarizer_orientation_deg]
        N_polar_angles = 1
        measurements = np.zeros([N_views, cnx, cny, N_channels])

    M = np.zeros([N_polar_angles, 3])
    for angle_index, polarizer_angle in enumerate(polarizer_orientation_deg):
        M[angle_index, ...] = np.array(
            [1, np.cos(2 * np.deg2rad(polarizer_angle)), np.sin(2 * np.deg2rad(polarizer_angle))])

    # here we have the matrix M which calculate intensities from (I, Q, U) elements.

    for wavelength_index in range(N_channels):

        S_per_channel = S[..., wavelength_index]

        for view_index in range(N_views):

            stokes_I = np.squeeze(S_per_channel[view_index, 0, ...]).copy()
            stokes_Q = np.squeeze(S_per_channel[view_index, 1, ...]).copy()
            if (num_stokes == 3):
                stokes_U = np.squeeze(S_per_channel[view_index, 2, ...]).copy()
            else:
                stokes_U = np.zeros_like(stokes_I)

            if (num_stokes == 4):
                raise Exception("I don't know what to do with the circular polarization part yet.")
                stokes_V = np.squeeze(S_per_channel[view_index, 3, ...]).copy()
            else:
                stokes_V = np.zeros_like(stokes_I)

            Sfull = np.vstack([stokes_I.flatten(),
                               stokes_Q.flatten(),
                               stokes_U.flatten()])
            # stokes_V.flatten()]) # full stocks vector

            g = 0.5 * np.dot(M, Sfull)  # size of g is [N_polar_angles x (cnx*cny) ]
            measurements[:, view_index, ..., wavelength_index] = g.reshape([N_polar_angles, cnx, cny], order='C')

    Intensities = []
    measurements = np.split(measurements, N_polar_angles, axis=0)
    for angle_index, polarizer_angle in enumerate(polarizer_orientation_deg):
        I = np.split(np.squeeze(measurements[angle_index]), N_views, axis=0)
        A = []
        for i in range(N_views):
            A.append(np.squeeze(I[i]))
        Intensities.append(A)

    if (N_polar_angles == 1):

        return Intensities[0], M
    else:

        return Intensities, M


def retrieve_stokes_from_measurments(measurements, M):
    """
    Retrieve stokes representation form at least 3 different (polaizer angle) measurements.

    """
    N_polar_angles = M.shape[0]
    assert N_polar_angles == len(measurements), "measurements do not consistent with the matrix M."
    assert N_polar_angles > 3, "At leas 3 measurements are needed."
    num_stokes = 3

    if (isinstance(measurements, list)):
        N_views = len(measurements[0])
        cnx, cny = measurements[0][0].shape[0:2]
        if (measurements[0][0].ndim == 3):
            N_channels = measurements[0][0].shape[2]
        else:
            N_channels = 1
    else:
        N_views = len(measurements)
        measurements = [measurements]
        cnx, cny = measurements[0].shape[0:2]
        if (measurements[0].ndim == 3):
            N_channels = measurements[0].shape[2]
        else:
            N_channels = 1

    Retrieved = []
    for view_index in range(N_views):
        intensites_per_view = [measurements[i][view_index].flatten() for i in range(N_polar_angles)]
        intensites_per_view = np.stack(intensites_per_view)

        S = 2 * np.dot(np.linalg.pinv(M), intensites_per_view)
        S = S.reshape([num_stokes, cnx, cny, -1], order='C')
        Retrieved.append(S)

    return Retrieved


def update_images_in_sensor_dict(images_per_sensor, sensor_dict):
    sensor_dict_out = copy.deepcopy(sensor_dict)

    assert len(sensor_dict) == 1, "Currently doesn't soppurt more than 1 instrument"
    for instrument in sensor_dict_out:
        sensor_list = sensor_dict_out[instrument]['sensor_list']
        assert len(sensor_list) == len(images_per_sensor), "len(sensor_list) does not match len(images_per_sensor)"
        for sensor_index, sensor_image in enumerate(images_per_sensor):
            sensor = sensor_list[sensor_index]
            curr_image = np.squeeze(sensor_image)
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['I'].data = curr_image[0].flatten(order='F')
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['Q'].data = curr_image[1].flatten(order='F')
            sensor_dict_out[instrument]['sensor_list'][sensor_index]['U'].data = curr_image[2].flatten(order='F')
    return sensor_dict_out


def add_noise_to_images_in_camera_plane(run_params, sensor_dict, sun_zenith, sat_names, cnx, cny):
    num_stokes = len(run_params['stokes'])
    N_channels = len(run_params['wavelengths'])
    Rsat = run_params['Rsat']
    GSD = run_params['GSD']
    cancel_noise = run_params['cancel_noise']
    radiances_per_imager_meridian_frame = np.array(create_images_list(sensor_dict, run_params['stokes'], sat_names))
    if (num_stokes >= 3):
        imagers, use_stokes, stokes_weights, wavelength_averaging = setup_imagers(run_params, sun_zenith)
        gain_std_percents, global_bias_std_percents, forward_dir_uncertainty_addition = get_uncertainties(
            run_params['uncertainty_options'])
        for imager_id, imager in imagers.items():
            # Update the resolution of each Imager with respect to this simulation pixels number [nx,ny]
            # (from run params + view tuning). In addition, we update Imager's FOV.
            imager.update_sensor_size_with_number_of_pixels(cnx, cny)
            imager.set_gain_uncertainty(gain_std_percents)
            imager.set_bias_uncertainty(global_bias_std_percents)
        # setup imagers list. right now applies for only one imager/instrument type.
        imagers_list = [copy.deepcopy(imagers['imager_id_0']) for _ in np.arange(len(sat_names))]

        radiances_per_imager_cam_frame, sensor_dict_camera_frame = convertStocks_vectorbase(sensor_dict, Rsat, GSD, method='meridian2camera')

        # Like sony polarized sensor.
        polarizer_orientations_deg = [0, 45, 90, 135]  # imitate lucid camera with filters to 0°, 45°, 90° and 135°.
        # TODO - add unsertainties for the angles for each pixel?
        Intensities_before_noise, M = imitate_measurements_with_polarizer_at(list(radiances_per_imager_cam_frame),
                                                                polarizer_orientations_deg)
        # Matrix M calculates intensities from (I, Q, U) elements.
        # It will be used to convert back to (I, Q, U) elements.

        # -------------------------------------------------------------
        # --------------- apply noise here: ---------------------------
        # The noise is added here during the conversion between
        # normalized radiance to grayscale.
        # -------------------------------------------------------------
        # old issue - source_imager.adjust_exposure_time(Intensities) # consider this since it is important to not
        # be in saturation or in low snr levels.

        # len(Intensities) - number of polarization angles.
        # len(Intensities[0]) - number of views per this imager per polarization angle of the polarizer.
        N_polar_angles = len(Intensities_before_noise)
        N_views = len(Intensities_before_noise[0])
        assert N_views == len(
            sat_names), "Something went wrong in the passage of the simulated stokes vector through simulated polarizers."

        """
        Remainder - here we per imager loop.
        The loop over all imagers channels is inside the conversion method.
        We should do the loops over the polarization angles and different views.
        The method convert_radiance_to_graylevel does the following:
        1. converts photons to electrons.
        2. add global bias (uncertainty) to the signal in electrons level.
        3. add noises: photonic and camera.
        4. add gain uncertainty for each pixel.
        5. convert to grayscales (assume linear responce).
        6. Quantize and clip in the relevant digital range [0,2^bits].

        """

        GRAY_SCALE_IMAGES = np.zeros([N_polar_angles, N_views, cnx, cny, N_channels])
        RADIANCE_IMAGES = np.zeros([N_polar_angles, N_views, cnx, cny, N_channels])

        # loop over angles of the simulated polarizer angles:
        for pol_ang_index in range(N_polar_angles):
            # loop over all views
            for sat_id, sat_name in enumerate(sat_names):
                # note that sat_id it is not the number (i) of sat(i). If for instance
                # sat_names = sat4, sat7 the pairs are (sat_id = 0 sat_name = sat4), (sat_id = 1 sat_name = sat7)

                # get the right imager from the list of setup imagers per this imager id:
                this_sat_imager = imagers_list[sat_id]
                # update each imager individualty:
                this_sat_imager.adjust_exposure_time(Intensities_before_noise, sat_id)

                image_per_imager_per_sat_per_pol_angle, radiance_to_graylevel_scale = \
                    this_sat_imager.convert_radiance_to_graylevel(Intensities_before_noise[pol_ang_index][sat_id], cancel_noise = cancel_noise)
                GRAY_SCALE_IMAGES[pol_ang_index,sat_id,...] = image_per_imager_per_sat_per_pol_angle

                radiance_to_electrons_scale = radiance_to_graylevel_scale/this_sat_imager.electrons2grayscale_factor

                # image_per_imager_per_sat.shape = (nx,ny,channels)
                # radiance_to_graylevel_scale.shape = (channels,)
                # Back to normalized radiances but here it is with the added noise whith scales to radiance
                radiance_per_imager_per_sat_per_pol_angle = image_per_imager_per_sat_per_pol_angle * (1/radiance_to_graylevel_scale)
                electrons_per_sat_per_band_per_pol_angle = radiance_per_imager_per_sat_per_pol_angle*radiance_to_electrons_scale
                # TODO - Ensure the feasability of the obove step with Yoav.
                RADIANCE_IMAGES[pol_ang_index,sat_id,...] = radiance_per_imager_per_sat_per_pol_angle

                a = np.squeeze(radiance_per_imager_per_sat_per_pol_angle) - Intensities_before_noise[pol_ang_index][sat_id]
                b = a/Intensities_before_noise[pol_ang_index][sat_id]
                NOISE_AMPLITURE_RATIO = 100*b.max()

        Intensities_after_noise = []
        I_list = np.split(RADIANCE_IMAGES, N_polar_angles, axis=0)
        for pol_ang_index in range(N_polar_angles):
            I = np.split(np.squeeze(I_list[pol_ang_index]), N_views, axis=0)
            A = []
            for i in range(N_views):
                A.append(np.squeeze(I[i]))
            Intensities_after_noise.append(A)
        # draw_scatter_plot(np.swapaxes(np.array(Intensities_before_noise),0,1),
        #                   np.swapaxes(np.array(Intensities_after_noise),0,1),
        #                   ['pol 0', 'pol 45', 'pol 90', 'pol 135'])
        # Again:
        # len(Intensities) - number of polarization angles.
        # len(Intensities[0]) - number of views per this imager per polarization angle of the polarizer.

        if N_polar_angles > 3:
            # Here at least 3 measurements are needed
            # Relevant for cases:
            # 1. imager_type == 'Polarized_sensor'.
            # 2. 3 or more imagers per satellite with imager_type == 'Polarized_filter'.
            radiances_per_imager_retreived_camera_frame = retrieve_stokes_from_measurments(Intensities_after_noise, M)
            # draw_scatter_plot(radiances_per_imager_cam_frame, np.squeeze(np.array(radiances_per_imager_retreived_camera_frame)),
            #                   ['I cam', 'Q cam', 'U cam'])
            # insert noisy images into sensor dict:
            sensor_dict_noisy_camera_frame = update_images_in_sensor_dict(
                radiances_per_imager_retreived_camera_frame, sensor_dict_camera_frame)
            radiances_per_imager_back_meridian_frame, sensor_dict_noisy_back_meridian_frame = convertStocks_vectorbase(
                sensor_dict_noisy_camera_frame, Rsat, GSD, method='camera2meridian')

            # shape of radiances_per_imager_back_meridian_frame[i] is
            # the same as of radiances_per_imager_retreived_camera_frame,
            # it is (num_stokes, cnx, cny, channels)
            # visualize_Stocks(radiances_per_imager_back_meridian_frame,\
            # sun_zenith,source_imager_wavelengths,projections.names,add_Ip = False, add_dolp = True, add_aolp = True, add2title = 'with noise - Meridian')
        # draw_scatter_plot(radiances_per_imager_meridian_frame, radiances_per_imager_back_meridian_frame, ['I mer', 'Q mer', 'U mer'])
    else: # to if(num_stokes >= 3):
        # TODO
        NotImplementedError()

    return sensor_dict_noisy_back_meridian_frame


def add_airmspi_noise(sensor_dict, stokes_list, names):
    """
        AirMSPI noise modeled accroding to:
            [1] Van Harten, G., Diner, D.J., Daugherty, B.J., Rheingans, B.E., Bull, M.A., Seidel, F.C.,
                Chipman, R.A., Cairns, B., Wasilewski, A.P. and Knobelspiesse, K.D., 2018.
                Calibration and validation of airborne multiangle spectropolarimetric imager (AirMSPI) polarization
                measurements. Applied optics, 57(16), pp.4499-4513.
            [2] Diner, D.J., Davis, A., Hancock, B., Geier, S., Rheingans, B., Jovanovic, V., Bull, M.,
                Rider, D.M., Chipman, R.A., Mahler, A.B. and McClain, S.C., 2010.
                First results from a dual photoelastic-modulator-based polarimetric camera.
                Applied optics, 49(15), pp.2929-2946.

        Apply the AirMSPI poisson noise model according to the modulation and de-modulation matrices.

        Parameters
        ----------
        measurements: shdom.Measurements
            input clean measurements

        Returns
        -------
        images: list of images
            A list of images (multiview camera)
        uncertainties: list of image pixel uncertainties
           A list of uncertainty images (multiview camera)

        Notes
        -----
        Non-polarized bands are not implemented.
    """
    # set airmspi parameters:
    full_well = 200000
    # Table 5 of [1]
    bandwidths = [45, 46, 47]
    optical_throughput = [0.516, 0.605, 0.602]
    quantum_efficiencies = [0.4, 0.35, 0.13]
    polarized_bands = [0.47, 0.66, 0.862]

    num_subframes = 23
    p = np.linspace(0.0, 1.0, num_subframes + 1)
    p1 = p[0:-1]
    p2 = p[1:]
    x = 0.5 * (p1 + p2 - 1)
    delta0_list = [4.472, 3.081, 2.284]
    r = 0.0
    eta = 0.009

    p, correlation, w, reflectance_to_electrons = dict(), dict(), dict(), dict()
    for wavelength, delta0, ot, qe, bw in zip(polarized_bands, delta0_list, optical_throughput,
                                              quantum_efficiencies, bandwidths):
        # Define z'(x_n) (Eq. 8in [1])
        z_idx = np.pi * x != eta
        z = np.full_like(x, r)
        z[z_idx] = -2 * delta0 * np.sin(np.pi * x - eta)[z_idx] * np.sqrt(
            1 + r ** 2 / np.tan(np.pi * x - eta)[z_idx])

        # Define s_n (Eq. 9 in [1])
        s = np.ones(shape=(num_subframes))
        s_idx = z_idx if r == 0 else np.ones_like(x, dtype=np.bool)
        s[s_idx] = (np.tan(np.pi * x - eta) ** 2 - r)[s_idx] / (np.tan(np.pi * x - eta) ** 2 + r)[s_idx]

        # Define F(x_n) (Eq. 7 in [1])
        f = j0(z) + (1 / 3) * (np.pi * (p2 - p1) / 2) ** 2 * delta0 ** 2 * (1 - r ** 2) * (
                s * jv(2, z) - np.cos(2 * (np.pi * x - eta)) * j0(z))

        # P modulation matrix for I, Q, U with and idealized modulator (without the linear correction factor)
        # Eq. 15 of [1]
        pq = np.vstack((np.ones_like(x), f, np.zeros_like(x))).T
        pu = np.vstack((np.ones_like(x), np.zeros_like(x), f)).T
        p[wavelength] = np.vstack((pq, pu))
        correlation[wavelength] = np.matmul(p[wavelength].T, p[wavelength])

        # W demodulation matrix (Eq. 16 of [1])
        w[wavelength] = np.linalg.pinv(p[wavelength])

        # Transform rho into S (Eq. (24) of [1])
        reflectance_to_electrons[wavelength] = (1.408 * 10 ** 18 * ot * qe * bw) / \
                                                    ((1000 * wavelength) ** 4 * (
                                                        np.exp(2489.7 / (1000 * wavelength))) - 1)

    # ---------------------------------------------------------------------------------------------------------------
    # function start:
    clean_images_list = create_images_list(sensor_dict, stokes_list, names)
    instruments = list(sensor_dict.keys())
    num_sats_per_instrument = len(names)

    images = []
    uncertainties = []
    for image_ind, image in enumerate(clean_images_list):
        instrument_ind = int(image_ind/num_sats_per_instrument)
        instrument = instruments[instrument_ind]
        wavelength = float(instrument)/1000

        if len(stokes_list) == 3:
            if wavelength not in polarized_bands:
                raise AttributeError('wavelength {} is not in AirMSPI polarized channels ({})'.format(
                    wavelength, polarized_bands)
                )

            # Reflectance at 0, 45 degrees concatenated (total of 46 subframe measurements)
            reflectance = np.matmul(p[wavelength], np.rollaxis(image, 1))

            # Electrons from reflectance
            electrons = reflectance_to_electrons[wavelength] * reflectance

            # Adjust gain induced by exposure, gain, lens size etc to make maximum signal reach a max well
            gain = full_well / electrons.max()
            electrons = np.round(electrons * gain)

            # Apply Poisson noise
            electrons = np.random.poisson(electrons)

            # Compute the Poisson induced uncertainty
            uncertainty = np.sqrt(electrons / (reflectance_to_electrons[wavelength] * gain))
            correlated_uncertainty = np.dot(
                p[wavelength].T,
                np.rollaxis(p[wavelength][None, :, None] / uncertainty[..., None], -1)
            )

            # Back to I, Q, U using W demodulation matrix
            noisy_image = np.rollaxis(np.matmul(w[wavelength], electrons), 1) / \
                          (reflectance_to_electrons[wavelength] * gain)

        else:
            raise NotImplementedError
        uncertainties.append(correlated_uncertainty)
        images.append(noisy_image)
    sensor_dict_noisy = update_images_in_sensor_dict(images, sensor_dict)
    return sensor_dict_noisy, uncertainties


def main():
    file_name = 'Radiances_to_calibrate_stokes_conversions_1.nc'
    sensor_dict, solver_dict, rte_grid = at3d.util.load_forward_model(file_name)
    method = 'meridian2camera'
    fontsize = 16

    r_sat = 500  # km
    GSD = 0.02
    sensor_dict_camera = convertStocks(sensor_dict, r_sat, GSD, method)
    # stokes_converted_back = convertStocks(sensor_dict, r_sat, method)

    # see images:
    for instrument in sensor_dict:
        meridian_sensor_images = sensor_dict.get_images(instrument)
        camera_sensor_images = sensor_dict_camera.get_images(instrument)

        sensor_list = sensor_dict[instrument]['sensor_list']

        PNCHANNELS = 1  # polarized channels
        pol_channels = ['I']
        if 'Q' in list(meridian_sensor_images[0].keys()) and 'U' in list(meridian_sensor_images[0].keys()):
            print("The images are polarized")
            PNCHANNELS = 3
            pol_channels = ['I', 'Q', 'U']

        nrows = 2
        LN = len(meridian_sensor_images)
        if LN % nrows == 0:
            ncols = int(LN / nrows)
        else:
            if ((LN / nrows) > int(LN / nrows)):
                ncols = int(LN / nrows) + 1
            else:
                ncols = int(LN / nrows)

                # ------------------------------
        fontsize = 16

        for sensor_index, (meridian_sensor_image, camera_sensor_image) in enumerate(
                zip(meridian_sensor_images, camera_sensor_images)):
            stokes_maridian = np.dstack([meridian_sensor_image[pol_channel].data for pol_channel in pol_channels])
            stokes_camera = np.dstack([camera_sensor_image[pol_channel].data for pol_channel in pol_channels])

            fig = plt.figure(figsize=(20, 10))
            ii = 1
            for pol_index, pol_channel in enumerate(pol_channels):
                ax = fig.add_subplot(4, 3, ii)

                img_meridian = stokes_maridian[..., pol_index]
                img_camera = stokes_camera[..., pol_index]

                min_ = min(img_meridian.min(), img_camera.min())
                max_ = max(img_meridian.max(), img_camera.max())

                im = ax.imshow(img_meridian, cmap='jet', vmin=min_, vmax=max_)
                title = "meridian {} {}".format(sensor_index, pol_channel)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

                # -------camera----------------------------
                ax = fig.add_subplot(4, 3, ii + 3)

                im = ax.imshow(img_camera, cmap='jet', vmin=min_, vmax=max_)
                title = "camera {} {}".format(sensor_index, pol_channel)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

                # -------dolp----------------------------
                ax = fig.add_subplot(4, 3, ii + 3)

                im = ax.imshow(img_camera, cmap='jet', vmin=min_, vmax=max_)
                title = "camera {} {}".format(sensor_index, pol_channel)
                ax.set_title(title, fontsize=fontsize)

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.01)
                plt.colorbar(im, cax=cax)
                ax.set_axis_off()

                ii = ii + 1
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------

            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------
            # ------------------------------------------------------------

    plt.show()
    print('done')


if __name__ == '__main__':
    main()