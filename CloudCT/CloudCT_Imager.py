import os
import sys
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import at3d

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import AxesGrid
import time
import glob
import json

import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline
from collections import OrderedDict
from math import log, exp, tan, atan, acos, pi, ceil, atan2
from itertools import chain
from collections import OrderedDict
# import xarray as xr
import copy

# importing functools for reduce()
import functools
# importing operator for operator functions
import operator

# -------------------------------------------------------------------------------
# ----------------------CONSTANTS------------------------------------------
# -------------------------------------------------------------------------------
integration_Dlambda = 1
# h is the Planck's constant, c the speed of light,
h = 6.62607004e-34  # J*s is the Planck constant
c = 3.0e8  # m/s speed of litght
k = 1.38064852e-23  # J/K is the Boltzmann constant
r_earth = 6371.0e3  # m
M_earth = 5.98e24  # kg
G = 6.673e-11  # gravit in N m2/kg2

def plank(llambda, T=5800):
    # https://en.wikipedia.org/wiki/Planck%27s_law
    a = 2.0 * h * (c ** 2)
    b = (h * c) / (llambda * k * T)
    spectral_radiance = a / ((llambda ** 5) * (np.exp(b) - 1.0))
    return spectral_radiance

def SatSpeed(orbit=100e3):
    """
    Determine the speed, (later, acceleration and orbital period) of the satellite.
    """
    Rsat = orbit  # in meters.
    R = r_earth + Rsat  # the orbital radius
    # The orbital speed of the satellite using the following equation:
    # v = SQRT [ (G*M_earth ) / R ]
    V = np.sqrt((G * M_earth) / R)  # units of [m/sec]
    return V
"""
This packege helps to define the imager to be simulated.
First, you need to define the sensor with its parameters. These parameters should be in the specs or given by by the companies.
Seconed, you neet to define the lens with its parameters. These parameters should be in the specs or given by by the companies.

Both the lens and the sensor can use efficiencies tables in csv format.
Sensor efficiency is QE - quantum effciency.
Lens efficienc is its TRANSMISSION.

After the defination of Sensor and Lens objects, you need to define an Imager object.
Imager object takes the following inputs:
* Sensor
* Lens
* scene spectrum -  which define only the range e.g. scene_spectrum=[400,1700].

The imager object can be set with uniform (simple model) QE and lens transtission. It is done for example by:
imager.assume_sensor_QE(45)
imager.assume_lens_UNITY_TRANSMISSION()
The important step here is to call the update method to apdate the imager with the new parameters e.g. imager.update().


Than, the Imager object can be used.
The recomended usage is as following (1-2 are needed for the rendering, the rest are aslo needed for radiometric manipulations):
1. Set Imager altitude (e.g imager.set_Imager_altitude(H=500) # in km).
2. Calculate the Imager footprint at nadir (e.g. imager.get_footprints_at_nadir()). The footprint is needed for the rendering simulation.
3. Get maximum exposur time. It is the exposure in which the motion blur would be less than 1 pixel (e.g. imager.max_exposure_time).
The maximum exposur time derivied only from the pixel footprint and orbital speed considerations. Currently the footprint is calculated for the nadir view.


TOADD...
Notes:
1. If the lens is undefined, use set_pixel_footprint(val) after set_Imager_altitude(H) to requaire 
certaine footprint, then, calcalate which folac length and diameter of the lens you need.

2.
"""
def float_round(x):
    """Round a float or np.float32 to a 3 digits float"""
    if type(x) == np.float32:
        x = x.item()
    return round(x, 3)


def list_flatten(l, a=None):
    # check a
    if a is None:
        # initialize with empty list
        a = []

    for i in l:
        if isinstance(i, list):
            list_flatten(i, a)
        else:
            a.append(i)
    return a


ColoR = ['b', 'g', 'r', 'm', 'c', 'k', 'y', 'pink', 'deeppink', 'indigo', 'khaki', 'salmon']
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

"""
pixel pitch - microns
lens diameter - mm
focal length - mm
fov - rad
camera footprint - km
pixel footprint - km
orbital speed - km/sec
max esposure time - micron ses

"""


class EFFICIENCY(object):
    """
    EFFICIENCY is a class that represent how the part is transmit light power. It can be related to QE of a sensor or Lens transmission.
    It is an OrderedDict: Keys are channels, values are effciencies. Each efficiency is pandas table.
    """

    def __init__(self, EFFICIENCYs=None, CHANNELs=None):
        """
        Parameters:

        EFFICIENCYs - list of effciencys - the effciency in range [0,1] relative to the spectrum.
        effciency - It is pandas table, column #1 is wavelenths in nm, column #2 is the effciency in range [0,1].

        CHANNELs - list of channels e.g. ['r','g','b'].

        QEs -list of quantum effciency tables, quantum effciency measures the prob-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the quantum effciency in range [0,1].


        """
        self._EFFICIENCYs = OrderedDict()
        if (EFFICIENCYs is not None):
            if CHANNELs is not None:
                if isinstance(CHANNELs, list):
                    assert len(CHANNELs) == len(
                        EFFICIENCYs), "the number of channels must be as the number of effciencies."
                    for channel_index, channel in enumerate(CHANNELs):
                        assert isinstance(EFFICIENCYs[channel_index],
                                          pd.DataFrame), "Here the effciency must by pandas type."
                        self._EFFICIENCYs[channel] = EFFICIENCYs[channel_index]

                else:  # there is only one channel - so they get default names 0,1,2,..
                    assert isinstance(EFFICIENCYs, pd.DataFrame), "Here the effciency must by pandas type."
                    self._EFFICIENCYs[CHANNELs] = EFFICIENCYs


            else:  # given channels are None
                if isinstance(EFFICIENCYs, pd.DataFrame):
                    self._EFFICIENCYs[0] = EFFICIENCYs
                else:
                    for channel_index, band in enumerate(bands):
                        assert isinstance(EFFICIENCYs[channel_index],
                                          pd.DataFrame), "Here the effciency must by pandas type."
                        self._EFFICIENCYs[channel_index] = EFFICIENCYs[channel_index]

    def add_EFFICIENCY(self, EFFICIENCY, channel=None):
        assert isinstance(EFFICIENCY, pd.DataFrame), "Here the effciency must by pandas type."
        if channel is not None:
            self._EFFICIENCYs[channel] = EFFICIENCY
        else:
            channels = self._EFFICIENCYs.keys()
            self._EFFICIENCYs[channels[-1] + 1] = EFFICIENCY

    def multiply_efficiency_by_scalar(self, scalar):
        """
        Multiply efficiency by just a scalar.
        Inputs:
            scalar - non negative scalar
        """
        assert scalar > 0, "Must be positive"
        EFFICIENCY_CHANNELS = self.channels
        for index, channel in enumerate(EFFICIENCY_CHANNELS):
            self._EFFICIENCYs[channel]['<Efficiency>'] *= scalar

    def Export_EFFICIENCY_table(self, channel, csv_table_path=None):
        """
        save EFFICIENCY at a channel to scv file "csv_table_path".
        Notes
        -----
        CSV format should be as follows:
        <wavelength [nm]>	<Efficiency>
        lambda1	         efficiency1
        .                    .
        .                    .
        """
        assert csv_table_path is not None, "You must provide EFFICIENCY table file"
        E = self._EFFICIENCYs[channel]
        E.to_csv(csv_table_path, index=False)  # use index=False to avoide indexindg od raws.

    def Load_EFFICIENCY_table(self, csv_table_path=None, channel='red'):
        """
        Load EFFICIENCY from scv file "csv_table_path".
        For example lens transmission or QE - quantum effciency, which measures the prob-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the quantum effciency in range [0,1].

        Notes
        -----
        CSV format should be as follows:
        <wavelength [nm]>	<Efficiency>
        lambda1	         efficiency1
        .                    .
        .                    .
        """
        assert csv_table_path is not None, "You must provide QE table file"
        E = pd.read_csv(csv_table_path)
        E[E < 0] = 0  # if mistakly there are negative values.
        g = np.array(E['<Efficiency>']) > 2
        if any(g):  # probabli the values are in the range [0,100], but the user must be careful here.
            E['<Efficiency>'] = E['<Efficiency>'] / 100  # effciency in range [0,1].

        wavelengths = E['<wavelength [nm]>'].values
        self._EFFICIENCYs[channel] = E

    def get_Efficiency(self, channel):
        """
        Returns Efficiency in range [0,1] and spectrum.
        """
        if (channel in self._EFFICIENCYs.keys()):
            wavelengths = self._EFFICIENCYs[channel]['<wavelength [nm]>'].values
            Efficiency = self._EFFICIENCYs[channel]['<Efficiency>'].values
            return Efficiency.copy(), wavelengths.copy()
        else:
            raise Exception('No channel {} exists in the effciencys object'.format(channel))

    def get_spectrum(self, channel, in_microns=False):
        """
        Returns spectrum in nm.
        """
        if (channel in self._EFFICIENCYs.keys()):
            wavelengths = self._EFFICIENCYs[channel]['<wavelength [nm]>'].values
            if (in_microns):
                w = wavelengths.copy()
                return float_round(1e-3 * w)
            else:
                return wavelengths.copy()
        else:
            raise Exception('No channel {} exists in the effciencys object'.format(channel))

    @property
    def channels(self):
        """
        Retunrs the CHANNELs.
        """
        return list(self._EFFICIENCYs.keys())

    def create_Efficiency_from_spectrum_and_values(self, spectrum, values):
        """
        Inputs:
        spectrum - np array of the wavelengths [nm]
        values - np array of the values per wavelength (like filter responce).
            values in the range [0, 1]

        Output is a pd.DataFrame which represents efficiency.
        """
        assert values.max() <= 1, "Values must be in [0,1] range."
        df = pd.DataFrame(
            data={"<wavelength [nm]>": spectrum, '<Efficiency>': values},
            index=None)

        return df

    def assume_Efficiency(self, EFFICIENCY, spectrum, channel='red'):
        """
        Some times we don't know the QE or it is given for the whole spectrum as one number, so use that number here.

        EFFICIENCY - value in the range [0, 100]
        spectrum - NOT a SPECTRUM class but a list of 2 elements or a scalar
        """

        start = spectrum[0]
        stop = spectrum[1]
        if (start == stop):

            self._EFFICIENCYs[channel]['<Efficiency>'] = EFFICIENCY / 100
            self._EFFICIENCYs[channel]['<wavelength [nm]>'] = spectrum

        else:

            spectrum = np.linspace(start, stop, 4)
            df = pd.DataFrame(
                data={"<wavelength [nm]>": spectrum, '<Efficiency>': (EFFICIENCY / 100) * np.ones_like(spectrum)},
                index=None)

            self._EFFICIENCYs[channel] = df

    def adjust_to_spectrum(self, spectrum, integration_Dlambda=10):
        """
        Adjust the EFFICEINCY to given spectrum. It just interpulates the existing spectrum to the new one.
        It is like addind an optical filter, wider or shorter.
        TODO: right now the filter is ideal, later to give the transmittion of the filter as a function of wavelength.
        Input:
            spectrum - Class SPECTRUM.

        """
        # chack channel overlap:
        if ((len(self.channels) == 1) and (self.channels[0] == 'wide_range')):
            # the EFFICIENCY_CHANNEL is 'wide_range', it is used in Lens for example to define wide spectrum and here to cut it in parts:
            efficeincy_values, efficeincy_spectrum = self.get_Efficiency('wide_range')
            # define interpulator:
            E = InterpolatedUnivariateSpline(efficeincy_spectrum, efficeincy_values)
            self._EFFICIENCYs = OrderedDict()  # get rid of the 'wide_range' channel.
            for index, spectrum_channel in enumerate(spectrum.channels):
                spectrum_band = spectrum.get_BAND(spectrum_channel)
                channel_spectrum = spectrum.get_wavelength_vector(spectrum_channel,
                                                                  integration_Dlambda=integration_Dlambda)

                # per channel: interpulate spectrum of efficeincy to spectrum of spectrum object:
                E_interpol = E(channel_spectrum)
                # overwite the old Efficiency:
                if np.isscalar(channel_spectrum) or channel_spectrum.size == 1:
                    df = pd.DataFrame(data={"<wavelength [nm]>": [channel_spectrum], '<Efficiency>': [E_interpol]},
                                      index=None)
                else:
                    df = pd.DataFrame(data={"<wavelength [nm]>": channel_spectrum, '<Efficiency>': E_interpol},
                                      index=None)

                self._EFFICIENCYs[spectrum_channel] = df

        else:

            EFFICIENCY_CHANNELS = self.channels
            for index, channel in enumerate(EFFICIENCY_CHANNELS):
                assert channel in spectrum.channels, "No overlap of channel in your definitions"
                channel_index_in_spectrum = list(EFFICIENCY_CHANNELS).index(channel)
                spectrum_channel = spectrum.channels[channel_index_in_spectrum]
                spectrum_band = spectrum.get_BAND(spectrum_channel)
                channel_spectrum = spectrum.get_wavelength_vector(spectrum_channel,
                                                                  integration_Dlambda=integration_Dlambda)
                # get spectrum and effciency values:
                efficeincy_values, efficeincy_spectrum = self.get_Efficiency(spectrum_channel)

                # per channel: interpulate spectrum of efficeincy to spectrum of spectrum object:
                E = InterpolatedUnivariateSpline(efficeincy_spectrum, efficeincy_values)
                E_interpol = E(channel_spectrum)
                # overwite the old Efficiency:
                if np.isscalar(channel_spectrum) or channel_spectrum.size == 1:
                    df = pd.DataFrame(data={"<wavelength [nm]>": [channel_spectrum], '<Efficiency>': [E_interpol]},
                                      index=None)
                else:
                    df = pd.DataFrame(data={"<wavelength [nm]>": channel_spectrum, '<Efficiency>': E_interpol},
                                      index=None)

                self._EFFICIENCYs[spectrum_channel] = df

    def show_EFFICIENCY(self, name='efficiency'):

        """
        name = QE or transmission for example.
        """

        if not plt.get_fignums():
            f, ax = plt.subplots(1, 1)

        ax = plt.gca()

        for index, channel in enumerate(self._EFFICIENCYs.keys()):
            E = self._EFFICIENCYs[channel]['<Efficiency>']
            S = self._EFFICIENCYs[channel]['<wavelength [nm]>']  # spectrum

            plt.plot(np.unique(S), E, linewidth=2, label='{} at {}'.format(name, channel), color=ColoR[index])

            plt.ylim([0, 1.1])
            # plt.xlim(self._scene_spectrum)

            plt.xlabel('wavelength [nm]', fontsize=16)
            plt.ylabel('Efficiencies [unitless]', fontsize=16)
            plt.title('Efficiencies')

            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            plt.grid(True)

        plt.legend()

    # -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

class SPECTRUM(object):
    """
    SPECTRUM: clase to handle a list of 2 elements (floats) e.g [ [lambda1, lambda2], ... ] in nm
    or, for simple case (just for premitive simulation)
    SPECTRUM hendles:
    a list of 1 element (floats) [ [lambda1], [lambda2], ... ] in nm.

    usage example:
    s = SPECTRUM(channels=['b','g','r1','r2'], bands=[[100,200], [300,400], [500,600], [700]])
    s.add_BAND(channel='swir', band = [1660,1670])
    s.add_BAND(channel='ir', band = [2000])

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    s.show_BANDs()

    plt.show()
    """

    def __init__(self, channels=None, bands=None):
        """
        Parameters:

        CHANNELS - list - e.g. ['r','g','b'] or any other names.

        BANDS - list of bands in nm e.g. [[550, 650], [580, 700] ...]
        or
        a polychromatic list of one wavelength in nm e.g. [[550], [680] ...].

        """
        self._BANDs_nm = OrderedDict()
        self._BANDs_microns = OrderedDict()

        if bands is not None:
            if channels is not None:
                assert len(channels) == len(bands), "cahnnels and bands must have same size."
                if isinstance(channels, list):
                    for channel_index, channel in enumerate(channels):
                        assert isinstance(bands[channel_index], list), "Here bands elements must be lists"
                        self._BANDs_nm[channel] = bands[channel_index] if len(bands[channel_index]) == 2 else \
                            bands[channel_index][0]
                        self._BANDs_microns[channel] = [float_round(1e-3 * i) for i in bands[channel_index]] if len(
                            bands[channel_index]) == 2 else float_round(1e-3 * bands[channel_index][0])
                else:  # bands is scalar:
                    self._BANDs_nm[channels] = bands
                    self._BANDs_microns[channel] = float_round(1e-3 * bands)
            else:  # cnahhels are missing so they get default names 0,1,2,..
                if isinstance(bands, list):
                    for channel_index, band in enumerate(bands):
                        assert isinstance(bands[channel_index], list), "Here bands elements must be lists"
                        self._BANDs_nm[channel_index] = bands[channel_index] if len(bands[channel_index]) == 2 else \
                            bands[channel_index][0]
                        self._BANDs_microns[channel] = [float_round(1e-3 * i) for i in bands[channel_index]] if len(
                            bands[channel_index]) == 2 else float_round(1e-3 * bands[channel_index][0])

                else:
                    self._BANDs_nm[0] = bands
                    self._BANDs_microns[0] = float_round(1e-3 * bands)

    def is_valid_spectrum_for_imager(self):
        """
        Check if all the channels are either bands or monochromatic.
        An Imager that will use this spectrum can not use spectrum with a cahannel which is monochromatic and other channel which is a band.
        If the spectrum has monochromatic chanels the imager is 'Simple imager'. Otherwise it is 'Real Imager'

        Returns:
            First output is True or False to say valide or not.
            Seconed output is a string:
               Simple imager  - 'simple'
               Real Imager    - 'real'
        """
        N = len(self._BANDs_nm.keys())
        C = 0
        for channel, band in self._BANDs_nm.items():
            if (isinstance(band, list)):
                C += 1
        if (C == 0):
            print("All the channels are monochromatic")
            return True, 'simple'
        elif (C == N):
            print("All the channels are bands")
            return True, 'real'
        else:
            raise Exception(
                "An Imager can not be defined with this spectrum. If you need to use it in Imager, the must be either bands or monochromatic")

        return False, None  # worst case

    def add_BAND(self, channel=None, band=None):
        """
        channel ot add and band to add.
        """
        if not np.isscalar(band):
            assert len(band) < 3, "bad band values."

        if channel is not None:
            self._BANDs_nm[channel] = band if len(band) == 2 else band[0]
            self._BANDs_microns[channel] = [float_round(1e-3 * i) for i in band] if len(band) == 2 else float_round(
                1e-3 * band[0])
        else:
            channels = self._BANDs_nm.keys()
            self._BANDs_nm[channels[-1] + 1] = band if len(band) == 2 else band[0]
            self._BANDs_microns[channels[-1] + 1] = [float_round(1e-3 * i) for i in band] if len(
                band) == 2 else float_round(1e-3 * band[0])

    def get_BAND(self, channel):
        """
        get the bend - list of 2 elements.
        """
        return self._BANDs_nm[channel]

    def get_BAND_microns(self, channel):
        """
        get the bend - list of 2 elements.
        """
        return self._BANDs_microns[channel]

    @property
    def channels(self):
        """
        Retunrs the CHANNELs.
        """
        channels = list(self._BANDs_nm.keys())
        return channels

    def get_center_wavelength(self, channel, in_microns=False):
        """
        Central waveband will be relevant when we use spectrally-averaged atmospheric parameters.

        in_microns=True means that the output will be in microns units.
        """
        if (channel in self._BANDs_nm.keys()):
            if (not np.isscalar(self._BANDs_nm[channel])):
                assert len(self._BANDs_nm[channel]) == 2, "This band must be represented with a list of [stert, end]"
                band_nm = self._BANDs_nm[channel]
                central_wavelength_nm = 1e3 * float_round(at3d.core.get_center_wavelen(band_nm[0], band_nm[1]))

                band_microns = self._BANDs_microns[channel]
                central_wavelength_microns = float_round(at3d.core.get_center_wavelen(band_microns[0], band_microns[1]))
            else:  # the band is polychromatic, so:
                central_wavelength_nm = self._BANDs_nm[channel]
                central_wavelength_microns = self._BANDs_microns[channel]
        else:
            raise Exception('No channel {} exists in the SPECTRUM object'.format(channel))

        if (in_microns):
            return central_wavelength_microns
        else:
            return central_wavelength_nm

    def calculate_solar_irradiance_at_channel(self, channel, SZA, integration_Dlambda=10):
        """
        calculate the solar irradiance that would reach the TOA at the spesific band. The band is picked by channel.
        It is done very simple, just black body radiation.

        To model the irradiance at a certain time of the day, we must
        multiple the irradiance at TOA by the cosine of the Sun zenith angle, it is also known as
        solar zenith angle (SZA) (1). Thus, the solar spectral irradiance at The TOA at
        a certain time is, self._LTOA = self._LTOA*cos(180-sun_zenith)

        input:
        channel

        SZA - is the zenith angle: float,
             Solar beam zenith angle in range (90,180]


        Output:
            LTOA - irradiance that would reach the TOA at the spesific band.
            lambdas_nm - wavelegths vector in nm.

        """
        LTOA = []
        lambdas_nm = self.get_wavelength_vector(channel, integration_Dlambda)
        if lambdas_nm.size == 1:
            lambdas_nm = np.asscalar(lambdas_nm)
        if np.isscalar(lambdas_nm):
            lambdas_nm = [lambdas_nm]

        for wavelength_nm in lambdas_nm:
            LTOA.append(6.8e-5 * 1e-9 * plank(1e-9 * wavelength_nm))  # units fo W/(m^2 nm)
        # I am assuming a solid angle of 6.8e-5 steradian for the source (the solar disk).
        assert 90.0 < SZA <= 180.0, 'Solar zenith:{} is not in range (90, 180] (photon direction in degrees)'.format(
            SZA)
        Cosain = np.cos(np.deg2rad((180 - SZA)))
        LTOA = np.array(LTOA) * Cosain
        return LTOA, lambdas_nm

    def approximate_TOA_radiance_at_channel(self, channel, rho=0.1, transmittion=1, SZA=180, VISUALIZE=False,
                                            integration_Dlambda=10):
        """
        calculate the hypotetic radiance that would reach the spaceborn view point.
        1. In the simple option, it is done very simple, just black body radiation and lamberation reflection be the clouds.
        The rho is the reflectance of the clouds (simple albedo).

        Note that the best way to do that is to use pyshdom. Here it is just rough estimation.
        It is a primitive radiance calculation, later consider pyshdom or libradtran.

        Inputs:
        ---------------------------------
        TODO

        SZA - is the zenith angle: float,
             Solar beam zenith angle in range (90,180]

        """
        LTOA, lambdas_nm = self.calculate_solar_irradiance_at_channel(channel, SZA, integration_Dlambda)
        radiance = transmittion * (rho * LTOA) / np.pi  # it is of units W/(m^2 st nm)

        if (VISUALIZE):
            f, ax = plt.subplots(1, 1, figsize=(8, 8))

            plt.plot(lambdas_nm, 1000 * radiance, label='rediance on the lens W/(m^2 st um)')
            plt.plot(lambdas_nm, 1000 * LTOA, label='black body radiation W/(m^2 um)')

            plt.ylim([0, 1.1 * max(1000 * radiance)])

            plt.xlabel('wavelength [nm]', fontsize=16)
            plt.ylabel('Intensity', fontsize=16)
            plt.title('The approximated radiance at the imager')

            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            plt.grid(True)
            plt.legend()

        return radiance, lambdas_nm

    def get_wavelength_vector(self, channel, integration_Dlambda=10, in_microns=False):
        """
        input
        integration_Dlambda: float , units nm.
            It is the Dlambda that will be used in intgration and the delta in the interpoolations on the spectrum.

        in_microns=True means that the output will be in microns units.
        """
        step = integration_Dlambda  # nm
        if isinstance(self._BANDs_nm[channel], list):

            start, stop = self._BANDs_nm[channel]
            lambdas = np.linspace(start, stop, int(((stop - start) / step) + 1))
            if (in_microns):
                lambdas = float_round(1e-3 * lambdas)

            return lambdas

        else:
            lambdas = self._BANDs_nm[channel]
            if (in_microns):
                return np.array(float_round(1e-3 * lambdas))
            else:
                return np.array(lambdas)

    def show_BANDs(self, add_TOA_irradiance=False):
        if not plt.get_fignums():
            f, ax = plt.subplots(1, 1)

        for index, (channel, band) in enumerate(self._BANDs_nm.items()):
            # highlighting regions
            if (isinstance(band, list)):

                plt.axvspan(band[0], band[1], color=ColoR[index], alpha=0.45)
                central = 0.5 * (band[0] + band[1])
                plt.annotate("channel {}".format(channel), xy=(central, 1), xycoords='data', \
                             xytext=(central + 10, 1 + 0.05 * index), textcoords='data',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

                if (add_TOA_irradiance):
                    LTOA, lambdas_nm = self.calculate_solar_irradiance_at_channel(channel)
                    plt.plot(lambdas_nm, 1000 * LTOA, label='black body radiation W/(m^2 um)')


            else:
                plt.axvspan(band - 1, band + 1, color=ColoR[index], alpha=0.45)
                central = band
                plt.annotate("channel {}".format(channel), xy=(central, 1), xycoords='data', \
                             xytext=(central + 10, 1 + 0.05 * index), textcoords='data',
                             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

            plt.xlabel('wavelength [nm]', fontsize=16)
            plt.ylim([0, 1.1 + 0.05 * index])

            if (add_TOA_irradiance):
                plt.ylim([0, 1.1 * max(1000 * LTOA)])
                plt.ylabel('Intensity', fontsize=16)

            ax.xaxis.set_tick_params(labelsize=16)
            ax.yaxis.set_tick_params(labelsize=16)
            plt.grid(True)


class SensorFPA(object):

    def __init__(self, QE=None, PIXEL_SIZE=5.5, FULLWELL=None, CHeight=1000, CWidth=1000,
                 READOUT_NOISE=100,DARK_CURRENT_NOISE=None, TEMP=15, BitDepth=8, TYPE='SIMPLE'):
        """
        Parameters:
        QE - quantum effciency, which measures the prob-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the quantum effciency in range [0,1].
        PIXEL_SIZE: float
            it is the pixel pitch in microns. It assumses symetric pitch.
        FULLWELL: int
            pixel full well, how much eletrons a pixel can generate befor it saturates.
        CHeight: int
            Number of pixels in camera x axis
        CWidth: int
            Number of pixels in camera y axis
        READOUT_NOISE: int
            Sensor readout noise in units of [electrons]. It is random noise which does not depend on exposure time. It is due to read
            circuits and output stages which add a random fluctuation to the output voltage.
        DARK_CURRENT_NOISE: float
            Dark current shot noise in units of [electrons/sec]. This is the output signal of
            the device with no ambient illumination. It consists of thermally generated electrons within
            the sensor which are accumulated during signal integration (self._Exposure time).
            Dark signal is a function of temperature. Therefor, the temperatur must be also given in consistence with the noise.
        TEMP: float
             temperatue in celsius. It is important for the dark noise. It must be consistent with the dark noise.
        BitDepth: int
            bit depth of the sensor.
        TYPE: str
            'VIS' or 'SWIR'.
        """

        self._CWidth = CWidth
        self._CHeight = CHeight
        self._PIXEL_SIZE = PIXEL_SIZE  # PIXEL_SIZE im microns
        if ((CHeight is None) or (CWidth is None)):
            self._SENSOR_SIZE = None
        else:
            self._SENSOR_SIZE = np.array([CHeight, CWidth]) * self._PIXEL_SIZE

        self._Exposure = 10  # mirco sec, IS IT NECESSARY HERE?
        self._Gain = 0  # 0 means no gain.

        if (QE is None):
            self._QE = None
            self._CHANNELS = []
        else:
            assert isinstance(QE, EFFICIENCY), "QE must be shdom.EFFICIENCY type!"
            self._CHANNELS = QE.channels
            self._QE = QE

        self._FULLWELL = FULLWELL
        # A common Rule of thumb is Full "well " ~ 1000 p^2, where p^2 is a pixel area in microns^2.
        if FULLWELL is None:

            self._FULLWELL = 1000 * self._PIXEL_SIZE ** 2  # e-
            self._SNR = np.sqrt(self._FULLWELL)

        else:
            self._SNR = np.sqrt(self._FULLWELL)

        # Sensor's SNR is not equal to Imager SNR sinse the SNR of a siglan also depends on the signal itself.

        self._READOUT_NOISE = READOUT_NOISE
        self._DARK_CURRENT_NOISE = None
        self._DARK_CURRENT_NOISE_TABLE = None
        self._TEMP = TEMP  # temperatue in celsius
        self._NOISE_FLOOR = None
        self._DR = None
        # self._NOISE_FLOOR = self._READOUT_NOISE + self._DARK_CURRENT_NOISE*(exposure*temp) , will be calculated later.
        # self._DR = self._FULLWELL/self._NOISE_FLOOR, will be calculated later.

        # Noise floor of the camera contains: self._READOUT_NOISE and self._DARK_CURRENT_NOISE.
        # Noise floor increases with the sensitivity setting of the sensor e.g. gain, exposure time and sensors temperature.
        # The Noise floor is important to define the Dynamic range (DN) of the sensor.
        # Dynamic range is defined as the ratio of the largest signal that an image sensor can handle to the
        # readout noise of the camera.  Readout noise of the camera can be classified a "Dark Noise"
        # which is measured during dark recording in specific temperature (e.g. room temp.)
        # thus the Dark noise has a term of dark current shot noise.

        self.BitDepth = BitDepth  # bit depth of the sensor.
        self._TYPE = TYPE
        self._alpha = (2 ** self.BitDepth) / self._FULLWELL
        # For a sensor having a linear radiometric response, the conversion between pixel electrons to grayscale is by a fixxed ratio self._alpha
        self._QUANTIZATION_NOISE_VARIANCE = 1 / (12 * (self._alpha))

    def Load_DARK_NOISE_table(self, csv_table_path=None):
        """
        Load DARK NOISE as a function of temperature from scv file "csv_table_path".
        DARK_CURRENT_NOISE -
            Dark current shot noise in units of [electrons/sec]. This is the output signal of
            the device with no ambient illumination. It consists of thermally generated electrons within
            the sensor which are accumulated during signal integration (self._Exposure time).
            Dark signal is a function of temperature. Therefor, the temperatur must be also given in consistence with the noise.

        temperatue in celsius. It is important for the dark noise. It must be consistent with the dark noise.

        It is pandas table, column #1 is noise [electrons/sec], column #2 is the temperatue in celsius.

        """
        assert csv_table_path is not None, "You must provide dark noise table file"
        self._DARK_CURRENT_NOISE_TABLE = pd.read_csv(csv_table_path)
        # update the dark nose with the loaded table values:
        DN = InterpolatedUnivariateSpline(self._DARK_CURRENT_NOISE_TABLE['temp'].values,
                                          self._DARK_CURRENT_NOISE_TABLE['noise [electrons/sec]'].values)
        DN_interpol = DN(self._TEMP)
        self._DARK_CURRENT_NOISE = DN_interpol

    def show_QE(self):
        assert self._QE is not None, "QE is still None"
        self._QE.show_EFFICIENCY(name='QE')

    @property
    def dark_noise_table(self):
        assert self._DARK_CURRENT_NOISE is not None, "You did not set the temperature of the sensor or the dark noise table."
        return self._DARK_CURRENT_NOISE_TABLE

    @property
    def QUANTIZATION_NOISE_VARIANCE(self):
        return self._QUANTIZATION_NOISE_VARIANCE

    @property
    def bits(self):
        """
        Retunrs the BitDepth of a pixel.
        """
        return self.BitDepth

    def get_QE(self):
        """
        Quantom Efficiency in range [0,1]
        """
        return self._QE

    def set_QE(self, val):
        assert isinstance(val, EFFICIENCY), "QE must be shdom.EFFICIENCY type!"

        self._QE = val
        self._CHANNELS = val.channels

    @property
    def channels(self):
        return self._CHANNELS

    @property
    def full_well(self):
        return self._FULLWELL

    @full_well.setter
    def full_well(self, val):
        self._FULLWELL = val

    def set_DR_IN_DB(self, DR):
        self._DR = 10 ** (DR / 20)

    def get_DR_IN_DB(self):
        return 20 * np.log10(self._DR)  # in DB

    def set_SNR_IN_DB(self, SNR):
        self._SNR = 10 ** (SNR / 20)

    def get_SNR_IN_DB(self):
        return 20 * np.log10(self._SNR)  # in DB

    def set_exposure_time(self, time):
        """
        time must be in microns
        """
        self._Exposure = time  # mirco sec

    def set_gain(self, gain):
        assert gain > 0, "gain must be positive."
        self._Gain = gain  # 0 means no gain.

    @property
    def QE(self):
        """
        Quantom Efficiency in range [0,1]
        """
        return self._QE

    @property
    def pixel_size(self):
        return self._PIXEL_SIZE

    @property
    def sensor_type(self):
        return self._TYPE

    @property
    def DynamicRange(self):
        return self._DR

    @property
    def SNR(self):
        return self._SNR

    @SNR.setter
    def SNR(self, val):
        self._SNR = val

    @property
    def sensor_size(self):
        """
        Retunrs the sensor size. it is np.array with 2 elements relative to [H,W]
        """
        return self._SENSOR_SIZE

    @sensor_size.setter
    def sensor_size(self, val):
        """
        Set the sensor size. it is np.array with 2 elements relative to [H,W]
        """
        self._SENSOR_SIZE = val

    @property
    def nx(self):
        """
        CHeight: int
            Number of pixels in camera x axis
        """
        return self._CHeight

    @property
    def ny(self):
        """
        CWidth: int
            Number of pixels in camera y axis
        """
        return self._CWidth

    @property
    def alpha(self):
        return self._alpha

    def get_sensor_resolution_in_lp_per_mm(self):
        """
        The resolution of the sensor, also referred to as the image space resolution for the system,
        can be calculated by multiplying the pixel size in um by 2 (to create a pair), and dividing that into 1000 to convert to mm.
        The highest frequency which can be resolved by a sensor, the Nyquist frequency, is effectively two pixels or one line pair.
        https://www.edmundoptics.com/knowledge-center/application-notes/imaging/resolution/
        """
        return 1000 / (2 * self._PIXEL_SIZE)


class LensSimple(object):

    def __init__(self, TRANSMISSION=None, FOCAL_LENGTH=100.0, DIAMETER=10.0):
        """
        Parameters:
        TRANSMISSION - Class TRANSMISSION - measures the transmittion of the lens as a function of wavelength-
            ability for an electron to be produced per incident photon.
            It is pandas table, column #1 is wavelenths in nm, column #2 is the transmittion [0,1].
        FOCAL_LENGTH: floate
            The focaal length of the lens in mm.
        DIAMETER: floate
            The diameter of the lens in mm.
        """

        if (TRANSMISSION is None):
            self._TRANSMISSION = None
            self._CHANNELS = []
        else:
            assert isinstance(TRANSMISSION, EFFICIENCY), "TRANSMISSION must be shdom.EFFICIENCY type!"
            self._CHANNELS = TRANSMISSION.channels
            self._TRANSMISSION = TRANSMISSION

        # -----------------------------------
        if FOCAL_LENGTH is not None:
            self._FOCAL_LENGTH = FOCAL_LENGTH  # mm
        else:
            self._FOCAL_LENGTH = None

        if DIAMETER is not None:
            self._DIAMETER = DIAMETER  # mm
        else:
            self._DIAMETER = None
        self._wave_diffraction = None  # in micro meters, it will be the 1e-3*min(2.44*self._SPECTRUM)*(self._FOCAL_LENGTH/self._DIAMETER)

        if (TRANSMISSION is not None):
            wave_diffractions = []
            for channel in TRANSMISSION.channels:
                transmission, spectrum = self._TRANSMISSION.get_Efficiency(channel)
                wave_diffractions.append(1e-3 * max(2.44 * spectrum) * (self._FOCAL_LENGTH / self._DIAMETER))
                # https://www.edmundoptics.com/knowledge-center/application-notes/imaging/limitations-on-resolution-and-contrast-the-airy-disk/

            self._wave_diffraction = max(wave_diffractions)
            print(
                "----> Spot size because of the diffraction is {}[micro m]".format(float_round(self._wave_diffraction)))

    def show_TRANSMISSION(self):
        assert self._TRANSMISSION is not None, "TRANSMISSION is still None"
        self._TRANSMISSION.show_EFFICIENCY(name='LENS TRANSMISSION')

    def get_spectrum(self, channel=None):
        return self._TRANSMISSION.get_spectrum(channel)

    def get_TRANSMISSION(self, channel):
        """
        TRANSMISSION in range [0,1]
        """
        return self._TRANSMISSION

    def set_TRANSMISSION(self, val):
        assert isinstance(val, EFFICIENCY), "QE must be shdom.EFFICIENCY type!"

        self._TRANSMISSION = val
        self._CHANNELS = val.channels

    @property
    def channels(self):
        return self._CHANNELS

    @property
    def TRANSMISSION(self):
        """
        LENS TRANSMISSION in range [0,1]
        """
        return self._TRANSMISSION

    @property
    def T(self):
        """
        Lens TRANSMISSION in range [0,1] in the whole wavelength range.
        It is different from property TRANSMISSION. The TRANSMISSION is EFFICEINCY object.
        """
        t = []
        w = []
        for index, channel in enumerate(self._CHANNELS):
            t_channel, wavelengths = self._TRANSMISSION.get_Efficiency(channel)
            t.append(t_channel)
            w.append(wavelengths)

        t = np.concatenate(t, axis=0)
        w = np.concatenate(w, axis=0)

        ind = np.argsort(w, axis=0)  # sorts along first axis (down)
        t = np.take_along_axis(t, ind, axis=0)  # same as np.sort(x, axis=0)
        w = np.take_along_axis(w, ind, axis=0)  # same as np.sort(x, axis=0)

        w, unique_indices = np.unique(w, return_index=True)
        t = t[unique_indices]

        return t, w

    @property
    def imager_type(self):
        return self._type

    @property
    def diameter(self):
        return self._DIAMETER

    @diameter.setter
    def diameter(self, val):
        self._DIAMETER = val

    @property
    def focal_length(self):
        return self._FOCAL_LENGTH


class Imager(object):
    def __init__(self, sensor=None, lens=None, scene_spectrum=None,
                 integration_Dlambda=1, temp=20, system_efficiency=1, TYPE='UNKNOWN_IMAGER'):
        """
        Parameters:
        sensor - sensor class.
        lens - lens class.
        scene_spectrum - SPECTRUM class.
        integration_Dlambda: float
            It is the Dlambda that will be used in intgration and the delta in the interpoolations on the spectrum.
            The scene_spectrum will be defined later on.
            will be dectated by QE of sensor and TRANSMMITION of the lens.

        temp: float
             temperatue in celsius.
        system_efficiency: float
             range in [0,1], it is the camera system efficiency due to optics losses and sensor reflection (it is not a part of QE).

        VERY IMPORTANT NOTES:
        1. The channels name of the sensor must overlap with the name of the spectrum class.
        2. In lens channels it is not obligation since the lens can be defined on wide range of wavelengths. So the general channel
        name in the lens object should be 'wide_range' for buth vis or swir options.
        For example, use lens_transmission.assume_Efficiency(90, spectrum = [400,800], channel='wide_range')
        or as in the sensor case, use (same channels names as in spectrum):
        lens_transmission.assume_Efficiency(90, spectrum = [x1,x2], channel='red')
        lens_transmission.assume_Efficiency(90, spectrum = [y1,y2], channel='green')
        .
        .


        3. usage:
           3.1 first use set_Imager_altitude(H) to set the altitude. This calculates footprints and exposure time.


        """
        assert sensor is not None, "SENSOR OBJECT MUST BE PROVIEDE."
        assert lens is not None, "LENS OBJECT MUST BE PROVIEDE."
        assert scene_spectrum is not None, "SPECTRUM OBJECT MUST BE PROVIEDE."

        # Here we have defined sensor, lens and scene_spectrum
        # ------------------------------------------------------------------------------
        # ----------------------------imager stuff:-------------------------------------
        # ------------------------------------------------------------------------------
        self._type = TYPE  # imager type can be important for the simulation type.
        self._temp = temp  # celsius
        self._H = None  # km, will be defined by function set_Imager_altitude()
        self._orbital_speed = None  # [km/sec], will be defined by function set_Imager_altitude()
        self._max_exposure_time = None  # [micro sec], will be defined by function set_Imager_altitude()
        self._exposure_time = None  # the actual exposure time, it must be less than the max exposure time.
        self._integration_Dlambda = integration_Dlambda

        assert isinstance(scene_spectrum, SPECTRUM), "scene_spectrum must be shdom.SPECTRUM type!"
        self._scene_spectrum = scene_spectrum
        self._channels = self._scene_spectrum.channels
        # The channels and wavelength vectors will be provided by scene_spectrum object.
        # use: get_wavelength_vector(self, channel, integration_Dlambda = 10, in_microns=False or True)
        #     get_center_wavelen(self,channel, in_microns=False)
        #     calculate_solar_irradiance_at_channel(self,channel)
        #     approximate_TOA_radiance_at_channel(self,channel,rho=0.1,transmittion=1, VISUALIZE = False)

        # ------------------------------------------------------------------------------
        # ----------------------------sensor stuff:-------------------------------------
        # ------------------------------------------------------------------------------
        self._sensor = sensor
        self._DR = self._sensor.DynamicRange  # dynamic range.
        self._SNR = self._sensor.SNR
        self._READ_NOISE = self._sensor._READOUT_NOISE
        self._NOISE_FLOOR = None  # sinse the exposure time may be unkwon.
        self.set_gain_uncertainty()  # here it is set as default gain_uncertainty - no uncertainty.
        # The uncertainties should be apdated in main simulator.
        self.set_bias_uncertainty()

        """
        Note that:
            self._NOISE_FLOOR = READOUT_NOISE + DARK_CURRENT_NOISE*(exposure*temp) , will be calculated later.
            self._DR = FULLWELL/NOISE_FLOOR, will be updated later.    
        """
        if ((self._temp is not None) and (self._sensor.dark_noise_table is not None)):
            self._dark_noise_table = self._sensor.dark_noise_table
            DN = InterpolatedUnivariateSpline(self._dark_noise_table['temp'].values,
                                              self._dark_noise_table['noise [electrons/sec]'].values)
            DN_interpol = DN(self._temp)
            self._DARK_NOISE = float_round(
                np.isscalar(DN_interpol))  # it is better then just self._sensor._DARK_CURRENT_NOISE
            # Since imager temperature may be differena then sensor's.
        else:
            print('There is no DARK_NOISE definition since ethier imager temperatur or dark noise table are missing.')
            self._dark_noise_table = None
            self._DARK_NOISE = None
            # the DARK CURRENT NOISE = self._DARK_NOISE*1e-6*self._exposure_time
        # Sensor's QE part:
        self._Imager_QE = copy.deepcopy(self._sensor.QE)  # it is not as self._sensor.QE
        self._Imager_QE.adjust_to_spectrum(self._scene_spectrum, integration_Dlambda=self._integration_Dlambda)
        # self._Imager_QE it is similar to self._sensor.QE but interpooleted on the full self._scene_spectrum.
        # scene_spectrum per channel imitates ideal filter of that channel.
        assert self._channels == self._Imager_QE.channels, "Channels must overlap"
        # ------------------------------------------------------------------------------
        # ----------------------------lens stuff:---------------------------------------
        # ------------------------------------------------------------------------------
        self._lens = lens
        self._ETA = system_efficiency  # It is the camera system efficiency, I don't know yet how to set its value,
        self._FOV = None  # [rad], will be defined by function set_Imager_altitude()
        self._Imager_EFFICIANCY = copy.deepcopy(self._lens.TRANSMISSION)  # it is not as self._lens._TRANSMISSION
        self._Imager_EFFICIANCY.multiply_efficiency_by_scalar(self._ETA)
        self._Imager_EFFICIANCY.adjust_to_spectrum(self._scene_spectrum, integration_Dlambda=self._integration_Dlambda)
        assert self._channels == self._Imager_EFFICIANCY.channels, "Channels must overlap"
        # ------------------------------------------------------------------------------
        # ----------------------------more imager stuff:-------------------------------------
        # ------------------------------------------------------------------------------

        # diffraction
        self._diffraction_scalar = 1.22
        # calculate diffraction for every channel:
        self._wave_diffraction = OrderedDict()  # it is the spot on the sensor plane
        self._centeral_wavelength_in_microns = OrderedDict()  # will be used by pyshdom mie tables.

        for channel in self._channels:
            lambdas = self._scene_spectrum.get_wavelength_vector(channel, integration_Dlambda=self._integration_Dlambda)
            self._centeral_wavelength_in_microns[channel] = self._scene_spectrum.get_center_wavelength(channel,
                                                                                                       in_microns=True)
            if np.isscalar(lambdas) or lambdas.size == 1:
                wave_diffraction = 1e-3 * (2 * self._diffraction_scalar * lambdas) * (
                        self._lens.focal_length / self._lens.diameter)
                wave_diffraction = wave_diffraction[0]
            else:
                wave_diffraction = 1e-3 * max(2 * self._diffraction_scalar * lambdas) * (
                        self._lens.focal_length / self._lens.diameter)

            self._wave_diffraction[channel] = wave_diffraction  # [micro m]
            print("---->  chennel {}: Spot size because of the diffraction is {}[micro m]".format(channel, float_round(
                wave_diffraction)))
            if (wave_diffraction > self._sensor.pixel_size):
                print("---->  Pixel size is {}[micro m]".format(float_round(self._sensor.pixel_size)))
                print("Diffreaction limit is met in chennel {}, be careful with the calculations!".format(channel))

        """
        # https://www.edmundoptics.com/knowledge-center/application-notes/imaging/limitations-on-resolution-and-contrast-the-airy-disk/
        self._pixel_size_diffration = self._wave_diffraction

        """
        self._pixel_footprint = None  # km, will be defined by function set_Imager_altitude()
        self._camera_footprint = None  # km, will be defined by function set_Imager_altitude()
        self._diffraction_Dtheta = None  # it will be updated in calculate footprints as self._pixel_footprint/self._H
        #  It can be shown that, for a circular aperture of diameter D, the first minimum in the
        # diffraction pattern occurs at Dtheta = 1.22 lambdas / lens diameter. Thus
        # lens diameter limitation with respect to diffraction = 1.22 lambdas / Dtheta. This is the
        # self._lens_diameter_diffration. The lens diameter in mm that whould fit diffraction requairment.
        # self._lens_diameter_diffration = 1e-6*(self._diffraction_scalar*self._lambdas)/self._diffraction_Dtheta # in mm
        self._lens_diameter_diffration = None

        # do I need it? self._lens_diameter_diffration = 0 # becouse it will be used in max(,) test to extract the minimum diameter.

        # ------------------------------------------------------------------------------
        # ----------------------------Solar setup stuff:-------------------------------------
        # ------------------------------------------------------------------------------
        self._SZA = 180

        """
        1.
        To know how raniance converted to electrons we define GAMMA.
        The GAMMA depends on the wavelength. 
        The number of electrons i_e generated by photons at wavelength lambda during exposure time Dt is
        i_e = GAMMA_lambda * I_lambda * Dt,
        where I_lambda is the radiance [W/m^2 Sr] that reachs a pixel.
        GAMMA_lambda = pi*eta*((D/(2*f))^2)*QE_lambda * (lambda/(h*c))*p^2
        * p - pixel sixe.
        * h- Planck's constant, c - speed of light.
        * D -lens diameter, f- focal length.
        * eta - camera system efficiency due to optics losses and sensor reflection (it is not a part of QE).
        The units of GAMMA_lambda are [electrons * m^2 * Sr /joule ].

        2.
        We can calculate I_lambda for wavelengths range since the calculation of pixel responce to light requires 
        an integral over solar spectral band. Thus RT simulations should be applied multiple times to calculated.
        But we there is an alternative:
        An alternative way is to use spectrally-averaged quantities, it is valid when wavelength dependencies within a
        spectral band are weak (e.g. in narrow band and absence absorption within the band).
        This alternative uses only one run of RT simulation per spectral band.
        So here, we define self.of_unity_flux_radiance. It is the radiance at the lens which calculated with RT 
        simulation when the solar irradiance at the TOA is 1 [W/m^2] and the spectrally-dependent parameters of the atmospheric model are spectrally-averaged.

        """

    @classmethod
    def copy_imager(cls):
        """
        Copy an Imager object.
        """
        obj = cls.__new__(cls)  # Does not call __init__

    def update(self):
        """
        Update should be called each time the optics or imager altitude is changed:
        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------
        Update each time lens is updated:
        - self._wave_diffraction
        - self._FOV
        - self._pixel_footprint
        - self._camera_footprint
        - self._diffraction_Dtheta
        - self._max_exposure_time
        - self._exposure_time
        - self._lens_diameter_diffration
        -
        -
        -


        --------------------------------------------------------------------------------
        --------------------------------------------------------------------------------
        """
        # diffraction
        # calculate diffraction for every channel:
        self._wave_diffraction = OrderedDict()  # it is the spot on the sensor plane
        self._centeral_wavelength_in_microns = OrderedDict()  # will be used by pyshdom mie tables.

        for channel in self._channels:
            lambdas = self._scene_spectrum.get_wavelength_vector(channel, integration_Dlambda=integration_Dlambda)
            self._centeral_wavelength_in_microns[channel] = self._scene_spectrum.get_center_wavelen(channel,
                                                                                                    in_microns=True)

            wave_diffraction = 1e-3 * max(2 * self._diffraction_scalar * lambdas) * (
                    self._lens.focal_length / self._lens.diameter)
            self._wave_diffraction[channel] = wave_diffraction  # [micro m]
            print("---->  chennel {}: Spot size because of the diffraction is {}[micro m]".format(channel, float_round(
                wave_diffraction)))
            if (wave_diffraction > self._sensor.pixel_size):
                print("Diffreaction limit is met in chennel {}, be carful with the calculations!".format(channel))

        # footprints, exposure and camera FOV:
        self.set_Imager_altitude(self._H)

        # check if we still have valide optics:
        self.IS_VALIDE_LENS_DIAMETER()

    def set_solar_beam_zenith_angle(self, SZA):
        """
        SZA - is the zenith angle: float,
            Solar beam zenith angle in range (90,180]

        To model the irradiance at a certain time of the day, we must
        multiple the irradiance at TOA by the cosine of the Sun zenith angle, it is also known as
        solar zenith angle (SZA) (1). Thus, the solar spectral irradiance at The TOA at
        a certain time is, self._LTOA = self._LTOA*cos(180-sun_zenith)
        """
        assert 90.0 < SZA <= 180.0, 'Solar zenith:{} is not in range (90, 180] (photon direction in degrees)'.format(
            SZA)
        self._SZA = SZA

    def change_temperature(self, val):
        """
        Change the temperature in celsius.
        """
        assert self._dark_noise_table is not None, "At this point you must set the dark noise table of the sensor."
        DN = InterpolatedUnivariateSpline(self._dark_noise_table['temp'].values,
                                          self._dark_noise_table['noise [electrons/sec]'].values)
        DN_interpol = DN(val)
        self._temp = val
        self._DARK_NOISE = float_round(
            self._sensor._DARK_CURRENT_NOISE))  # it is better then just self._sensor._DARK_CURRENT_NOISE
        # Since imager temperature may be different then sensor's.

    def set_Imager_altitude(self, H):
        """
        H must be in km
        """
        self._H = H  # km
        self._orbital_speed = 1e-3 * SatSpeed(orbit=self._H*1e3)  # units of [m/sec] converted to [km/sec]
        print("----> Speed in {}[km] orbit is {}[km/sec]".format(self._H, float_round(self._orbital_speed)))
        P = 0.5  # is the ration of the GSD that we require not to blur during the exposure time.
        # in the old version it was 1.

        # At this point the lens must be defined:
        self.calculate_footprints()

        # bound the exposure time:
        # Let self._orbital_speed be the speed of the satellites. To avoid motion blur, it is important that:
        self._max_exposure_time = 1e6 * P * self._pixel_footprint / self._orbital_speed  # in micron sec
        self.set_exposure_time(self._max_exposure_time - 1)  # set to max exposur time -1 micron.

        # camera FOV:
        # Let self._camera_footprint be the footprint of the camera at nadir view in x axis. The field of view of the camera in radians is,
        self._FOV = 2 * np.arctan(self._camera_footprint / 2 * self._H)

        if (self._DARK_NOISE is not None):
            self._NOISE_FLOOR = self._READ_NOISE + (self._DARK_NOISE * 1e-6 * self._exposure_time)
            self._DR = self._sensor.full_well / self._NOISE_FLOOR

        print(
            "The exposure time is set here to be the maximum exposure time. The maximum exposure time is calculated to avoide pixel bluring due to motion.")

    def get_gray_lavel_maximum(self):
        """
        It is the maximal gray level that pixel can get.
        """
        return 2 ** self._sensor.bits

    def IS_VALIDE_LENS_DIAMETER(self, for_this_radiance=None):
        """
        Test if the lens diameter is compatible with the exposure time and all the rest.

        inputs:
        --------------------------------------------------
        for_this_radiance - scalar, np.array or None:
            If none the radiance will be calculated in the simplest way, just to assume simple labertion reflection of a surface with
            reflectens rho = 0.2 and atmospheric transmission of 0.8

            If scalar or np.array, it should be calculated radience (shdom or libradtran).
        """
        minimum_lens_diameter_no_diffraction = []
        for channel in self._channels:
            lambdas = self._scene_spectrum.get_wavelength_vector(channel, integration_Dlambda=self._integration_Dlambda)

            if np.isscalar(lambdas) or lambdas.size == 1:
                wave_diffraction = 1e-3 * (2 * self._diffraction_scalar * lambdas) * (
                        self._lens.focal_length / self._lens.diameter)
                wave_diffraction = wave_diffraction[0]
            else:
                wave_diffraction = 1e-3 * max(2 * self._diffraction_scalar * lambdas) * (
                        self._lens.focal_length / self._lens.diameter)

            print("---->  chennel {}: Spot size because of the diffraction is {}[micro m]".format(channel, float_round(
                wave_diffraction)))
            if (wave_diffraction > self._sensor.pixel_size):
                print("Diffreaction limit is met in chennel {}, be carful with the calculations!".format(channel))
                print("Spot size is {}, pixel size is {}".format(float_round(wave_diffraction),
                                                                 float_round(self._sensor.pixel_size)))

            # assume low radiance just for simple test:
            # TODO - bring here radiance from shdom
            if for_this_radiance is None:
                radiance, _ = self._scene_spectrum.approximate_TOA_radiance_at_channel(channel, rho=0.35,
                                                                                       transmittion=0.8, SZA=self._SZA,
                                                                                       integration_Dlambda=self._integration_Dlambda)
            else:
                radiance = for_this_radiance
            # sensor QE and optics transmission:
            Imager_QE_channel, _ = self._Imager_QE.get_Efficiency(channel)
            Imager_EFFICIANCY_channel, _ = self._Imager_EFFICIANCY.get_Efficiency(channel)

            G1 = (1 / (h * c)) * self._exposure_time * (((np.pi) / 4) / (self._lens._FOCAL_LENGTH ** 2)) * (
                    self._sensor.pixel_size ** 2)
            if np.isscalar(lambdas) or lambdas.size == 1:
                G2 = radiance * Imager_EFFICIANCY_channel * Imager_QE_channel * lambdas
            else:
                G2 = np.trapz((radiance * Imager_EFFICIANCY_channel * Imager_QE_channel) * lambdas, x=lambdas)
            G = 1e-21 * G1 * G2

            minimum_channel_lens_diameter = 1000 * ((self._sensor.full_well / G) ** 0.5)  # in mm
            minimum_lens_diameter_no_diffraction.append(minimum_channel_lens_diameter)

        minimum_lens_diameter_no_diffraction = np.array(minimum_lens_diameter_no_diffraction).max()
        # maybe the diffraction limits the diameter, so:
        minimum_lens_diameter_due_to_diffraction = self._lens_diameter_diffration
        minimum_lens_diameter = max(minimum_lens_diameter_due_to_diffraction, minimum_lens_diameter_no_diffraction)
        if minimum_lens_diameter > self._lens.diameter:
            print('Lens diameter is invalide due to low dynamic range or diffraction limit is met.')
            print("Minimum lens diameter due to diffraction is {}".format(minimum_lens_diameter_due_to_diffraction))
            print("Minimum lens diameter due to dynamic range is {}".format(minimum_lens_diameter_no_diffraction))
            print('Lens diameter is {} but should be {}.'.format(self._lens.diameter, minimum_lens_diameter))
            # raise Exception("Lens diameter is invalide due to low dynamic range or diffraction limit is met.")

    def adjust_exposure_time_from_electrons(self, electrons, C=0.9):
        """
        TODO
        """
        t = 1e6 * C * self._sensor.full_well / electrons
        if (t < self._exposure_time):
            self._exposure_time = t - 10  # - 10 micros it is a margin
            print("The adjusted exposure time becouce of saturation is {} micro sec.".format(
                float_round(self._exposure_time)))

        else:
            # means we far from full well, t > self._exposure_time
            self._exposure_time = t - 10
            print("The adjusted exposure time becouce of low signal is {} micro sec.".format(
                float_round(self._exposure_time)))
            print(
                "If you still want to change the exposure time, use set_exposure_time() method and do not use adjust_exposure_time().")
        return self._exposure_time

    def adjust_exposure_time(self, images, sat_id, C=0.9):
        """
        This method adjusts the exposure time such that the pixel reached its full well for the maximal value of the raciances rendered by RTE.
        However, if the current exposure time does not cause full well, there will be now adjustment.

        Parameters:
        Input:
        images - np.array or list of np.arrays, it is the images that represent radiances that reache the lens where the simulation of that radiance considered
        solar flux of 1 [W/m^2] and spectrally-dependent parameters of the atmospheric model are spectrally-averaged.
        C scalar in range 0-1. If it is 1, the pixel will reach full well. C is the procent of the full well that will be reached.

        """
        max_of_all_images = np.array(images).max()

        GAMMA_lambda = self.get_GAMMA_lambda()
        # GAMMA_lambda - EFFICIENCY object.

        # next, calculate the integral of GAMMA_lambda with respect to lambda.
        TIMES = []
        WELLS = []

        for channel_index, channel in enumerate(self._channels):
            GAMMA_lambda_channel, lambdas_channel = GAMMA_lambda.get_Efficiency(channel)
            LTOA, lambdas_channel_test = self._scene_spectrum.calculate_solar_irradiance_at_channel(channel, self._SZA,
                                                                                                    self._integration_Dlambda)
            assert np.all(lambdas_channel_test == lambdas_channel), "Inconsistency in the wavelengths."
            if np.isscalar(lambdas_channel) or lambdas_channel.size == 1:
                GAMMA_lambda_INTEGRAL_lambda = GAMMA_lambda_channel * LTOA
                GAMMA_lambda_INTEGRAL_lambda = np.asscalar(GAMMA_lambda_INTEGRAL_lambda)
            else:
                GAMMA_lambda_INTEGRAL_lambda = np.trapz(GAMMA_lambda_channel * LTOA, x=lambdas_channel)
            # units of GAMMA_lambda_INTEGRAL_lambda are [electrons*st/sec]

            if (len(self._channels) > 1):
                max_of_images_per_channel = np.array(images)[..., channel_index].max()

            else:
                # max_of_images_per_channel = np.array(images).max()
                max_of_images_per_channel = np.array(images)[:,sat_id,:,:].max()

            # electrons_number = 1e-6*self._exposure_time*INTEGRAL*image
            # The 1e-6* scales electrons_number to units of [electrons]
            # Thus:

            t = 1e6 * C * self._sensor.full_well / (
                        GAMMA_lambda_INTEGRAL_lambda * max_of_images_per_channel)  # in micro sec
            test_reached_well = 1e-6 * GAMMA_lambda_INTEGRAL_lambda * max_of_images_per_channel * self._exposure_time  # how much electrons generated for current exposur time
            WELLS.append(test_reached_well)
            TIMES.append(t)

        test_reached_well = max(WELLS)
        t = min(TIMES)  # consistent to max(WELLS)

        # test_reached_well_percents = 100*(test_reached_well/self._sensor.full_well)

        # print(" \n\n\n\n-------------- Exposure time consideration of {}:-----------".format(self.imager_type))
        # print(" The exposure time was set to {} micro sec.".format(float_round(self._exposure_time)))
        if (t < self._exposure_time):
            self._exposure_time = t - 10  # - 10 micros it is a margin
            # print("The adjusted exposure time becouce of saturation is {} micro sec.".format(
            #     float_round(self._exposure_time)))


        else:
            # means we far from full well, t > self._exposure_time
            self._exposure_time = t - 10
            # print("The adjusted exposure time becouce of low signal is {} micro sec.".format(
            #     float_round(self._exposure_time)))
            # print(
            #     "If you still want to change the exposure time, use set_exposure_time() method and do not use adjust_exposure_time().")

        # double check:
        for channel_index, channel in enumerate(self._channels):
            GAMMA_lambda_channel, lambdas_channel = GAMMA_lambda.get_Efficiency(channel)
            LTOA, lambdas_channel_test = self._scene_spectrum.calculate_solar_irradiance_at_channel(channel, self._SZA,
                                                                                                    self._integration_Dlambda)
            assert np.all(lambdas_channel_test == lambdas_channel), "Inconsistency in the wavelengths."
            if np.isscalar(lambdas_channel) or lambdas_channel.size == 1:
                GAMMA_lambda_INTEGRAL_lambda = GAMMA_lambda_channel * LTOA
            else:
                GAMMA_lambda_INTEGRAL_lambda = np.trapz(GAMMA_lambda_channel * LTOA, x=lambdas_channel)
            # units of GAMMA_lambda_INTEGRAL_lambda are [electrons*st/sec]
            if (len(self._channels) > 1):
                max_of_images_per_channel = np.array(images)[..., channel_index].max()
            else:
                # max_of_images_per_channel = np.array(images).max()
                max_of_images_per_channel = np.array(images)[:, sat_id, :, :].max()
                # electrons_number = 1e-6*self._exposure_time*INTEGRAL*image
            # The 1e-6* scales electrons_number to units of [electrons]
            # Thus:
            channel_reached_well = 1e-6 * GAMMA_lambda_INTEGRAL_lambda * max_of_images_per_channel * self._exposure_time  # how much electrons generated for current exposur time
            # print("At channel {} at maximum, the well is {}%.".format(channel, 100 * (
            #             channel_reached_well / self._sensor.full_well)))

        # print((40 * "-") + "\n")
        # print((40 * "-") + "\n")
        # print((40 * "-") + "\n")

    def get_GAMMA_lambda(self):
        """
        Output:
             GAMMA_lambda - EFFICIENCY object.

        --------------------------------------------------
        To know how raniance converted to electrons we define GAMMA.
        The GAMMA depends on the wavelength.
        The number of electrons i_e generated by photons at wavelength lambda during exposure time Dt is
        i_e = GAMMA_lambda * I_lambda * Dt,
        where I_lambda is the radiance [W/m^2 Sr] that reachs a pixel.
        GAMMA_lambda = pi*eta*((D/(2*f))^2)*QE_lambda * (lambda/(h*c))*p^2
        * p - pixel sixe.
        * h- Planck's constant, c - speed of light.
        * D -lens diameter, f- focal length.
        * eta - camera system efficiency due to optics losses and sensor reflection (it is not a part of QE).
        The units of GAMMA_lambda are [electrons * m^2 * Sr /joule ].
        """
        GAMMA_lambda = EFFICIENCY()

        channels = self._Imager_QE.channels
        for channel in channels:
            Imager_QE_channel, lambdas_channel = self._Imager_QE.get_Efficiency(channel)
            Imager_EFFICIANCY_channel, _ = self._Imager_EFFICIANCY.get_Efficiency(channel)

            GAMMA_lambda_channel = 1e-12 * np.pi * Imager_EFFICIANCY_channel * (
                    (self._lens.diameter / (2 * self._lens._FOCAL_LENGTH)) ** 2) * Imager_QE_channel * (
                                           1e-9 * lambdas_channel / (h * c)) * (self._sensor.pixel_size ** 2)
            # The 1e-12* scales GAMMA_lambda to units of [electrons*m^2*st/joule]

            df = pd.DataFrame(data={"<wavelength [nm]>": lambdas_channel, '<Efficiency>': GAMMA_lambda_channel},
                              index=None)
            GAMMA_lambda.add_EFFICIENCY(df, channel=channel)

        return GAMMA_lambda

    def add_noise(self, electrons_number):
        """
        TODO
        Currently, we use:
        * Gaussian distribution for the read noise and dark noise. TODO consider more accurate model for both.


        """
        # photon noise, by Poisson noise:
        electrons_number = np.random.poisson(electrons_number)

        # dark noise:
        DARK_NOISE_mean = (self._DARK_NOISE * 1e-6 * self._exposure_time)
        DARK_NOISE_variance = DARK_NOISE_mean  # since itcomes from poisson distribution.
        DN_noise = np.random.normal(
            loc=DARK_NOISE_mean,
            scale=DARK_NOISE_variance ** 0.5,
            # The scale parameter controls the standard deviation of the normal distribution.
            size=electrons_number.shape
        ).astype(np.int)

        electrons_number += DN_noise

        # read noise:
        # TODO ask Yoav if it is needed and how to model it?
        READ_NOISE_mean = (self._READ_NOISE ** 2)
        READ_NOISE_variance = (self._READ_NOISE ** 2)  # since it comes from poisson distribution.
        READ_noise = np.random.normal(
            loc=0,  # TODO ask Yoav
            scale=READ_NOISE_variance ** 0.5,
            # The scale parameter controls the standard deviation of the normal distribution.
            size=electrons_number.shape
        ).astype(np.int)

        electrons_number += READ_noise

        electrons_number = np.clip(electrons_number, a_min=0, a_max=None)
        return electrons_number.astype(np.float)

    def convert_radiance_to_graylevel(self, image, C=0.9, cancel_noise=False, limit_integral_in_lambda_range_nm=None):
        """
        This method convert radiances to grayscals. In addition, it returns the scale that would convert radiance to grayscale BUT without noise addition in the electrons lavel.
        The user decides if to use that as normalization or not.

        Parameters:
        Input:
        image - np.array , it is the image that represent radiances that reache the lens where the simulation of that radiance considered
            solar flux of 1 [W/m^2] and spectrally-dependent parameters of the atmospheric model are spectrally-averaged.

        C - float in range [0,1], should be consistent with the C of self.adjust_exposure_time.
            Than, imager operates such that the most brigth pixel
            reaches C (e.g. 90%) of its full well.

        cancel_noise: bool
            If it is True (only for debug) do not use any noise addition.

        limit_integral_in_lambda_range_nm - if it is None, the integral (convert radiance to electrons) will be on the imager's spectrum.
            If it is a range, force that range.
        Output:
        gray_scales - list of images in grayscale

        radiance_to_graylevel_scale - foalt,
              The scale that would convert radiance to grayscale BUT without noise addition in the electrons lavel
        """
        if limit_integral_in_lambda_range_nm is not None:
            assert len(limit_integral_in_lambda_range_nm) == 2, "Must be 2 elements list."

        max_of_image = image.max()
        gray_level_bound = 2 ** self._sensor.bits

        # -------------------------------------------------------
        # -------------------------------------------------------
        # -------------------------------------------------------
        GAMMA_lambda = self.get_GAMMA_lambda()
        # GAMMA_lambda - EFFICIENCY object.

        # next, calculate the integral of GAMMA_lambda with respect to lambda.
        if (len(self._channels) > 1):
            assert len(self._channels) == image.shape[2], "The image does not match to imagers channels."
        else:
            # interduce theird axis for the channel if there is only one channel
            image = image[..., np.newaxis]

        graylevel_scales = []
        gray_scales = []

        for channel_index, channel in enumerate(self._channels):
            GAMMA_lambda_channel, lambdas_channel = GAMMA_lambda.get_Efficiency(channel)
            LTOA, lambdas_channel_test = self._scene_spectrum.calculate_solar_irradiance_at_channel(channel, self._SZA,
                                                                                                    self._integration_Dlambda)
            assert np.all(lambdas_channel_test == lambdas_channel), "Inconsistency in the wavelengths."
            if np.isscalar(lambdas_channel) or lambdas_channel.size == 1:
                GAMMA_lambda_INTEGRAL_lambda = GAMMA_lambda_channel * LTOA
                GAMMA_lambda_INTEGRAL_lambda = np.asscalar(GAMMA_lambda_INTEGRAL_lambda)
            else:
                if limit_integral_in_lambda_range_nm is not None:
                    E1 = InterpolatedUnivariateSpline(lambdas_channel, GAMMA_lambda_channel)
                    E2 = InterpolatedUnivariateSpline(lambdas_channel, LTOA)
                    step = self._integration_Dlambda  # nm
                    start = limit_integral_in_lambda_range_nm[0]
                    stop = limit_integral_in_lambda_range_nm[1]
                    limited_lambda = np.linspace(start, stop, int(((stop - start) / step) + 1))

                    GAMMA_lambda_channel_interpulated = E1(limited_lambda)
                    LTOA_interpulated = E2(limited_lambda)
                    y = LTOA_interpulated * GAMMA_lambda_channel_interpulated
                    GAMMA_lambda_INTEGRAL_lambda = np.trapz(y, x=limited_lambda)
                else:
                    GAMMA_lambda_INTEGRAL_lambda = np.trapz(GAMMA_lambda_channel * LTOA, x=lambdas_channel)
            # units of INTEGRAL are [electrons*st/sec]

            radiance_to_graylevel_scale = self._sensor.alpha * (
                        1e-6 * self._exposure_time) * GAMMA_lambda_INTEGRAL_lambda
            graylevel_scales.append(radiance_to_graylevel_scale)

            electrons_number = 1e-6 * self._exposure_time \
                               * GAMMA_lambda_INTEGRAL_lambda * image[..., channel_index]
            # The 1e-6* scales electrons_number to units of [electrons]
            if not cancel_noise:
                electrons_number = np.round(electrons_number)
                # add bias:
                electrons_number = self.add_bais(electrons_number)
                # Here is the place to put the noise, since the noise is on the electrons levels:

                electrons_number = self.add_noise(electrons_number)
                # add gain uncertainty
                electrons_number = self.add_gain(electrons_number)
                electrons_number = np.round(electrons_number)

            assert self._sensor.full_well > electrons_number.max(), "You reached full well, maybe you have saturation. Set less exposure time!"
            # convert to grayscale:
            gray_scale = self._sensor.alpha * electrons_number
            # For a sensor having a linear radiometric response, the conversion between pixel electrons to grayscale is by a fixed ratio self._alpha
            # Quantisize and cut overflow values.
            if not cancel_noise:
                gray_scale = np.round(gray_scale).astype(np.int)
                gray_scale = np.clip(gray_scale, a_min=0, a_max=gray_level_bound)

            gray_scales.append(gray_scale)

        gray_scales = np.stack(gray_scales, axis=2)
        radiance_to_graylevel_scales = np.hstack(graylevel_scales)
        return gray_scales, radiance_to_graylevel_scales

    def set_exposure_time(self, time, force_exposur=False):
        """
        This method set the exposure time of the imager. The exposure time must be less than the
        the extrime time (max exposure time).

        Parameters:
        Inpute:
        time - float, units nicro sec.

        """
        if (time is not None):
            if not force_exposur:
                assert time < self._max_exposure_time, "The exposure time must be less than the maximum exposure time!"
            self._exposure_time = time

            if (self._DARK_NOISE is not None):
                DARK_NOISE = self._DARK_NOISE * 1e-6 * self._exposure_time
                self._NOISE_FLOOR = self._READ_NOISE + DARK_NOISE
                self._DR = self._sensor.full_well / self._NOISE_FLOOR

                # SNR:
                QUANTIZATION_NOISE_VARIANCE = self._sensor.QUANTIZATION_NOISE_VARIANCE

                self._SNR = (self._sensor.full_well) / np.sqrt(
                    self._sensor.full_well + DARK_NOISE + (self._READ_NOISE ** 2) + (
                            QUANTIZATION_NOISE_VARIANCE ** 2))  # see https://www.photometrics.com/learn/imaging-topics/signal-to-noise-ratio
                print("----------------------------------------------------------------------------")
                print("--------------------Since you set the exposure time, you get:---------------")
                print("----------------------------------------------------------------------------")
                print("----> Exposure bound is {}[micro sec]".format(float_round(self._max_exposure_time)))
                print("----> You set exposure time to {}[micro sec]".format(float_round(self._exposure_time)))
                print("----> Dynamic range (at full well) changed to {} or {}[db]".format(float_round(self._DR),
                                                                                          float_round(
                                                                                              20 * np.log10(self._DR))))
                print("----> Noise floor changed to {}[electrons]".format(float_round(self._NOISE_FLOOR)))
                print("----> SNR (at full well) changed to {} or {}[db]".format(float_round(self._SNR), float_round(
                    self._sensor.get_SNR_IN_DB())))

    def update_sensor_size_with_number_of_pixels(self, nx, ny):
        """
        Set/update the sensor size by using new [nx,ny] resolution of an Imager. It can be done for instance, if the simulated resolution is smaller than the resolution from a spec.
        TODO - Be carfule here, this method doesn't update any other parameters.
        """
        self._sensor.sensor_size = np.array([nx, ny]) * self._sensor.pixel_size

        self.calculate_footprints()
        # camera FOV:
        # Let self._camera_footprint be the footprint of the camera at nadir view in x axis. The field of view of the camera in radians is,
        self._FOV = 2 * np.arctan(self._camera_footprint / (2 * self._H))
        self.set_gain_uncertainty()

    def set_gain_uncertainty(self, gain_std_percents=0):
        """
        Only sets the gain_uncertainty term not add it to the signal as add_gain does.
        This term will multiply electrons.

        input:
           gain_std_percents - float - max gain in procents.

        """

        nx, ny = self.get_sensor_resolution()
        if gain_std_percents == 0:
            self._gain_uncertainty_term = np.zeros([nx, ny])  # it will multiply electrons
        else:
            self._gain_uncertainty_term = np.random.normal(0.0, gain_std_percents / 100, (nx, ny))  #

        self._gain_std_percents = gain_std_percents

    def reset_gain_uncertainty(self, new_value=None):
        """
        If new_value is none, reset uncertainty with the same std but with new random value.
        Else, use the new_value and save it further.
        """
        if new_value is not None:
            self._gain_std_percents = new_value

        nx, ny = self.get_sensor_resolution()
        if self._gain_std_percents == 0:
            self._gain_uncertainty_term = np.zeros([nx, ny])  # it will multiply electrons
        else:
            self._gain_uncertainty_term = np.random.normal(0.0, self._gain_std_percents / 100, (nx, ny))  #

    def add_gain(self, electrons_number_image):
        """
        Add the gian uncertainty to each pixel. It is random term.
        """
        electrons_number_gain = self._gain_uncertainty_term
        electrons_number_image *= (1.0 + electrons_number_gain)

        return electrons_number_image

    def set_bias_uncertainty(self, global_bias_std_percents=0):
        """
        Only sets the bias uncertainty term not add it to the signal as add_bias does.
        This term will be used in electrons level.

        input:
           global_bias_std_percents - float - random but yet global bias in procents.

        """
        if global_bias_std_percents == 0:
            self._bias_uncertainty_term = 0
        else:
            self._bias_uncertainty_term = \
                global_bias_std_percents * np.random.random_sample() / 100

        self._global_bias_std_percents = global_bias_std_percents

    def reset_bias_uncertainty(self, new_value=None):
        """
        If new_value is none, reset uncertainty with the same std but with new random value.
        Else, use the new_value and save it further.
        """
        if new_value is not None:
            self._global_bias_std_percents = new_value

        if self._global_bias_std_percents == 0:
            self._bias_uncertainty_term = 0
        else:
            self._bias_uncertainty_term = \
                self._global_bias_std_percents * np.random.random_sample() / 100

    def add_bais(self, electrons_number_image):
        """
        Add global bias. It is random term.
        TODO - implement local biases.
        """
        mean_electrons = np.mean(electrons_number_image)
        electrons_number_bias = int(mean_electrons * self._bias_uncertainty_term)
        electrons_number_image += electrons_number_bias

        return electrons_number_image

    def get_sensor_resolution(self):
        """
        Just get the [nx,ny] of the Imager's sensor.
        """
        return [int(i / self._sensor.pixel_size) for i in self._sensor.sensor_size]

    def calculate_footprints(self):
        """
        Calculats footprints in km:
        HEre the footprint is calculated from the pixel size.
        """
        self._pixel_footprint = 1e-3 * (self._H * self._sensor.pixel_size) / self._lens._FOCAL_LENGTH  # km
        self._camera_footprint = 1e-3 * (self._H * self._sensor.sensor_size) / self._lens._FOCAL_LENGTH  # km
        # here self._camera_footprint is a np.array with 2 elements, relative to [H,W]. Which element to take?
        # currently I take the minimal volue:
        self._camera_footprint = max(self._camera_footprint)
        self._diffraction_Dtheta = self._pixel_footprint / self._H

        lens_diameter_diffration = []
        for channel in self._channels:
            lambdas = self._scene_spectrum.get_wavelength_vector(channel, integration_Dlambda=self._integration_Dlambda)
            a = 1e-6 * (self._diffraction_scalar * lambdas) / self._diffraction_Dtheta  # in mm
            lens_diameter_diffration.append(a.max())
        self._lens_diameter_diffration = np.array(lens_diameter_diffration).max()

    def set_system_efficiency(self, val):
        """
        Set the camera system efficiency. The camera system efficiency is due to optics losses and sensor reflection (it is not a part of QE).
        """
        assert val > 1 or val < 0, "system efficiency must be in the [0,1] range."
        self._ETA = val
        self._Imager_EFFICIANCY.multiply_efficiency_by_scalar(self._ETA)
        self._Imager_EFFICIANCY.adjust_to_spectrum(self._scene_spectrum, integration_Dlambda=self._integration_Dlambda)

    def get_system_efficiency(self):
        return self._ETA

    def get_footprints_at_nadir(self):
        """
        Get pixel footprint and camera footprint at nadir view only.
        """
        return self._pixel_footprint, self._camera_footprint

    def IS_HAS_POLARIZATION(self):
        """
        Check if the imager has polarization capabilities.
        Just check if the work 'Polarized' in imager type.
        The user must be awar of that fact.
        """
        if 'Polarized' in self._type:
            return True
        else:
            return False

    @property
    def max_exposure_time(self):
        return self._max_exposure_time

    @property
    def imager_type(self):
        return self._type

    @property
    def info(self):
        imager_type = self.imager_type
        channels = self.channels
        str_ = ''
        for channel in channels:
            spectrum_band_nm = self.get_band_at_channel(channel)
            D_band_nm = spectrum_band_nm[1] - spectrum_band_nm[0]
            str_ += str(channel) + "_{}-{}nm_".format(spectrum_band_nm[0], spectrum_band_nm[1])
        channels = str_.rstrip('_')
        # channels = functools.reduce(operator.add,[str(j)+"_" for j in channels]).rstrip('_')
        return "Imager_type_{}_channels_{}--".format(imager_type, channels)

    def get_centeral_wavelength_in_microns_at_channel(self, channel):
        return self._centeral_wavelength_in_microns[channel]

    def get_band_in_microns_at_channel(self, channel):
        return self._scene_spectrum.get_BAND_microns(channel)

    def get_band_at_channel(self, channel):
        return self._scene_spectrum.get_BAND(channel)

    def split_band_to_mini_bands_in_micron_at_channel(self, channel, mini_band):
        """
        Splits one band to N mini bands with mini_band (input) [nm] band widths.
        Returns:
        mini_spectrum_bands_micron - list of mini bands, i.e. alist of 2 elements lists.
        centeral_wavelengths_micron - list of central wavelengths.
        """

        spectrum_band_nm = self.get_band_at_channel(channel)
        D_band_nm = spectrum_band_nm[1] - spectrum_band_nm[0]
        samples = int(np.ceil(D_band_nm / mini_band) + 1)
        mini_samples_bands_nm = np.arange(spectrum_band_nm[0], spectrum_band_nm[1]
                                          , step=mini_band)
        if mini_samples_bands_nm[-1] is not spectrum_band_nm[1]:
            # there is residual.
            mini_samples_bands_nm = np.append(mini_samples_bands_nm, spectrum_band_nm[1])

        mini_spectrum_bands_micron = []
        centeral_wavelengths_micron = []
        for index, sample in enumerate(mini_samples_bands_nm[:-1]):
            sampled_band_nm = [sample, mini_samples_bands_nm[index + 1]]
            if index < samples - 2:
                assert (sampled_band_nm[1] - sampled_band_nm[0]) == mini_band, "Wrong spliting of bands."
            tmp = [i / 1000 for i in sampled_band_nm]
            mini_spectrum_bands_micron.append(tmp)
            centeral_wavelength = core.get_center_wavelen(tmp[0], tmp[1])
            centeral_wavelength = float_round(centeral_wavelength)
            centeral_wavelengths_micron.append(centeral_wavelength)

        return mini_spectrum_bands_micron, centeral_wavelengths_micron

    def reduce_channels_to_channel(self, channel):
        """
        The function should be used for the minibands rendering and only then.
        It set the channels to be only one channel.
        """
        self._channels = [channel]

    @property
    def scene_spectrum(self):
        return self._scene_spectrum

    @property
    def channels(self):
        return self._channels

    @scene_spectrum.setter
    def scene_spectrum(self, scene_spectrum):
        assert isinstance(scene_spectrum, SPECTRUM), "The spectrum is a must be of type SPECTRUM"
        self._scene_spectrum = scene_spectrum
        self._update()

    @property
    def orbital_speed(self):
        return self._orbital_speed

    @orbital_speed.setter
    def orbital_speed(self, val):
        self._orbital_speed = val  # [km/sec]
        self._update()

    @property
    def max_noise_floor(self):
        return self._NOISE_FLOOR

    @property
    def max_dynamic_range(self):
        return self._DR

    @property
    def electrons2grayscale_factor(self):
        return self._sensor.alpha

    def show_sensor_QE(self):
        self._sensor.QE.show_EFFICIENCY(name='QE')

    @property
    def FOV(self):
        """
        returns the Field of view of a camera.
        """
        return self._FOV  # be carfule, it is in radiance.

    @property
    def gain_std_percents(self):
        """
        TODO
        """
        return self._gain_std_percents

    @property
    def global_bias_std_percents(self):
        """
        TODO
        """
        return self._global_bias_std_percents

    @property
    def sun_zenith(self):
        return self._SZA


# -------------------------------------------------------------
# -------------------------------------------------------------
# ------------END IMAGER---------------------------------------
# -------------------------------------------------------------


def test_polarized_imager():
    """
    The information based on: https://www.flir.eu/products/blackfly-s-gige/?model=BFS-PGE-51S5P-C:

    Dark current noise = ?, since we don't know this info. we use the dark current noise as in Gecko.

    Read noise = 2.31 electrons RMSE

    full well = 10.5 [ke]

    channels = [470, 505, 625]

    BitDepth = 8 or 10 or 12, we choose 10.

    """
    SONY = {'PIXEL_SIZE': 3.45, 'FULLWELL': 10.5e3, 'CHeight': 2048, 'CWidth': 2048,
            'READOUT_NOISE': 2.31, 'DARK_CURRENT_NOISE': 3.51, 'TEMP': 25, 'BitDepth': 10}

    # Define sensor:
    sensor = SensorFPA(PIXEL_SIZE=SONY['PIXEL_SIZE'], FULLWELL=SONY['FULLWELL'], CHeight=SONY['CHeight'],
                       CWidth=SONY['CWidth'],
                       READOUT_NOISE=SONY['READOUT_NOISE'],
                       DARK_CURRENT_NOISE=SONY['DARK_CURRENT_NOISE'], TEMP=SONY['TEMP'], BitDepth=SONY['BitDepth'])

    SENSOR_QE_CSV_FILE_RED = './red.csv'
    SENSOR_QE_CSV_FILE_GREEN = './green.csv'
    SENSOR_QE_CSV_FILE_BLUE = './blue.csv'
    SENSOR_DARK_NOISE_CSV_FILE = 'GECKO_DARK_NOISE.csv'

    qe = EFFICIENCY()  # define QE object
    qe.Load_EFFICIENCY_table(csv_table_path=SENSOR_QE_CSV_FILE_BLUE, channel='blue')
    qe.Load_EFFICIENCY_table(csv_table_path=SENSOR_QE_CSV_FILE_GREEN, channel='green')
    qe.Load_EFFICIENCY_table(csv_table_path=SENSOR_QE_CSV_FILE_RED, channel='red')

    # qe.show_EFFICIENCY() # - just for test
    # set sensor efficiency:
    sensor.set_QE(qe)
    sensor.Load_DARK_NOISE_table(SENSOR_DARK_NOISE_CSV_FILE)
    # sensor.show_QE() # - just for test

    # Define lens:
    lens = LensSimple(FOCAL_LENGTH=86.2, DIAMETER=20.0)  # lengths in mm
    lens_transmission = EFFICIENCY()
    lens_transmission.assume_Efficiency(90, spectrum=[400, 800], channel='wide_range')
    lens.set_TRANSMISSION(lens_transmission)
    # lens.show_TRANSMISSION() # - just for test

    # create imager:
    # set spectrum:
    scene_spectrum = SPECTRUM(channels=['blue', 'green', 'red'], bands=[[450, 500], [520, 570], [620, 670]])
    # merge quantum effciency with the defined spectrum:
    imager = Imager(sensor=sensor, lens=lens, scene_spectrum=scene_spectrum)
    imager.set_solar_beam_zenith_angle(SZA=165)
    imager.change_temperature(15)
    imager.set_Imager_altitude(H=500)  # in km
    imager.IS_VALIDE_LENS_DIAMETER()

    # imager.update()

    ## set geometry:
    # H = 500 # km
    # imager.set_Imager_altitude(H=H)
    ## calclate footprints:
    # pixel_footprint, camera_footprint = imager.get_footprints_at_nadir()
    # max_esposure_time = imager.max_exposure_time
    # pixel_footprint = shdom.float_round(pixel_footprint)
    # camera_footprint = shdom.float_round(camera_footprint)
    # max_esposure_time = shdom.float_round(max_esposure_time)
    # print("At nadir:\n Pixel footprint is {}[km]\n Camera footprint is {}[km]\n Max esposure time {}[micro sec]\n"
    # .format(pixel_footprint, camera_footprint, max_esposure_time))

    # imager.calculate_scene_radiance(TYPE='SHDOM')
    # imager.ExportConfig(file_name = '../CloudCT_notebooks/lusid_like_config_20mGSD.json')

    plt.show()


if __name__ == '__main__':
    test_polarized_imager()