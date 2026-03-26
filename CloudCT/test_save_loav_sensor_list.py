import sys
import os
import warnings
import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io as sio
from scipy.ndimage import affine_transform
import xarray as xr
import netCDF4 as nc

# Custom imports
import at3d

# Assumes this path exists in your environment
sys.path.append('../CloudCT_utils/')
try:
    from CloudCTUtils import *
except ImportError:
    print("Warning: CloudCTUtils not found. Ensure the path is correct.")

# ==========================================
# 1. IO Functions (Save / Load)
# ==========================================
def save_sensor_list(file_name, sensor_list):
    if not isinstance(sensor_list, list):
        raise TypeError("`sensor_list` must be a standard Python list.")

    initial_file_name = file_name
    base_name, ext = os.path.splitext(initial_file_name)
    counter = 1
    
    while os.path.exists(file_name):
        file_name = f"{base_name}_{counter}{ext}"
        counter += 1

    if file_name != initial_file_name:
        warnings.warn(f"Saving to alternate file: '{file_name}'", category=RuntimeWarning)

    save_file = nc.Dataset(file_name, 'w')
    save_file.close()

    for i, image in enumerate(sensor_list):
        group_path = f"sensor_list/{i}"
        image.to_netcdf(file_name, mode='a', group=group_path)

    print(f" -> Successfully saved {len(sensor_list)} sensors to '{file_name}'")
    return file_name


def load_sensor_list(file_name):
    dataset = nc.Dataset(file_name)

    if 'sensor_list' not in dataset.groups:
        raise KeyError(f"No 'sensor_list' group found in {file_name}")

    list_group = dataset.groups['sensor_list'].groups
    loaded_list = []

    indices = sorted([int(k) for k in list_group.keys()])

    for i in indices:
        sensor_dataset = xr.open_dataset(
            xr.backends.NetCDF4DataStore(dataset[f'sensor_list/{i}'])
        )
        
        # Restore boolean types
        if 'stokes' in sensor_dataset:
            sensor_dataset['stokes'] = (['stokes_index'], sensor_dataset['stokes'].data.astype(bool))
        if 'use_subpixel_rays' in sensor_dataset:
            sensor_dataset['use_subpixel_rays'] = sensor_dataset['use_subpixel_rays'].data.astype(bool)
            
        loaded_list.append(sensor_dataset)

    print(f" -> Successfully loaded {len(loaded_list)} sensors from '{file_name}'")
    return loaded_list

# ==========================================
# 2. Configuration
# ==========================================
@dataclass
class SimConfig:
    Rsat: float = 500  # km
    GSD: float = 0.5 * 0.02  # km
    sats_number: int = 10
    
    wavelengths_micron: float = 0.672
    sun_azimuth: float = 0.0
    sun_zenith: float = 180 - 21.993519
    solar_flux: float = 1.0
    
    dx: float = 0.02
    dy: float = 0.02
    dz: float = 0.02    
    nx: int = 32
    ny: int = 32
    nz: int = 32
    
    n_jobs: int = 60
    maxiter: int = 150
    
    path_atmosphere: str = '../data/ancillary/AFGL_summer_mid_lat.nc'
    path_mie: str = '../mie_tables'

    @property
    def solarmu(self):
        return np.cos(np.deg2rad(self.sun_zenith))

# ==========================================
# 3. Geometry & Grid Helpers
# ==========================================
def create_grid(cfg: SimConfig):
    xgrid = np.linspace(0, float_round(cfg.dx * cfg.nx) - cfg.dx, cfg.nx)
    ygrid = np.linspace(0, float_round(cfg.dy * cfg.ny) - cfg.dy, cfg.ny)
    zgrid = np.linspace(0, float_round(cfg.dz * cfg.nz) - cfg.dz, cfg.nz)
    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')

    mask = np.zeros_like(X)
    mask[1,1,1] = 1 # Tiny arbitrary point just so scatterer isn't fully empty
        
    lwc = 0.1 * mask
    veff = 0.1 * mask
    reff = 10.0 * mask
    return xgrid, ygrid, zgrid, lwc, reff, veff

def get_lookat_point(cfg: SimConfig):
    return np.array([0.5 * cfg.nx * cfg.dx, 0.5 * cfg.ny * cfg.dy, 0.68 * cfg.nz * cfg.dz])

def pack_scatterer(dx, nx, dy, ny, zgrid, lwc, reff, veff):
    scatterer = at3d.grid.make_grid(dx, nx, dy, ny, zgrid)
    shape = (scatterer.sizes['x'], scatterer.sizes['y'], scatterer.sizes['z'])
    
    for name, data in zip(['density', 'reff', 'veff'], [lwc, reff, veff]):
        field_container = np.full(shape, np.nan)
        indices = np.where(data > 0)
        field_container[indices] = data[indices]
        scatterer[name] = (['x', 'y', 'z'], field_container)
        
    return scatterer

def create_sensor_dict(cfg: SimConfig, lookat):
    L = max(cfg.nx * cfg.dx, cfg.ny * cfg.dy) * 5 
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / cfg.Rsat))
    cny = int(np.floor(L / cfg.GSD))
    cnx = int(1 * np.floor(L / cfg.GSD))

    center_ground = [0.5 * cfg.nx * cfg.dx, 0.5 * cfg.ny * cfg.dy, 0]
    sat_positions, _, _, _ = StringOfPearls(
        SATS_NUMBER=cfg.sats_number, orbit_altitude=cfg.Rsat,
        move_nadir_x=center_ground[0], move_nadir_y=center_ground[1]
    )

    sensor_dict = at3d.containers.SensorsDict()
    up_list = np.array(len(sat_positions) * [0, 1, 0]).reshape(-1, 3)

    for i, (pos, up) in enumerate(zip(sat_positions, up_list)):
        sensor = at3d.sensor.perspective_projection(
            wavelength=cfg.wavelengths_micron, fov=fov,
            x_resolution=cnx, y_resolution=cny,
            position_vector=pos, lookat_vector=lookat,
            up_vector=up, stokes=['I']
        )
        sensor_dict.add_sensor('CloudCT', sensor)

    return sensor_dict

# ==========================================
# 4. Radiative Transfer Core
# ==========================================
def prepare_optical_properties(cfg: SimConfig, scatterer, zgrid):
    atmos = xr.open_dataset(cfg.path_atmosphere)
    reduced_atmos = atmos.sel({'z': atmos.coords['z'].data[atmos.coords['z'].data <= 4.0]})
    merged_z = at3d.grid.combine_z_coordinates([reduced_atmos, scatterer])
    
    rte_grid = at3d.grid.make_grid(cfg.dx, scatterer.x.data.size, cfg.dy, scatterer.y.data.size, merged_z)
    scatterer_rte = at3d.grid.resample_onto_grid(rte_grid, scatterer)

    safe_mkdirs(cfg.path_mie)
    wavelength_band = (cfg.wavelengths_micron, cfg.wavelengths_micron)
    
    mie_table = at3d.mie.get_mono_table(
        'Water', wavelength_band,
        max_integration_radius=65.0,
        minimum_effective_radius=0.1,
        relative_dir=cfg.path_mie,
        verbose=False
    )
    
    # 3. Optical Property Generator
    gen = at3d.medium.OpticalPropertyGenerator(
        'cloud',
        {cfg.wavelengths_micron: mie_table},
        at3d.size_distribution.gamma,
        particle_density=1.0,
        interpolation_mode='exact',
        density_normalization='density',
        reff=np.linspace(1, 30.0, 50),
        veff=np.linspace(0.01, 0.15, 15)
    )
    
    
    
    opt_props = gen(scatterer_rte)
    rayleigh = at3d.rayleigh.to_grid([cfg.wavelengths_micron], atmos, rte_grid)
    
    return opt_props, rayleigh

def run_simulation(cfg: SimConfig, scatterer, sensor_dict, sun_azimuth, sun_zenith, zgrid, surface_type="lambertian"):
    opt_props, rayleigh = prepare_optical_properties(cfg, scatterer, zgrid)
    
    solarmu = np.cos(np.deg2rad(sun_zenith))
    source = at3d.source.solar(
        wavelength=cfg.wavelengths_micron, solarmu=solarmu,
        solar_azimuth=sun_azimuth, solarflux=cfg.solar_flux, skyrad=0.0
    )

    solvers = at3d.containers.SolversDict()
    wvl = cfg.wavelengths_micron
    
    medium = {
        #'cloud': opt_props[wvl], 
        'rayleigh': rayleigh[wvl]
    }
    
    numerical_config = at3d.configuration.get_config()
    
    if surface_type == "ocean":
        surface_model = at3d.surface.ocean_unpolarized(5.0, 0.1)
    else:
        surface_model = at3d.surface.lambertian(0.05)
        
    solver = at3d.solver.RTE(
        numerical_params=numerical_config, surface=surface_model,
        source=source, medium=medium, num_stokes=1
    )
    solvers.add_solver(wvl, solver)

    print(f"Generating Sun at azimuth {sun_azimuth} deg, zenith {sun_zenith}...")    
    print(f"Solving RTE for {len(sensor_dict['CloudCT']['sensor_list'])} sensors...")
    sensor_dict.get_measurements(solvers, n_jobs=cfg.n_jobs, verbose=True)
    
    return sensor_dict.get_images('CloudCT')

# ==========================================
# 5. Visualization
# ==========================================
def plot_comparison(images1, images2, channels=['I']):
    n_sats = len(images1)
    ncols = n_sats
    nrows = 3 
    
    for channel in channels:
        fig = plt.figure(figsize=(18, 9))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        all_data = []
        for img_set in [images1, images2]:
            for s_img in img_set:
                all_data.append(s_img[channel].data)
        
        vmax = np.max(all_data)
        vmin = 0 if channel == 'I' else np.min(all_data)
        cmap = 'gray'

        for i in range(n_sats):
            d1 = images1[i][channel].data
            d2 = images2[i][channel].data
            
            # Row 1: Original
            ax1 = fig.add_subplot(nrows, ncols, i + 1)
            im1 = ax1.imshow(d1, cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_title(f"Sat {i}", fontsize=10)
            ax1.axis('off')
            
            # Row 2: Loaded
            ax2 = fig.add_subplot(nrows, ncols, i + 1 + ncols)
            im2 = ax2.imshow(d2, cmap=cmap, vmin=vmin, vmax=vmax)
            ax2.axis('off')
            
            # Row 3: Difference (%)
            safe_d1 = np.where(np.abs(d1) < 1e-3, 1e-3, d1)
            diff = 100 * np.abs(d1 - d2) / np.abs(safe_d1)
            
            ax3 = fig.add_subplot(nrows, ncols, i + 1 + 2*ncols)
            im3 = ax3.imshow(diff, cmap='jet') 
            ax3.set_title("Diff %", fontsize=8)
            ax3.axis('off')
            
            if i == n_sats - 1:
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        fig.suptitle(f"Save/Load Verification - Stokes: {channel}", size=16)

# ==========================================
# 6. Main Execution (The IO Test)
# ==========================================
if __name__ == "__main__":
    
    # --- 1. Init & Setup ---
    cfg = SimConfig()
    lookat_pt = get_lookat_point(cfg)
    
    print("Generating base scene...")
    xgrid, ygrid, zgrid, lwc, reff, veff = create_grid(cfg)
    base_scatterer = pack_scatterer(cfg.dx, cfg.nx, cfg.dy, cfg.ny, zgrid, lwc, reff, veff)
    base_sensors = create_sensor_dict(cfg, lookat_pt)
    
    # --- 2. Run Simulation ---
    print("\n>>> Running Simulation: OCEAN GLINT SURFACE")
    original_images = run_simulation(cfg, base_scatterer, base_sensors, 
                                     cfg.sun_azimuth, cfg.sun_zenith, zgrid, 
                                     surface_type="ocean")   
    
    # --- 3. Test Save & Load ---
    test_filename = "ocean_test_output.nc"
    
    print("\n>>> Testing File I/O...")
    saved_file = save_sensor_list(test_filename, original_images)
    loaded_images = load_sensor_list(saved_file)

    # --- 4. Plot Verification ---
    print("\nPlotting results (Row 1: Original, Row 2: Loaded, Row 3: Difference)...")
    
    # Compare Original against Loaded
    plot_comparison(original_images, loaded_images, channels=['I'])
        
    print("Done. If difference plots are empty, save/load was flawless.")
    plt.show()