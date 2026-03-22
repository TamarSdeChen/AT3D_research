import sys
import os
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
import mayavi.mlab as mlab

# Custom imports
import at3d

# Assumes this path exists in your environment
sys.path.append('../CloudCT_utils/')
try:
    from CloudCTUtils import *
except ImportError:
    print("Warning: CloudCTUtils not found. Ensure the path is correct.")

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class SimConfig:
    # Satellite Orbit
    Rsat: float = 500  # km
    GSD: float = 0.5 * 0.02  # km
    sats_number: int = 10
    
    # Sun
    wavelengths_micron: float = 0.672
    sun_azimuth: float = np.random.uniform(-179, 179)
    sun_zenith: float = 180 - 21.993519 # np.random.uniform(140, 179)
    sun_azimuth: float = 0
    #sun_zenith: float = 150
    solar_flux: float = 1.0
    
    # Geometry / Grid
    dx: float = 0.2
    dy: float = 0.2
    dz: float = 0.2
    dx: float = 0.02
    dy: float = 0.02
    dz: float = 0.02    
    nx: int = 32
    ny: int = 32
    nz: int = 32
    
    # Computation
    n_jobs: int = 60
    maxiter: int = 150
    
    # Paths
    path_atmosphere: str = '../data/ancillary/AFGL_summer_mid_lat.nc'
    path_mie: str = '../mie_tables'

    @property
    def solarmu(self):
        return np.cos(np.deg2rad(self.sun_zenith))

# ==========================================
# 2. Geometry & Grid Helpers
# ==========================================
def create_grid(cfg: SimConfig):
    """Generates the 3D meshgrid and the base cloud shape (blob)."""
    xgrid = np.linspace(0, float_round(cfg.dx * cfg.nx) - cfg.dx, cfg.nx)
    ygrid = np.linspace(0, float_round(cfg.dy * cfg.ny) - cfg.dy, cfg.ny)
    zgrid = np.linspace(0, float_round(cfg.dz * cfg.nz) - cfg.dz, cfg.nz)

    X, Y, Z = np.meshgrid(xgrid, ygrid, zgrid, indexing='ij')

    # CREATE AN EMPTY MASK - NO CLOUDS
    mask = np.zeros_like(X)
    mask [1,1,1] = 1
        
    # Physical properties
    lwc = 0.1 * mask
    veff = 0.1 * mask
    reff = 10.0 * mask

    return xgrid, ygrid, zgrid, lwc, reff, veff

def get_lookat_point(cfg: SimConfig):
    """Calculates the LookAt point (bottom center of medium)."""
    return np.array([0.5 * cfg.nx * cfg.dx, 0.5 * cfg.ny * cfg.dy, 0.68 * cfg.nz * cfg.dz])


def pack_scatterer(dx, nx, dy, ny, zgrid, lwc, reff, veff):
    """Packs arrays into an AT3D Scatterer Grid object."""
    # Create empty grid
    scatterer = at3d.grid.make_grid(dx, nx, dy, ny, zgrid)
    
    # Populate data (handling non-zeros to avoid memory waste/density issues)
    # Initialize with NaNs
    shape = (scatterer.sizes['x'], scatterer.sizes['y'], scatterer.sizes['z'])
    
    for name, data in zip(['density', 'reff', 'veff'], [lwc, reff, veff]):
        field_container = np.full(shape, np.nan)
        indices = np.where(data > 0)
        field_container[indices] = data[indices]
        scatterer[name] = (['x', 'y', 'z'], field_container)
        
    return scatterer

# ==========================================
# 3. Sensor Setup
# ==========================================
def create_sensor_dict(cfg: SimConfig, lookat):
    """
    Creates the string of pearls sensor setup.
    If rotation_matrix is provided, it applies the transformation to the sensors.
    """
    
    # Calculate FOV and Resolution based on footprint
    L = max(cfg.nx * cfg.dx, cfg.ny * cfg.dy) * 5 # Tuned L
    fov = 2 * np.rad2deg(np.arctan(0.5 * L / cfg.Rsat))
    cny = int(np.floor(L / cfg.GSD))
    cnx = int(1 * np.floor(L / cfg.GSD))

    # Generate String of Pearls positions
    # Note: Using CENTER_OF_MEDIUM for Nadir calculation logic from original code
    center_ground = [0.5 * cfg.nx * cfg.dx, 0.5 * cfg.ny * cfg.dy, 0]
    
    sat_positions, _, _, _ = StringOfPearls(
        SATS_NUMBER=cfg.sats_number,
        orbit_altitude=cfg.Rsat,
        move_nadir_x=center_ground[0],
        move_nadir_y=center_ground[1]
    )

    sensor_dict = at3d.containers.SensorsDict()
    up_list = np.array(len(sat_positions) * [0, 1, 0]).reshape(-1, 3)


    for i, (pos, up) in enumerate(zip(sat_positions, up_list)):
        
        curr_pos = pos
        curr_lookat = lookat
        curr_up = up

        sensor = at3d.sensor.perspective_projection(
            wavelength=cfg.wavelengths_micron,
            fov=fov,
            x_resolution=cnx,
            y_resolution=cny,
            position_vector=curr_pos,
            lookat_vector=curr_lookat,
            up_vector=curr_up,
            stokes=['I']        )
        sensor_dict.add_sensor('CloudCT', sensor)

    return sensor_dict

# ==========================================
# 4. Radiative Transfer Core
# ==========================================
def prepare_optical_properties(cfg: SimConfig, scatterer, zgrid):
    """Generates optical properties from LWC/Reff using Mie tables."""
    
    # 1. Load Atmosphere & Merge Grids
    atmos = xr.open_dataset(cfg.path_atmosphere)
    reduced_atmos = atmos.sel({'z': atmos.coords['z'].data[atmos.coords['z'].data <= 4.0]})
    merged_z = at3d.grid.combine_z_coordinates([reduced_atmos, scatterer])
    
    rte_grid = at3d.grid.make_grid(
        cfg.dx, scatterer.x.data.size,
        cfg.dy, scatterer.y.data.size,
        merged_z
    )
    
    # Resample Cloud to RTE grid
    scatterer_rte = at3d.grid.resample_onto_grid(rte_grid, scatterer)

    # 2. Mie Tables
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
    at3d.checks.check_optical_properties(opt_props[cfg.wavelengths_micron])
    
    # 4. Rayleigh
    rayleigh = at3d.rayleigh.to_grid([cfg.wavelengths_micron], atmos, rte_grid)
    
    return opt_props, rayleigh

def run_simulation(cfg: SimConfig, scatterer, sensor_dict, sun_azimuth, sun_zenith, zgrid, surface_type="lambertian"):
    """
    Sets up the solver and runs the simulation for a specific configuration.
    Accepts 'lambertian' or 'ocean' as surface_type.
    """
    
    # 1. Get Optical Properties
    opt_props, rayleigh = prepare_optical_properties(cfg, scatterer, zgrid)
    
    # 2. Source Definition
    solarmu = np.cos(np.deg2rad(sun_zenith))
    source = at3d.source.solar(
        wavelength=cfg.wavelengths_micron,
        solarmu=solarmu,
        solar_azimuth=sun_azimuth,
        solarflux=cfg.solar_flux, 
        skyrad=0.0
    )

    # 3. Configure Solvers
    solvers = at3d.containers.SolversDict()
    wvl = cfg.wavelengths_micron
    
    medium = {
        # 'cloud': opt_props[wvl], # Cloud is empty anyway
        'rayleigh': rayleigh[wvl]
    }
    
    numerical_config = at3d.configuration.get_config()
    
    # --- CONFIGURE SURFACE DYNAMICALLY ---
    if surface_type == "ocean":
        print("Configuring surface: Ocean Unpolarized (Cox-Munk)")
        ocean_wind_speed = 5.0 # m/s
        surface_model = at3d.surface.ocean_unpolarized(ocean_wind_speed,0.1)
    else:
        print("Configuring surface: Lambertian")
        surface_albedo = 0.05 # Typical dark ocean diffuse albedo
        surface_model = at3d.surface.lambertian(surface_albedo)
        
    solver = at3d.solver.RTE(
        numerical_params=numerical_config,
        surface=surface_model,
        source=source,
        medium=medium,
        num_stokes=1
    )
    solvers.add_solver(wvl, solver)

    # 4. Execute
    print(f"Generating Sun at azimuth {sun_azimuth} deg, and zenith {sun_zenith}...")    
    print(f"Solving RTE for {len(sensor_dict['CloudCT']['sensor_list'])} sensors...")
    sensor_dict.get_measurements(solvers, n_jobs=cfg.n_jobs, verbose=True)
    
    return sensor_dict.get_images('CloudCT')

# ==========================================
# 5. Visualization
# ==========================================
def plot_comparison(images1, images2, channels=['I', 'Q', 'U']):
    """Plots original vs transformed images and their difference."""
    
    n_sats = len(images1)
    ncols = n_sats
    nrows = 3 # Org, Transformed, Diff
    
    for channel in channels:
        fig = plt.figure(figsize=(20, 10))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        
        # Determine global min/max for scaling
        all_data = []
        for img_set in [images1, images2]:
            for s_img in img_set:
                all_data.append(s_img[channel].T.data)
        
        vmax = np.max(all_data)
        vmin = 0 if channel == 'I' else np.min(all_data)
        cmap = 'gray'

        for i in range(n_sats):
            d1 = images1[i][channel].T.data
            d2 = images2[i][channel].T.data
            
            # Row 1: Original
            ax1 = fig.add_subplot(nrows, ncols, i + 1)
            im1 = ax1.imshow(d1, cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.set_title(f"Sat {i}", fontsize=10)
            ax1.axis('off')
            
            # Row 2: Transformed
            ax2 = fig.add_subplot(nrows, ncols, i + 1 + ncols)
            im2 = ax2.imshow(d2, cmap=cmap, vmin=vmin, vmax=vmax)
            ax2.axis('off')
            
            # Row 3: Difference (%)
            # Avoid division by zero
            safe_d1 = np.where(np.abs(d1) < 1e-9, 1e-9, d1)
            diff = 100 * np.abs(d1 - d2) / np.abs(safe_d1)
            
            ax3 = fig.add_subplot(nrows, ncols, i + 1 + 2*ncols)
            im3 = ax3.imshow(diff, cmap='jet') # Heatmap for error
            ax3.set_title("Diff %", fontsize=8)
            ax3.axis('off')
            
            # Colorbars (only on last column to save space)
            if i == n_sats - 1:
                plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
                plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        fig.suptitle(f"Stokes Parameter: {channel}", size=16)
        

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    # ---  Visualization (Optional) ---
    IFVISUALIZE = True
    
    # --- Init ---
    cfg = SimConfig()
    lookat_pt = get_lookat_point(cfg)
    
    # --- 1. Generate Base Scene ---
    print("Generating base scene...")
    xgrid, ygrid, zgrid, lwc, reff, veff = create_grid(cfg)
    base_scatterer = pack_scatterer(cfg.dx, cfg.nx, cfg.dy, cfg.ny, zgrid, lwc, reff, veff)
    base_sensors = create_sensor_dict(cfg, lookat_pt)
    
    
    # --- 2. Run Simulations ---
    print("\n>>> Running Simulation 1: LAMBERTIAN SURFACE")
    imgs_lambertian = run_simulation(cfg, base_scatterer, base_sensors, 
                                     cfg.sun_azimuth, cfg.sun_zenith, zgrid, 
                                     surface_type="lambertian")

    print("\n>>> Running Simulation 2: OCEAN GLINT SURFACE")
    imgs_ocean = run_simulation(cfg, base_scatterer, base_sensors, 
                                cfg.sun_azimuth, cfg.sun_zenith, zgrid, 
                                surface_type="ocean")    
    
    # --- 3. Compare Results ---
    print("Plotting results...")
    # Note for plot_comparison: 
    # Row 1 will be Lambertian
    # Row 2 will be Ocean
    # Row 3 will be the Difference
    channels = ['I'] # only show not polarized
    plot_comparison(imgs_lambertian, imgs_ocean, channels=channels)
        
    
    print("Done.")
    plt.show()