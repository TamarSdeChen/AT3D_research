import numpy as np
import matplotlib.pyplot as plt
from Py6S import *

def run_6s_altitude_comparison():
    # 1. Initialize 6S
    s = SixS()

    # 2. Match AT3D Environment (Mid-Latitude Summer, Rayleigh only)
    s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    s.wavelength = Wavelength(0.445)

    # 3. Match AT3D Ocean Surface (Wind = 5 m/s, Pigment = 0.1)
    s.ground_reflectance = GroundReflectance.HomogeneousOcean(
        wind_speed=5.0, 
        wind_azimuth=0.0, 
        salinity=34.3, 
        pigment_concentration=0.1
    )

    # 4. Target is at sea level
    s.altitudes.set_target_sea_level()

    # 5. Geometry Setup
    sza = 21.99
    sun_azimuth = 0.0
    mu_s = np.cos(np.deg2rad(sza)) # For normalizing reflectance to radiance
    
    view_zeniths = np.linspace(-40, 40, 81)
    
    # Data storage for both altitudes
    radiances_toa = []
    radiances_5km = []

    print(f"Running 6S Sweep for SZA = {sza}°, Wind = 5.0 m/s...")
    print("Calculating for both TOA and 5 km altitude. This may take a moment...")

    for vz in view_zeniths:
        # Determine forward/backward scatter azimuths
        if vz >= 0:
            vza = vz
            vaa = 180.0 # Forward scatter (Glint)
        else:
            vza = abs(vz)
            vaa = 0.0   # Backscatter (Dark ocean)
            
        # Set base geometry
        s.geometry = Geometry.User()
        s.geometry.solar_z = sza
        s.geometry.solar_a = sun_azimuth
        s.geometry.view_z = vza
        s.geometry.view_a = vaa
        s.geometry.month = 6
        s.geometry.day = 21

        # --- RUN 1: Top of Atmosphere (Satellite Level) ---
        s.altitudes.set_sensor_satellite_level()
        s.run()
        # Convert TOA Reflectance to Normalized Radiance
        norm_rad_toa = (s.outputs.pixel_reflectance * mu_s) / np.pi
        radiances_toa.append(norm_rad_toa)

        # --- RUN 2: 5 km Altitude ---
        s.altitudes.set_sensor_custom_altitude(5.0)
        s.run()
        # Convert 5km Reflectance to Normalized Radiance
        norm_rad_5km = (s.outputs.pixel_reflectance * mu_s) / np.pi
        radiances_5km.append(norm_rad_5km)

    # 6. Plot the Results
    plt.figure(figsize=(12, 7))
    
    # Plot both lines
    plt.plot(view_zeniths, radiances_toa, 'b-', linewidth=2.5, label='TOA (Satellite Level)')
    plt.plot(view_zeniths, radiances_5km, 'g--', linewidth=2.5, label='5 km Altitude')
    
    # Mark the specular point
    plt.axvline(x=sza, color='r', linestyle=':', alpha=0.8, label=f'Specular Point (VZA={sza}°)')
    
    plt.title("Ocean Sunglint Normalized Radiance: TOA vs 5 km Altitude", fontsize=15)
    plt.xlabel("Viewing Zenith Angle (Degrees)\n<-- Backscatter (Dark)  |  Forward Scatter (Glint) -->", fontsize=12)
    plt.ylabel("Normalized Radiance ($I / F_0$)", fontsize=12)
    
    plt.grid(True, alpha=0.5)
    plt.legend(fontsize=12)
    plt.xlim(-40, 40)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_6s_altitude_comparison()