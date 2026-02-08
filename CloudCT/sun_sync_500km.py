import csv
import numpy as np
from skyfield.api import load, EarthSatellite, Topos
from skyfield.constants import AU_KM
from datetime import datetime, timedelta, timezone

# 1. Setup Constants and Orbit Parameters
ts = load.timescale()
data = load('de421.bsp')
earth_obj = data['earth']
sun_obj = data['sun']

altitude_km = 500.0
earth_radius_km = 6378.137
semi_major_axis = earth_radius_km + altitude_km

# Calculate required inclination for Sun-synchronous orbit
# Formula: cos(i) = - (2/3) * (R_earth / a)^3.5 * (T_sunsync / T_J2)
# For 500km, this is roughly 97.4 degrees
inclination = 97.40  

# 2. Define the "Satellite"
# We'll use a TLE-like structure to define the orbit in Skyfield
line1 = '1 99999U          26036.00000000  .00000000  00000-0  00000-0 0    01'
line2 = f'2 99999 {inclination:8.4f} 000.0000 0001000 000.0000 000.0000 15.21900000    01'
satellite = EarthSatellite(line1, line2, 'SSO_500km', ts)

# 3. Simulation Time Setup
# One orbit at 500km takes ~94.6 minutes
start_time = ts.utc(2026, 2, 5, 12, 0, 0)
minutes = np.arange(0, 95, 1) # Measure every minute for one orbit
times = start_time + (minutes / 1440.0)

# 4. Calculate Sun Geometry
print(f"{'Time':<20} | {'Lat':<8} | {'Lon':<8} | {'Sun Zenith':<12} | {'Sun Azimuth':<12}")
print("-" * 75)
time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_path = f"/wdata/tamarsd/AT3D_research/CloudCT/Figures/sun_sync/sun_sync_500km_results_{time_stap}.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Time", "Lat", "Lon", "Sun_Zenith_deg", "Sun_Azimuth_deg"])

    for t in times:
        # Get satellite position in Geocentric coordinates (ITRF)
        geocentric = satellite.at(t)
        subpoint = geocentric.subpoint()

        # Create a temporary "Topos" (observer) at the satellite's current location
        # Note: altitude is 0 because we want the angle relative to the ground point directly below
        observer_loc = Topos(latitude_degrees=subpoint.latitude.degrees,
                             longitude_degrees=subpoint.longitude.degrees,
                             elevation_m=altitude_km * 1000)

        # Calculate Sun position relative to the satellite
        # We observe the Sun from the satellite's position
        difference = (earth_obj + observer_loc).at(t).observe(sun_obj).apparent()
        alt, az, distance = difference.altaz()

        # Zenith angle = 90 - Altitude
        zenith = 90.0 - alt.degrees

        time_str = t.utc_strftime('%H:%M:%S')
        lat_deg = subpoint.latitude.degrees
        lon_deg = subpoint.longitude.degrees
        az_deg = az.degrees

        print(f"{time_str:<20} | {lat_deg:>7.2f} | {lon_deg:>7.2f} | {zenith:>11.2f}° | {az_deg:>11.2f}°")
        writer.writerow([time_str, f"{lat_deg:.2f}", f"{lon_deg:.2f}", f"{zenith:.2f}", f"{az_deg:.2f}"])