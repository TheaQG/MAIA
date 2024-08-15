''' 
    NOT IN USE
    This script is used to process the MAIA data for the temperature and sea ice cover data.
    The data is processed by calculating the area-weighted mean of the sea ice cover data over a specified latitude and longitude range.
    The data is then saved to a new .nc file.
    

'''
import netCDF4 as nc
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob



# Define the path to the data
path = '/Volumes/1TB-FREECOM/MAIA_data/'

# Plot for a single time step?
PLOT_SINGLE_TIME = False

# Define the latitude and longitude of the data to use
lat_use = [90, 70] # degrees North
lon_use = [-45, 45] # degrees West/East

# Specify the latitude and longitude slice (replace with your actual values)
lat_slice = slice(lat_use[0], lat_use[1])  # e.g., slice(60, 70)
lon_slice = slice(lon_use[0], lon_use[1])  # e.g., slice(-45, 45)



# Define the pattern of the files to use
file_pattern = os.path.join(path, 'ERA5_download__sea_ice_cover_*__6hr_mean.nc')

# Lists to store integral and times
temperatures = []
times = []

sorted_file_paths = sorted(glob.glob(file_pattern))
for file_path in sorted_file_paths:

    with xr.open_dataset(file_path) as ds:
        # Store the data of interest in a new variable
        #print(ds)
        t2m = ds['siconc']
        

        # Select the data for the specified slice
        temperature_slice = t2m.sel(latitude=lat_slice, longitude=lon_slice)

        # Calculate the weights for the slice - this assumes that the latitude is in degrees
        weights = np.cos(np.deg2rad(ds['latitude'].sel(latitude=lat_slice)))
        weights.name = "weights"

        # Normalizing weights so they sum to 1 over the slice
        weights /= weights.sum()

        # Calculate the weighted mean over the slice
        t2m_weighted_mean = temperature_slice.weighted(weights).mean(dim=['latitude', 'longitude'])
        
        print('Length of temperatures in file: ' + str(len(t2m_weighted_mean.values)))
        # Append to the list of integrals
        temperatures.append(t2m_weighted_mean.values)

        time = ds['time'][:]
        # Extract the times and store in a list
        times.append(pd.to_datetime(time.values))
        if len(temperatures) == 2:
            print(np.concatenate(temperatures))
        ds.close()


temperatures_array = np.concatenate(temperatures)
times_array = np.concatenate(times)

print(temperatures_array.shape)
print(times_array.shape)

new_ds = xr.Dataset({
    'areaMean_t2m': (['time'], temperatures_array),
    'time': (['time'], times_array)
})

PATH_SAVE = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed'
fn_new = 'MAIA_AreaMean_SIC_1994-2022__6hr_mean.nc'

new_ds.to_netcdf(os.path.join(PATH_SAVE, fn_new))



plt.plot(temperatures_array-273.15)
plt.show()








# lat_use = [90, 70] # degrees North
# lon_use = [-45, 45] # degrees West/East

# # Specify the latitude and longitude slice (replace with your actual values)
# lat_slice = slice(lat_use[0], lat_use[1])  # e.g., slice(60, 70)
# lon_slice = slice(lon_use[0], lon_use[1])  # e.g., slice(-45, 45)

# # Select the data for the specified slice
# temperature_slice = temperature.sel(latitude=lat_slice, longitude=lon_slice)

# # Calculate the weights for the slice - this assumes that the latitude is in degrees
# weights = np.cos(np.deg2rad(ds['latitude'].sel(latitude=lat_slice)))
# weights.name = "weights"

# # Normalizing weights so they sum to 1 over the slice
# weights /= weights.sum()

# # Calculate the weighted mean over the slice
# t2m_weighted_mean = temperature_slice.weighted(weights).mean(dim=['latitude', 'longitude'])


# plt.plot(t2m_weighted_mean.values-273.15)
# plt.show()