
'''
This script is used to examine and visualize the MAIA time series data.
The MAIA data has been processed and saved in NetCDF files of 6-hourly means for the years 1994-2022.
NWVF is already integrated over the longitude [-45E, 45E] at latitude 70N.
Temperature and sea ice cover are area-weighted means.

It reads the data from NetCDF files, normalizes the data, and plots the time series for three variables: 
- longitudinal integral of NWVF
- area-weighted 2 meter temperature mean
- area-weighted sea ice cover mean.

The script also saves the generated plot as an image file.

Author: Thea Q
Date: [Current Date]
'''

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def NormalizeData(data):
    '''
    Normalize the input data using min-max normalization.

    Parameters:
        data (numpy.ndarray): Input data array.

    Returns:
        numpy.ndarray: Normalized data array.
    '''
    return (data - np.min(data)) / (np.max(data) - np.min(data))

PATH_DATA = '../../Data/Processed_MAIA/'
PATH_FIGS = '../../Figures/'
# File paths for the NetCDF files
fn_nwvf = PATH_DATA + 'MAIA_nwvf_integrals_1994-2022__6hr_mean.nc'
fn_t2m = PATH_DATA + 'MAIA_AreaMean_t2m_1994-2022__6hr_mean.nc'
fn_sic = PATH_DATA + 'MAIA_AreaMean_SIC_1994-2022__6hr_mean.nc'

# Read the NetCDF files
ds_nwvf = nc.Dataset(fn_nwvf)
ds_t2m = nc.Dataset(fn_t2m)
ds_sic = nc.Dataset(fn_sic)

# Convert time values to timestamps
hrs_after_1994 = ds_nwvf['time'][:]
timestamps = pd.to_datetime(hrs_after_1994, unit='h', origin=pd.Timestamp('1994-01-01 02:00:00'))

# Create a figure with subplots
fig, ax = plt.subplots(3, 1, figsize=(12,8), sharex=True, )
fig.suptitle('MAIA time series', fontsize=16)

# Plot the time series for each variable
ax[0].plot(timestamps, ds_nwvf.variables['nwvf_integral'][:], linewidth=0.7, color='k')
ax[0].set_title('Longitudinal integral of NWVF [-45E, 45E] at latitude 70N ')
ax[0].set_ylabel('[kg m$^{-1}$ s$^{-1}$]')

ax[1].plot(timestamps,ds_t2m.variables['areaMean_t2m'][:], linewidth=0.7, color='k')
ax[1].set_title('Area-weighted 2 meter temperature mean')
ax[1].set_ylabel('[K]')

ax[2].plot(timestamps, ds_sic.variables['areaMean_t2m'][:], linewidth=0.7, color='k')
ax[2].set_title('Area-weighted sea ice cover mean')
ax[2].set_xlim([min(timestamps), max(timestamps)])
ax[2].set_xlabel('Year')
ax[2].set_ylabel('[%]')

fig.tight_layout()

# Save the figure as an image file
fig.savefig(PATH_FIGS + 'MAIA_time_series.png', dpi=300)

# Display the plot
plt.show()

