
import netCDF4 as nc
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


fn_nwvf = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc'
fn_t2m = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/MAIA_AreaMean_t2m_1994-2022__6hr_mean.nc'
fn_sic = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/MAIA_AreaMean_SIC_1994-2022__6hr_mean.nc'


ds_nwvf = nc.Dataset(fn_nwvf)
ds_t2m = nc.Dataset(fn_t2m)
ds_sic = nc.Dataset(fn_sic)

hrs_after_1994 = ds_nwvf['time'][:]
timestamps = pd.to_datetime(hrs_after_1994, unit='h', origin=pd.Timestamp('1994-01-01 02:00:00'))

fig, ax = plt.subplots(3, 1, figsize=(12,8), sharex=True, )
fig.suptitle('MAIA time series', fontsize=16)

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

fig.savefig('/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/MAIA_time_series.png', dpi=300)
plt.show()

# print(ds_nwvf['time'][:])
# ds_nwvf_norm = NormalizeData(ds_nwvf.variables['nwvf_integral'][:])
# ds_t2m = NormalizeData(ds_t2m.variables['areaMean_t2m'][:])
# ds_sic = NormalizeData(ds_sic.variables['areaMean_t2m'][:])

# fig, ax = plt.subplots(1,1, figsize=(12,4))

# ax.plot(timestamps, ds_nwvf_norm, linewidth=0.7, color='g', label='NWVF')
# ax.set_xlabel('Time')
# ax.set_ylabel('Normalized NWVF', color='g')

# ax2 = ax.twinx()
# ax2.plot(timestamps, ds_t2m+1.5, linewidth=0.7, color='k', label='T2m')
# ax2.set_ylabel('Normalized T2m', color='k')
# ax2.set_ylim([0, 2.5])
# #ax.plot(ds_sic, linewidth=0.7, color='b', label='SIC')
# #ax.set_xlim([1994,2023])

# ax.set_title('MAIA time series normalized')

# # Remove labels on y-axis
# #ax.set_yticklabels([])
# # Set the x-axis tick labels to beggining of every year
# #ax.set_xticks(np.arange(0, len(ds_nwvf.variables['nwvf_integral'][:]), 365*4))

# fig.tight_layout()
# plt.show()

