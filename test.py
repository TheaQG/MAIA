import netCDF4 as nc
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob



# Full ERA5, on DANRA grid (2048x2048) - To be saved in /Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_FullERA5_danra_grid
fn_ERA5_DG = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/ERA5/1991/Daily/data_t2m/ERA5_t2m_daily_1991_danraGrid.nc'
# ERA5 on full DANRA grid (589x789) - To be saved in /Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5_danra_grid
fn_era5_small = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/ERA5/1991/Daily/data_t2m/ERA5_t2m_daily_1991_small.nc'
# ERA5 on full ERA5 grid (321x601) - To be saved in /Users/au728490/Documents/PhD_AU/Python_Scripts/Data/Data_DiffMod/data_ERA5_era5_grid
fn_ERA5 = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/ERA5/1991/Daily/data_t2m/ERA5_t2m_daily_1991.nc'

data_ERA5_DG = nc.Dataset(fn_ERA5_DG)['t2m'][0,:,:]
print(data_ERA5_DG.shape)
data_ERA5 = nc.Dataset(fn_ERA5)['t2m'][0,:,:]
print(data_ERA5.shape)
data_ERA5_small = nc.Dataset(fn_era5_small)['t2m'][0,:,:]
print(data_ERA5_small.shape)

fig, ax = plt.subplots(1,3, figsize=(15,5))
ax[0].imshow(np.flipud(data_ERA5_DG))
ax[1].imshow(data_ERA5)
ax[2].imshow(np.flipud(data_ERA5_small))

fig.tight_layout()
plt.show()


'''
Dataloader for ERA5 data should have option to crop to smaller grid.
Also save ERA5 to a coarser grid to save space. Then can be interpolated to necesary resolution in dataloader/in model.
'''