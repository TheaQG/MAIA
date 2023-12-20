import netCDF4 as nc
import numpy as np
import pandas as pd
#import cartopy.crs as ccrs
#import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob

PLOT_FIG = True
VERBOSE = True

# Import NWVF and aerosol data
# NWVF
nwvf_EC_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/nwvf_EC_Villum.csv', index_col=0, parse_dates=True)
nwvf_EC_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/nwvf_EC_Zeppelin.csv', index_col=0, parse_dates=True)
nwvf_SO4_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/nwvf_SO4_Villum.csv', index_col=0, parse_dates=True)
nwvf_SO4_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/nwvf_SO4_Zeppelin.csv', index_col=0, parse_dates=True)

# NWVF full dataset
nwvf_ts_raw = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/nwvf_ts.csv', index_col=0, parse_dates=True)

# Remove dates where value larger than 0.6e6 (outliers)
nwvf_ts = nwvf_ts_raw[nwvf_ts_raw['nwvf'] < 0.6e6]

# Compute a 7-day rolling mean
nwvf_ts_roll = nwvf_ts.rolling(7).mean()

# Plot
fig, ax = plt.subplots(3,1, figsize=(15,8), sharex=True)
ax[0].plot(nwvf_ts_raw, color='k', linewidth=0.7)
ax[0].set_title('NWVF raw')
ax[1].plot(nwvf_ts, color='k', linewidth=0.7)
ax[1].set_title('NWVF filtered')
ax[2].plot(nwvf_ts_roll, color='k', linewidth=0.7)
ax[2].set_title('NWVF rolling mean')
ax[2].set(xlabel='Date')
fig.tight_layout()

# Aerosol
df_EC_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/df_EC_Villum.csv', index_col=0, parse_dates=True)
df_EC_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/df_EC_Zeppelin.csv', index_col=0, parse_dates=True)
df_SO4_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/df_SO4_Villum.csv', index_col=0, parse_dates=True)
df_SO4_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/df_SO4_Zeppelin.csv', index_col=0, parse_dates=True)



# Create climatology
# NWVF full dataset, monthly and weekly
# To datetime
nwvf_ts.index = pd.to_datetime(nwvf_ts.index)
nwvf_ts_clim = nwvf_ts.groupby(nwvf_ts.index.month).mean()
nwvf_ts_clim_std = nwvf_ts.groupby(nwvf_ts.index.month).std()

nwvf_ts_clim_week = nwvf_ts.groupby(nwvf_ts.index.week).mean()
nwvf_ts_clim_week_std = nwvf_ts.groupby(nwvf_ts.index.week).std()

nwvf_ts_clim_day = nwvf_ts.groupby(nwvf_ts.index.day_of_year).mean()
nwvf_ts_clim_day_std = nwvf_ts.groupby(nwvf_ts.index.day_of_year).std()

fig, ax = plt.subplots(3,1, figsize=(15,8))
ax[0].plot(nwvf_ts_clim.index, nwvf_ts_clim['nwvf'], color='k', linewidth=1)
ax[0].errorbar(nwvf_ts_clim.index, nwvf_ts_clim['nwvf'], yerr=nwvf_ts_clim_std['nwvf'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4)
ax[0].set_title('Monthly climatology')
ax[0].set(xlabel='Month', ylabel='NWVF')
ax[1].plot(nwvf_ts_clim_week.index, nwvf_ts_clim_week['nwvf'], color='k', linewidth=1)
ax[1].errorbar(nwvf_ts_clim_week.index, nwvf_ts_clim_week['nwvf'], yerr=nwvf_ts_clim_week_std['nwvf'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4)
ax[1].set_title('Weekly climatology')
ax[1].set(xlabel='Week', ylabel='NWVF')
ax[2].plot(nwvf_ts_clim_day.index, nwvf_ts_clim_day['nwvf'], color='k', linewidth=1)
ax[2].errorbar(nwvf_ts_clim_day.index, nwvf_ts_clim_day['nwvf'], yerr=nwvf_ts_clim_day_std['nwvf'], fmt='.', color='k', capsize=1, elinewidth=0.5, ms=4)
ax[2].set_title('Daily climatology')
ax[2].set(xlabel='Day of year', ylabel='NWVF')
fig.tight_layout()
#plt.show()




#######################
# MONTHLY CLIMATOLOGY #
#######################

# NWVF
nwvf_EC_Villum_clim = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.month).mean()
nwvf_EC_Villum_clim_std = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.month).std()
nwvf_EC_Zeppelin_clim = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.month).mean()
nwvf_EC_Zeppelin_clim_std = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.month).std()
nwvf_SO4_Villum_clim = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.month).mean()
nwvf_SO4_Villum_clim_std = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.month).std()
nwvf_SO4_Zeppelin_clim = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.month).mean()
nwvf_SO4_Zeppelin_clim_std = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.month).std()

# Aerosol
df_EC_Villum_clim = df_EC_Villum.groupby(df_EC_Villum.index.month).mean()
df_EC_Villum_clim_std = df_EC_Villum.groupby(df_EC_Villum.index.month).std()
df_EC_Zeppelin_clim = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.month).mean()
df_EC_Zeppelin_clim_std = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.month).std()
df_SO4_Villum_clim = df_SO4_Villum.groupby(df_SO4_Villum.index.month).mean()
df_SO4_Villum_clim_std = df_SO4_Villum.groupby(df_SO4_Villum.index.month).std()
df_SO4_Zeppelin_clim = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.month).mean()
df_SO4_Zeppelin_clim_std = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.month).std()


# Plot
fig, ax = plt.subplots(2,2, figsize=(15,8), sharex=True)
ax[0,0].plot(nwvf_EC_Villum_clim.index, nwvf_EC_Villum_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[0,0].errorbar(nwvf_EC_Villum_clim.index, nwvf_EC_Villum_clim['nwvf_time_integral'], yerr=nwvf_EC_Villum_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,0].legend(loc='upper left')
ax[0,0].set_title('Villum EC')
ax2 = ax[0,0].twinx()
ax2.plot(df_EC_Villum_clim.index, df_EC_Villum_clim['EC'], color='r', linewidth=1, label='EC')
#ax2.errorbar(df_EC_Villum_clim.index, df_EC_Villum_clim['EC'], yerr=df_EC_Villum_clim_std['EC'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[0,1].plot(nwvf_EC_Zeppelin_clim.index, nwvf_EC_Zeppelin_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[0,1].errorbar(nwvf_EC_Zeppelin_clim.index, nwvf_EC_Zeppelin_clim['nwvf_time_integral'], yerr=nwvf_EC_Zeppelin_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,1].legend(loc='upper left')
ax[0,1].set_title('Zeppelin EC')
ax2 = ax[0,1].twinx()
ax2.plot(df_EC_Zeppelin_clim.index, df_EC_Zeppelin_clim['EC'], color='r', linewidth=1, label='EC')
#ax2.errorbar(df_EC_Zeppelin_clim.index, df_EC_Zeppelin_clim['EC'], yerr=df_EC_Zeppelin_clim_std['EC'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[1,0].plot(nwvf_SO4_Villum_clim.index, nwvf_SO4_Villum_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[1,0].errorbar(nwvf_SO4_Villum_clim.index, nwvf_SO4_Villum_clim['nwvf_time_integral'], yerr=nwvf_SO4_Villum_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,0].legend(loc='upper left')
ax[1,0].set_title('Villum SO4')
ax2 = ax[1,0].twinx()
ax2.plot(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], color='r', linewidth=1, label='SO4')
#ax2.errorbar(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], yerr=df_SO4_Villum_clim_std['SO4'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

ax[1,1].plot(nwvf_SO4_Zeppelin_clim.index, nwvf_SO4_Zeppelin_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[1,1].errorbar(nwvf_SO4_Zeppelin_clim.index, nwvf_SO4_Zeppelin_clim['nwvf_time_integral'], yerr=nwvf_SO4_Zeppelin_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,1].legend(loc='upper left')
ax[1,1].set_title('Zeppelin SO4')
ax2 = ax[1,1].twinx()
ax2.plot(df_SO4_Zeppelin_clim.index, df_SO4_Zeppelin_clim['SO4'], color='r', linewidth=1, label='SO4')
#ax2.errorbar(df_SO4_Zeppelin_clim.index, df_SO4_Zeppelin_clim['SO4'], yerr=df_SO4_Zeppelin_clim_std['SO4'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

fig.tight_layout()
#plt.show()




######################
# WEEKLY CLIMATOLOGY #
######################

# NWVF
nwvf_EC_Villum_clim = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.week).mean()
nwvf_EC_Villum_clim_std = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.week).std()
nwvf_EC_Zeppelin_clim = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.week).mean()
nwvf_EC_Zeppelin_clim_std = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.week).std()
nwvf_SO4_Villum_clim = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.week).mean()
nwvf_SO4_Villum_clim_std = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.week).std()
nwvf_SO4_Zeppelin_clim = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.week).mean()
nwvf_SO4_Zeppelin_clim_std = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.week).std()

# Aerosol
df_EC_Villum_clim = df_EC_Villum.groupby(df_EC_Villum.index.week).mean()
df_EC_Villum_clim_std = df_EC_Villum.groupby(df_EC_Villum.index.week).std()
df_EC_Zeppelin_clim = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.week).mean()
df_EC_Zeppelin_clim_std = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.week).std()
df_SO4_Villum_clim = df_SO4_Villum.groupby(df_SO4_Villum.index.week).mean()
df_SO4_Villum_clim_std = df_SO4_Villum.groupby(df_SO4_Villum.index.week).std()
df_SO4_Zeppelin_clim = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.week).mean()
df_SO4_Zeppelin_clim_std = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.week).std()


# Plot
fig, ax = plt.subplots(2,2, figsize=(15,8), sharex=True)
ax[0,0].plot(nwvf_EC_Villum_clim.index, nwvf_EC_Villum_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[0,0].errorbar(nwvf_EC_Villum_clim.index, nwvf_EC_Villum_clim['nwvf_time_integral'], yerr=nwvf_EC_Villum_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,0].legend(loc='upper left')
ax[0,0].set_title('Villum EC')
ax2 = ax[0,0].twinx()
ax2.plot(df_EC_Villum_clim.index, df_EC_Villum_clim['EC'], color='r', linewidth=1, label='EC')
#ax2.errorbar(df_EC_Villum_clim.index, df_EC_Villum_clim['EC'], yerr=df_EC_Villum_clim_std['EC'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[0,1].plot(nwvf_EC_Zeppelin_clim.index, nwvf_EC_Zeppelin_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[0,1].errorbar(nwvf_EC_Zeppelin_clim.index, nwvf_EC_Zeppelin_clim['nwvf_time_integral'], yerr=nwvf_EC_Zeppelin_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,1].legend(loc='upper left')
ax[0,1].set_title('Zeppelin EC')
ax2 = ax[0,1].twinx()
ax2.plot(df_EC_Zeppelin_clim.index, df_EC_Zeppelin_clim['EC'], color='r', linewidth=1, label='EC')
#ax2.errorbar(df_EC_Zeppelin_clim.index, df_EC_Zeppelin_clim['EC'], yerr=df_EC_Zeppelin_clim_std['EC'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[1,0].plot(nwvf_SO4_Villum_clim.index, nwvf_SO4_Villum_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[1,0].errorbar(nwvf_SO4_Villum_clim.index, nwvf_SO4_Villum_clim['nwvf_time_integral'], yerr=nwvf_SO4_Villum_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,0].legend(loc='upper left')
ax[1,0].set_title('Villum SO4')
ax2 = ax[1,0].twinx()
ax2.plot(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], color='r', linewidth=1, label='SO4')
#ax2.errorbar(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], yerr=df_SO4_Villum_clim_std['SO4'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

ax[1,1].plot(nwvf_SO4_Zeppelin_clim.index, nwvf_SO4_Zeppelin_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[1,1].errorbar(nwvf_SO4_Zeppelin_clim.index, nwvf_SO4_Zeppelin_clim['nwvf_time_integral'], yerr=nwvf_SO4_Zeppelin_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,1].legend(loc='upper left')
ax[1,1].set_title('Zeppelin SO4')
ax2 = ax[1,1].twinx()
ax2.plot(df_SO4_Zeppelin_clim.index, df_SO4_Zeppelin_clim['SO4'], color='r', linewidth=1, label='SO4')
#ax2.errorbar(df_SO4_Zeppelin_clim.index, df_SO4_Zeppelin_clim['SO4'], yerr=df_SO4_Zeppelin_clim_std['SO4'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

fig.tight_layout()
#plt.show()



#####################
# DAILY CLIMATOLOGY #
#####################

# NWVF
nwvf_EC_Villum_clim = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.day_of_year).mean()
nwvf_EC_Villum_clim_std = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.day_of_year).std()
nwvf_EC_Zeppelin_clim = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.day_of_year).mean()
nwvf_EC_Zeppelin_clim_std = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.day_of_year).std()
nwvf_SO4_Villum_clim = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.day_of_year).mean()
nwvf_SO4_Villum_clim_std = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.day_of_year).std()
nwvf_SO4_Zeppelin_clim = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.day_of_year).mean()
nwvf_SO4_Zeppelin_clim_std = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.day_of_year).std()

# Aerosol
df_EC_Villum_clim = df_EC_Villum.groupby(df_EC_Villum.index.day_of_year).mean()
df_EC_Villum_clim_std = df_EC_Villum.groupby(df_EC_Villum.index.day_of_year).std()
df_EC_Zeppelin_clim = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.day_of_year).mean()
df_EC_Zeppelin_clim_std = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.day_of_year).std()
df_SO4_Villum_clim = df_SO4_Villum.groupby(df_SO4_Villum.index.day_of_year).mean()
df_SO4_Villum_clim_std = df_SO4_Villum.groupby(df_SO4_Villum.index.day_of_year).std()
df_SO4_Zeppelin_clim = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.day_of_year).mean()
df_SO4_Zeppelin_clim_std = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.day_of_year).std()


# Plot
fig, ax = plt.subplots(2,2, figsize=(15,8), sharex=True)
ax[0,0].plot(nwvf_EC_Villum_clim.index, nwvf_EC_Villum_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[0,0].errorbar(nwvf_EC_Villum_clim.index, nwvf_EC_Villum_clim['nwvf_time_integral'], yerr=nwvf_EC_Villum_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,0].legend(loc='upper left')
ax[0,0].set_title('Villum EC')
ax2 = ax[0,0].twinx()
ax2.plot(df_EC_Villum_clim.index, df_EC_Villum_clim['EC'], color='r', linewidth=1, label='EC')
#ax2.errorbar(df_EC_Villum_clim.index, df_EC_Villum_clim['EC'], yerr=df_EC_Villum_clim_std['EC'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[0,1].plot(nwvf_EC_Zeppelin_clim.index, nwvf_EC_Zeppelin_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[0,1].errorbar(nwvf_EC_Zeppelin_clim.index, nwvf_EC_Zeppelin_clim['nwvf_time_integral'], yerr=nwvf_EC_Zeppelin_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,1].legend(loc='upper left')
ax[0,1].set_title('Zeppelin EC')
ax2 = ax[0,1].twinx()
ax2.plot(df_EC_Zeppelin_clim.index, df_EC_Zeppelin_clim['EC'], color='r', linewidth=1, label='EC')
#ax2.errorbar(df_EC_Zeppelin_clim.index, df_EC_Zeppelin_clim['EC'], yerr=df_EC_Zeppelin_clim_std['EC'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[1,0].plot(nwvf_SO4_Villum_clim.index, nwvf_SO4_Villum_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[1,0].errorbar(nwvf_SO4_Villum_clim.index, nwvf_SO4_Villum_clim['nwvf_time_integral'], yerr=nwvf_SO4_Villum_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,0].legend(loc='upper left')
ax[1,0].set_title('Villum SO4')
ax2 = ax[1,0].twinx()
ax2.plot(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], color='r', linewidth=1, label='SO4')
#ax2.errorbar(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], yerr=df_SO4_Villum_clim_std['SO4'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

ax[1,1].plot(nwvf_SO4_Zeppelin_clim.index, nwvf_SO4_Zeppelin_clim['nwvf_time_integral'], color='k', linewidth=1, label='NWVF')
#ax[1,1].errorbar(nwvf_SO4_Zeppelin_clim.index, nwvf_SO4_Zeppelin_clim['nwvf_time_integral'], yerr=nwvf_SO4_Zeppelin_clim_std['nwvf_time_integral'], fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,1].legend(loc='upper left')
ax[1,1].set_title('Zeppelin SO4')
ax2 = ax[1,1].twinx()
ax2.plot(df_SO4_Zeppelin_clim.index, df_SO4_Zeppelin_clim['SO4'], color='r', linewidth=1, label='SO4')
#ax2.errorbar(df_SO4_Zeppelin_clim.index, df_SO4_Zeppelin_clim['SO4'], yerr=df_SO4_Zeppelin_clim_std['SO4'], fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

fig.tight_layout()
plt.show()