import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr

# Import txt file (with header)
data = pd.read_csv('TS_AerosolBurden_ArcticSlice.txt')
# Save data in pd series
EC = data['BC']
t = data['date']

# Add date (from data) to pd.series
t = pd.to_datetime(t)

# Make EC and t into dataframe 
EC = pd.DataFrame(EC)
t = pd.DataFrame(t)

# Merge EC and t into one dataframe
EC_df = pd.concat([t, EC], axis=1)



# Read ERA5 NWVF (6 hourly) data from .nc file
NWVF_ERA5 = xr.open_dataset('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc')

daily_NWVF = NWVF_ERA5['nwvf_integral'].resample(time='1D').sum()
daily_NWVF_df = daily_NWVF.to_dataframe()

daily_NWVF_df.reset_index(inplace=True)

# Filter data to only after 1997 and before 2022-07-01
EC_df = EC_df[EC_df['date'] > '1997-01-01']
EC_df = EC_df[EC_df['date'] < '2022-07-01']
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] < '2022-07-01']
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] > '1997-01-01']



# Plot EC and NWVF in different subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axs[0].plot(EC_df['date'], EC_df['BC'], label='EC', color='k', lw=1)
axs[0].set_title('EC')
axs[0].set_ylabel('EC [kg/m2]')
axs[0].set_xlabel('Time')
axs[0].grid()

axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['nwvf_integral'], label='NWVF', color='k', lw=1)
axs[1].set_title('NWVF')
axs[1].set_ylabel('NWVF [kg/m2]')
axs[1].set_xlabel('Time')
# Set x-axis to [first_day, last_day]
axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()

fig.tight_layout()


# Calculate correlation between EC and NWVF
correlation = np.corrcoef(EC_df['BC'], daily_NWVF_df['nwvf_integral'])
print('Correlation between EC and NWVF:')
print(correlation)


# Compute a weekly moving average of EC and NWVF
EC_df['BC_WA'] = EC_df['BC'].rolling(window=7).mean()
daily_NWVF_df['NWVF_WA'] = daily_NWVF_df['nwvf_integral'].rolling(window=7).mean()

# Compute a weekly average of EC and NWVF (not moving) - different weeks for different years
EC_df_weekly = EC_df.resample('W', on='date').mean()
daily_NWVF_df_weekly = daily_NWVF_df.resample('W', on='time').mean()


# Plot EC and NWVF in different subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axs[0].plot(EC_df['date'], EC_df['BC_WA'], label='EC moving weekly average', color='k', alpha=0.5, lw=1)
axs[0].plot(EC_df_weekly.index, EC_df_weekly['BC'], label='EC weekly average', color='k', lw=1)
axs[0].set_title('EC with 7-day moving average')
axs[0].set_ylabel('EC [kg/m2]')
axs[0].grid()

axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['NWVF_WA'], label='NWVF moving weekly average', color='k', alpha=0.5, lw=1)
axs[1].plot(daily_NWVF_df_weekly.index, daily_NWVF_df_weekly['nwvf_integral'], label='NWVF weekly average', color='k', lw=1)
axs[1].set_title('NWVF with 7-day moving average')
axs[1].set_ylabel('NWVF [kg/m2]')
axs[1].set_xlabel('Time')
# Set x-axis to [first_day, last_day]
axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()

fig.tight_layout()





# Compute a monthly moving average of EC and NWVF
EC_df['BC_MA'] = EC_df['BC'].rolling(window=30).mean()
daily_NWVF_df['NWVF_MA'] = daily_NWVF_df['nwvf_integral'].rolling(window=30).mean()

# Compute a monthly average of EC and NWVF (not moving) - different months for different years
EC_df_monthly = EC_df.resample('M', on='date').mean()
daily_NWVF_df_monthly = daily_NWVF_df.resample('M', on='time').mean()


# Plot EC and NWVF in different subplots
fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

axs[0].plot(EC_df['date'], EC_df['BC_MA'], label='EC', color='k', alpha=0.5, lw=1)
axs[0].plot(EC_df_monthly.index, EC_df_monthly['BC'], label='EC monthly average', color='k', lw=1)
axs[0].set_title('EC with 30-day moving average')
axs[0].set_ylabel('EC [kg/m2]')
axs[0].grid()

axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['NWVF_MA'], label='NWVF', color='k', alpha=0.5, lw=1)
axs[1].plot(daily_NWVF_df_monthly.index, daily_NWVF_df_monthly['nwvf_integral'], label='NWVF monthly average', color='k', lw=1)
axs[1].set_title('NWVF with 30-day moving average')
axs[1].set_ylabel('NWVF [kg/m2]')
axs[1].set_xlabel('Time')
# Set x-axis to [first_day, last_day]
axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()

fig.tight_layout()




# Use low-pass filter to smooth the data (NWVF only)
from scipy.signal import butter, filtfilt

def butter_lowpass_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Filter requirements
order = 6
fs = 1       # sample rate, days
cutoff = 1/30  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter(order, cutoff, btype='low', analog=False)

# Apply the filter to NWVF data
NWVF_filtered = butter_lowpass_filter(daily_NWVF_df['nwvf_integral'], cutoff, fs, order)

# Calculate a weekly average of the filtered NWVF data
NWVF_filtered_weekly = pd.DataFrame(NWVF_filtered, columns=['nwvf_integral'])
# Add time column to NWVF_filtered_weekly (same as daily_NWVF_df['time'])
NWVF_filtered_weekly['time'] = daily_NWVF_df['time']
print(NWVF_filtered_weekly)
NWVF_filtered_weekly = NWVF_filtered_weekly.resample('W', on='time').mean()

print(NWVF_filtered_weekly)

# Plot the filtered NWVF data and EC data in different plots
fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

axs[0].plot(EC_df['date'], EC_df['BC'], label='EC', color='k', lw=1)
axs[0].set_title('EC')
axs[0].set_ylabel('EC [kg/m2]')
axs[0].set_xlabel('Time')
axs[0].grid()

axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['nwvf_integral'], label='NWVF', color='k', alpha=0.5, lw=1)
axs[1].plot(daily_NWVF_df['time'], NWVF_filtered, label='NWVF filtered', color='k', lw=1)
axs[1].plot(NWVF_filtered_weekly.index, NWVF_filtered_weekly['nwvf_integral'], label='NWVF filtered weekly average', color='b', lw=1, linestyle='--')
axs[1].set_title('NWVF filtered')
axs[1].set_ylabel('NWVF [kg/m2]')
# Set x-axis to [first_day, last_day]
axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()

axs[2].plot(NWVF_filtered_weekly.index, NWVF_filtered_weekly['nwvf_integral'], label='NWVF filtered weekly average', color='k', lw=1, linestyle='--')
axs[2].set_title('NWVF filtered')
axs[2].set_ylabel('NWVF [kg/m2]')
# Set x-axis to [first_day, last_day]
axs[2].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[2].grid()

fig.tight_layout()

plt.show()


