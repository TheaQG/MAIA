'''

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr


SAVE_FIGS = False



aerosol_str = 'TotalMass'
region_str = '_small'
n_window = 14
shift = 100#int(n_window/2) 

PATH_DATA = '../../Data/'
PATH_FIGS = '../../Figures/AverageAndShifts/' + aerosol_str + '/'

# Import txt file (with header)
data = pd.read_csv(PATH_DATA + 'ModelData/TS_AerosolBurden_ArcticSlice' + region_str + '.txt')



# Save data in pd series (most important aerosols: ['BC', 'Dust', 'SO4', 'TotalMass'])
t = data['date']

# Add date (from data) to pd.series
t = pd.to_datetime(t)

# Make aerosol and t into dataframe 
t = pd.DataFrame(t)

# If 'TotalMass' is chosen, the sum of all aerosols is calculated
if aerosol_str == 'TotalMass':
    aerosol = data['BC'] + data['Dust'] + data['SO4']
    # Add 'TotalMass' header
    aerosol = pd.DataFrame(aerosol, columns=['TotalMass'])
else:
    aerosol = data[aerosol_str]
# Merge aerosol and t into one dataframe
aerosol_df = pd.concat([t, aerosol], axis=1)



# Read ERA5 NWVF (6 hourly) data from .nc file
NWVF_ERA5 = xr.open_dataset(PATH_DATA + 'Processed_MAIA/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc')

daily_NWVF = NWVF_ERA5['nwvf_integral'].resample(time='1D').sum()
daily_NWVF_df = daily_NWVF.to_dataframe()

daily_NWVF_df.reset_index(inplace=True)

# Filter data to only after 1997 and before 2022-07-01
aerosol_df = aerosol_df[aerosol_df['date'] > '1997-01-01']
aerosol_df = aerosol_df[aerosol_df['date'] < '2022-07-01']
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] < '2022-07-01']
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] > '1997-01-01']

# Calculate correlation between aerosol and NWVF
correlation = np.corrcoef(aerosol_df[aerosol_str], daily_NWVF_df['nwvf_integral'])
print(f'\nCorrelation between {aerosol_str} and NWVF (raw):')
print(correlation)
print('\n')



















# Plot aerosol and NWVF in different subplots
fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

axs[0].plot(aerosol_df['date'], aerosol_df[aerosol_str], label='Aerosol', color='teal', lw=0.8)
axs[0].set_title(f'Aerosol (correlation with NWVF: {correlation[0, 1]:.4f})')
axs[0].set_ylabel('aerosol [kg/m2]')
axs[0].grid()
axs[0].legend()

axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['nwvf_integral'], label='NWVF', color='teal', lw=0.8)
axs[1].set_title('NWVF')
axs[1].set_ylabel('NWVF [kg/m2]')
axs[1].set_xlabel('Time')
# Set x-axis to [first_day, last_day]
axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()
axs[1].legend()

fig.tight_layout()

# Save the figure as an image file
if SAVE_FIGS:
    plt.savefig(PATH_FIGS + aerosol_str + '_NWVF_raw_comparison' + region_str + '.png')
























# Compute a weekly moving average of aerosol and NWVF
aerosol_df['BC_WA'] = aerosol_df[aerosol_str].rolling(window=n_window).mean()
daily_NWVF_df['NWVF_WA'] = daily_NWVF_df['nwvf_integral'].rolling(window=n_window).mean()

# Compute a weekly average of aerosol and NWVF (not moving) - different weeks for different years
aerosol_df_weekly = aerosol_df.resample('W', on='date').mean()
daily_NWVF_df_weekly = daily_NWVF_df.resample('W', on='time').mean()

# Calculate correlation between aerosol and NWVF
correlation_weekly = np.corrcoef(aerosol_df_weekly[aerosol_str], daily_NWVF_df_weekly['nwvf_integral'])
print(f'\nCorrelation between aerosol and NWVF ({n_window}-day average):')
print(correlation_weekly)
print('\n')

# Calculate correlation between aerosol and NWVF (moving average)
correlation_moving = np.corrcoef(aerosol_df['BC_WA'][n_window-1:], daily_NWVF_df['NWVF_WA'][n_window-1:])
print(f'\nCorrelation between aerosol and NWVF ({n_window}-day moving average):')
print(correlation_moving)
print('\n')


# Plot aerosol and NWVF in different subplots
fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

axs[0].plot(aerosol_df_weekly.index, aerosol_df_weekly[aerosol_str], label=f'Aerosol {n_window}-day average (correlation with NWVF: {correlation_weekly[0, 1]:.4f})', color='teal', alpha=1, lw=0.8)#, marker='.', ms=2)
axs[0].plot(aerosol_df['date'], aerosol_df['BC_WA'], label=f'Aerosol moving {n_window}-day average (correlation with NWVF: {correlation_moving[0, 1]:.4f})', color='orangered', lw=1)
axs[0].set_title(f'{aerosol_str} with {n_window}-day moving average')
axs[0].set_ylabel(f'{aerosol_str} [kg/m2]')
axs[0].grid()
axs[0].legend()

axs[1].plot(daily_NWVF_df_weekly.index, daily_NWVF_df_weekly['nwvf_integral'], label=f'NWVF {n_window}-day average', color='teal', alpha=1, lw=0.8)#, marker='.', ms=2)
axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['NWVF_WA'], label=f'NWVF moving {n_window}-day average', color='orangered', lw=1)
axs[1].set_title(f'NWVF with {n_window}-day moving average')
axs[1].set_ylabel('NWVF [kg/m2]')
axs[1].set_xlabel('Time')
# Set x-axis to [first_day, last_day]
axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()
axs[1].legend()

fig.tight_layout()

# Save the figure as an image file
if SAVE_FIGS:
    plt.savefig(PATH_FIGS + aerosol_str + '_NWVF_' + str(n_window) + '_day_average' + region_str + '.png')














# Shift the N-day average of NWVF by int(N/2) days to the left
daily_NWVF_df['NWVF_WA_shifted'] = daily_NWVF_df['NWVF_WA'].shift(-shift)

# Calculate correlation between aerosol and NWVF (N-day moving average) (only for the overlapping period)
correlation_moving_shifted = np.corrcoef(aerosol_df['BC_WA'][n_window-1:-shift], daily_NWVF_df['NWVF_WA_shifted'][n_window-1:-shift])

print(f'\nCorrelation between aerosol and NWVF ({n_window}-day moving average, shifted by {shift} days):')
print(correlation_moving_shifted)

# Plot aerosol and NWVF in different subplots
fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

axs[0].plot(aerosol_df_weekly.index, aerosol_df_weekly[aerosol_str], label=f'Aerosol {n_window}-day average (correlation with NWVF: {correlation_weekly[0, 1]:.4f})', color='teal', alpha=1, lw=0.8)
axs[0].plot(aerosol_df['date'], aerosol_df['BC_WA'], label=f'Aerosol moving {n_window}-day average (correlation with NWVF: {correlation_moving_shifted[0, 1]:.4f})', color='orangered', lw=1)
axs[0].set_title(f'{aerosol_str} with {n_window}-day moving average')
axs[0].set_ylabel(f'{aerosol_str} [kg/m2]')
axs[0].grid()
axs[0].legend()

axs[1].plot(daily_NWVF_df_weekly.index, daily_NWVF_df_weekly['nwvf_integral'], label=f'NWVF {n_window}-day average', color='teal', alpha=0.8, lw=0.8)
axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['NWVF_WA_shifted'], label=f'NWVF moving {n_window}-day average (shifted by {shift} days)', color='orangered', lw=1)
axs[1].set_title(f'NWVF with {n_window}-day moving average (shifted by {shift} days)')
axs[1].set_ylabel('NWVF [kg/m2]')
axs[1].set_xlabel('Time')
# Set x-axis to [first_day, last_day]

axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
axs[1].grid()
axs[1].legend()

fig.tight_layout()


# Save the figure as an image file
if SAVE_FIGS:
    plt.savefig(PATH_FIGS + aerosol_str + '_NWVF_' + str(n_window) + '_day_average_' + str(shift) + '_days_shift' + region_str + '.png')
































# # Use low-pass filter to smooth the data (NWVF only)
# from scipy.signal import butter, filtfilt

# def butter_lowpass_filter(data, cutoff, fs, order):
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

# # Filter requirements
# order = 6
# fs = 1       # sample rate, days
# cutoff = 1/30  # desired cutoff frequency of the filter, Hz

# # Get the filter coefficients so we can check its frequency response.
# b, a = butter(order, cutoff, btype='low', analog=False)

# # Apply the filter to NWVF data
# NWVF_filtered = butter_lowpass_filter(daily_NWVF_df['nwvf_integral'], cutoff, fs, order)

# # Calculate a weekly average of the filtered NWVF data
# NWVF_filtered_weekly = pd.DataFrame(NWVF_filtered, columns=['nwvf_integral'])
# # Add time column to NWVF_filtered_weekly (same as daily_NWVF_df['time'])
# NWVF_filtered_weekly['time'] = daily_NWVF_df['time']
# print(NWVF_filtered_weekly)
# NWVF_filtered_weekly = NWVF_filtered_weekly.resample('W', on='time').mean()

# print(NWVF_filtered_weekly)

# # Plot the filtered NWVF data and EC data in different plots
# fig, axs = plt.subplots(3, 1, figsize=(20, 9), sharex=True)

# axs[0].plot(EC_df['date'], EC_df[aerosol_str], label='Aerosol', color='k', lw=1)
# axs[0].set_title('Aerosol')
# axs[0].set_ylabel('Aerosol [kg/m2]')
# axs[0].set_xlabel('Time')
# axs[0].grid()
# axs[0].legend()

# axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['nwvf_integral'], label='NWVF', color='k', alpha=0.5, lw=1)
# axs[1].plot(daily_NWVF_df['time'], NWVF_filtered, label='NWVF filtered', color='k', lw=1)
# axs[1].plot(NWVF_filtered_weekly.index, NWVF_filtered_weekly['nwvf_integral'], label='NWVF filtered weekly average', color='b', lw=1, linestyle='--')
# axs[1].set_title('NWVF filtered with Butter low-pass filter')
# axs[1].set_ylabel('NWVF [kg/m2]')
# # Set x-axis to [first_day, last_day]
# axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
# axs[1].grid()
# axs[1].legend()

# axs[2].plot(NWVF_filtered_weekly.index, NWVF_filtered_weekly['nwvf_integral'], label='NWVF filtered weekly average', color='k', lw=1, linestyle='--')
# axs[2].set_title('NWVF low-pass filtered weekly average')
# axs[2].set_ylabel('NWVF [kg/m2]')
# # Set x-axis to [first_day, last_day]
# axs[2].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
# axs[2].grid()
# axs[2].legend()

# fig.tight_layout()

# # Save the figure as an image file
# if SAVE_FIGS:
#     plt.savefig(PATH_FIGS + aerosol_str + '_NWVF_ButterFilter_' + region_str + '.png')

plt.show()


