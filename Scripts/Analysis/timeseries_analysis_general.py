'''
    Script to analyze NWVF and any other timeseries data in general.
'''

import os

import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp

from utils import Climatology, EventAnalysis
# Save figures?
SAVE_FIGS = True

##################################
# LOADING AND PREPROCESSING DATA #
##################################

# Define start and end date
start_date_str = '1997-01-01'
end_date_str = '2013-12-31'

# Define paths to data and figures
PATH_DATA = '../../Data/'
PATH_FIGS = '../../Figures/AnomalyTimeseriesAnalysis/'
# Create folder if it does not exist
if not os.path.exists(PATH_FIGS):
    os.makedirs(PATH_FIGS)

nwvf_filename = 'ModelData/TS_Clouds_ArcticSlice.txt' #'Processed_MAIA/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc'
model_data_filename = 'ModelData/TS_Clouds_ArcticSlice.txt'


# NWVF DATA

# Read ERA5 NWVF (6 hourly) data from .nc file
NWVF_ERA5 = pd.read_csv(PATH_DATA + nwvf_filename) #xr.open_dataset(PATH_DATA + nwvf_filename)

print(NWVF_ERA5)
# Resample NWVF data to daily data and save in dataframe

#daily_NWVF = NWVF_ERA5['nwvf_integral'].resample(time='1D').sum()

daily_NWVF = NWVF_ERA5['pvq']#.resample(time='1D').sum()
daily_dates = NWVF_ERA5['date']#.resample(time='1D').mean()
print(daily_NWVF)
# Create a dataframe with the daily NWVF data
daily_NWVF_df = pd.DataFrame(daily_NWVF)
# Change the name of the column to 'nwvf_integral'
daily_NWVF_df.columns = ['nwvf_integral']

daily_NWVF_df['time'] = daily_dates
# To datetime
daily_NWVF_df['time'] = pd.to_datetime(daily_NWVF_df['time'])

# Reset index of daily_NWVF_df
daily_NWVF_df.reset_index(inplace=True)

# Filter data to only after 1997 and before 2022-07-01
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] >= start_date_str]
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] <= end_date_str]

print(daily_NWVF_df)

# MODEL DATA
# Read model data
model_data = pd.read_csv(PATH_DATA + model_data_filename)

# Convert date column to datetime
model_data['date'] = pd.to_datetime(model_data['date'])

# Filter data to only after 1997 and before 2022-07-01
model_data = model_data[model_data['date'] >= start_date_str]
model_data = model_data[model_data['date'] <= end_date_str]

# IF MORE FILTERING OF DATA NEEDED, DO IT HERE (e.g., selecting only some columns, etc.)

# Get the number of columns in the model data to know how many timeseries we have
N_timeseries = model_data.shape[1] - 1

# Select N_timeseries nice colors for plotting
colors = plt.cm.get_cmap('tab10', N_timeseries)


# Save time data in datetime format in dataframe and crop data to only after 1997 and before 2022-07-01
t = pd.DataFrame(pd.to_datetime(model_data['date']))
t = t[t['date'] >= start_date_str]
t = t[t['date'] <= end_date_str]






###############################
# CLIMATOLOGIES AND ANOMALIES #
###############################

# Number of days to smooth over
n_days_clim = 30

climatology_NWVF = Climatology(daily_NWVF_df, 'nwvf_integral', 'time', n_days=n_days_clim)
NWVF_climatology_smooth = climatology_NWVF.climatology_smooth
NWVF_anomalies = climatology_NWVF.calculate_anomalies(daily_NWVF_df)


# Set up figures to plot timeseries, climatologies, timeseries of anomalies with historgrams on side

# FIGURE 1: Timeseries of NWVF and model data
fig1, axs1 = plt.subplots(N_timeseries+1, 1, figsize=(15, N_timeseries*3), sharex=True)
fig1.suptitle('Timeseries of NWVF and model data')
for i, ax in enumerate(axs1):
    print('\n\n')
    print(i)
    print('\n\n')
    if i == 0:
        ax.plot(daily_NWVF_df['time'], daily_NWVF_df['nwvf_integral'], label='NWVF', color='black', linewidth=0.5)
        ax.set_ylabel('NWVF')
    else:
        ax.plot(model_data['date'], model_data[model_data.columns[i]], label=model_data.columns[i], linewidth=0.5, color=colors(i-1))
        ax.set_ylabel(model_data.columns[i])
    ax.set_xlim([model_data['date'].iloc[0], model_data['date'].iloc[-1]])
    
    ax.grid()
    if i == len(axs1) - 1:
        ax.set_xlabel('Date')






# FIGURE 2: Climatologies of NWVF and model data
fig2, axs2 = plt.subplots(N_timeseries+1, 1, figsize=(15, N_timeseries*3), sharex=True)
fig2.suptitle('Climatologies of NWVF and model data')
# Set x-axis to be days of the year
for i, ax in enumerate(axs2):

    if i == 0:
        ax.set_ylabel('NWVF')
    else:
        ax.set_ylabel(model_data.columns[i])
    if i == len(axs2) - 1:
        ax.set_xlabel('Day of the year')

    ax.set_xticks(np.arange(0, 366, 30))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan'])
    ax.set_xlim([1, 366])
    # Set grid
    ax.grid()
    # Set labels and titles 
    ax.set_xlabel('Day of the year')





# FIGURE 3: Timeseries of anomalies of NWVF and model data with histograms on side
fig3, axs3 = plt.subplots(N_timeseries+1, 2, figsize=(15, N_timeseries*3), sharey='row', gridspec_kw={'width_ratios': [3, 1], 'wspace': 0}, sharex='col')
fig3.suptitle('Anomalies of NWVF and model data')
# Set x-axis to be days of the year
for i, ax in enumerate(axs3):
    if i == 0:
        ax[0].set_ylabel('NWVF')
    else:
        ax[0].set_ylabel(model_data.columns[i])
    if i == len(axs3) - 2:
        ax[0].set_xlabel('Date')
    if i == len(axs3) - 1:
        ax[1].set_xlabel('Density')
    ax[0].grid()
    ax[1].grid()

    # set x-axis of time series to time interval only
    ax[0].set_xlim([model_data['date'].iloc[0], model_data['date'].iloc[-1]])






# Start with plotting NWVF, climatology
axs2[0].plot(NWVF_climatology_smooth.index, NWVF_climatology_smooth['nwvf_integral', 'mean'], label='NWVF', color='black', lw=1)

print(NWVF_anomalies)
axs3[0, 0].plot(NWVF_anomalies['time'], NWVF_anomalies['anomaly'], label='NWVF', color='black', lw=1)
axs3[0, 1].hist(NWVF_anomalies['anomaly'], bins=30, color='black', alpha=0.5, orientation='horizontal', label='NWVF') # density=True,

model_data_anomaly_dfs = []
model_data_climatology_smooth_dfs = []

for i, data_str in enumerate(model_data.columns[1:]):

    # Get specific model data
    data = model_data[data_str]

    # Merge model data with t in one df
    data_df = pd.concat([t, data], axis=1)

    # Calculate climatologies
    climatology_model = Climatology(data_df, data_str, 'date', n_days=n_days_clim)

    # Compute smoothed climatology
    model_climatology_smooth = climatology_model.climatology_smooth
    model_data_climatology_smooth_dfs.append(model_climatology_smooth)
    # Plot smoothed climatology in figure 2
    axs2[i+1].plot(model_climatology_smooth.index, model_climatology_smooth[data_str, 'mean'], label=data_str, lw=1, color=colors(i))

    # Calculate anomalies
    data_anomalies = climatology_model.calculate_anomalies(data_df)
    model_data_anomaly_dfs.append(data_anomalies)
    # Plot anomalies
    axs3[i+1, 0].plot(data_anomalies['date'], data_anomalies['anomaly'], label=data_str, lw=1, color=colors(i))
    # Plot histograms
    axs3[i+1, 1].hist(data_anomalies['anomaly'], bins=30, alpha=0.5, orientation='horizontal', label=data_str, color=colors(i)) # density=True,


# Tight layout for all figures
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()

# Save figures
if SAVE_FIGS:
    fig1.savefig(PATH_FIGS + 'Timeseries_NWVF_Clouds.png', dpi=600)
    fig2.savefig(PATH_FIGS + 'Climatologies_NWVF_Clouds.png', dpi=600)
    fig3.savefig(PATH_FIGS + 'Anomalies_NWVF_Clouds.png', dpi=600)




##################
# EVENT ANALYSIS #
##################
region_str = '_small'
sigma_ns = 1
# Loop through all data anomaly dataframes and perform event analysis
for i, data_anomaly_df in enumerate(model_data_anomaly_dfs):
    # Get the data name
    data_name = model_data.columns[i+1]
    # Create an EventAnalysis object
    event_analysis = EventAnalysis(data_anomaly_df, 'anomaly', NWVF_anomalies, 'anomaly', n_sigmas=sigma_ns)
    # Plot the data
    fig, (ax1, ax2) = event_analysis.plot_all_data(bins_nwvf=32,
                                                   bins_aerosol=32,
                                                   show_figs=False,
                                                   save_figs=False,
                                                   save_path=None,
                                                   plot_events_pos=True,
                                                   plot_events_neg=True
                                                   )
    fig.suptitle(f'Anomaly timeseries, {data_name}')
    
    PATH_data = PATH_FIGS + '/' + data_name + '/'
    if not os.path.exists(PATH_data):
        os.makedirs(PATH_data)

    if SAVE_FIGS:
        fig.savefig(PATH_data + 'AnomalyTimeseries_' + data_name + region_str + '.png', dpi=600)

    # Mask the data based on the statistical parameters (e.g. 1 sigma)
    masked_nwvf_data_pos, masked_nwvf_data_neg, masked_data_pos, masked_data_neg = event_analysis.mask_data_by_stats()
    # Plot histograms of masked data
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f'Event analysis of anomalies, {data_name}')

    # Get the data from the histograms (values and bins) for CDF calculation
    ax[0].hist(NWVF_anomalies['anomaly'],
                bins=32,
                alpha=0.9,
                label='All data',
                ec='darkblue',
                fc='lightblue')
    
    ax[0].hist(masked_nwvf_data_pos,
                bins=9,
                alpha=0.9,
                label=F'+{sigma_ns} sigma, {len(masked_nwvf_data_pos)} events',
                ec='darkgreen',
                fc='lightgreen')

    ax[0].set_title('NWVF')
    ax[0].legend()


    ax[1].hist(data_anomaly_df['anomaly'],
                bins=32,
                alpha=0.9,
                label='All data',
                ec='darkblue',
                fc='lightblue'
                )
    
    ax[1].hist(masked_data_pos,
                bins=10,
                alpha=0.9,
                label=F'+{sigma_ns} sigma, {len(masked_data_pos)} events',
                ec='darkgreen',
                fc='lightgreen'
                )
    ax[1].set_title(data_name)


    ax[2].hist(data_anomaly_df['anomaly'],
                bins=32,
                alpha=0.9,
                label='All data',
                ec='darkblue',
                fc='lightblue',
                log=True
                )
    ax[2].hist(masked_data_pos,
            bins=17,
            alpha=0.9,
            label=F'+{sigma_ns} sigma, {len(masked_data_pos)} events',
            ec='darkgreen',
            fc='lightgreen',
            log=True
            )

    ax[2].set_title('Data, log-scale')
    ax[2].legend()

    fig.tight_layout()

    # save figures
    if SAVE_FIGS:
        fig.savefig(PATH_data + '/' + f'EventAnalysis_{data_name}' + region_str + '.png', dpi=600)


    # Use np.sort to get the CDF values 
    x_nwvf = np.sort(NWVF_anomalies['anomaly'])
    f_nwvf = np.arange(len(x_nwvf)) / float(len(x_nwvf))
    x_data = np.sort(data_anomaly_df['anomaly'])
    f_data = np.arange(len(x_data)) / float(len(x_data))
    x_masked_nwvf = np.sort(masked_nwvf_data_pos)
    f_masked_nwvf = np.arange(len(x_masked_nwvf)) / float(len(x_masked_nwvf))
    x_masked_data = np.sort(masked_data_pos)
    f_masked_data = np.arange(len(x_masked_data)) / float(len(x_masked_data))


    # Make KS-test on data vs. masked data
    ks_data, p_data = ks_2samp(data_anomaly_df['anomaly'], masked_data_pos)
    print(f'KS-test all data vs. >{sigma_ns} sigma: KS={ks_data:.4f}, p={p_data:.8f}')


    # Plot CDFs
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle(f'CDF of anomalies, {data_name}, KS-test of all data vs. >{sigma_ns} sigma: KS={ks_data:.4f}, p={p_data:.8f}')

    ax[0].plot(x_nwvf, f_nwvf, label='NWVF', color='teal')
    # Plot a vertical line at the KS-test value
    ax[0].axvline(x=x_masked_nwvf[0], color='green', linestyle='--', label=F'Cut-off >{sigma_ns} sigma')
    # ax[0].plot(x_masked_nwvf, f_masked_nwvf, label='NWVF masked', color='green')
    ax[0].set_title('NWVF')
    ax[0].legend()

    ax[1].plot(x_data, f_data, label=data_name, color='teal')
    ax[1].plot(x_masked_data, f_masked_data, label=data_name + ' masked', color='green')
    ax[1].set_title(data_name)
    ax[1].legend()

    fig.tight_layout()

    # save figures
    if SAVE_FIGS:
        fig.savefig(PATH_data + '/' + f'CDF_{data_name}' + region_str + '.png', dpi=600)



plt.show()