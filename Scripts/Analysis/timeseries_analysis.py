import os

import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import matplotlib.pyplot as plt

# Save figures?
SAVE_FIGS = True


class Climatology:
    '''
        Class to calculate climatologies and anomalies for aerosol and NWVF data.
        Methods:
        --------
        calculate_climatology()
            - Calculate the daily mean and std of both NWVF and aerosol (1. jan, 2. jan,..., 31. dec)
        extend_climatology()
            - Extend the climatology to include February 29
        calculate_smoothed_climatology()
            - Make a (circular) smoothed version of the climatology and add column with day of year in format 'MM-DD' and account for leap years
        calculate_anomalies()
            - Calculate anomalies for a given year

    '''
    def __init__(self, df, value_column, date_column, n_days=7):
        self.df = df
        self.value_column = value_column
        self.date_column = date_column
        self.n_days = n_days
        self.climatology = self.calculate_climatology()
        self.climatology_full = self.extend_climatology()
        self.climatology_smooth = self.calculate_smoothed_climatology(self.n_days)

    def calculate_climatology(self):
        # Calculate the daily mean and std of both NWVF and aerosol (1. jan, 2. jan,..., 31. dec)
        self.df['day'] = self.df[self.date_column].dt.dayofyear
        climatology = self.df.groupby('day').agg(['mean', 'std'])
        return climatology

    def extend_climatology(self):
        # Extend the climatology to include February 29
        # Fill February 29 with average of Feb 28 and Mar 1 if missing
        climatology_template = pd.DataFrame(index=pd.date_range('2000-01-01', '2000-12-31'))
        climatology_template['day_of_year'] = climatology_template.index.dayofyear
        climatology_template = climatology_template.set_index('day_of_year')

        # Merge climatology with template
        climatology_full = climatology_template.join(self.climatology, how='left')
        climatology_full = climatology_full.interpolate()  # Fill Feb 29 if missing
        return climatology_full

    def calculate_smoothed_climatology(self, n_days):
        # Make a (circular) smoothed version of the climatology and add column with day of year in format 'MM-DD' and account for leap years
        # Define number of days to pad the climatology with at the beginning and end of the year
        pad_length = n_days // 2

        # Create circular padding by appending and prepending daata from climatology
        climatology_padded = pd.concat([self.climatology_full.iloc[-pad_length:], self.climatology_full, self.climatology_full.iloc[:pad_length]])
        climatology_smooth_padded = climatology_padded.rolling(window=n_days, center=True).mean()
        climatology_smooth = climatology_smooth_padded.iloc[pad_length:-pad_length]
        climatology_smooth['day'] = pd.to_datetime(climatology_smooth.index, format='%j').strftime('%m-%d')
        return climatology_smooth

    def calculate_anomalies(self, df):
        anomalies = pd.DataFrame()
        for year in df[self.date_column].dt.year.unique():
            # Extract data for the specific year
            year_data = df[df[self.date_column].dt.year == year].copy()
            # print('\nYear:', year)
            # print('Length:', len(year_data))

            # Align day of year including leap year
            year_data['day_of_year'] = year_data[self.date_column].dt.dayofyear
            if len(year_data) == 366:
                # Check for leap year
                climatology_to_use = self.climatology
            else:
                # Drop Feb 29 for non-leap years
                climatology_to_use = self.climatology[self.climatology.index != 60]
            # Calculate anomalies
            year_data['anomaly'] = year_data[self.value_column].values - climatology_to_use[self.value_column, 'mean'].values
            anomalies = pd.concat([anomalies, year_data])
        return anomalies


class EventAnalysis:
    '''
        Class to perform event analysis on aerosol and nwvf data.
        Methods:
        --------
        calculate_statistical_params()
            - Calculate the statistical parameters for nwvf and aerosol data
        mask_data_by_stats()
            - Mask the data based on the statistical parameters
        plot_all_data()
            - Plot all the data (time series)
    '''
    def __init__(self, aerosol_data, aerosol_name, nwvf_data, nwvf_name, n_sigmas=1):
        '''
            Parameters:
            -----------
            aerosol_data : pd.DataFrame
                DataFrame containing the aerosol data.
            aerosol_name : str
                Name of the aerosol data.
            nwvf_data : pd.DataFrame
                DataFrame containing the nwvf data.
            nwvf_name : str
                Name of the nwvf data.
            n_sigmas : int
                Number of standard deviations to use for the threshold.
        '''
        self.aerosol_data = aerosol_data
        self.aerosol_name = aerosol_name
        self.nwvf_data = nwvf_data
        self.nwvf_name = nwvf_name
        self.n_sigmas = n_sigmas
        self.aerosol_stats, self.nwvf_stats = self.calculate_statistical_params()
        import matplotlib.pyplot as plt
    
    def calculate_statistical_params(self):
        '''
            Calculate the statistical parameters for both aerosol and nwvf data.
        '''

        # Calculate the statistical parameters for aerosol data
        aerosol_stats = {'mean': self.aerosol_data[self.aerosol_name].mean(), 'std': self.aerosol_data[self.aerosol_name].std()}
        # Calculate the statistical parameters for nwvf data
        nwvf_stats = {'mean': self.nwvf_data[self.nwvf_name].mean(), 'std': self.nwvf_data[self.nwvf_name].std()}

        return aerosol_stats, nwvf_stats
    
    def mask_data_by_stats(self):
        '''
            Mask the data based on the statistical parameters.
        '''

        # Calculate the statistical parameters
        self.calculate_statistical_params()

        # Mask the data based on the statistical parameters
        self.masked_nwvf_data_pos = {}
        self.masked_nwvf_data_neg = {}

        self.masked_aerosol_data_pos = {}
        self.masked_aerosol_data_neg = {}

        # Get the seasonal data
        data_nwvf = self.nwvf_data[self.nwvf_name]
        data_aerosol = self.aerosol_data[self.aerosol_name]
        
        # Get the statistical parameters
        mean = self.nwvf_stats['mean']
        std = self.nwvf_stats['std']

        # Create a mask (to mask both nwvf and aerosol data)
        mask_pos = np.where(data_nwvf > mean + self.n_sigmas * std, True, False)
        mask_neg = np.where(data_nwvf < mean - self.n_sigmas * std, True, False)

        # Mask the nwvf data
        masked_nwvf_pos = data_nwvf[mask_pos]
        masked_nwvf_neg = data_nwvf[mask_neg]

        # Mask the aerosol data
        masked_aerosol_pos = data_aerosol[mask_pos]
        masked_aerosol_neg = data_aerosol[mask_neg]

        # Store the masked data in a dictionary
        self.masked_nwvf_data_pos = masked_nwvf_pos
        self.masked_nwvf_data_neg = masked_nwvf_neg

        self.masked_aerosol_data_pos = masked_aerosol_pos
        self.masked_aerosol_data_neg = masked_aerosol_neg

        return self.masked_nwvf_data_pos, self.masked_nwvf_data_neg, self.masked_aerosol_data_pos, self.masked_aerosol_data_neg
    
    def coinciding_events(self):
        '''
            Find the coinciding events between aerosol and nwvf data.
        '''

        # Mask the data based on the statistical parameters
        masked_nwvf_data_pos, masked_nwvf_data_neg, masked_aerosol_data_pos, masked_aerosol_data_neg = self.mask_data_by_stats()

        # Find the coinciding events
        coinciding_events_pos = np.where(masked_nwvf_data_pos == masked_aerosol_data_pos, True, False)
        coinciding_events_neg = np.where(masked_nwvf_data_neg == masked_aerosol_data_neg, True, False)

        return coinciding_events_pos, coinciding_events_neg

    def plot_all_data(self, bins_nwvf=32, bins_aerosol=32, show_figs=True, save_figs=False, save_path=None, plot_events_pos=True,
                      plot_events_neg=True):
        '''
            Plot aerosol and nwvf data (not masked) in time series and corresponding histograms.ß
        '''

        # Get the general statistics
        aerosol_stats = self.aerosol_stats
        nwvf_stats = self.nwvf_stats

        fig, (ax1, ax2) = plt.subplots(nrows=2,
                                       ncols=2,
                                       figsize=(15, 8),
                                       gridspec_kw={'width_ratios': [3, 1], 'wspace': 0},
                                       sharey='row'
                                       )

        fig.suptitle(f'{self.aerosol_name} and NWVF data')
        N_SIG = self.n_sigmas

        # NWVF time series
        ax1[0].plot(self.nwvf_data['time'], self.nwvf_data[self.nwvf_name], label='NWVF', color='k')
        ax1[0].set_title('NWVF')
        # Draw horizontal lines at mean and mean +/- n_sigmas * std
        ax1[0].axhline(nwvf_stats['mean'], color='b', linestyle='--', linewidth=1, label='Mean')
        if plot_events_pos:
            ax1[0].axhline(nwvf_stats['mean'] + N_SIG * nwvf_stats['std'], color='b', linestyle=':', linewidth=1, label=f'+ {N_SIG} sigma')
        if plot_events_neg:
            ax1[0].axhline(nwvf_stats['mean'] - N_SIG * nwvf_stats['std'], color='b', linestyle=':', linewidth=1, label=f'- {N_SIG} sigma')
        ax1[0].legend()

        # NWVF histogram (approx. gaussian)
        ax1[1].hist(self.nwvf_data[self.nwvf_name], bins=bins_nwvf, ec='k', fc='none', orientation='horizontal')
        ax1[1].tick_params(axis='y', left=False)

        # Aerosol time series
        ax2[0].plot(self.aerosol_data['date'], self.aerosol_data[self.aerosol_name], label=self.aerosol_name, color='k')
        ax2[0].set_title(f'{self.aerosol_name} Aerosols')
        # Draw horizontal lines at mean and mean + n_sigmas * std (not gaussian)
        ax2[0].axhline(aerosol_stats['mean'], color='b', linestyle='--', linewidth=1, label='Mean')
        ax2[0].axhline(aerosol_stats['mean'] + N_SIG * aerosol_stats['std'], color='b', linestyle=':', linewidth=1, label=f'+ {N_SIG} sigma')
        ax2[0].axhline(aerosol_stats['mean'] - N_SIG * aerosol_stats['std'], color='b', linestyle=':', linewidth=1, label=f'- {N_SIG} sigma')
        ax2[0].legend()

        # Aerosol histogram (not gaussian)
        ax2[1].hist(self.aerosol_data[self.aerosol_name], bins=bins_aerosol, ec='k', fc='none', orientation='horizontal')
        ax2[1].tick_params(axis='y', left=False)

        # fig.tight_layout()

        if show_figs:
            plt.show()

        return fig, (ax1, ax2)
    
    





##################################
# LOADING AND PREPROCESSING DATA #
##################################

# Choose aerosol
aerosol_types = ['TotalMass', 'BC', 'Dust', 'SO4']
#aerosol_str = aerosol_types[0]

# Choose region (pizza slice, '', or band, '_small')
region_str = '_small'

# Define start and end date
start_date_str = '1997-01-01'
end_date_str = '2021-12-31'

# Define paths to data and figures
PATH_DATA = '../../Data/'
PATH_FIGS = '../../Figures/AnomalyTimeseriesAnalysis/'
# Create folder if it does not exist
if not os.path.exists(PATH_FIGS):
    os.makedirs(PATH_FIGS)



# AEROSOL DATA

# Import aerosol txt file (with header)
aerosol_data = pd.read_csv(PATH_DATA + 'ModelData/TS_AerosolBurden_ArcticSlice' + region_str + '.txt')

# Save time data in datetime format in dataframe and crop data to only after 1997 and before 2022-07-01
t = pd.DataFrame(pd.to_datetime(aerosol_data['date']))
t = t[t['date'] >= start_date_str]
t = t[t['date'] <= end_date_str]

# Create new column 'TotalMass' which is the sum of all columns except 'date'
aerosol_data['TotalMass'] = aerosol_data['BC'] + aerosol_data['Dust'] + aerosol_data['SO4'] + aerosol_data['OC'] + aerosol_data['SS'] + aerosol_data['PM25']

# Create a df with only aerosols in aerosol_types and date column
aerosol_data = aerosol_data[['date'] + aerosol_types]


# Filter data to only after 1997 and before 2022-07-01
aerosol_data = aerosol_data[aerosol_data['date'] >= start_date_str] 
aerosol_data = aerosol_data[aerosol_data['date'] <= end_date_str]


# NWVF DATA

# Read ERA5 NWVF (6 hourly) data from .nc file
NWVF_ERA5 = xr.open_dataset(PATH_DATA + 'Processed_MAIA/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc')

# Resample NWVF data to daily data and save in dataframe
daily_NWVF = NWVF_ERA5['nwvf_integral'].resample(time='1D').sum()
daily_NWVF_df = daily_NWVF.to_dataframe()

# Reset index of daily_NWVF_df
daily_NWVF_df.reset_index(inplace=True)

# Filter data to only after 1997 and before 2022-07-01
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] >= start_date_str]
daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] <= end_date_str]









###############################
# CLIMATOLOGIES AND ANOMALIES #
###############################

n_days_clim = 30
# Loop through aerosols in df and calculate climatologies and anomalies


# Figure to plot all smoothed aerosol climatologies (one subplot) and smoothed NWVF climatology (one subplot)
fig_clim, axs_clim = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
axs_clim[0].set_title('Smoothed climatology, aerosols')
# Set x-axis to [first_day, last_day]
axs_clim[0].set_xlim([1, 366])
# Add grid and axis labels
axs_clim[0].grid()
axs_clim[0].set_ylabel('aerosol [kg/m2]')
axs_clim[0].set_xlabel('Time')
axs_clim[1].set_title('Smoothed climatology, NWVF')
axs_clim[1].grid()
axs_clim[1].set_ylabel('NWVF [kg/m2]')
axs_clim[1].set_xlabel('Time')



# Figure to plot all aerosols (one subplot) and NWVF (one subplot) anomalies
fig_anom, axs_anom = plt.subplots(2, 1, figsize=(18, 6), sharex=True)
axs_anom[0].set_title('Anomalies, aerosols')
# Set x-axis to [first_day, last_day]
axs_anom[0].set_xlim([t['date'].iloc[0], t['date'].iloc[-1]])
# Add grid and axis labels
axs_anom[0].grid()
axs_anom[0].set_ylabel('aerosol anomaly [kg/m2]')
axs_anom[0].set_xlabel('Time')
axs_anom[1].set_title('Anomalies, NWVF')
axs_anom[1].grid()
axs_anom[1].set_ylabel('NWVF anomaly [kg/m2]')
axs_anom[1].set_xlabel('Time')


# Figure to plot histograms of anomalies
fig_hist, axs_hist = plt.subplots(2, 1, figsize=(10, 8))
axs_hist[0].set_title('Anomalies, aerosols')
axs_hist[0].set_ylabel('Frequency')
axs_hist[0].set_xlim([-0.00006, 0.00006]) # Make x-axis symmetric
axs_hist[0].grid()
axs_hist[1].set_title('Anomalies, NWVF')
axs_hist[1].set_ylabel('Frequency')
axs_hist[1].set_xlabel('Anomaly from climatology')
axs_hist[1].set_xlim([-1.7e6, 1.7e6]) # Make x-axis symmetric
axs_hist[1].grid()

# Define colors for plotting
colors = ['teal', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Empty list to store aerosol dataframes
aerosol_anomaly_dfs = []
aerosol_climatology_smooth_dfs = []

for aerosol_str in aerosol_types:
    # Get color for plotting
    color = colors.pop(0)
    
    # Get specified aerosol data
    aerosol = aerosol_data[aerosol_str]

    # Merge aerosol and t into one dataframe
    aerosol_df = pd.concat([t, aerosol], axis=1)
    
    # Create climatology object
    climatology = Climatology(aerosol_df, aerosol_str, 'date', n_days=n_days_clim)

    # Compute smoothed climatology
    climatology_smooth = climatology.climatology_smooth
    aerosol_climatology_smooth_dfs.append(climatology_smooth)
    # Plot smoothed climatology
    axs_clim[0].plot(climatology_smooth.index, climatology_smooth[aerosol_str, 'mean'], label=aerosol_str, lw=0.8, color=color)
    

    # Calculate anomalies for both aerosol and NWVF
    aerosol_anomalies = climatology.calculate_anomalies(aerosol_df)
    aerosol_anomaly_dfs.append(aerosol_anomalies)
    # Plot anomalies
    axs_anom[0].plot(aerosol_anomalies['date'], aerosol_anomalies['anomaly'], label=aerosol_str, lw=0.8, color=color)
    # Plot histograms
    axs_hist[0].hist(aerosol_anomalies['anomaly'], bins=50, color=color, alpha=0.7, label=aerosol_str)


# Calculate NWVF climatology and anomalies'
NWVF_climatology = Climatology(daily_NWVF_df, 'nwvf_integral', 'time', n_days=n_days_clim)
NWVF_climatology_smooth = NWVF_climatology.climatology_smooth
NWVF_anomalies = NWVF_climatology.calculate_anomalies(daily_NWVF_df)
# Plot NWVF climatology
axs_clim[1].plot(NWVF_climatology_smooth.index, NWVF_climatology_smooth['nwvf_integral', 'mean'], label='NWVF', lw=0.8, color='teal')
# Plot NWVF anomalies
axs_anom[1].plot(NWVF_anomalies['time'], NWVF_anomalies['anomaly'], label='NWVF', lw=0.8, color='teal')
# Plot histograms
axs_hist[1].hist(NWVF_anomalies['anomaly'], bins=50, color='teal', alpha=0.7, label='NWVF')

# Add legends
axs_clim[0].legend()
axs_clim[1].legend()
axs_anom[0].legend()
axs_anom[1].legend()

fig_clim.tight_layout()
fig_anom.tight_layout()
print(PATH_FIGS)
# Save figures 
if SAVE_FIGS:
    fig_clim.savefig(PATH_FIGS + 'SmoothedClimatologies' + region_str + '.png', dpi=600)
    fig_anom.savefig(PATH_FIGS + 'Anomalies' + region_str + '.png', dpi=600)
    fig_hist.savefig(PATH_FIGS + 'Histograms' + region_str + '.png', dpi=600)









##################
# EVENT ANALYSIS #
##################

# Loop through all aerosol anomaly dataframes and perform event analysis
for i, aerosol_anomaly_df in enumerate(aerosol_anomaly_dfs):
    # Get the aerosol name
    aerosol_name = aerosol_types[i]
    # Create an EventAnalysis object
    event_analysis = EventAnalysis(aerosol_anomaly_df, 'anomaly', NWVF_anomalies, 'anomaly', n_sigmas=2)
    # Plot the data
    fig, (ax1, ax2) = event_analysis.plot_all_data(bins_nwvf=32, bins_aerosol=32, show_figs=False, save_figs=False, save_path=None, plot_events_pos=True,
                        plot_events_neg=True)
    fig.suptitle(f'Anomaly timeseries, {aerosol_name}')
    
    PATH_AEROSOL = PATH_FIGS + '/' + aerosol_name
    if not os.path.exists(PATH_AEROSOL):
        os.makedirs(PATH_AEROSOL)

    if SAVE_FIGS:
        fig.savefig(PATH_AEROSOL + 'AnomalyTimeseries_' + aerosol_name + region_str + '.png', dpi=600)

    # Mask the data based on the statistical parameters
    masked_nwvf_data_pos, masked_nwvf_data_neg, masked_aerosol_data_pos, masked_aerosol_data_neg = event_analysis.mask_data_by_stats()
    # Plot histograms of masked data
    fig, ax = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f'Event analysis, {aerosol_name}')

    ax[0].hist(NWVF_anomalies['anomaly'],
                    bins=32,
                    alpha=0.9,
                    label=f'All data',
                    ec='darkblue',
                    fc='lightblue')
    ax[0].hist(masked_nwvf_data_pos,
            bins=9,
            alpha=0.9,
            label=F'+{2} sigma, {len(masked_nwvf_data_pos)} events',
            ec='darkgreen',
            fc='lightgreen')
    ax[0].hist(masked_nwvf_data_neg,
            bins=8,
            alpha=0.9,
            label=f'-{2} sigma, {len(masked_nwvf_data_neg)} events',
            ec='darkred',
            fc='lightcoral')
    ax[0].set_title('NWVF')
    ax[0].legend()
    ax[1].hist(aerosol_anomaly_df['anomaly'],
            bins=32,
            alpha=0.9,
            label=f'All data',
            ec='darkblue',
            fc='lightblue'
            )
    ax[1].hist(masked_aerosol_data_pos,
        bins=10,
        alpha=0.9,
        label=F'+{2} sigma, {len(masked_aerosol_data_pos)} events',
        ec='darkgreen',
        fc='lightgreen'
        )
    ax[1].hist(masked_aerosol_data_neg,
        bins=10,
        alpha=0.9,
        label=f'-{2} sigma, {len(masked_aerosol_data_neg)} events',
        ec='darkred',
        fc='lightcoral'
        )
    ax[1].set_title('Aerosol')



    ax[2].hist(aerosol_anomaly_df['anomaly'],
                bins=32,
                alpha=0.9,
                label=f'All data',
                ec='darkblue',
                fc='lightblue',
                log=True
                )
    ax[2].hist(masked_aerosol_data_pos,
            bins=17,
            alpha=0.9,
            label=F'+{2} sigma, {len(masked_aerosol_data_pos)} events',
            ec='darkgreen',
            fc='lightgreen',
            log=True
            )
    ax[2].hist(masked_aerosol_data_neg,
            bins=7,
            alpha=0.9,
            label=f'-{2} sigma, {len(masked_aerosol_data_neg)} events',
            ec='darkred',
            fc='lightcoral',
            log=True
            )
    ax[2].set_title('Aerosol, log-scale')
    ax[2].legend()

    fig.tight_layout()


    # save figures
    if SAVE_FIGS:
        fig.savefig(PATH_AEROSOL + '/' + f'EventAnalysis_{aerosol_name}' + region_str + '.png', dpi=600)

    plt.show()




























































# # Save data in pd series (most important aerosols: ['BC', 'Dust', 'SO4', 'TotalMass'])
# # If 'TotalMass' is chosen, the sum of all aerosols is calculated
# if aerosol_str == 'TotalMass':
#     aerosol = aerosol_data['BC'] + aerosol_data['Dust'] + aerosol_data['SO4']
#     # Add 'TotalMass' header
#     aerosol = pd.DataFrame(aerosol, columns=['TotalMass'])
# else:
#     aerosol = aerosol_data[aerosol_str]
# # Merge aerosol and t into one dataframe
# aerosol_df = pd.concat([t, aerosol], axis=1)

# # Filter data to only after 1997 and before 2022-07-01
# aerosol_df = aerosol_df[aerosol_df['date'] >= start_date_str]
# aerosol_df = aerosol_df[aerosol_df['date'] <= end_date_str]


# # NWVF DATA

# # Read ERA5 NWVF (6 hourly) data from .nc file
# NWVF_ERA5 = xr.open_dataset(PATH_DATA + 'Processed_MAIA/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc')

# # Resample NWVF data to daily data and save in dataframe
# daily_NWVF = NWVF_ERA5['nwvf_integral'].resample(time='1D').sum()
# daily_NWVF_df = daily_NWVF.to_dataframe()

# # Reset index of daily_NWVF_df
# daily_NWVF_df.reset_index(inplace=True)

# # Filter data to only after 1997 and before 2022-07-01
# daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] >= start_date_str]
# daily_NWVF_df = daily_NWVF_df[daily_NWVF_df['time'] <= end_date_str]









# ###############################
# # COMPUTE CLIMATOLOGY (DAILY) #
# ###############################

# # Calculate the daily mean and std of both NWVF and aerosol (1. jan, 2. jan,..., 31. dec)
# # Aerosol
# aerosol_df['day'] = aerosol_df['date'].dt.dayofyear
# aerosol_climatology = aerosol_df.groupby('day').agg(['mean', 'std'])
# # NWVF
# daily_NWVF_df['day'] = daily_NWVF_df['time'].dt.dayofyear
# NWVF_climatology = daily_NWVF_df.groupby('day').agg(['mean', 'std'])


# # Extend the climatology to include February 29
# # Fill February 29 with average of Feb 28 and Mar 1 if missing
# climatology_template = pd.DataFrame(index=pd.date_range('2000-01-01', '2000-12-31'))
# climatology_template['day_of_year'] = climatology_template.index.dayofyear
# climatology_template = climatology_template.set_index('day_of_year')


# # Merge climatology with template
# aerosol_climatology_full = climatology_template.join(aerosol_climatology, how='left')
# aerosol_climatology_full = aerosol_climatology_full.interpolate()  # Fill Feb 29 if missing

# NWVF_climatology_full = climatology_template.join(NWVF_climatology, how='left')
# NWVF_climatology_full = NWVF_climatology_full.interpolate()  # Fill Feb 29 if missing






# #########################################
# # COMPUTE CIRCULAR SMOOTHED CLIMATOLOGY #
# #########################################

# # Make a (circular) smoothed version of the climatology and add column with day of year in format 'MM-DD' and account for leap years
# n_days = 7
# # Define number of days to pad the climatology with at the beginning and end of the year
# pad_length = n_days // 2

# # Create circular padding by appending and prepending daata from climatology
# # Aerosol
# aerosol_climatology_padded = pd.concat([aerosol_climatology_full.iloc[-pad_length:], aerosol_climatology_full, aerosol_climatology_full.iloc[:pad_length]])
# aerosol_climatology_smooth_padded = aerosol_climatology_padded.rolling(window=n_days, center=True).mean()
# aerosol_climatology_smooth = aerosol_climatology_smooth_padded.iloc[pad_length:-pad_length]
# aerosol_climatology_smooth['day'] = pd.to_datetime(aerosol_climatology_smooth.index, format='%j').strftime('%m-%d')

# # NWVF
# NWVF_climatology_padded = pd.concat([NWVF_climatology_full.iloc[-pad_length:], NWVF_climatology_full, NWVF_climatology_full.iloc[:pad_length]])
# NWVF_climatology_smooth_padded = NWVF_climatology_padded.rolling(window=n_days, center=True).mean()
# NWVF_climatology_smooth = NWVF_climatology_smooth_padded.iloc[pad_length:-pad_length]
# NWVF_climatology_smooth['day'] = pd.to_datetime(NWVF_climatology_smooth.index, format='%j').strftime('%m-%d')


# # Function to get anomalies for a given year
# def calculate_anomalies(df, climatology, value_column, date_column):
#     anomalies = pd.DataFrame()
#     for year in df[date_column].dt.year.unique():
#         # Extract data for the specific year
#         year_data = df[df[date_column].dt.year == year].copy()
#         print('\nYear:', year)
#         print('Length:', len(year_data))

#         # Align day of year including leap year
#         year_data['day_of_year'] = year_data[date_column].dt.dayofyear
#         if len(year_data) == 366:  # Check for leap year
#             climatology_to_use = climatology
#         else:
#             # Drop Feb 29 for non-leap years
#             climatology_to_use = climatology[climatology.index != 60]
#         # Calculate anomalies
#         year_data['anomaly'] = year_data[value_column].values - climatology_to_use[value_column, 'mean'].values
#         anomalies = pd.concat([anomalies, year_data])
#     return anomalies

# # Calculate anomalies for both aerosol and NWVF
# aerosol_anomalies = calculate_anomalies(aerosol_df, aerosol_climatology_smooth, aerosol_str, 'date')
# NWVF_anomalies = calculate_anomalies(daily_NWVF_df, NWVF_climatology_smooth, 'nwvf_integral', 'time')















# #####################################
# # PLOT TIMESERIES AND CLIMATOLOGIES #
# #####################################

# # Plot aerosol and NWVF in different subplots
# fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

# axs[0].plot(aerosol_df['date'], aerosol_df[aerosol_str], label='Aerosol', color='teal', lw=0.8)
# # axs[0].set_title(f'Aerosol (correlation with NWVF: {correlation[0, 1]:.4f})')
# axs[0].set_ylabel('aerosol [kg/m2]')
# axs[0].grid()
# axs[0].legend()

# axs[1].plot(daily_NWVF_df['time'], daily_NWVF_df['nwvf_integral'], label='NWVF', color='teal', lw=0.8)
# axs[1].set_title('NWVF')
# axs[1].set_ylabel('NWVF [kg/m2]')
# axs[1].set_xlabel('Time')
# # Set x-axis to [first_day, last_day]
# axs[1].set_xlim([daily_NWVF_df['time'].iloc[0], daily_NWVF_df['time'].iloc[-1]])
# axs[1].grid()
# axs[1].legend()

# fig.tight_layout()

# # Save the figure as an image file
# if SAVE_FIGS:
#     plt.savefig(PATH_FIGS + aerosol_str + '_NWVF_raw_comparison' + region_str + '.png')



# # Plot the climatology
# fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

# axs[0].plot(aerosol_climatology.index, aerosol_climatology[aerosol_str]['mean'], label='Aerosol', color='teal', lw=0.8)
# axs[0].fill_between(aerosol_climatology.index, aerosol_climatology[aerosol_str]['mean'] - aerosol_climatology[aerosol_str]['std'],
#                     aerosol_climatology[aerosol_str]['mean'] + aerosol_climatology[aerosol_str]['std'], color='teal', alpha=0.2)
# axs[0].set_title(f'Aerosol climatology')
# axs[0].set_ylabel('aerosol [kg/m2]')
# axs[0].grid()
# axs[0].legend()

# axs[1].plot(NWVF_climatology.index, NWVF_climatology['nwvf_integral']['mean'], label='NWVF', color='teal', lw=0.8)
# axs[1].fill_between(NWVF_climatology.index, NWVF_climatology['nwvf_integral']['mean'] - NWVF_climatology['nwvf_integral']['std'],
#                     NWVF_climatology['nwvf_integral']['mean'] + NWVF_climatology['nwvf_integral']['std'], color='teal', alpha=0.2)
# axs[1].set_title('NWVF climatology')
# axs[1].set_ylabel('NWVF [kg/m2]')
# axs[1].set_xlabel('Day of year')
# axs[1].grid()
# axs[1].legend()

# fig.tight_layout()


# # Plot the smoothed climatology
# fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

# fig.suptitle(f'Smoothed over {n_days} days')
# axs[0].plot(aerosol_climatology_smooth.index, aerosol_climatology_smooth[aerosol_str, 'mean'], label='Aerosol', color='teal', lw=0.8)
# axs[0].fill_between(aerosol_climatology_smooth.index, aerosol_climatology_smooth[aerosol_str, 'mean'] - aerosol_climatology_smooth[aerosol_str, 'std'],
#                     aerosol_climatology_smooth[aerosol_str, 'mean'] + aerosol_climatology_smooth[aerosol_str, 'std'], color='teal', alpha=0.2)
# axs[0].set_title(f'Aerosol smoothed climatology')
# axs[0].set_ylabel('aerosol [kg/m2]')
# axs[0].grid()
# axs[0].legend()

# axs[1].plot(NWVF_climatology_smooth.index, NWVF_climatology_smooth['nwvf_integral', 'mean'], label='NWVF', color='teal', lw=0.8)
# axs[1].fill_between(NWVF_climatology_smooth.index, NWVF_climatology_smooth['nwvf_integral', 'mean'] - NWVF_climatology_smooth['nwvf_integral', 'std'],
#                     NWVF_climatology_smooth['nwvf_integral', 'mean'] + NWVF_climatology_smooth['nwvf_integral', 'std'], color='teal', alpha=0.2)
# axs[1].set_title(f'Aerosol smoothed climatology')
# axs[1].set_ylabel('aerosol [kg/m2]')
# axs[1].grid()
# axs[1].legend()


# # Plot the anomalies
# fig, axs = plt.subplots(2, 1, figsize=(18, 6), sharex=True)

# axs[0].plot(aerosol_anomalies['date'], aerosol_anomalies['anomaly'], label='Aerosol', color='teal', lw=0.8)
# axs[0].set_title(f'Aerosol anomalies')
# axs[0].set_ylabel('aerosol anomaly [kg/m2]')
# axs[0].grid()
# axs[0].legend()

# axs[1].plot(NWVF_anomalies['time'], NWVF_anomalies['anomaly'], label='NWVF', color='teal', lw=0.8)
# axs[1].set_title('NWVF anomalies')
# axs[1].set_ylabel('NWVF anomaly [kg/m2]')
# axs[1].set_xlabel('Time')
# axs[1].grid()
# axs[1].legend()

# fig.tight_layout()


# # plt.close()
# # Subtract the smoothed climatology from each year in the timeseries to get the anomaly
# # Aerosol




# plt.show()


