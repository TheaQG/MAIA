'''
    Script to process aerosol measurements from Villum and Zeppelin stations.
    Observational data (EC and SO4) is matched to model data (NWVF) by date.
    NWVF is summed over a number of days (default 7 days) to match the observational data.

    This script processes aerosol measurements from Villum and Zeppelin stations, matching observational data (EC and SO4) to model data (NWVF) by date. The NWVF data is summed over a number of days (default 7 days) to align with the observational data.

    Key Features:
        - Data Import: Reads aerosol data from CSV files and NWVF data from a NetCDF file.
        - Data Processing:
            - Converts timestamps from NWVF data to datetime format.
            - Saves processed NWVF data to a CSV file.
            - Converts EC data to numeric and datetime formats, correcting erroneous dates.
            - Replaces missing values in EC Zeppelin data with NaN.
    
    File Paths:
        - Aerosol Data:
            - EC Villum: EC_Villum.csv
            - EC Zeppelin: EC_Zeppelin.csv
            - SO4 Villum: SO4_Villum.csv
            - SO4 Zeppelin: SO4_Zeppelin.csv
        - NWVF Data: MAIA_nwvf_integrals_1994-2022__6hr_mean.nc
        - Processed NWVF Data Output: nwvf_ts.csv
    
    Configuration:
    PLOT_FIG: Boolean to control plotting.
    VERBOSE: Boolean to control verbosity.
'''
import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOT_FIG = True
VERBOSE = True




###################################################
# PROCESS RAW DATA TO ACCOUNT FOR MISSING VALUES  #
###################################################

DATA_PATH_AERO = '../../Data/Aerosol_StationMeasurements/'
DATA_PATH_PROCESSED = '../../Data/Processed_MAIA/'
FIG_PATH = '../../Figures/ObsResults/StationResults/'
# Define aerosol data file paths
fn_EC_Villum = DATA_PATH_AERO + 'EC_Villum.csv'
fn_EC_Zeppelin = DATA_PATH_AERO + 'EC_Zeppelin.csv'
fn_SO4_Villum = DATA_PATH_AERO + 'SO4_Villum.csv'
fn_SO4_Zeppelin = DATA_PATH_AERO + 'SO4_Zeppelin.csv'
# Define northward water vapour flux file path
fn_nwvf = DATA_PATH_PROCESSED + 'MAIA_nwvf_integrals_1994-2022__6hr_mean.nc'

# Read in all datasets
ds_nwvf = nc.Dataset(fn_nwvf) # NWVF
df_EC_Villum = pd.read_csv(fn_EC_Villum) # EC(elemental carbon), Villum station
df_EC_Zeppelin = pd.read_csv(fn_EC_Zeppelin) # EC(elemental carbon), Zeppelin station
df_SO4_Villum = pd.read_csv(fn_SO4_Villum, sep=';', decimal=',') # SO4, Villum station
df_SO4_Zeppelin = pd.read_csv(fn_SO4_Zeppelin) # SO4, Zeppelin station

# Extract time stamps from NWVF datasets and  convert to datetime format
nwvf_hrs_after_1994 = ds_nwvf['time'][:]
nwvf_timestamps = pd.to_datetime(nwvf_hrs_after_1994, unit='h', origin=pd.Timestamp('1994-01-01 02:00:00'))

# Save nwvf and timestamps to csv file
nwvf_data = ds_nwvf.variables['nwvf_integral'][:]
nwvf_ts = pd.DataFrame({'time': nwvf_timestamps, 'nwvf': nwvf_data})
nwvf_ts.to_csv(DATA_PATH_PROCESSED + 'nwvf_ts.csv', index=False, header=True)


# Convert EC Villum data to numeric and datetime format. Correct one erroneous date.
df_EC_Villum['EC'] = pd.to_numeric(df_EC_Villum['EC'].str.replace(',', '.'), errors='coerce')
df_EC_Villum['date'] = pd.to_datetime(df_EC_Villum['date'])
df_EC_Villum.at[550, 'date'] = '2019-02-24' # Correct erroneous date from 2018-02-24 to 2019-02-24

# EC Zeppelin, replace previous missing values (9.999)
# with NaN and convert to numeric and datetime format
df_EC_Zeppelin['EC'].replace(9.999, np.nan, inplace=True)
df_EC_Zeppelin['EC'] = pd.to_numeric(df_EC_Zeppelin['EC'], errors='coerce')*1000
df_EC_Zeppelin['date'] = pd.to_datetime(df_EC_Zeppelin['date'])

# Convert SO4 Villum data to numeric and datetime format. 
# Format with hours and minutes due to multiple measurements per day.
df_SO4_Villum['date'] = pd.to_datetime(df_SO4_Villum['date'], format='%d/%m/%Y %H.%M')
df_SO4_Villum['SO4'] = pd.to_numeric(df_SO4_Villum['SO4'], errors='coerce')

# SO4 Zeppelin, replace previous missing values (9.999, 9.99, 99.999, 99.99, 99.9)
# with NaN and convert to numeric and datetime format
df_SO4_Zeppelin['SO4'].replace(9.999, np.nan, inplace=True)
df_SO4_Zeppelin['SO4'].replace(9.99, np.nan, inplace=True)
df_SO4_Zeppelin['SO4'].replace(99.999, np.nan, inplace=True)
df_SO4_Zeppelin['SO4'].replace(99.99, np.nan, inplace=True)
df_SO4_Zeppelin['SO4'].replace(99.9, np.nan, inplace=True)
df_SO4_Zeppelin['SO4'] = pd.to_numeric(df_SO4_Zeppelin['SO4'], errors='coerce')
df_SO4_Zeppelin['date'] = pd.to_datetime(df_SO4_Zeppelin['date'])


# Drop rows with NaN values
df_EC_Villum.dropna(inplace=True)
df_EC_Zeppelin.dropna(inplace=True)
df_SO4_Villum.dropna(inplace=True)
df_SO4_Zeppelin.dropna(inplace=True)




###############################################################
# COMPUTE DIFFERENCE IN DAYS BETWEEN CONSECUTIVE MEASUREMENTS #
###############################################################


# Add a column with the difference in days between consecutive measurements.Fill first value w 0. Count unique values
df_EC_Villum['days_diff'] = df_EC_Villum['date'].diff().dt.days
df_EC_Villum['days_diff'].fillna(0, inplace=True)
unique_diffs_EC_Villum, counts_EC_Villum = np.unique(df_EC_Villum['days_diff'], return_counts=True)

df_EC_Zeppelin['days_diff'] = df_EC_Zeppelin['date'].diff().dt.days
df_EC_Zeppelin['days_diff'].fillna(0, inplace=True)
unique_diffs_EC_Zeppelin, counts_EC_Zeppelin = np.unique(df_EC_Zeppelin['days_diff'], return_counts=True)

df_SO4_Villum['days_diff'] = df_SO4_Villum['date'].diff().dt.days
df_SO4_Villum['days_diff'].fillna(0, inplace=True)
unique_diffs_SO4_Villum, counts_SO4_Villum = np.unique(df_SO4_Villum['days_diff'], return_counts=True)

df_SO4_Zeppelin['days_diff'] = df_SO4_Zeppelin['date'].diff().dt.days
df_SO4_Zeppelin['days_diff'].fillna(0, inplace=True)
unique_diffs_SO4_Zeppelin, counts_SO4_Zeppelin = np.unique(df_SO4_Zeppelin['days_diff'], return_counts=True)

# Save processed aerosol dataframes to csv files
df_EC_Villum.to_csv(DATA_PATH_PROCESSED + 'df_EC_Villum.csv', index=False, header=True)
df_EC_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'df_EC_Zeppelin.csv', index=False, header=True)
df_SO4_Villum.to_csv(DATA_PATH_PROCESSED + 'df_SO4_Villum.csv', index=False, header=True)
df_SO4_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'df_SO4_Zeppelin.csv', index=False, header=True)

if VERBOSE:
    # Print some statistics
    print('EC Villum: mean days between measurements: ', df_EC_Villum['days_diff'].mean())
    print(df_EC_Villum)
    print('EC Villum: unique differences btw days: ',
          dict(zip(unique_diffs_EC_Villum, counts_EC_Villum)))
    print('\n')

    print('EC Zeppelin: mean days between measurements: ', df_EC_Zeppelin['days_diff'].mean())
    print(df_EC_Zeppelin)
    print('EC Zeppelin: unique differences btw days: ',
          dict(zip(unique_diffs_EC_Zeppelin, counts_EC_Zeppelin)))
    print('\n')

    print('SO4 Villum: mean days between measurements: ', df_SO4_Villum['days_diff'].mean())
    print(df_SO4_Villum)
    print('SO4 Villum: unique differences btw days: ',
          dict(zip(unique_diffs_SO4_Villum, counts_SO4_Villum)))
    print('\n')

    print('SO4 Zeppelin: mean days between measurements: ', df_SO4_Zeppelin['days_diff'].mean())
    print(df_SO4_Zeppelin)
    print('SO4 Zeppelin: unique differences btw days: ',
          dict(zip(unique_diffs_SO4_Zeppelin, counts_SO4_Zeppelin)))
    print('\n')

    print(nwvf_ts)
    print('\n\n\n')



##################################################
# PROCESS MODEL DATA TO MATCH OBSERVATIONAL DATA #
##################################################

def process_data(observational_data, model_data, max_sum_days = 14, default_sum_days=7):
    '''
        Function to abbreviate model data to match the observational data.
        The model data is abbreviated by summing the model data over a number of days (default 7 days).
        The number of days to sum over is determined by the observational data.
        The observational data is then matched to the model data by date.

        Parameters:
        -----------
        observational_data: pandas dataframe
            Dataframe containing observational data.
        model_data: pandas dataframe
            Dataframe containing model data.
        max_sum_days: int
            Maximum number of days to sum over. 
        default_sum_days: int
            Default number of days to sum over. 
    '''

    # Make sure observational data is in datetime format
    observational_data['date'] = pd.to_datetime(observational_data['date'])
    # Standardize timestamps to midnight and aggregate into means if necessary
    observational_data['date'] = observational_data['date'].dt.normalize()

    # Make sure model data is also datetime format
    model_data['time'] = pd.to_datetime(model_data['time'])

    # Get the name of the column containing the values to sum over
    value_str = observational_data.columns[1]

    # List to store the results 
    results = []

    # Loop over all rows in observational data
    for index, row in observational_data.iterrows():
        # Get timestamp and days since last measurement from observational data
        obs_ts = row['date']
        obs_days_diff = row['days_diff']


        # Check if previous index exists in the DataFrame
        if index > 0 and (index - 1) in observational_data.index:
            prev_days_diff = observational_data.loc[index - 1, 'days_diff']
        else:
            prev_days_diff = max_sum_days
        
        # If days since last measurement is larger than max_sum_days, use default_sum_days instead
        if obs_days_diff > max_sum_days:
            # Set the start time for the integration period to be the observation time minus the default_sum_days
            start_time = obs_ts - pd.Timedelta(days=default_sum_days)
        else:
            # Set the start time for the integration period to be the observation time minus the number of days since last measurement
            start_time = obs_ts - pd.Timedelta(days=obs_days_diff)

        # Filter the model data for the relevant time period (between start_time and current obs_ts)
        relevant_model_data = model_data[(model_data['time'] >= start_time) & (model_data['time'] < obs_ts)]

        # Sum the model data over the relevant time period
        integral = relevant_model_data['nwvf'].sum()

        # Append the result to the results list
        results.append((obs_ts, integral))


    return pd.DataFrame(results, columns=['date', 'nwvf_time_integral'])

# Process model data in order to match the observational data, all
nwvf_EC_Villum = process_data(df_EC_Villum, nwvf_ts)
nwvf_EC_Zeppelin = process_data(df_EC_Zeppelin, nwvf_ts)
nwvf_SO4_Villum = process_data(df_SO4_Villum, nwvf_ts)
nwvf_SO4_Zeppelin = process_data(df_SO4_Zeppelin, nwvf_ts)

# Save processed model data to csv files
nwvf_EC_Villum.to_csv(DATA_PATH_PROCESSED + 'nwvf_EC_Villum.csv',index=False, header=True)
nwvf_EC_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'nwvf_EC_Zeppelin.csv', index=False, header=True)
nwvf_SO4_Villum.to_csv(DATA_PATH_PROCESSED + 'nwvf_SO4_Villum.csv', index=False, header=True)
nwvf_SO4_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'nwvf_SO4_Zeppelin.csv', index=False, header=True)




################################################
# COMPUTE WEEKLY SUMS OF NWVF AND AEROSOL DATA #
################################################


def weekly_sum(df):
    '''
        Function to sum the data in a dataframe over a week.
        The dataframe must contain a column named 'date' with datetime format.
        The function returns a dataframe containing the weekly sums.
    '''
    # Convert date column to datetime format and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # Resample the data frame into weekly sums
    df_weekly = df.resample('W').sum()

    return df_weekly

# NWVF weekly sums
nwvf_EC_Villum_weekly = weekly_sum(nwvf_EC_Villum)
nwvf_EC_Villum_weekly_nonzero = nwvf_EC_Villum_weekly[nwvf_EC_Villum_weekly['nwvf_time_integral'] != 0]
nwvf_EC_Zeppelin_weekly = weekly_sum(nwvf_EC_Zeppelin)
nwvf_EC_Zeppelin_weekly_nonzero = nwvf_EC_Zeppelin_weekly[nwvf_EC_Zeppelin_weekly['nwvf_time_integral'] != 0]
nwvf_SO4_Villum_weekly = weekly_sum(nwvf_SO4_Villum)
nwvf_SO4_Villum_weekly_nonzero = nwvf_SO4_Villum_weekly[nwvf_SO4_Villum_weekly['nwvf_time_integral'] != 0]
nwvf_SO4_Zeppelin_weekly = weekly_sum(nwvf_SO4_Zeppelin)
nwvf_SO4_Zeppelin_weekly_nonzero = nwvf_SO4_Zeppelin_weekly[nwvf_SO4_Zeppelin_weekly['nwvf_time_integral'] != 0]

weekly_nwvfs = [nwvf_EC_Villum_weekly,
                nwvf_EC_Zeppelin_weekly,
                nwvf_SO4_Villum_weekly,
                nwvf_SO4_Zeppelin_weekly]
weekly_nwvfs_nonzero = [nwvf_EC_Villum_weekly_nonzero,
                        nwvf_EC_Zeppelin_weekly_nonzero,
                        nwvf_SO4_Villum_weekly_nonzero,
                        nwvf_SO4_Zeppelin_weekly_nonzero]

# Aerosol weekly sums
EC_Villum_weekly = weekly_sum(df_EC_Villum)
EC_Villum_weekly_nonzero = EC_Villum_weekly[EC_Villum_weekly['EC'] != 0]
EC_Zeppelin_weekly = weekly_sum(df_EC_Zeppelin)
EC_Zeppelin_weekly_nonzero = EC_Zeppelin_weekly[EC_Zeppelin_weekly['EC'] != 0]
SO4_Villum_weekly = weekly_sum(df_SO4_Villum)
SO4_Villum_weekly_nonzero = SO4_Villum_weekly[SO4_Villum_weekly['SO4'] != 0]
SO4_Zeppelin_weekly = weekly_sum(df_SO4_Zeppelin)
SO4_Zeppelin_weekly_nonzero = SO4_Zeppelin_weekly[SO4_Zeppelin_weekly['SO4'] != 0]

weekly_earosols = [EC_Villum_weekly,
                   EC_Zeppelin_weekly,
                   SO4_Villum_weekly,
                   SO4_Zeppelin_weekly]
weekly_earosols_nonzero = [EC_Villum_weekly_nonzero,
                            EC_Zeppelin_weekly_nonzero,
                            SO4_Villum_weekly_nonzero,
                            SO4_Zeppelin_weekly_nonzero]


# Save weekly sums, aerosols and NWVF in same csv file
weekly_nwvf_EC_Villum = pd.concat([nwvf_EC_Villum_weekly, EC_Villum_weekly], axis=1)
weekly_nwvf_EC_Villum.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_EC_Villum.csv', index=True, header=True)

weekly_nwvf_EC_Zeppelin = pd.concat([nwvf_EC_Zeppelin_weekly, EC_Zeppelin_weekly], axis=1)
weekly_nwvf_EC_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_EC_Zeppelin.csv', index=True, header=True)

weekly_nwvf_SO4_Villum = pd.concat([nwvf_SO4_Villum_weekly, SO4_Villum_weekly], axis=1)
weekly_nwvf_SO4_Villum.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_SO4_Villum.csv', index=True, header=True)

weekly_nwvf_SO4_Zeppelin = pd.concat([nwvf_SO4_Zeppelin_weekly, SO4_Zeppelin_weekly], axis=1)
weekly_nwvf_SO4_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_SO4_Zeppelin.csv',index=True,header=True)



# Save weekly sums, aerosols and NWVF in same csv file, non-zero values only
weekly_nwvf_EC_Villum_nonzero = pd.concat([nwvf_EC_Villum_weekly_nonzero, EC_Villum_weekly_nonzero], axis=1)
weekly_nwvf_EC_Villum_nonzero.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_EC_Villum_nonzero.csv', index=True, header=True)

weekly_nwvf_EC_Zeppelin_nonzero = pd.concat([nwvf_EC_Zeppelin_weekly_nonzero, EC_Zeppelin_weekly_nonzero], axis=1)
weekly_nwvf_EC_Zeppelin_nonzero.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_EC_Zeppelin_nonzero.csv', index=True, header=True)

weekly_nwvf_SO4_Villum_nonzero = pd.concat([nwvf_SO4_Villum_weekly_nonzero, SO4_Villum_weekly_nonzero], axis=1)
weekly_nwvf_SO4_Villum_nonzero.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_SO4_Villum_nonzero.csv', index=True, header=True)

weekly_nwvf_SO4_Zeppelin_nonzero = pd.concat([nwvf_SO4_Zeppelin_weekly_nonzero, SO4_Zeppelin_weekly_nonzero], axis=1)
weekly_nwvf_SO4_Zeppelin_nonzero.to_csv(DATA_PATH_PROCESSED + 'weekly_nwvf_SO4_Zeppelin_nonzero.csv', index=True, header=True)


df_EC_Villum = df_EC_Villum.reset_index()
df_SO4_Villum = df_SO4_Villum.reset_index()
df_EC_Zeppelin = df_EC_Zeppelin.reset_index()
df_SO4_Zeppelin = df_SO4_Zeppelin.reset_index()

nwvf_EC_Villum = nwvf_EC_Villum.reset_index()
nwvf_SO4_Villum = nwvf_SO4_Villum.reset_index()
nwvf_EC_Zeppelin = nwvf_EC_Zeppelin.reset_index()
nwvf_SO4_Zeppelin = nwvf_SO4_Zeppelin.reset_index()


################
# PLOT FIGURES #
################

# Plot EC, SO4 and resampled NWVF in separate subplots - color coded by station and aerosol type
fig, ax = plt.subplots(3,1, figsize=(15,9), sharex=True)

ax[0].plot(df_EC_Villum['date'], df_EC_Villum['EC'], linewidth=0.7, color='k', label='Villum', marker='o', linestyle='-', markersize=2)
ax[0].plot(df_EC_Zeppelin['date'], df_EC_Zeppelin['EC'], linewidth=0.7, color='r', label='Zeppelin', marker='o', linestyle='-', markersize=2)
ax[0].set_title('EC')
ax[0].set_ylabel('EC []')
ax[0].legend()

ax[1].plot(df_SO4_Villum['date'], df_SO4_Villum['SO4'], linewidth=0.7, color='b', label='Villum', marker='o', linestyle='-', markersize=2)
ax[1].plot(df_SO4_Zeppelin['date'], df_SO4_Zeppelin['SO4'], linewidth=0.7, color='g', label='Zeppelin', marker='o', linestyle='-', markersize=2)
ax[1].set_title('SO4')
ax[1].set_ylabel('SO4 []')
ax[1].legend()

ax[2].plot(nwvf_EC_Villum['date'], nwvf_EC_Villum['nwvf_time_integral'], linewidth=0.7, color='k', label='EC Villum', marker='o', linestyle='-', markersize=2)
ax[2].plot(nwvf_EC_Zeppelin['date'], nwvf_EC_Zeppelin['nwvf_time_integral'], linewidth=0.7, color='r', label='EC Zeppelin', marker='o', linestyle='-', markersize=2)
ax[2].plot(nwvf_SO4_Villum['date'], nwvf_SO4_Villum['nwvf_time_integral'], linewidth=0.7, color='b', label='SO4 Villum', marker='o', linestyle='-', markersize=2)
ax[2].plot(nwvf_SO4_Zeppelin['date'], nwvf_SO4_Zeppelin['nwvf_time_integral'], linewidth=0.7, color='g', label='SO4 Zeppelin', marker='o', linestyle='-', markersize=2)
ax[2].set_ylim([min(nwvf_ts['nwvf'])-1e7, 1.5e7])
ax[2].set_title('NWVF time integral')
ax[2].set_ylabel('NWVF time integral [kg m$^{-1}$]')
ax[2].set_xlabel('Year')
ax[2].legend()
fig.tight_layout()



# Store processed NWVF and aerosol data in lists
nwvf_processed_data = [nwvf_EC_Villum, nwvf_EC_Zeppelin, nwvf_SO4_Villum, nwvf_SO4_Zeppelin]
aerosol_data = [df_EC_Villum, df_EC_Zeppelin, df_SO4_Villum, df_SO4_Zeppelin]
aerosol_names = ['EC_Villum', 'EC_Zeppelin', 'SO4_Villum', 'SO4_Zeppelin']


# Plot NWVF and corresponding aerosol data in separate subplots, with different y-axes
fig, ax = plt.subplots(4,1, figsize=(15,9), sharex=True)
fig.suptitle('NWVF time integral and corresponding aerosol data', fontsize=16)

for i, nwvf_data in enumerate(nwvf_processed_data):
    ax[i].plot(nwvf_data['date'], nwvf_data['nwvf_time_integral'], linewidth=0.7, color='k', label='NWVF', marker='o', linestyle='-', markersize=2)
    ax[i].set_title(aerosol_names[i].split('_')[0] + ' ' + aerosol_names[i].split('_')[1])
    ax[i].set_ylabel('NWVF time integral')
    ax[i].set_ylim([-2e7, 1.5e7])
    ax2 = ax[i].twinx()
    ax2.plot(aerosol_data[i]['date'], aerosol_data[i][aerosol_names[i].split('_')[0]], linewidth=0.7, color='r', label=aerosol_names[i].split('_')[0], marker='o', linestyle='-', markersize=2)
    ax2.set_ylabel(aerosol_names[i].split('_')[0])
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')


    ax[i].legend(loc='upper left')
    ax2.legend(loc='upper right')
fig.tight_layout()
fig.savefig(FIG_PATH + 'nwvf_aerosol__matched.png', dpi=600, bbox_inches='tight')
 

#
fig, ax = plt.subplots(4,1, figsize=(15,9), sharex=True)
fig.suptitle('NWVF aerosol weekly sums', fontsize=16)

for i, df_mod, df_obs in zip(range(len(weekly_nwvfs)), weekly_nwvfs_nonzero, weekly_earosols_nonzero):
    ax[i].plot(df_mod.index, df_mod['nwvf_time_integral'], linewidth=0.7, color='k', label='NWVF', marker='o', linestyle='-', markersize=2)
    ax[i].set_title(aerosol_names[i].split('_')[0] + ' ' + aerosol_names[i].split('_')[1])
    ax[i].set_ylabel('NWVF time integral')
    ax[i].set_ylim([-2e7, 1.5e7])
    ax2 = ax[i].twinx()
    ax2.plot(df_obs.index, df_obs[aerosol_names[i].split('_')[0]], linewidth=0.7, color='r', label=aerosol_names[i].split('_')[0], marker='o', linestyle='-', markersize=2)
    ax2.set_ylabel(aerosol_names[i].split('_')[0])
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')

    ax[i].legend(loc='upper left')
    ax2.legend(loc='upper right')

fig.tight_layout()
fig.savefig(FIG_PATH + 'nwvf_aerosol__matched__weekly_sums.png', dpi=600, bbox_inches='tight')







#################################################
# COMPUTE MONTHLY SUMS OF NWVF AND AEROSOL DATA #
#################################################


def monthly_sum(df):
    '''
        Function to sum the data in a dataframe over a week.
        The dataframe must contain a column named 'date' with datetime format.
        The function returns a dataframe containing the weekly sums.
    '''
    # Convert date column to datetime format and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    # Resample the data frame into weekly sums
    df_monthly = df.resample('M').sum()

    return df_monthly

# NWVF monthly sums
nwvf_EC_Villum_monthly = monthly_sum(nwvf_EC_Villum)
nwvf_EC_Villum_monthly_nonzero = nwvf_EC_Villum_monthly[nwvf_EC_Villum_monthly['nwvf_time_integral'] != 0]
nwvf_EC_Zeppelin_monthly = monthly_sum(nwvf_EC_Zeppelin)
nwvf_EC_Zeppelin_monthly_nonzero = nwvf_EC_Zeppelin_monthly[nwvf_EC_Zeppelin_monthly['nwvf_time_integral'] != 0]
nwvf_SO4_Villum_monthly = monthly_sum(nwvf_SO4_Villum)
nwvf_SO4_Villum_monthly_nonzero = nwvf_SO4_Villum_monthly[nwvf_SO4_Villum_monthly['nwvf_time_integral'] != 0]
nwvf_SO4_Zeppelin_monthly = monthly_sum(nwvf_SO4_Zeppelin)
nwvf_SO4_Zeppelin_monthly_nonzero = nwvf_SO4_Zeppelin_monthly[nwvf_SO4_Zeppelin_monthly['nwvf_time_integral'] != 0]

monthly_nwvfs = [nwvf_EC_Villum_monthly,
                nwvf_EC_Zeppelin_monthly,
                nwvf_SO4_Villum_monthly,
                nwvf_SO4_Zeppelin_monthly]
monthly_nwvfs_nonzero = [nwvf_EC_Villum_monthly_nonzero,
                        nwvf_EC_Zeppelin_monthly_nonzero,
                        nwvf_SO4_Villum_monthly_nonzero,
                        nwvf_SO4_Zeppelin_monthly_nonzero]

# Aerosol monthly sums
EC_Villum_monthly = monthly_sum(df_EC_Villum)
EC_Villum_monthly_nonzero = EC_Villum_monthly[EC_Villum_monthly['EC'] != 0]
EC_Zeppelin_monthly = monthly_sum(df_EC_Zeppelin)
EC_Zeppelin_monthly_nonzero = EC_Zeppelin_monthly[EC_Zeppelin_monthly['EC'] != 0]
SO4_Villum_monthly = monthly_sum(df_SO4_Villum)
SO4_Villum_monthly_nonzero = SO4_Villum_monthly[SO4_Villum_monthly['SO4'] != 0]
SO4_Zeppelin_monthly = monthly_sum(df_SO4_Zeppelin)
SO4_Zeppelin_monthly_nonzero = SO4_Zeppelin_monthly[SO4_Zeppelin_monthly['SO4'] != 0]

monthly_earosols = [EC_Villum_monthly,
                   EC_Zeppelin_monthly,
                   SO4_Villum_monthly,
                   SO4_Zeppelin_monthly]
monthly_earosols_nonzero = [EC_Villum_monthly_nonzero,
                            EC_Zeppelin_monthly_nonzero,
                            SO4_Villum_monthly_nonzero,
                            SO4_Zeppelin_monthly_nonzero]


# Save monthly sums, aerosols and NWVF in same csv file
monthly_nwvf_EC_Villum = pd.concat([nwvf_EC_Villum_monthly, EC_Villum_monthly], axis=1)
monthly_nwvf_EC_Villum.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_EC_Villum.csv', index=True, header=True)

monthly_nwvf_EC_Zeppelin = pd.concat([nwvf_EC_Zeppelin_monthly, EC_Zeppelin_monthly], axis=1)
monthly_nwvf_EC_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_EC_Zeppelin.csv', index=True, header=True)

monthly_nwvf_SO4_Villum = pd.concat([nwvf_SO4_Villum_monthly, SO4_Villum_monthly], axis=1)
monthly_nwvf_SO4_Villum.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_SO4_Villum.csv', index=True, header=True)

monthly_nwvf_SO4_Zeppelin = pd.concat([nwvf_SO4_Zeppelin_monthly, SO4_Zeppelin_monthly], axis=1)
monthly_nwvf_SO4_Zeppelin.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_SO4_Zeppelin.csv', index=True, header=True)



# Save monthly sums, aerosols and NWVF in same csv file, non-zero values only
monthly_nwvf_EC_Villum_nonzero = pd.concat([nwvf_EC_Villum_monthly_nonzero, EC_Villum_monthly_nonzero], axis=1)
monthly_nwvf_EC_Villum_nonzero.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_EC_Villum_nonzero.csv', index=True, header=True)

monthly_nwvf_EC_Zeppelin_nonzero = pd.concat([nwvf_EC_Zeppelin_monthly_nonzero, EC_Zeppelin_monthly_nonzero], axis=1)
monthly_nwvf_EC_Zeppelin_nonzero.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_EC_Zeppelin_nonzero.csv', index=True, header=True)

monthly_nwvf_SO4_Villum_nonzero = pd.concat([nwvf_SO4_Villum_monthly_nonzero, SO4_Villum_monthly_nonzero], axis=1)
monthly_nwvf_SO4_Villum_nonzero.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_SO4_Villum_nonzero.csv', index=True, header=True)

monthly_nwvf_SO4_Zeppelin_nonzero = pd.concat([nwvf_SO4_Zeppelin_monthly_nonzero, SO4_Zeppelin_monthly_nonzero], axis=1)
monthly_nwvf_SO4_Zeppelin_nonzero.to_csv(DATA_PATH_PROCESSED + 'monthly_nwvf_SO4_Zeppelin_nonzero.csv', index=True, header=True)


df_EC_Villum = df_EC_Villum.reset_index()
df_SO4_Villum = df_SO4_Villum.reset_index()
df_EC_Zeppelin = df_EC_Zeppelin.reset_index()
df_SO4_Zeppelin = df_SO4_Zeppelin.reset_index()

nwvf_EC_Villum = nwvf_EC_Villum.reset_index()
nwvf_SO4_Villum = nwvf_SO4_Villum.reset_index()
nwvf_EC_Zeppelin = nwvf_EC_Zeppelin.reset_index()
nwvf_SO4_Zeppelin = nwvf_SO4_Zeppelin.reset_index()




################
# PLOT FIGURES #
################


#
fig, ax = plt.subplots(4,1, figsize=(15,9), sharex=True)
fig.suptitle('NWVF aerosol monthly sums', fontsize=16)

for i, df_mod, df_obs in zip(range(len(monthly_nwvfs)),
                             monthly_nwvfs_nonzero, monthly_earosols_nonzero):
    ax[i].plot(df_mod.index, df_mod['nwvf_time_integral'],
               linewidth=0.7,
               color='k',
               label='NWVF',
               marker='o',
               linestyle='-',
               markersize=2)
    ax[i].set_title(aerosol_names[i].split('_')[0] + ' ' + aerosol_names[i].split('_')[1])
    ax[i].set_ylabel('NWVF time integral')
    ax[i].set_ylim([-2e7, 1.5e7])
    ax2 = ax[i].twinx()
    ax2.plot(df_obs.index, df_obs[aerosol_names[i].split('_')[0]],
             linewidth=0.7,
             color='r',
             label=aerosol_names[i].split('_')[0],
             marker='o',
             linestyle='-',
             markersize=2)
    ax2.set_ylabel(aerosol_names[i].split('_')[0])
    ax2.yaxis.label.set_color('r')
    ax2.tick_params(axis='y', colors='r')

    ax[i].legend(loc='upper left')
    ax2.legend(loc='upper right')

fig.tight_layout()
fig.savefig(FIG_PATH + 'nwvf_aerosol__matched__monthly_sums.png',
            dpi=600, bbox_inches='tight')



plt.show()
