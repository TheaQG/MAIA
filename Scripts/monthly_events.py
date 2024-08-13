import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from event_analysis_class import EventAnalysis

if __name__ == '__main__':

    # NWVF data
    data_NWVF = xr.open_dataset('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc')
    
    # Resaple by creating daily sum
    daily_NWVF = data_NWVF['nwvf_integral'].resample(time='1D').sum()
    df_NWVF = pd.DataFrame({'time': daily_NWVF['time'],
                            'nwvf_time_integral': daily_NWVF.data})

    df_NWVF['time'] = pd.to_datetime(df_NWVF['time'])
    
    # Filter data to only after 1997 and before 2022-07-01
    df_NWVF = df_NWVF[df_NWVF['time'] < '2022-07-01']
    df_NWVF = df_NWVF[df_NWVF['time'] > '1997-01-01']
    
    df_NWVF.set_index('time', inplace=True)

    # Aerosol data
    df_aerosol = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/'
                          +'TS_AerosolBurden_ArcticSlice_small.txt',
                          index_col=0, parse_dates=True).drop(columns=['OC', 'SS', 'PM25'])

    # Add a column, that is total sum of all aerosol components
    df_aerosol['Total'] = df_aerosol.sum(axis=1)

    df_aerosol.reset_index(inplace=True)

    # Filter data to only after 1997 and before 2022-07-01
    df_aerosol = df_aerosol[df_aerosol['date'] < '2022-07-01']
    df_aerosol = df_aerosol[df_aerosol['date'] > '1997-01-01'] 

    df_aerosol.set_index('date', inplace=True)



    n_sigmas = 2

    # Check how many events for NWVF exist for each month
    threshold = df_NWVF['nwvf_time_integral'].mean() + n_sigmas*df_NWVF['nwvf_time_integral'].std()
    df_NWVF['event'] = df_NWVF['nwvf_time_integral'] > threshold

    df_NWVF['month'] = df_NWVF.index.month

    # Count number of events per month
    df_NWVF['event'] = df_NWVF['event'].astype(int)

    df_NWVF_monthly = df_NWVF.groupby('month').sum()

    # Plot the number of events per month
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.bar(df_NWVF_monthly.index, df_NWVF_monthly['event'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of events')
    ax.set_title('Number of NWVF events per month')
    




    # Make the same analysis, but for separate months in separate years
    df_NWVF['event'] = df_NWVF['nwvf_time_integral'] > threshold

    df_NWVF['year'] = df_NWVF.index.year
    df_NWVF['month'] = df_NWVF.index.month

    # Count number of events per month
    df_NWVF['event'] = df_NWVF['event'].astype(int)

    df_NWVF_monthly = df_NWVF.groupby(['year', 'month']).sum()


    
    
    

    x = np.linspace(df_NWVF['year'].min(), df_NWVF['year'].max(), len(df_NWVF_monthly))

    # Plot the number of events per month
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(x, df_NWVF_monthly['event'])
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of events')
    ax.set_title('Number of NWVF events per month')
    plt.show()

    
