import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from Scripts.Analysis.event_analysis_class import EventAnalysis

if __name__ == '__main__':
    # Create dataframe with different seasons (based on model_aerosol_climatology.py)
    # Five columns (nwvf_time_integral, Dust, SO4, BC, OC), two rows, one for each season
    # Each season is defined by a list of months

    
    # Define the seasons
    s1_nwvf = [7, 8, 9, 10, 11]
    s2_nwvf = [2, 3, 4, 5, 6] 

    s1_dust = [3, 4, 5, 6]
    s2_dust = [1, 8, 9, 10, 11, 12]

    s1_so4 = [4, 5, 6, 7, 8]
    s2_so4 = [1, 2, 10, 11, 12]

    s1_bc = [4, 5, 6, 7]
    s2_bc = [1, 9, 10, 11, 12]

    s1_tot = [4, 5, 6, 7]
    s2_tot = [1, 9, 10, 11, 12]


    df_seasons = pd.DataFrame(index=['Season1', 'Season2'], columns=['Dust', 'SO4', 'BC', 'OC', 'Total'])
    df_seasons.loc['Season1'] = [s1_dust, s1_so4, s1_bc, s1_bc, s1_tot]
    df_seasons.loc['Season2'] = [s2_dust, s2_so4, s2_bc, s2_bc, s2_tot]








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

    # Loop through the different aerosols and perform the event analysis
    
    for aerosol in df_aerosol.columns:

        df_aerosol_temp = df_aerosol[[aerosol]]
        seasons = {'Season1': df_seasons.loc['Season1'][aerosol], 'Season2': df_seasons.loc['Season2'][aerosol]}


        event_analysis = EventAnalysis(df_aerosol_temp,
                                       aerosol,
                                       df_NWVF,
                                       'nwvf_time_integral',
                                       seasons,
                                       n_sigmas)

        bins_nwvf = {'Season1': [200,20,5], 'Season2': [50, 6, 4]}
        bins_aerosol = {'Season1': [100, 20, 5], 'Season2':[200, 20, 10]}

        event_analysis.plot_seasonal_data(bins_nwvf,
                                          bins_aerosol,
                                          'ArcticSlice_small',
                                          show_figs=True,
                                          save_figs=True,
                                          save_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/Figures/MERRA/ArcticSlice_small/',
                                          plot_events_neg=False
                                          )

        # Plot all data
        event_analysis.plot_all_data('ArcticSlice_small', show_figs=True, save_figs=True, save_path='/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/Figures/MERRA/ArcticSlice_small/', plot_events_neg=False)
        

