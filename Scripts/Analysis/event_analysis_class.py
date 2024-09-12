'''
    This file contains a class called EventAnalysis that is used to perform event analysis 
    on aerosol and nwvf (non-wind vector) data. The class has several methods:
        - get_seasonal_data(): This method retrieves the seasonal data for both aerosol 
                                and nwvf data by cropping the data based on the specified seasons.
        - calculate_stat_params(): This method calculates the statistical parameters (mean and 
                                standard deviation) for both aerosol and nwvf data.
        - mask_data_by_stats(): This method masks the data based on the statistical parameters. 
                                It creates masks for both positive and negative deviations from 
                                the mean and applies them to the nwvf and aerosol data.
        - plot_seasonal_data(): This method plots the seasonal data in histograms, both with and 
                                without masking. It takes dictionaries of bins for the nwvf and 
                                aerosol data for each season as input.
        - plot_all_data(): This method is not implemented yet.

To use the EventAnalysis class, you need to provide the aerosol and nwvf data as pandas DataFrames, along with their respective names. You also need to specify the seasons as a dictionary, where each key is the name of a season and the value is a list of months corresponding to that season. Additionally, you can specify the number of standard deviations to use for the threshold when masking the data.

To create an instance of the EventAnalysis class, you can use the following code:
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr



class EventAnalysis:
    '''
        Class to perform event analysis on aerosol and nwvf data.
        Methods:
        --------
        get_seasonal_data()
            - Get the seasonal data for both aerosol and nwvf data
        calculate_stat_params()
            - Calculate the statistical parameters for nwvf and aerosol data
        mask_data_by_stats()
            - Mask the data based on the statistical parameters
        plot_seasonal_data()
            - Plot the seasonal data
        plot_all_data()
            - Plot all the data (time series)
    '''
    def __init__(self, aerosol_data, aerosol_name, nwvf_data, nwvf_name, seasons, n_sigmas=1):
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
            seasons : dict
                Dictionary containing the seasons (as lists of months) with the key as the season name.
            n_sigmas : int
                Number of standard deviations to use for the threshold.
        '''
        self.aerosol_data = aerosol_data
        self.aerosol_name = aerosol_name
        self.nwvf_data = nwvf_data
        self.nwvf_name = nwvf_name
        self.seasons = seasons
        self.n_seasons = len(seasons)
        self.n_sigmas = n_sigmas
        import matplotlib.pyplot as plt

        
    def get_seasonal_data(self):
        ''' 
            Get seasonal data of both aerosol and nwvf data.
            Crops data based on the seasons.
        '''

        # Aerosol dictionary for storing seasonal data
        self.aerosol_data_seasonal = {}
        # NWVF dictionary for storing seasonal data
        self.nwvf_data_seasonal = {}


        for season in self.seasons:
            # Get the months in the season
            months_in_season = self.seasons[season]
            # Crop the data based on the months in the season
            self.aerosol_data_seasonal[season] = self.aerosol_data[self.aerosol_data.index.month.isin(months_in_season)][self.aerosol_name]
            self.nwvf_data_seasonal[season] = self.nwvf_data[self.nwvf_data.index.month.isin(months_in_season)][self.nwvf_name]
        
        return self.aerosol_data_seasonal, self.nwvf_data_seasonal
    
    def calculate_seasonal_statistical_params(self):
        '''
            Calculate the statistical parameters for nwvf data.
            Used to test correlation between aerosol and nwvf data.
        '''

        # Get seasonal data
        self.get_seasonal_data()

        # Loop through seasonal nwvf data
        self.nwvf_stats = {}
        for season in self.nwvf_data_seasonal:
            # Get the seasonal data
            data = self.nwvf_data_seasonal[season]
            # Calculate the statistical parameters
            mean = data.mean()
            std = data.std()
            self.nwvf_stats[season] = {'mean': mean, 'std': std}

        return self.nwvf_stats
    
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
        self.calculate_seasonal_statistical_params()

        # Mask the data based on the statistical parameters
        self.masked_nwvf_data_pos = {}
        self.masked_nwvf_data_neg = {}

        self.masked_aerosol_data_pos = {}
        self.masked_aerosol_data_neg = {}

        for season in self.nwvf_data_seasonal:
            # Get the seasonal data
            data_nwvf = self.nwvf_data_seasonal[season]
            data_aerosol = self.aerosol_data_seasonal[season]
            
            # Get the statistical parameters
            mean = self.nwvf_stats[season]['mean']
            std = self.nwvf_stats[season]['std']

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
            self.masked_nwvf_data_pos[season] = masked_nwvf_pos
            self.masked_nwvf_data_neg[season] = masked_nwvf_neg

            self.masked_aerosol_data_pos[season] = masked_aerosol_pos
            self.masked_aerosol_data_neg[season] = masked_aerosol_neg

        return self.masked_nwvf_data_pos, self.masked_nwvf_data_neg, self.masked_aerosol_data_pos, self.masked_aerosol_data_neg

    def plot_seasonal_data(self, bins_nwvf, bins_aerosol, station_name, show_figs=True, save_figs=False, save_path=None, plot_events_pos=True, plot_events_neg=True):
        '''
            Plot the seasonal data in histograms - with and without masking.

            Parameters:
            -----------
            bins_nwvf : dict
                Dictionary of list of bins for the nwvf data for each season.
                Ex.: bins_nwvf = {'Season1': [50, 20, 15], 'Season2': [60, 18, 12]}
            bins_aerosol : dict
                Dictionary of list of bins for the aerosol data for each season.
                Ex.: bins_aerosol = {'Season1': [50, 20, 15], 'Season2': [60, 18, 12]}
        '''

        # Mask the data based on the statistical parameters
        masked_nwvf_data_pos, masked_nwvf_data_neg, masked_aerosol_data_pos, masked_aerosol_data_neg = self.mask_data_by_stats()

        # Get the statistics to plot as well
        nwvf_stats = self.calculate_seasonal_statistical_params()

        # Store the figures and axes
        self.figures = []
        self.axes = []
        

        # Plot a figure for each season (with two subplots, for nwvf and aerosol data)
        for i, season in enumerate(list(self.nwvf_data_seasonal.keys())):
            
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            fig.suptitle(f'{self.aerosol_name} {station_name}: {season}, +/- {self.n_sigmas} $\sigma$ events')
            
            
            # NWVF DATA PLOT
            
            # Bins for the histograms
            bins_nwvf_season = bins_nwvf[season]

            ax[0].hist(self.nwvf_data_seasonal[season],
                       bins=bins_nwvf_season[0],
                       alpha=0.9,
                       label=f'All data, {len(self.nwvf_data_seasonal[season])} events',
                       ec='darkblue',
                       fc='lightblue')
            if plot_events_pos:
                ax[0].hist(masked_nwvf_data_pos[season],
                        bins=bins_nwvf_season[1],
                        alpha=0.9,
                        label=F'+{self.n_sigmas} sigma, {len(masked_nwvf_data_pos[season])} events',
                        ec='darkgreen',
                        fc='lightgreen')
            if plot_events_neg:
                ax[0].hist(masked_nwvf_data_neg[season],
                        bins=bins_nwvf_season[2],
                        alpha=0.9,
                        label=f'-{self.n_sigmas} sigma, {len(masked_nwvf_data_neg[season])} events',
                        ec='darkred',
                        fc='lightcoral')
            # Draw vertical lines at mean and mean +/- n_sigmas * std
            ax[0].axvline(nwvf_stats[season]['mean'], color='k', linestyle='--', linewidth=1, label='Mean')
            if plot_events_pos:
                ax[0].axvline(nwvf_stats[season]['mean'] + self.n_sigmas * nwvf_stats[season]['std'], color='k', linestyle=':', linewidth=1, label=f'Mean + {self.n_sigmas} sigma')
            if plot_events_neg:
                ax[0].axvline(nwvf_stats[season]['mean'] - self.n_sigmas * nwvf_stats[season]['std'], color='k', linestyle=':', linewidth=1, label=f'Mean - {self.n_sigmas} sigma')

            ax[0].set_title('NWVF')
            ax[0].legend()


            # AEROSOL DATA PLOT

            # Bins for the histograms
            bins_aerosol_season = bins_aerosol[season]

            ax[1].hist(self.aerosol_data_seasonal[season],
                       bins=bins_aerosol_season[0],
                       alpha=0.9,
                       label=f'All data, {len(self.aerosol_data_seasonal[season])} events',
                       ec='darkblue',
                       fc='lightblue')
            if plot_events_pos:
                ax[1].hist(masked_aerosol_data_pos[season],
                        bins=bins_aerosol_season[1],
                        alpha=0.9,
                        label=F'+{self.n_sigmas} sigma, {len(masked_aerosol_data_pos[season])} events',
                        ec='darkgreen',
                           fc='lightgreen')
            if plot_events_neg:
                ax[1].hist(masked_aerosol_data_neg[season],
                        bins=bins_aerosol_season[2],
                        alpha=0.9,
                        label=f'-{self.n_sigmas} sigma, {len(masked_aerosol_data_neg[season])} events',
                        ec='darkred',
                        fc='lightcoral')
            
            ax[1].set_title(self.aerosol_name)
            ax[1].legend()

            fig.tight_layout()

            self.figures.append(fig)
            self.axes.append(ax)

            if save_figs:
                try:
                    file_path = save_path + f'{self.aerosol_name}_NWVF_{station_name}_hist_{season}_{self.n_sigmas}.png'

                    print(f'Saving figure: {file_path}')
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                except Exception as e:
                    print(f'Error saving figure: {e}')

        
        if show_figs:
            plt.show()
        

        return self.figures, self.axes
    
    def plot_all_data(self, station_name, bins_nwvf=32, bins_aerosol=32, show_figs=True, save_figs=False, save_path=None, plot_events_pos=True, plot_events_neg=True):
        '''
            Plot aerosol and nwvf data (not masked) in time series and corresponding histograms.

        '''

        # Get the general statistics
        aerosol_stats, nwvf_stats = self.calculate_statistical_params()

        fig, (ax1, ax2) = plt.subplots(nrows=2,
                                       ncols=2,
                                       figsize=(15, 8),
                                       gridspec_kw={'width_ratios': [3, 1], 'wspace': 0},
                                       sharey='row'
                                       )

        fig.suptitle(f'{self.aerosol_name} and NWVF {station_name} data')
        N_SIG = self.n_sigmas

        # NWVF time series
        ax1[0].plot(self.nwvf_data, label='NWVF', color='k')
        ax1[0].set_title('NWVF')
        # Draw horizontal lines at mean and mean +/- n_sigmas * std
        ax1[0].axhline(nwvf_stats['mean'], color='b', linestyle='--', linewidth=1, label='Mean')
        if plot_events_pos:
            ax1[0].axhline(nwvf_stats['mean'] + N_SIG * nwvf_stats['std'], color='b', linestyle=':', linewidth=1, label=f'+ {N_SIG} sigma')
        if plot_events_neg:
            ax1[0].axhline(nwvf_stats['mean'] - N_SIG * nwvf_stats['std'], color='b', linestyle=':', linewidth=1, label=f'- {N_SIG} sigma')
        ax1[0].legend()

        # NWVF histogram (approx. gaussian)
        ax1[1].hist(self.nwvf_data, bins=bins_nwvf, ec='k', fc='none', orientation='horizontal')
        ax1[1].tick_params(axis='y', left=False)

        # Aerosol time series
        ax2[0].plot(self.aerosol_data, label=self.aerosol_name, color='k')
        ax2[0].set_title(f'{self.aerosol_name} Aerosols')
        # Draw horizontal lines at mean and mean + n_sigmas * std (not gaussian)
        ax2[0].axhline(aerosol_stats['mean'], color='b', linestyle='--', linewidth=1, label='Mean')
        ax2[0].axhline(aerosol_stats['mean'] + N_SIG * aerosol_stats['std'], color='b', linestyle=':', linewidth=1, label=f'+ {N_SIG} sigma')
        ax2[0].legend()

        # Aerosol histogram (not gaussian)
        ax2[1].hist(self.aerosol_data, bins=bins_aerosol, ec='k', fc='none', orientation='horizontal')
        ax2[1].tick_params(axis='y', left=False)

        fig.tight_layout()

        if show_figs:
            plt.show()

        if save_figs:
            print(f'Saving figure: {save_path + f"{self.aerosol_name}_NWVF_{station_name}_timeseries_hist.png"}')
            fig.savefig(save_path + f'{self.aerosol_name}_NWVF_{station_name}_timeseries_hist.png', dpi=300, bbox_inches='tight')


        return fig, (ax1, ax2)





if __name__ == '__main__':
    
    PATH_DATA = '../../Data/'
    PATH_FIGS = '../../Figures/'

    # Load EC data 
    data_EC = pd.read_csv(PATH_DATA + 'ModelData/TS_AerosolBurden_ArcticSlice_small.txt',
                          index_col=0, parse_dates=True).drop(columns=['Dust', 'OC', 'BC', 'SS', 'PM25'])

    df_EC = data_EC

    # Load NWVF data, get daily sum and create data frame
    data_NWVF = xr.open_dataset(PATH_DATA + '/Processed_MAIA/MAIA_nwvf_integrals_1994-2022__6hr_mean.nc')
    # Resaple by creating daily sum
    daily_NWVF = data_NWVF['nwvf_integral'].resample(time='1D').sum()
    df_NWVF = pd.DataFrame({'time': daily_NWVF['time'],
                            'nwvf_time_integral': daily_NWVF.data})
    
    df_NWVF['time'] = pd.to_datetime(df_NWVF['time'])

    df_EC.reset_index(inplace=True)

    # Filter data to only after 1997 and before 2022-07-01
    df_EC = df_EC[df_EC['date'] < '2022-07-01']
    df_EC = df_EC[df_EC['date'] > '1997-01-01'] 
    df_NWVF = df_NWVF[df_NWVF['time'] < '2022-07-01']
    df_NWVF = df_NWVF[df_NWVF['time'] > '1997-01-01']

    df_EC.set_index('date', inplace=True)
    df_NWVF.set_index('time', inplace=True)
    # print(df_NWVF.head())
    # print(df_EC.head())
    # print(df_EC.keys())

    # Define seasons
    season1 = [12, 1, 2, 3, 4] # DJFMA
    season2 = [7, 8, 9, 10] # JAS
    
    # Define the number of standard deviations
    n_sigmas = 2
    
    aerosol_dfs = [df_EC]#, df_EC_Zep, df_SO4_Vil, df_SO4_Zep]
    nwvf_dfs = [df_NWVF]#, nwvf_EC_Zep, nwvf_SO4_Vil, nwvf_SO4_Zep]
    aerosol_names = ['SO4']#, 'EC', 'BC', 'SO4']
    station_names = ['Arcticslice']#, 'Zeppelin', 'Villum', 'Zeppelin']
    nwvf_season1_bins = [[200, 20, 5]]#, [32, 16, 10], [40, 20, 8], [80, 40, 16]]
    aerosol_season1_bins =[[100, 20, 5]]#, [32, 20, 20], [40, 16, 16], [80, 30, 30]] 
    nwvf_season2_bins = [[50, 6, 4]]#, [24, 10, 8], [24, 12, 12], []]
    aerosol_season2_bins =[[200, 20, 10]]#, [32, 32, 20], [24, 20, 20], []] 

    seasons = {'Season1': season1, 'Season2': season2}

    path_save = PATH_FIGS + 'ObsResults/MERRA_aerosols_v_ERA5_NWVF/'

    # Loop through different stations/aerosol sets
    for i in range(len(aerosol_dfs)):
        if i == 3:
            seasons = {'Season1': season1}

        event_analysis = EventAnalysis(aerosol_dfs[i], aerosol_names[i], nwvf_dfs[i], 'nwvf_time_integral', seasons, n_sigmas)

        # Define bins pairs
        bins_nwvf = {'Season1': nwvf_season1_bins[i], 'Season2': nwvf_season2_bins[i]}
        bins_aerosol = {'Season1': aerosol_season1_bins[i], 'Season2': aerosol_season2_bins[i]}

        # Plot the seasonal data
        event_analysis.plot_seasonal_data(bins_nwvf, bins_aerosol, station_names[i], show_figs=True, save_figs=True, save_path=path_save)

        # Plot all data
        event_analysis.plot_all_data(station_names[i], show_figs=True, save_figs=True, save_path=path_save)
        




    # Create a three-day moving average and test again
    df_EC_WA = df_EC['SO4'].rolling(window=3).mean()
    df_NWVF_WA = df_NWVF['nwvf_time_integral'].rolling(window=3).mean()
    df_EC_WA = pd.DataFrame(df_EC_WA)
    df_NWVF_WA = pd.DataFrame(df_NWVF)
    print(df_EC_WA.head(10))
    print(df_NWVF_WA.head(10))


    event_analysis = EventAnalysis(df_EC_WA, 'SO4', df_NWVF_WA, 'nwvf_time_integral', seasons, n_sigmas)

    # Define bins pairs
    bins_nwvf = {'Season1': [200, 30, 5], 'Season2': [200, 30, 20]}
    bins_aerosol = {'Season1': [200, 100, 15], 'Season2': [200, 20, 15]}

    # Plot the seasonal data
    event_analysis.plot_seasonal_data(bins_nwvf, bins_aerosol, 'ArcticSlice_3Day', show_figs=True, save_figs=True, save_path=path_save)

    # Plot all data
    event_analysis.plot_all_data('ArcticSlice_3Day', show_figs=True, save_figs=True, save_path=path_save)




    # Compute a 7 day moving average of EC and NWVF
    df_EC_WA = df_EC['SO4'].rolling(window=7).mean()
    df_NWVF_WA = df_NWVF['nwvf_time_integral'].rolling(window=7).mean()
    df_EC_WA = pd.DataFrame(df_EC_WA)
    df_NWVF_WA = pd.DataFrame(df_NWVF)
    print(df_EC_WA.head(10))
    print(df_NWVF_WA.head(10))


    event_analysis = EventAnalysis(df_EC_WA, 'SO4', df_NWVF_WA, 'nwvf_time_integral', seasons, n_sigmas)

    # Define bins pairs
    bins_nwvf = {'Season1': [100, 20, 8], 'Season2': [100, 20, 20]}
    bins_aerosol = {'Season1': [100, 20, 10], 'Season2': [100, 20, 32]}

    # Plot the seasonal data
    event_analysis.plot_seasonal_data(bins_nwvf, bins_aerosol, 'ArcticSlice_7Days', show_figs=True, save_figs=True, save_path=path_save)

    # Plot all data
    event_analysis.plot_all_data('ArcticSlice_7Days', show_figs=True, save_figs=True, save_path=path_save)






    # # Load NWVF data (filtered to match aerosol data)
    # nwvf_EC_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                             + 'MAIA_processed/Processed_MAIA/nwvf_EC_Villum.csv',
    #                             index_col=0, parse_dates=True)
    # nwvf_EC_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                             + 'MAIA_processed/Processed_MAIA/nwvf_EC_Zeppelin.csv',
    #                             index_col=0, parse_dates=True)
    # nwvf_SO4_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                             + 'MAIA_processed/Processed_MAIA/nwvf_SO4_Villum.csv',
    #                             index_col=0, parse_dates=True)
    # nwvf_SO4_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                                 + 'MAIA_processed/Processed_MAIA/nwvf_SO4_Zeppelin.csv',
    #                                 index_col=0, parse_dates=True)

    # # Load aerosol data
    # df_EC_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                         + 'MAIA_processed/Processed_MAIA/df_EC_Villum.csv',
    #                         index_col=0, parse_dates=True).drop(columns='days_diff')
    # df_EC_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                             + 'MAIA_processed/Processed_MAIA/df_EC_Zeppelin.csv',
    #                             index_col=0, parse_dates=True).drop(columns='days_diff')
    # df_SO4_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                             + 'MAIA_processed/Processed_MAIA/df_SO4_Villum.csv',
    #                             index_col=0, parse_dates=True).drop(columns='days_diff')
    # df_SO4_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
    #                             + 'MAIA_processed/Processed_MAIA/df_SO4_Zeppelin.csv',
    #                             index_col=0, parse_dates=True).drop(columns='days_diff')
    
    # # Replace values above 2.0e7 with NaN in EC Vil data
    # nwvf_EC_Vil = nwvf_EC_Vil.where(nwvf_EC_Vil < 2.0e7, np.nan)
    # # Replace values above 2.0e7 with NaN in EC Zep data
    # nwvf_EC_Zep = nwvf_EC_Zep.where(nwvf_EC_Zep < 2.0e7, np.nan)
    # # Replace values above 2.0e7 with NaN in SO4 Vil data
    # nwvf_SO4_Vil = nwvf_SO4_Vil.where(nwvf_SO4_Vil < 2.0e7, np.nan)
    # # Replace values above 4.0e6 with NaN in SO4 Zep data
    # nwvf_SO4_Zep = nwvf_SO4_Zep.where(nwvf_SO4_Zep < 4.0e6, np.nan)

    # print(df_EC_Vil.head())
    # print(nwvf_EC_Vil.head())
    # print(df_EC_Vil.keys())
    # print(nwvf_EC_Vil.keys())

    
    # # Define seasons
    # season1 = [12, 1, 2, 3, 4] # DJFMA
    # season2 = [7, 8, 9, 10] # JAS
    
    # # Define the number of standard deviations
    # n_sigmas = 1
    
    # aerosol_dfs = [df_EC_Vil, df_EC_Zep, df_SO4_Vil, df_SO4_Zep]
    # nwvf_dfs = [nwvf_EC_Vil, nwvf_EC_Zep, nwvf_SO4_Vil, nwvf_SO4_Zep]
    # aerosol_names = ['EC', 'EC', 'SO4', 'SO4']
    # station_names = ['Villum', 'Zeppelin', 'Villum', 'Zeppelin']
    # nwvf_season1_bins = [[32, 20, 8], [32, 16, 10], [40, 20, 8], [80, 40, 16]]
    # aerosol_season1_bins =[[32, 20, 20], [32, 20, 20], [40, 16, 16], [80, 30, 30]] 
    # nwvf_season2_bins = [[32, 20, 10], [24, 10, 8], [24, 12, 12], []]
    # aerosol_season2_bins =[[32, 20, 32], [32, 32, 20], [24, 20, 20], []] 

    # seasons = {'Season1': season1, 'Season2': season2}

    # path_save = '/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/MAIA/'

    # # Loop through different stations/aerosol sets
    # for i in range(len(aerosol_dfs)):
    #     if i == 3:
    #         seasons = {'Season1': season1}

    #     event_analysis = EventAnalysis(aerosol_dfs[i], aerosol_names[i], nwvf_dfs[i], 'nwvf_time_integral', seasons, n_sigmas)

    #     # Define bins pairs
    #     bins_nwvf = {'Season1': nwvf_season1_bins[i], 'Season2': nwvf_season2_bins[i]}
    #     bins_aerosol = {'Season1': aerosol_season1_bins[i], 'Season2': aerosol_season2_bins[i]}

    #     # Plot the seasonal data
    #     event_analysis.plot_seasonal_data(bins_nwvf, bins_aerosol, station_names[i], show_figs=True, save_figs=True, save_path=path_save)

    #     # Plot all data
    #     event_analysis.plot_all_data(station_names[i], show_figs=True, save_figs=True, save_path=path_save)
        
