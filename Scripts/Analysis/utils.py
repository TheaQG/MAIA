'''

'''
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


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
        '''
        
        '''
        # Calculate the daily mean and std of both NWVF and aerosol (1. jan, 2. jan,..., 31. dec)
        self.df['day'] = self.df[self.date_column].dt.dayofyear
        climatology = self.df.groupby('day').agg(['mean', 'std'])
        return climatology

    def extend_climatology(self):
        '''
        
        '''
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
        '''
        
        '''
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
        '''
        
        '''
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
    