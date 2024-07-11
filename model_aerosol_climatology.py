'''
    This script computes the climatology of the NWVF and aerosol data 
    along with the correlation between the two.
'''
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from scipy.stats import pearsonr

class climatology:
    '''
        Class to compute the climatology of a dataset
        and plot if needed.
    '''
    def __init__(self, data, freq):
        self.data = data
        self.freq = freq

    def compute_rolling_mean(self, days_rolling):
        self.data = self.data.rolling(days_rolling).mean()
    
    def compute_climatology(self):
        if self.freq == 'monthly':
            self.clim = self.data.groupby(self.data.index.month).mean()
            self.clim_std = self.data.groupby(self.data.index.month).std()
        elif self.freq == 'weekly':
            self.clim = self.data.groupby(self.data.index.week).mean()
            self.clim_std = self.data.groupby(self.data.index.week).std()
        elif self.freq == 'daily':
            self.clim = self.data.groupby(self.data.index.day_of_year).mean()
            self.clim_std = self.data.groupby(self.data.index.day_of_year).std()
        else:
            raise ValueError('Frequency not recognized. Choose between monthly, weekly, daily.')
        
    def plot_climatology(self, WITH_ERRORBARS=True, SAVE_FIG=False, SHOW_FIG=True):
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax.plot(self.clim.index, self.clim['nwvf_time_integral'], color='k', linewidth=1)
        if WITH_ERRORBARS:
            ax.errorbar(self.clim.index, self.clim['nwvf_time_integral'],
                        yerr=self.clim_std['nwvf_time_integral'],
                        fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4)
        ax.set_title(f'{self.freq.capitalize()} climatology, all NWVF')
        ax.set(xlabel=f'{self.freq.capitalize()}', ylabel='NWVF')
        fig.tight_layout()
        
        if SAVE_FIG:
            fig.savefig(f'/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/MAIA/nwvf_climatology_{self.freq}.png',
                        dpi=600, bbox_inches='tight')
        if SHOW_FIG:
            plt.show()



if __name__ == '__main__':
    # Get the data



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

    print(df_aerosol.head())
    # Each aerosol needs to be split up into two seasons (different months for each aerosol)








    # Plot the raw data (all columns in aerosol data), one subplot per column + NWVF
    n_plots = len(df_aerosol.columns) + 1
    fig, axs = plt.subplots(n_plots, 1, figsize=(15, 2*n_plots), sharex=True)

    # Aerosol
    for i, col in enumerate(df_aerosol.columns):
        axs[i].plot(df_aerosol.index, df_aerosol[col], color='k', linewidth=1)
        axs[i].set(ylabel=df_aerosol[col].name)
        axs[i].grid()

    # NWVF
    axs[-1].plot(df_NWVF.index, df_NWVF['nwvf_time_integral'], color='k', linewidth=1)
    axs[-1].set(xlabel='Time', ylabel='NWVF')
    axs[-1].grid()
    # Set the x-axis to the same for all subplots
    axs[-1].set_xlim([df_NWVF.index[0], df_NWVF.index[-1]])

    fig.tight_layout()
    
    fig.savefig('/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/Figures/MERRA/ArcticSlice_small/merra_aerosol_nwvf_raw_data.png',
                dpi=600, bbox_inches='tight')



    clim_frequencies = ['monthly', 'daily']# 'weekly']

    # Compute the climatology of the NWVF data
    fig, axs = plt.subplots(len(clim_frequencies), 1, figsize=(15, 4*len(clim_frequencies)))

    for i, freq in enumerate(clim_frequencies):
        clim_NWVF = climatology(df_NWVF, freq)
        clim_NWVF.compute_climatology()
#        clim_NWVF.plot_climatology(SAVE_FIG=False, SHOW_FIG=False)
        axs[i].plot(clim_NWVF.clim.index, clim_NWVF.clim['nwvf_time_integral'], color='k', linewidth=1)
        axs[i].errorbar(clim_NWVF.clim.index, clim_NWVF.clim['nwvf_time_integral'],
                        yerr=clim_NWVF.clim_std['nwvf_time_integral'],
                        fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4)
        axs[i].set_title(f'{freq.capitalize()} climatology, all NWVF')
        axs[i].set(xlabel=f'{freq.capitalize()}', ylabel='NWVF')
        axs[i].grid()

    fig.tight_layout()
    fig.savefig('/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/Figures/MERRA/ArcticSlice_small/nwvf_climatology.png',
                dpi=600, bbox_inches='tight')









    # Compute the climatology of the aerosol data separately for each component
    clim_aerosol = {}

    for col in df_aerosol.columns:
        clim_aerosol[col] = {}
        for freq in clim_frequencies:
            clim_aerosol[col][freq] = climatology(df_aerosol[[col]], freq)
            clim_aerosol[col][freq].compute_climatology()
            # clim_aerosol[col][freq].plot_climatology(SAVE_FIG=False, SHOW_FIG=False)

    clim_NWVF = {}
    for freq in clim_frequencies:
        clim_NWVF[freq] = climatology(df_NWVF, freq)
        clim_NWVF[freq].compute_climatology()
        # clim_NWVF[freq].plot_climatology(SAVE_FIG=False, SHOW_FIG=False)


    color = 'r'

    # Plot the climatology of the aerosol data (monthly) with NWVF as reference
    fig, axs = plt.subplots(len(df_aerosol.columns)//2, 2, figsize=(17, 4*len(df_aerosol.columns)//2), sharex=True)
    freq = 'monthly'

    for ax, col in zip(axs.flat, df_aerosol.columns):
        # Plot aerosol and nwvf on separate y-axes
        ax.plot(clim_aerosol[col][freq].clim.index, clim_aerosol[col][freq].clim[col], color='k', linewidth=1)
        ax.set(ylabel=col)
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(clim_NWVF[freq].clim.index, clim_NWVF[freq].clim['nwvf_time_integral'], color=color, linewidth=1)
        # Make second y-axis ticks color match the line color
        ax2.yaxis.label.set_color(color)
        ax2.tick_params(axis='y', colors=color)

        # Only show y-axis label and ticks (ax2) on the last column
        if ax == axs[0, -1] or ax == axs[1, -1]:
            ax2.set(ylabel='NWVF')

        else:
            ax2.set(yticks=[])
            ax2.set(ylabel='')

        # Compute correlation
        corr, _ = pearsonr(clim_aerosol[col][freq].clim[col], clim_NWVF[freq].clim['nwvf_time_integral'])
        ax.set_title(f'{col} vs NWVF, {freq.capitalize()} climatology, corr={corr:.2f}')

    fig.tight_layout()
    fig.savefig('/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/Figures/MERRA/ArcticSlice_small/aerosol_nwvf_climatology_monthly.png',
                dpi=600, bbox_inches='tight')

    # Plot the climatology of the aerosol data (daily) with NWVF as reference
    fig, axs = plt.subplots(len(df_aerosol.columns)//2, 2, figsize=(17, 4*len(df_aerosol.columns)//2), sharex=True)
    freq = 'daily'

    for ax, col in zip(axs.flat, df_aerosol.columns):
        # Plot aerosol and nwvf on separate y-axes
        ax.plot(clim_aerosol[col][freq].clim.index, clim_aerosol[col][freq].clim[col], color='k', linewidth=1)
        ax.set(ylabel=col)
        ax2 = ax.twinx()
        ax2.plot(clim_NWVF[freq].clim.index, clim_NWVF[freq].clim['nwvf_time_integral'], color=color, linewidth=1)
        # Make second y-axis ticks color match the line color
        ax2.yaxis.label.set_color(color)
        ax2.tick_params(axis='y', colors=color)

        # Only show y-axis label and ticks (ax2) on the last column
        if ax == axs[0, -1] or ax == axs[1, -1]:
            ax2.set(ylabel='NWVF')

        else:
            ax2.set(yticks=[])
            ax2.set(ylabel='')

        ax2.grid()
        # Compute correlation
        corr, _ = pearsonr(clim_aerosol[col][freq].clim[col], clim_NWVF[freq].clim['nwvf_time_integral'])
        ax.set_title(f'{col} vs NWVF, {freq.capitalize()} climatology, corr={corr:.2f}')


    fig.tight_layout()

    fig.savefig('/Users/au728490/Documents/PhD_AU/Python_Scripts/MAIA_ERA5_Download/Figures/MERRA/ArcticSlice_small/aerosol_nwvf_climatology_daily.png',
                dpi=600, bbox_inches='tight')




    plt.show()
    