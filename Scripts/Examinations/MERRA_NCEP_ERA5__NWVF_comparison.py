import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#from ..Scripts.Analysis.event_analysis_class import EventAnalysis

if __name__ == '__main__':

    PATH_DATA = '../../Data/ModelData/'
    PATH_FIGS = '../../Figures/'

    # Import csv file with data
    data = pd.read_csv(PATH_DATA + 'NCEP_MERRA_Arctic_Slice_MoistureFlux.csv', sep=';', decimal=',')
    data_nwvf = pd.read_csv(PATH_DATA + 'nwvf_ts.csv', sep=',')

    # Plot the data for NCEP and MERRA in subplots
    fig, ax = plt.subplots(figsize=(20,8), sharex=True)
    fig.suptitle('Arctic moisture flux', fontsize=16)

    ax.plot(data.index, data['AMIP NINT Daily NCEP Nudged'], linewidth=1.1, color='#1f77b4', label='NCEP')
    ax.plot(data.index, data['AMIP NINT Daily Merra Nudged'], linewidth=1.1, color='#ff7f0e', label='MERRA')
    ax.plot(data_nwvf.index, data_nwvf['nwvf']*10**3, linewidth=1.1, color='#2ca02c', label='NWVF')
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    ax.set_xlim([min(data.index), max(data.index)/4])
    ax.set_ylim([-0.3*1e9, 0.51e9])
    ax.set_xlabel('Year')
    ax.set_ylabel('NWVF')
    ax.legend()
    fig.tight_layout()

    # Calculate correlation between NCEP, MERRA and NWVF
    corr = data['AMIP NINT Daily NCEP Nudged'].corr(data['AMIP NINT Daily Merra Nudged'])
    corr_nwvf = data['AMIP NINT Daily NCEP Nudged'].corr(data_nwvf['nwvf'])
    corr_nwvf_merra = data['AMIP NINT Daily Merra Nudged'].corr(data_nwvf['nwvf'])
    
    print('\n\nCorrelation between NCEP and MERRA: ', corr)
    print('Correlation between NCEP and NWVF: ', corr_nwvf)
    print('Correlation between MERRA and NWVF: ', corr_nwvf_merra)

    # Save the figure as an image file
    plt.savefig(PATH_FIGS + 'MERRA_NCEP_ERA5__NWVF.png')

    plt.show()