import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from event_analysis_class import EventAnalysis

if __name__ == '__main__':

    # Import csv file with data
    data = pd.read_csv('NCEP_MERRA_Arctic_Slice_MoistureFlux.csv', sep=';')

    print(data.head())
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(data['date'], data['AMIP NINT Daily NCEP Nudged'], color='k', lw=1)
    
    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(data['date'], data['AMIP NINT Daily Merra Nudged'], color='k', lw=1)

    plt.show()