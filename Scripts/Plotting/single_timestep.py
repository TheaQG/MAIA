'''
    This script contains a function that plots a single time step for a given parameter from a .nc file.
'''
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt



def plot_single_timestep(filename, variable, timestep, lon_bounds, lat_bounds):
    """
    Plot a single time step for a given parameter from a .nc file.

    Parameters:
    - filename: str, the path to the .nc file
    - variable: str, the parameter of interest to plot (e.g., 'temp', 'p0', 'p72.162')
    - timestep: int or str, the index or specific datetime of the timestep of interest
    - lon_bounds: tuple, the longitude bounds for the area cutoff (e.g., (-45, 45))
    - lat_bounds: tuple, the latitude bounds for the area cutoff (e.g., (60, 70))
    """
    
    with xr.open_dataset(filename) as ds:
        # If timestep is an integer, use isel; if it's a string (datetime), use sel
        if isinstance(timestep, int):
            data = ds[variable].isel(time=timestep.values)
        else:
            data = ds[variable].sel(time=timestep.values, method='nearest')

        # Select the data within the specified longitude and latitude bounds
        data = data.sel(longitude=slice(*lon_bounds), latitude=slice(*lat_bounds))

        # Define the projection
        projection = ccrs.Orthographic(central_longitude=np.mean(lon_bounds), central_latitude=np.mean(lat_bounds))

        # Plot the data
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=projection)
        ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], ccrs.PlateCarree())

        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.gridlines(draw_labels=True)

        # Plot the data
        data_plot = ax.pcolormesh(
            data['longitude'],
            data['latitude'],
            data.values,
            transform=ccrs.PlateCarree(),
            cmap='coolwarm'
        )

        # Add a colorbar and a title
        plt.colorbar(data_plot, orientation='horizontal', shrink=0.5, pad=0.05)
        plt.title(f'{variable} at timestep {timestep}')

        # Show the plot
        plt.show()

        # Close the dataset
        ds.close()
