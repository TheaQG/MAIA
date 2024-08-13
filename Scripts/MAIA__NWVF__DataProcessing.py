
import netCDF4 as nc
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.pyplot as plt
import os
import glob

# Define the path to the data
path = '/Volumes/1TB-FREECOM/MAIA_data/'

# Plot for a single time step?
PLOT_SINGLE_TIME = False

# Define the latitude and longitude of the data to use
lat_use = 70 # degrees North
lon_use = [-45, 45] # degrees West/East


# Define the resolutions of the data
res_lon = 0.25
res_lat = 0.25

# Define the circumference of the earth and the cosine of the latitude for the latitudinal adjustment
earth_circumference = 40075.0  # km
cosine_lat_adjust = np.cos(np.deg2rad(lat_use))

# Convert from degrees to km
res_lon_km = (res_lon / 360.0) * earth_circumference * cosine_lat_adjust


# Define the pattern of the files to use
file_pattern = os.path.join(path, 'ERA5_download__vertical_integral_of_northward_water_vapour_flux_*__6hr_mean.nc')

# Lists to store integral and times
integrals = []
times = []


for file_path in glob.glob(file_pattern):
    print(file_path)

    with xr.open_dataset(file_path) as ds:
        # Rename the variable to something more useful
        ds = ds.rename({'p72.162': 'nwvf'})

        # Store the data of interest in a new variable
        nwvf = ds['nwvf']
        
        for timestep in ds['time']:

            # Select the data at desired latitude 
            nwvf_xN = nwvf.sel(latitude=lat_use, method='nearest', time=timestep.values)

            # Select the data between desired longitudes
            nwvf_xN_xWxE = nwvf_xN.sel(longitude=slice(lon_use[0], lon_use[1]))


            # Calculate the latitudinal integral
            integral = nwvf_xN_xWxE.sum(dim='longitude') * res_lon_km

            # Append to the list of integrals
            integrals.append(integral.values)

            # Extract the times and store in a list
            times.append(pd.to_datetime(timestep.values))

        ds.close()


integrals_array = np.array(integrals)
times_array = np.array(times)

print(integrals_array)
print(times_array)

new_ds = xr.Dataset({
    'nwvf_integral': (['time'], integrals_array),
    'time': (['time'], times_array)
})

PATH_SAVE = '/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/MAIA_processed'
fn_new = 'MAIA_nwvf_integrals_1994-2022__6hr_mean.nc'

new_ds.to_netcdf(os.path.join(PATH_SAVE, fn_new))



plt.plot(integrals)


if PLOT_SINGLE_TIME:
    i = 0
    for file_path in glob.glob(file_pattern):
        if i == 0:
            print(file_path)

            with xr.open_dataset(file_path) as ds:
                
                # Rename the variable to something more useful
                ds = ds.rename({'p72.162': 'nwvf'})

                # Store the data of interest in a new variable
                nwvf = ds['nwvf']
                
                # Select a single time step
                nwvf_single_time = nwvf.isel(time=0)

                # Define the projection: Orthographic with central longitude 0 and central latitude 90
                projection = ccrs.Orthographic(central_longitude=0.0, central_latitude=90.0)

                # Plot the data
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(1,1,1, projection=projection)

                # Set the extent of the plot to all longitudes and latitudes between 60 and 90 degrees north
                ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

                # Add coastlines, borders and gridlines
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.gridlines(draw_labels=True)

                # Plot the data
                nwvf_plot = ax.pcolormesh(
                    nwvf_single_time['longitude'],
                    nwvf_single_time['latitude'],
                    nwvf_single_time.values,
                    transform=ccrs.PlateCarree(),
                    cmap='coolwarm'
                )

                # Add a contour line at 70N
                longitude_values_full = np.linspace(-180, 180, 1000)  # 1000 points make a smooth line
                longitude_values_45 = np.linspace(lon_use[0], lon_use[1], 1000)
                constant_latitude_full = np.full_like(longitude_values_full, lat_use)
                constant_latitude_45 = np.full_like(longitude_values_45, lat_use)

                # Add a curved line at 70 degrees North
                ax.plot(longitude_values_full, constant_latitude_full, color='black', linewidth=2, transform=ccrs.Geodetic())
                ax.plot(longitude_values_45, constant_latitude_45, color='red', linewidth=2, transform=ccrs.Geodetic())


                latitude_values = np.linspace(lat_use, 90, 500)  # 500 points for a smooth line
                constant_longitude_m45 = np.full_like(latitude_values, lon_use[0])  # -45 for 45W
                constant_longitude_45 = np.full_like(latitude_values, lon_use[1])  # 45 for 45E

                # Add a line at 45W from 70N to 90N
                ax.plot(constant_longitude_m45, latitude_values, color='red', linewidth=1.5, linestyle='--', transform=ccrs.Geodetic())
                ax.plot(constant_longitude_45, latitude_values, color='red', linewidth=1.5, linestyle='--', transform=ccrs.Geodetic())

                # Add a colorbar and label to colorbar
                cbar = plt.colorbar(nwvf_plot, orientation='horizontal', shrink=0.5, pad=0.05)
                cbar.set_label('[kg m-1 s-1]')

                # Add a title
                plt.title('ERA5: Vertical integral of northward water vapour flux\nTime: ' + str(nwvf_single_time['time'].values) + ' UTC')

                # Show the plot
                plt.show()

                # Save the plot
                fig.savefig('MAIA_NWVF_single_timestep.png', dpi=300, bbox_inches='tight')







# # Close the file
ds.close()




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
