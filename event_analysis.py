'''
    This script examines events of more than 2 sigma above the mean (for the season)
    for both the NWVF and the aerosol data. The events are then compared to see if
    there is a correlation between the two.

    Based on 'climatology.py' the following 'seasons' are defined:
        - Season 1: December, January, February, March, April (5 months)
        - Season 2: July, August, September, October (4 months)
    
    
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PLOT_FIG = True
SAVE_FIG = False
VERBOSE = True
WITH_ERRORBARS = True

# Load NWVF data (filtered to match aerosol data)
nwvf_EC_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                             + 'MAIA_processed/nwvf_EC_Villum.csv',
                             index_col=0, parse_dates=True)
nwvf_EC_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                               + 'MAIA_processed/nwvf_EC_Zeppelin.csv',
                               index_col=0, parse_dates=True)
nwvf_SO4_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                              + 'MAIA_processed/nwvf_SO4_Villum.csv',
                              index_col=0, parse_dates=True)
nwvf_SO4_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                                + 'MAIA_processed/nwvf_SO4_Zeppelin.csv',
                                index_col=0, parse_dates=True)

# Load aerosol data
df_EC_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                           + 'MAIA_processed/df_EC_Villum.csv',
                           index_col=0, parse_dates=True).drop(columns='days_diff')
df_EC_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                             + 'MAIA_processed/df_EC_Zeppelin.csv',
                             index_col=0, parse_dates=True).drop(columns='days_diff')
df_SO4_Vil = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                            + 'MAIA_processed/df_SO4_Villum.csv',
                            index_col=0, parse_dates=True).drop(columns='days_diff')
df_SO4_Zep = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                              + 'MAIA_processed/df_SO4_Zeppelin.csv',
                              index_col=0, parse_dates=True).drop(columns='days_diff')

# Replace values above 2.0e7 with NaN in EC Vil data
nwvf_EC_Vil = nwvf_EC_Vil.where(nwvf_EC_Vil < 2.0e7, np.nan)




# Plot the EC and NWVF Villum data
fig, (ax1, ax2) = plt.subplots(nrows=2,
                                 ncols = 2,
                                 figsize=(15, 8),
                                 gridspec_kw={'width_ratios': [3, 1], 'wspace': 0},
                                 sharey='row'
                                 )
fig.suptitle('EC and NWVF Villum data')

n_sig = 1
# NWVF time series
ax1[0].plot(nwvf_EC_Vil['nwvf_time_integral'], label='NWVF', color='k')
ax1[0].set_title('NWVF')
# Draw horizontal lines at mean +/- 2 sigma
ax1[0].axhline(nwvf_EC_Vil['nwvf_time_integral'].mean(), color='b', linestyle='--', label='Mean')
ax1[0].axhline(nwvf_EC_Vil['nwvf_time_integral'].mean()
               + n_sig*nwvf_EC_Vil['nwvf_time_integral'].std(),
               color='b', linestyle=':', label=f'+{n_sig}' + '$\sigma$')
ax1[0].axhline(nwvf_EC_Vil['nwvf_time_integral'].mean()
               - n_sig*nwvf_EC_Vil['nwvf_time_integral'].std(),
               color='b', linestyle=':', label=f'-{n_sig}' + '$\sigma$')
ax1[0].legend()

# Aerosol time series
ax2[0].plot(df_EC_Vil, label='Aerosol', color='k')
ax2[0].set_title('Aerosol')
# Draw horizontal lines at mean + 2 sigma
ax2[0].axhline(df_EC_Vil['EC'].mean(), color='b', linestyle='--', label='Mean')
ax2[0].axhline(df_EC_Vil['EC'].mean() + n_sig*df_EC_Vil['EC'].std(),
               color='b', linestyle=':', label=f'-{n_sig}' + '$\sigma$')
ax2[0].legend()

# NWVF histogram (approx. gaussian distribution)
ax1[1].hist(nwvf_EC_Vil,
               bins=32,
               color='k',
               ec='k',
               fc='none',
               orientation='horizontal'
               )
ax1[1].tick_params(axis='y', left=False)

# Aerosol histogram (approx.)
ax2[1].hist(df_EC_Vil,
                bins=32,
                color='k',
                ec='k',
                fc='none',
                orientation='horizontal'
                )
ax2[1].tick_params(axis='y', left=False)


# Define seasons
season1 = [12, 1, 2, 3, 4]
season2 = [7, 8, 9, 10]

# Define EC and NWVF Villum data for each season
nwvf_EC_Vil_season1 = nwvf_EC_Vil[nwvf_EC_Vil.index.month.isin(season1)]['nwvf_time_integral']
nwvf_EC_Vil_season2 = nwvf_EC_Vil[nwvf_EC_Vil.index.month.isin(season2)]['nwvf_time_integral']
df_EC_Vil_season1 = df_EC_Vil[df_EC_Vil.index.month.isin(season1)]['EC']
df_EC_Vil_season2 = df_EC_Vil[df_EC_Vil.index.month.isin(season2)]['EC']

# Calculate mean and std for each season, throughout the entire period

# EC VILLUM NWVF
nwvf_EC_Vil_season1_mean = nwvf_EC_Vil_season1.mean()
nwvf_EC_Vil_season1_std = nwvf_EC_Vil_season1.std()
nwvf_EC_Vil_season2_mean = nwvf_EC_Vil_season2.mean()
nwvf_EC_Vil_season2_std = nwvf_EC_Vil_season1.std()


# NWVF Calculate +/- 2 sigma
nwvf_EC_Vil_2std_season1 = nwvf_EC_Vil_season1_mean + n_sig*nwvf_EC_Vil_season1_std
nwvf_EC_Vil_neg2std_season1 = nwvf_EC_Vil_season1_mean - n_sig*nwvf_EC_Vil_season1_std
nwvf_EC_Vil_2std_season2 = nwvf_EC_Vil_season2_mean + n_sig*nwvf_EC_Vil_season2_std
nwvf_EC_Vil_neg2std_season2 = nwvf_EC_Vil_season2_mean - n_sig*nwvf_EC_Vil_season2_std

# Create four masks for NWVF data, two for each season 
mask_seas1_pos = np.where(nwvf_EC_Vil_season1 > nwvf_EC_Vil_2std_season1, True, False)
mask_seas1_neg = np.where(nwvf_EC_Vil_season1 < nwvf_EC_Vil_neg2std_season1, True, False)
mask_seas2_pos = np.where(nwvf_EC_Vil_season2 > nwvf_EC_Vil_2std_season2, True, False)
mask_seas2_neg = np.where(nwvf_EC_Vil_season2 < nwvf_EC_Vil_neg2std_season2, True, False)


# Find the corresponding NWVF data for the mask
nwvf_EC_Vil_season1_events_pos = nwvf_EC_Vil_season1[mask_seas1_pos]
nwvf_EC_Vil_season1_events_neg = nwvf_EC_Vil_season1[mask_seas1_neg]
nwvf_EC_Vil_season2_events_pos = nwvf_EC_Vil_season2[mask_seas2_pos]
nwvf_EC_Vil_season2_events_neg = nwvf_EC_Vil_season2[mask_seas2_neg]

# Find the corresponding aerosol data for the mask ()
df_EC_Vil_season1_events_pos = df_EC_Vil_season1[mask_seas1_pos]
df_EC_Vil_season1_events_neg = df_EC_Vil_season1[mask_seas1_neg]
df_EC_Vil_season2_events_pos = df_EC_Vil_season2[mask_seas2_pos]
df_EC_Vil_season2_events_neg = df_EC_Vil_season2[mask_seas2_neg]

# Plot season 1, three histograms (all data, positive events, negative events)
# 2 subplots, for NWVF and aerosol data
fig, ax = plt.subplots(2, 1, figsize=(15, 8))
fig.suptitle(f'Season 1, +/- {n_sig}' + ' $\sigma$ events')

# NWVF
ax[0].hist(nwvf_EC_Vil_season1, bins=32, label='All data'
              + ', '+str(len(nwvf_EC_Vil_season1))+' events', alpha=0.9, ec='darkblue', fc='lightblue')
ax[0].hist(nwvf_EC_Vil_season1_events_pos, bins=10, label=f'+{n_sig}' + '$\sigma$'
           + ', '+str(len(nwvf_EC_Vil_season1_events_pos))+' events', alpha=0.8, ec='darkred', fc='lightcoral')
ax[0].hist(nwvf_EC_Vil_season1_events_neg, bins=10, label=f'-{n_sig}' + '$\sigma$'
           + ', '+str(len(nwvf_EC_Vil_season1_events_neg))+' events', alpha=0.8, ec='darkgreen', fc='lightgreen')
# Draw vertical lines at mean and +/- 2 sigma
ax[0].axvline(nwvf_EC_Vil_season1_mean, color='k', linestyle='--', label='Mean')
ax[0].axvline(nwvf_EC_Vil_2std_season1, color='k', linestyle=':')
ax[0].axvline(nwvf_EC_Vil_neg2std_season1, color='k', linestyle=':')
ax[0].set_title('NWVF')
ax[0].legend()

# Aerosol
ax[1].hist(df_EC_Vil_season1, bins=32, label='All data'
              + ', '+str(len(df_EC_Vil_season1))+' events',
              alpha=0.9, ec='darkblue', fc='lightblue')
ax[1].hist(df_EC_Vil_season1_events_pos, bins=20, label=f'+{n_sig}' + '$\sigma$'
           + ', '+str(len(df_EC_Vil_season1_events_pos))+' events',
           alpha=0.8, ec='darkred', fc='lightcoral')
ax[1].hist(df_EC_Vil_season1_events_neg, bins=20, label=f'-{n_sig}' + '$\sigma$'
           + ', '+str(len(df_EC_Vil_season1_events_neg))+' events',
           alpha=0.8, ec='darkgreen', fc='lightgreen')
ax[1].set_title('Aerosol')
ax[1].legend()
fig.set_tight_layout(True)

#plt.show()


# Plot season 2, three histograms (all data, positive events, negative events)
# 2 subplots, for NWVF and aerosol data
fig, ax = plt.subplots(2, 1, figsize=(15, 8), sharex=False)
fig.suptitle(f'Season 2, +/- {n_sig}' + ' $\sigma$ events')

# NWVF
ax[0].hist(nwvf_EC_Vil_season2, bins=32, label='All data'
              + ', '+str(len(nwvf_EC_Vil_season2))+' events',
              alpha=0.9, ec='darkblue', fc='lightblue')
ax[0].hist(nwvf_EC_Vil_season2_events_pos, bins=10, label=f'+{n_sig}' + '$\sigma$'
              + ', '+str(len(nwvf_EC_Vil_season2_events_pos))+' events',
              alpha=0.8, ec='darkred', fc='lightcoral')
ax[0].hist(nwvf_EC_Vil_season2_events_neg, bins=10, label=f'-{n_sig}' + '$\sigma$'
                + ', '+str(len(nwvf_EC_Vil_season2_events_neg))+' events',
                alpha=0.8, ec='darkgreen', fc='lightgreen')
# Draw vertical lines at mean and +/- 2 sigma
ax[0].axvline(nwvf_EC_Vil_season2_mean, color='k', linestyle='--', label='Mean')
ax[0].axvline(nwvf_EC_Vil_2std_season2, color='k', linestyle=':')
ax[0].axvline(nwvf_EC_Vil_neg2std_season2, color='k', linestyle=':')
ax[0].set_title('NWVF')
ax[0].legend()


# Aerosol
ax[1].hist(df_EC_Vil_season2, bins=32, label='All data'
              + ', '+str(len(df_EC_Vil_season2))+' events', alpha=0.9, ec='darkblue', fc='lightblue')
ax[1].hist(df_EC_Vil_season2_events_pos, bins=20, label=f'+{n_sig}' + '$\sigma$'
              + ', '+str(len(df_EC_Vil_season2_events_pos))+' events', alpha=0.8, ec='darkred', fc='lightcoral')
ax[1].hist(df_EC_Vil_season2_events_neg, bins=32, label=f'+{n_sig}' + '$\sigma$'
                + ', '+str(len(df_EC_Vil_season2_events_neg))+' events',
                alpha=0.8, ec='darkgreen', fc='lightgreen')
ax[1].set_title('Aerosol')
ax[1].legend()
fig.set_tight_layout(True)

plt.show()


















# #  EC ZEPPELIN + NWVF
# nwvf_EC_Zep_season1_mean = nwvf_EC_Zep[nwvf_EC_Zep.index.month.isin(season1)].mean()
# nwvf_EC_Zep_season1_std = nwvf_EC_Zep[nwvf_EC_Zep.index.month.isin(season1)].std()
# nwvf_EC_Zep_season2_mean = nwvf_EC_Zep[nwvf_EC_Zep.index.month.isin(season2)].mean()
# nwvf_EC_Zep_season2_std = nwvf_EC_Zep[nwvf_EC_Zep.index.month.isin(season2)].std()

# df_EC_Zep_season1_mean = df_EC_Zep[df_EC_Zep.index.month.isin(season1)].mean()
# df_EC_Zep_season1_std = df_EC_Zep[df_EC_Zep.index.month.isin(season1)].std()
# df_EC_Zep_season2_mean = df_EC_Zep[df_EC_Zep.index.month.isin(season2)].mean()
# df_EC_Zep_season2_std = df_EC_Zep[df_EC_Zep.index.month.isin(season2)].std()



# # SO4 VILLUM + NWVF
# nwvf_SO4_Vil_season1_mean = nwvf_SO4_Vil[nwvf_SO4_Vil.index.month.isin(season1)].mean()
# nwvf_SO4_Vil_season1_std = nwvf_SO4_Vil[nwvf_SO4_Vil.index.month.isin(season1)].std()
# nwvf_SO4_Vil_season2_mean = nwvf_SO4_Vil[nwvf_SO4_Vil.index.month.isin(season2)].mean()
# nwvf_SO4_Vil_season2_std = nwvf_SO4_Vil[nwvf_SO4_Vil.index.month.isin(season2)].std()

# df_SO4_Vil_season1_mean = df_SO4_Vil[df_SO4_Vil.index.month.isin(season1)].mean()
# df_SO4_Vil_season1_std = df_SO4_Vil[df_SO4_Vil.index.month.isin(season1)].std()
# df_SO4_Vil_season2_mean = df_SO4_Vil[df_SO4_Vil.index.month.isin(season2)].mean()
# df_SO4_Vil_season2_std = df_SO4_Vil[df_SO4_Vil.index.month.isin(season2)].std()



# # SO4 ZEPPELIN + NWVF
# nwvf_SO4_Zep_season1_mean = nwvf_SO4_Zep[nwvf_SO4_Zep.index.month.isin(season1)].mean()
# nwvf_SO4_Zep_season1_std = nwvf_SO4_Zep[nwvf_SO4_Zep.index.month.isin(season1)].std()
# nwvf_SO4_Zep_season2_mean = nwvf_SO4_Zep[nwvf_SO4_Zep.index.month.isin(season2)].mean()
# nwvf_SO4_Zep_season2_std = nwvf_SO4_Zep[nwvf_SO4_Zep.index.month.isin(season2)].std()

# df_SO4_Zep_season1_mean = df_SO4_Zep[df_SO4_Zep.index.month.isin(season1)].mean()
# df_SO4_Zep_season1_std = df_SO4_Zep[df_SO4_Zep.index.month.isin(season1)].std()
# df_SO4_Zep_season2_mean = df_SO4_Zep[df_SO4_Zep.index.month.isin(season2)].mean()
# df_SO4_Zep_season2_std = df_SO4_Zep[df_SO4_Zep.index.month.isin(season2)].std()




















# #######################
# # MONTHLY CLIMATOLOGY #
# #######################

# # NWVF, groupby month
# nwvf_EC_Vil_clim = nwvf_EC_Vil.groupby(nwvf_EC_Vil.index.month).mean()
# nwvf_EC_Vil_clim_std = nwvf_EC_Vil.groupby(nwvf_EC_Vil.index.month).std()
# nwvf_EC_Zep_clim = nwvf_EC_Zep.groupby(nwvf_EC_Zep.index.month).mean()
# nwvf_EC_Zep_clim_std = nwvf_EC_Zep.groupby(nwvf_EC_Zep.index.month).std()
# nwvf_SO4_Vil_clim = nwvf_SO4_Vil.groupby(nwvf_SO4_Vil.index.month).mean()
# nwvf_SO4_Vil_clim_std = nwvf_SO4_Vil.groupby(nwvf_SO4_Vil.index.month).std()
# nwvf_SO4_Zep_clim = nwvf_SO4_Zep.groupby(nwvf_SO4_Zep.index.month).mean()
# nwvf_SO4_Zep_clim_std = nwvf_SO4_Zep.groupby(nwvf_SO4_Zep.index.month).std()

# # Aerosol,  groupby month
# df_EC_Vil_clim = df_EC_Vil.groupby(df_EC_Vil.index.month).mean()
# df_EC_Vil_clim_std = df_EC_Vil.groupby(df_EC_Vil.index.month).std()
# df_EC_Zep_clim = df_EC_Zep.groupby(df_EC_Zep.index.month).mean()
# df_EC_Zep_clim_std = df_EC_Zep.groupby(df_EC_Zep.index.month).std()
# df_SO4_Vil_clim = df_SO4_Vil.groupby(df_SO4_Vil.index.month).mean()
# df_SO4_Vil_clim_std = df_SO4_Vil.groupby(df_SO4_Vil.index.month).std()
# df_SO4_Zep_clim = df_SO4_Zep.groupby(df_SO4_Zep.index.month).mean()
# df_SO4_Zep_clim_std = df_SO4_Zep.groupby(df_SO4_Zep.index.month).std()




# ######################
# # WEEKLY CLIMATOLOGY #
# ######################

# # NWVF (groupby week)
# nwvf_EC_Vil_clim = nwvf_EC_Vil.groupby(nwvf_EC_Vil.index.week).mean()
# nwvf_EC_Vil_clim_std = nwvf_EC_Vil.groupby(nwvf_EC_Vil.index.week).std()
# nwvf_EC_Zep_clim = nwvf_EC_Zep.groupby(nwvf_EC_Zep.index.week).mean()
# nwvf_EC_Zep_clim_std = nwvf_EC_Zep.groupby(nwvf_EC_Zep.index.week).std()
# nwvf_SO4_Vil_clim = nwvf_SO4_Vil.groupby(nwvf_SO4_Vil.index.week).mean()
# nwvf_SO4_Vil_clim_std = nwvf_SO4_Vil.groupby(nwvf_SO4_Vil.index.week).std()
# nwvf_SO4_Zep_clim = nwvf_SO4_Zep.groupby(nwvf_SO4_Zep.index.week).mean()
# nwvf_SO4_Zep_clim_std = nwvf_SO4_Zep.groupby(nwvf_SO4_Zep.index.week).std()

# # Aerosol (groupby week)
# df_EC_Vil_clim = df_EC_Vil.groupby(df_EC_Vil.index.week).mean()
# df_EC_Vil_clim_std = df_EC_Vil.groupby(df_EC_Vil.index.week).std()
# df_EC_Zep_clim = df_EC_Zep.groupby(df_EC_Zep.index.week).mean()
# df_EC_Zep_clim_std = df_EC_Zep.groupby(df_EC_Zep.index.week).std()
# df_SO4_Vil_clim = df_SO4_Vil.groupby(df_SO4_Vil.index.week).mean()
# df_SO4_Vil_clim_std = df_SO4_Vil.groupby(df_SO4_Vil.index.week).std()
# df_SO4_Zep_clim = df_SO4_Zep.groupby(df_SO4_Zep.index.week).mean()
# df_SO4_Zep_clim_std = df_SO4_Zep.groupby(df_SO4_Zep.index.week).std()




# #####################
# # DAILY CLIMATOLOGY #
# #####################

# # NWVF, groupby day of year
# nwvf_EC_Vil_clim = nwvf_EC_Vil.groupby(nwvf_EC_Vil.index.day_of_year).mean()
# nwvf_EC_Vil_clim_std = nwvf_EC_Vil.groupby(nwvf_EC_Vil.index.day_of_year).std()
# nwvf_EC_Zep_clim = nwvf_EC_Zep.groupby(nwvf_EC_Zep.index.day_of_year).mean()
# nwvf_EC_Zep_clim_std = nwvf_EC_Zep.groupby(nwvf_EC_Zep.index.day_of_year).std()
# nwvf_SO4_Vil_clim = nwvf_SO4_Vil.groupby(nwvf_SO4_Vil.index.day_of_year).mean()
# nwvf_SO4_Vil_clim_std = nwvf_SO4_Vil.groupby(nwvf_SO4_Vil.index.day_of_year).std()
# nwvf_SO4_Zep_clim = nwvf_SO4_Zep.groupby(nwvf_SO4_Zep.index.day_of_year).mean()
# nwvf_SO4_Zep_clim_std = nwvf_SO4_Zep.groupby(nwvf_SO4_Zep.index.day_of_year).std()

# # Aerosol, groupby day of year
# df_EC_Vil_clim = df_EC_Vil.groupby(df_EC_Vil.index.day_of_year).mean()
# df_EC_Vil_clim_std = df_EC_Vil.groupby(df_EC_Vil.index.day_of_year).std()
# df_EC_Zep_clim = df_EC_Zep.groupby(df_EC_Zep.index.day_of_year).mean()
# df_EC_Zep_clim_std = df_EC_Zep.groupby(df_EC_Zep.index.day_of_year).std()
# df_SO4_Vil_clim = df_SO4_Vil.groupby(df_SO4_Vil.index.day_of_year).mean()
# df_SO4_Vil_clim_std = df_SO4_Vil.groupby(df_SO4_Vil.index.day_of_year).std()
# df_SO4_Zep_clim = df_SO4_Zep.groupby(df_SO4_Zep.index.day_of_year).mean()
# df_SO4_Zep_clim_std = df_SO4_Zep.groupby(df_SO4_Zep.index.day_of_year).std()
