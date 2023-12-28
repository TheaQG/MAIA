'''
    This script computes the climatology of the NWVF and aerosol data 
    along with the correlation between the two.
'''
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import pearsonr
PLOT_FIG = True
SAVE_FIG = False
VERBOSE = True
WITH_ERRORBARS = True

# Load NWVF data, filtered to match aerosol data
nwvf_EC_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                             + 'MAIA_processed/nwvf_EC_Villum.csv',
                             index_col=0, parse_dates=True)
nwvf_EC_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                               + 'MAIA_processed/nwvf_EC_Zeppelin.csv',
                               index_col=0, parse_dates=True)
nwvf_SO4_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                              + 'MAIA_processed/nwvf_SO4_Villum.csv',
                              index_col=0, parse_dates=True)
nwvf_SO4_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                                + 'MAIA_processed/nwvf_SO4_Zeppelin.csv',
                                index_col=0, parse_dates=True)

# Load aerosol data
df_EC_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                           + 'MAIA_processed/df_EC_Villum.csv',
                           index_col=0, parse_dates=True)
df_EC_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                             + 'MAIA_processed/df_EC_Zeppelin.csv',
                             index_col=0, parse_dates=True)
df_SO4_Villum = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                            + 'MAIA_processed/df_SO4_Villum.csv',
                            index_col=0, parse_dates=True)
df_SO4_Zeppelin = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/Data/'
                              + 'MAIA_processed/df_SO4_Zeppelin.csv',
                              index_col=0, parse_dates=True)



#############################
# EXAMINE FULL NWVF DATASET #
#############################


# Load NWVF full dataset
nwvf_ts_raw = pd.read_csv('/Users/au728490/Documents/PhD_AU/Python_Scripts/'
                          + 'Data/MAIA_processed/nwvf_ts.csv',
                          index_col=0, parse_dates=True)

# Remove dates where value larger than 0.6e6 (outliers)
nwvf_ts = nwvf_ts_raw[nwvf_ts_raw['nwvf'] < 0.6e6]
# Convert index to datetime
nwvf_ts.index = pd.to_datetime(nwvf_ts.index)

# Compute a 7-day rolling mean
nwvf_ts_roll = nwvf_ts.rolling(7).mean()

# Plot NWVF, raw (with default values), filtered (removing outliers) and rolling mean
fig, ax = plt.subplots(3,1, figsize=(15,8), sharex=True)
ax[0].plot(nwvf_ts_raw, color='k', linewidth=0.7)
ax[0].set_title('NWVF raw')
ax[1].plot(nwvf_ts, color='k', linewidth=0.7)
ax[1].set_title('NWVF filtered')
ax[2].plot(nwvf_ts_roll, color='k', linewidth=0.7)
ax[2].set_title('NWVF rolling mean')
ax[2].set(xlabel='Date')
fig.tight_layout()

if SAVE_FIG:
    fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                + 'MAIA/nwvf_raw_filtered_rolling.png',
                dpi=600, bbox_inches='tight')


####################################
# CREATE CLIMATOLOGY FOR NWVF ONLY #
####################################
# Climatology is 'the average year' (by day/week/month) of a variable, here pooling
# together all data of specific days/weeks/months and averaging them


# Monthly climatology
nwvf_ts_clim = nwvf_ts.groupby(nwvf_ts.index.month).mean()
nwvf_ts_clim_std = nwvf_ts.groupby(nwvf_ts.index.month).std()

# Weekly climatology
nwvf_ts_clim_week = nwvf_ts.groupby(nwvf_ts.index.week).mean()
nwvf_ts_clim_week_std = nwvf_ts.groupby(nwvf_ts.index.week).std()

# Daily climatology
nwvf_ts_clim_day = nwvf_ts.groupby(nwvf_ts.index.day_of_year).mean()
nwvf_ts_clim_day_std = nwvf_ts.groupby(nwvf_ts.index.day_of_year).std()


# Plot NWVF (raw, but filtered - i.e. not matched to observational data)
# climatology (monthly, weekly, daily)
fig, ax = plt.subplots(3,1, figsize=(15,8))

# Plot monthly climatology
ax[0].plot(nwvf_ts_clim.index,
           nwvf_ts_clim['nwvf'],
           color='k', linewidth=1)
ax[0].errorbar(nwvf_ts_clim.index,
               nwvf_ts_clim['nwvf'],
               yerr=nwvf_ts_clim_std['nwvf'],
               fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4)
ax[0].set_title('Monthly climatology')
ax[0].set(xlabel='Month', ylabel='NWVF')

# Plot weekly climatology
ax[1].plot(nwvf_ts_clim_week.index,
           nwvf_ts_clim_week['nwvf'],
           color='k', linewidth=1)
ax[1].errorbar(nwvf_ts_clim_week.index,
               nwvf_ts_clim_week['nwvf'],
               yerr=nwvf_ts_clim_week_std['nwvf'],
               fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4)
ax[1].set_title('Weekly climatology')
ax[1].set(xlabel='Week', ylabel='NWVF')

# Plot daily climatology
ax[2].plot(nwvf_ts_clim_day.index,
           nwvf_ts_clim_day['nwvf'],
           color='k', linewidth=1)
ax[2].errorbar(nwvf_ts_clim_day.index,
               nwvf_ts_clim_day['nwvf'],
               yerr=nwvf_ts_clim_day_std['nwvf'],
               fmt='.', color='k', capsize=1, elinewidth=0.5, ms=4)
ax[2].set_title('Daily climatology')
ax[2].set(xlabel='Day of year', ylabel='NWVF')

fig.tight_layout()

if SAVE_FIG:
    fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/'
                + 'Figures/MAIA/nwvf_climatology__M_W_D.png',
                dpi=600, bbox_inches='tight')
#plt.show()




#######################
# MONTHLY CLIMATOLOGY #
#######################

# NWVF, groupby month
nwvf_EC_Villum_clim = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.month).mean()
nwvf_EC_Villum_clim_std = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.month).std()
nwvf_EC_Zeppelin_clim = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.month).mean()
nwvf_EC_Zeppelin_clim_std = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.month).std()
nwvf_SO4_Villum_clim = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.month).mean()
nwvf_SO4_Villum_clim_std = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.month).std()
nwvf_SO4_Zeppelin_clim = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.month).mean()
nwvf_SO4_Zeppelin_clim_std = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.month).std()

# Aerosol,  groupby month
df_EC_Villum_clim = df_EC_Villum.groupby(df_EC_Villum.index.month).mean()
df_EC_Villum_clim_std = df_EC_Villum.groupby(df_EC_Villum.index.month).std()
df_EC_Zeppelin_clim = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.month).mean()
df_EC_Zeppelin_clim_std = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.month).std()
df_SO4_Villum_clim = df_SO4_Villum.groupby(df_SO4_Villum.index.month).mean()
df_SO4_Villum_clim_std = df_SO4_Villum.groupby(df_SO4_Villum.index.month).std()
df_SO4_Zeppelin_clim = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.month).mean()
df_SO4_Zeppelin_clim_std = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.month).std()

# Correlation, compute Pearson correlation coefficient and p-value
r_EC_Villum, p_EC_Villum = pearsonr(nwvf_EC_Villum_clim['nwvf_time_integral'],
                                    df_EC_Villum_clim['EC'])
r_EC_Zeppelin, p_EC_Zeppelin = pearsonr(nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
                                        df_EC_Zeppelin_clim['EC'])
r_SO4_Villum, p_SO4_Villum = pearsonr(nwvf_SO4_Villum_clim['nwvf_time_integral'],
                                      df_SO4_Villum_clim['SO4'])
r_SO4_Zeppelin, p_SO4_Zeppelin = pearsonr(nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
                                          df_SO4_Zeppelin_clim['SO4'])

# Plot NWVF and aerosol climatology (monthly)
fig, ax = plt.subplots(2,2, figsize=(15,8), sharex=True)
ax[0,0].plot(nwvf_EC_Villum_clim.index,
             nwvf_EC_Villum_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[0,0].errorbar(nwvf_EC_Villum_clim.index,
                    nwvf_EC_Villum_clim['nwvf_time_integral'],
                    yerr=nwvf_EC_Villum_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,0].legend(loc='upper left')
ax[0,0].set_title(f'Villum EC, r = {r_EC_Villum:.2f}, p = {p_EC_Villum:.2f}')
ax2 = ax[0,0].twinx()
ax2.plot(df_EC_Villum_clim.index,
         df_EC_Villum_clim['EC'],
         color='r', linewidth=1, label='EC')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
if WITH_ERRORBARS:
    ax2.errorbar(df_EC_Villum_clim.index,
                 df_EC_Villum_clim['EC'],
                 yerr=df_EC_Villum_clim_std['EC'],
                 fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.legend(loc='upper right')

ax[0,1].plot(nwvf_EC_Zeppelin_clim.index,
             nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[0,1].errorbar(nwvf_EC_Zeppelin_clim.index,
                    nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
                    yerr=nwvf_EC_Zeppelin_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,1].legend(loc='upper left')
ax[0,1].set_title(f'Zeppelin EC, r = {r_EC_Zeppelin:.2f}, p = {p_EC_Zeppelin:.2f}')
ax2 = ax[0,1].twinx()
ax2.plot(df_EC_Zeppelin_clim.index,
         df_EC_Zeppelin_clim['EC'],
         color='r', linewidth=1, label='EC')
if WITH_ERRORBARS:
    ax2.errorbar(df_EC_Zeppelin_clim.index,
                df_EC_Zeppelin_clim['EC'],
                yerr=df_EC_Zeppelin_clim_std['EC'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[1,0].plot(nwvf_SO4_Villum_clim.index,
             nwvf_SO4_Villum_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[1,0].errorbar(nwvf_SO4_Villum_clim.index,
                    nwvf_SO4_Villum_clim['nwvf_time_integral'],
                    yerr=nwvf_SO4_Villum_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,0].legend(loc='upper left')
ax[1,0].set_title(f'Villum SO4, r = {r_SO4_Villum:.2f}, p = {p_SO4_Villum:.2f}')
ax2 = ax[1,0].twinx()
ax2.plot(df_SO4_Villum_clim.index, df_SO4_Villum_clim['SO4'], color='r', linewidth=1, label='SO4')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
if WITH_ERRORBARS:
    ax2.errorbar(df_SO4_Villum_clim.index,
                df_SO4_Villum_clim['SO4'],
                yerr=df_SO4_Villum_clim_std['SO4'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.legend(loc='upper right')

ax[1,1].plot(nwvf_SO4_Zeppelin_clim.index,
             nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[1,1].errorbar(nwvf_SO4_Zeppelin_clim.index,
                    nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
                    yerr=nwvf_SO4_Zeppelin_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,1].legend(loc='upper left')
ax[1,1].set_title(f'Zeppelin SO4, r = {r_SO4_Zeppelin:.2f}, p = {p_SO4_Zeppelin:.2f}')
ax2 = ax[1,1].twinx()
ax2.plot(df_SO4_Zeppelin_clim.index,
         df_SO4_Zeppelin_clim['SO4'],
         color='r', linewidth=1, label='SO4')
if WITH_ERRORBARS:
    ax2.errorbar(df_SO4_Zeppelin_clim.index,
                df_SO4_Zeppelin_clim['SO4'],
                yerr=df_SO4_Zeppelin_clim_std['SO4'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

fig.tight_layout()

if SAVE_FIG:
    if WITH_ERRORBARS:
        fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                    + 'MAIA/nwvf_aerosol_climatology_monthly_with_errorbars.png',
                    dpi=600, bbox_inches='tight')
    else:
        fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                    + 'MAIA/nwvf_aerosol_climatology_monthly.png',
                    dpi=600, bbox_inches='tight')
#plt.show()




######################
# WEEKLY CLIMATOLOGY #
######################

# NWVF (groupby week)
nwvf_EC_Villum_clim = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.week).mean()
nwvf_EC_Villum_clim_std = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.week).std()
nwvf_EC_Zeppelin_clim = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.week).mean()
nwvf_EC_Zeppelin_clim_std = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.week).std()
nwvf_SO4_Villum_clim = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.week).mean()
nwvf_SO4_Villum_clim_std = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.week).std()
nwvf_SO4_Zeppelin_clim = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.week).mean()
nwvf_SO4_Zeppelin_clim_std = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.week).std()

# Aerosol (groupby week)
df_EC_Villum_clim = df_EC_Villum.groupby(df_EC_Villum.index.week).mean()
df_EC_Villum_clim_std = df_EC_Villum.groupby(df_EC_Villum.index.week).std()
df_EC_Zeppelin_clim = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.week).mean()
df_EC_Zeppelin_clim_std = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.week).std()
df_SO4_Villum_clim = df_SO4_Villum.groupby(df_SO4_Villum.index.week).mean()
df_SO4_Villum_clim_std = df_SO4_Villum.groupby(df_SO4_Villum.index.week).std()
df_SO4_Zeppelin_clim = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.week).mean()
df_SO4_Zeppelin_clim_std = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.week).std()

# Correlation, compute Pearson correlation coefficient and p-value
r_EC_Villum, p_EC_Villum = pearsonr(nwvf_EC_Villum_clim['nwvf_time_integral'],
                                    df_EC_Villum_clim['EC'])
r_EC_Zeppelin, p_EC_Zeppelin = pearsonr(nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
                                        df_EC_Zeppelin_clim['EC'])
r_SO4_Villum, p_SO4_Villum = pearsonr(nwvf_SO4_Villum_clim['nwvf_time_integral'],
                                      df_SO4_Villum_clim['SO4'])
r_SO4_Zeppelin, p_SO4_Zeppelin = pearsonr(nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
                                          df_SO4_Zeppelin_clim['SO4'])


# Plot NWVF and aerosol climatology (weekly)
fig, ax = plt.subplots(2,2, figsize=(15,8), sharex=True)
ax[0,0].plot(nwvf_EC_Villum_clim.index,
             nwvf_EC_Villum_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[0,0].errorbar(nwvf_EC_Villum_clim.index,
                    nwvf_EC_Villum_clim['nwvf_time_integral'],
                    yerr=nwvf_EC_Villum_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,0].legend(loc='upper left')
ax[0,0].set_title(f'Villum EC, r = {r_EC_Villum:.2f}, p = {p_EC_Villum:.2f}')
ax2 = ax[0,0].twinx()
ax2.plot(df_EC_Villum_clim.index,
         df_EC_Villum_clim['EC'],
         color='r', linewidth=1, label='EC')
if WITH_ERRORBARS:
    ax2.errorbar(df_EC_Villum_clim.index,
                df_EC_Villum_clim['EC'],
                yerr=df_EC_Villum_clim_std['EC'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[0,1].plot(nwvf_EC_Zeppelin_clim.index,
             nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[0,1].errorbar(nwvf_EC_Zeppelin_clim.index,
                    nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
                    yerr=nwvf_EC_Zeppelin_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,1].legend(loc='upper left')
ax[0,1].set_title(f'Zeppelin EC, r = {r_EC_Zeppelin:.2f}, p = {p_EC_Zeppelin:.2f}')
ax2 = ax[0,1].twinx()
ax2.plot(df_EC_Zeppelin_clim.index,
         df_EC_Zeppelin_clim['EC'],
         color='r', linewidth=1, label='EC')
if WITH_ERRORBARS:
    ax2.errorbar(df_EC_Zeppelin_clim.index,
                df_EC_Zeppelin_clim['EC'],
                yerr=df_EC_Zeppelin_clim_std['EC'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[1,0].plot(nwvf_SO4_Villum_clim.index,
             nwvf_SO4_Villum_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[1,0].errorbar(nwvf_SO4_Villum_clim.index,
                    nwvf_SO4_Villum_clim['nwvf_time_integral'],
                    yerr=nwvf_SO4_Villum_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,0].legend(loc='upper left')
ax[1,0].set_title(f'Villum SO4, r = {r_SO4_Villum:.2f}, p = {p_SO4_Villum:.2f}')
ax2 = ax[1,0].twinx()
ax2.plot(df_SO4_Villum_clim.index,
         df_SO4_Villum_clim['SO4'],
         color='r', linewidth=1, label='SO4')
if WITH_ERRORBARS:
    ax2.errorbar(df_SO4_Villum_clim.index,
                df_SO4_Villum_clim['SO4'],
                yerr=df_SO4_Villum_clim_std['SO4'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[1,1].plot(nwvf_SO4_Zeppelin_clim.index,
             nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[1,1].errorbar(nwvf_SO4_Zeppelin_clim.index,
                    nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
                    yerr=nwvf_SO4_Zeppelin_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,1].legend(loc='upper left')
ax[1,1].set_title(f'Zeppelin SO4, r = {r_SO4_Zeppelin:.2f}, p = {p_SO4_Zeppelin:.2f}')
ax2 = ax[1,1].twinx()
ax2.plot(df_SO4_Zeppelin_clim.index,
         df_SO4_Zeppelin_clim['SO4'],
         color='r', linewidth=1, label='SO4')
if WITH_ERRORBARS:
    ax2.errorbar(df_SO4_Zeppelin_clim.index,
                df_SO4_Zeppelin_clim['SO4'],
                yerr=df_SO4_Zeppelin_clim_std['SO4'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

fig.tight_layout()

if SAVE_FIG:
    if WITH_ERRORBARS:
        fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                    + 'MAIA/nwvf_aerosol_climatology_weekly_with_errorbars.png',
                    dpi=600, bbox_inches='tight')
    else:
        fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                    + 'MAIA/nwvf_aerosol_climatology_weekly.png',
                    dpi=600, bbox_inches='tight')

#plt.show()



#####################
# DAILY CLIMATOLOGY #
#####################

# NWVF, groupby day of year
nwvf_EC_Villum_clim = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.day_of_year).mean()
nwvf_EC_Villum_clim_std = nwvf_EC_Villum.groupby(nwvf_EC_Villum.index.day_of_year).std()
nwvf_EC_Zeppelin_clim = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.day_of_year).mean()
nwvf_EC_Zeppelin_clim_std = nwvf_EC_Zeppelin.groupby(nwvf_EC_Zeppelin.index.day_of_year).std()
nwvf_SO4_Villum_clim = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.day_of_year).mean()
nwvf_SO4_Villum_clim_std = nwvf_SO4_Villum.groupby(nwvf_SO4_Villum.index.day_of_year).std()
nwvf_SO4_Zeppelin_clim = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.day_of_year).mean()
nwvf_SO4_Zeppelin_clim_std = nwvf_SO4_Zeppelin.groupby(nwvf_SO4_Zeppelin.index.day_of_year).std()

# Aerosol, groupby day of year
df_EC_Villum_clim = df_EC_Villum.groupby(df_EC_Villum.index.day_of_year).mean()
df_EC_Villum_clim_std = df_EC_Villum.groupby(df_EC_Villum.index.day_of_year).std()
df_EC_Zeppelin_clim = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.day_of_year).mean()
df_EC_Zeppelin_clim_std = df_EC_Zeppelin.groupby(df_EC_Zeppelin.index.day_of_year).std()
df_SO4_Villum_clim = df_SO4_Villum.groupby(df_SO4_Villum.index.day_of_year).mean()
df_SO4_Villum_clim_std = df_SO4_Villum.groupby(df_SO4_Villum.index.day_of_year).std()
df_SO4_Zeppelin_clim = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.day_of_year).mean()
df_SO4_Zeppelin_clim_std = df_SO4_Zeppelin.groupby(df_SO4_Zeppelin.index.day_of_year).std()

# Correlation, compute Pearson correlation coefficient and p-value
r_EC_Villum, p_EC_Villum = pearsonr(nwvf_EC_Villum_clim['nwvf_time_integral'],
                                    df_EC_Villum_clim['EC'])
r_EC_Zeppelin, p_EC_Zeppelin = pearsonr(nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
                                        df_EC_Zeppelin_clim['EC'])
r_SO4_Villum, p_SO4_Villum = pearsonr(nwvf_SO4_Villum_clim['nwvf_time_integral'],
                                      df_SO4_Villum_clim['SO4'])
r_SO4_Zeppelin, p_SO4_Zeppelin = pearsonr(nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
                                          df_SO4_Zeppelin_clim['SO4'])


# Plot NWVF and aerosol climatology (daily)
fig, ax = plt.subplots(2,2, figsize=(15,8), sharex=True)
ax[0,0].plot(nwvf_EC_Villum_clim.index,
             nwvf_EC_Villum_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[0,0].errorbar(nwvf_EC_Villum_clim.index,
                    nwvf_EC_Villum_clim['nwvf_time_integral'],
                    yerr=nwvf_EC_Villum_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,0].legend(loc='upper left')
ax[0,0].set_title(f'Villum EC, r = {r_EC_Villum:.2f}, p = {p_EC_Villum:.2f}')
ax2 = ax[0,0].twinx()
ax2.plot(df_EC_Villum_clim.index,
         df_EC_Villum_clim['EC'],
         color='r', linewidth=1, label='EC')
if WITH_ERRORBARS:
    ax2.errorbar(df_EC_Villum_clim.index,
                df_EC_Villum_clim['EC'],
                yerr=df_EC_Villum_clim_std['EC'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[0,1].plot(nwvf_EC_Zeppelin_clim.index,
             nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[0,1].errorbar(nwvf_EC_Zeppelin_clim.index,
                     nwvf_EC_Zeppelin_clim['nwvf_time_integral'],
                     yerr=nwvf_EC_Zeppelin_clim_std['nwvf_time_integral'],
                     fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[0,1].legend(loc='upper left')
ax[0,1].set_title(f'Zeppelin EC, r = {r_EC_Zeppelin:.2f}, p = {p_EC_Zeppelin:.2f}')
ax2 = ax[0,1].twinx()
ax2.plot(df_EC_Zeppelin_clim.index,
         df_EC_Zeppelin_clim['EC'],
         color='r', linewidth=1, label='EC')
if WITH_ERRORBARS:
    ax2.errorbar(df_EC_Zeppelin_clim.index,
                 df_EC_Zeppelin_clim['EC'],
                 yerr=df_EC_Zeppelin_clim_std['EC'],
                 fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='EC')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[1,0].plot(nwvf_SO4_Villum_clim.index,
             nwvf_SO4_Villum_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[1,0].errorbar(nwvf_SO4_Villum_clim.index,
                    nwvf_SO4_Villum_clim['nwvf_time_integral'],
                    yerr=nwvf_SO4_Villum_clim_std['nwvf_time_integral'],
                    fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,0].legend(loc='upper left')
ax[1,0].set_title(f'Villum SO4, r = {r_SO4_Villum:.2f}, p = {p_SO4_Villum:.2f}')
ax2 = ax[1,0].twinx()
ax2.plot(df_SO4_Villum_clim.index,
         df_SO4_Villum_clim['SO4'],
         color='r', linewidth=1, label='SO4')
if WITH_ERRORBARS:
    ax2.errorbar(df_SO4_Villum_clim.index,
                df_SO4_Villum_clim['SO4'],
                yerr=df_SO4_Villum_clim_std['SO4'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

ax[1,1].plot(nwvf_SO4_Zeppelin_clim.index,
             nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
             color='k', linewidth=1, label='NWVF')
if WITH_ERRORBARS:
    ax[1,1].errorbar(nwvf_SO4_Zeppelin_clim.index,
                     nwvf_SO4_Zeppelin_clim['nwvf_time_integral'],
                     yerr=nwvf_SO4_Zeppelin_clim_std['nwvf_time_integral'],
                     fmt='.', color='k', capsize=2, elinewidth=0.7, ms=4, label='NWVF')
ax[1,1].legend(loc='upper left')
ax[1,1].set_title(f'Zeppelin SO4, r = {r_SO4_Zeppelin:.2f}, p = {p_SO4_Zeppelin:.2f}')
ax2 = ax[1,1].twinx()
ax2.plot(df_SO4_Zeppelin_clim.index,
         df_SO4_Zeppelin_clim['SO4'],
         color='r', linewidth=1, label='SO4')
if WITH_ERRORBARS:
    ax2.errorbar(df_SO4_Zeppelin_clim.index,
                df_SO4_Zeppelin_clim['SO4'],
                yerr=df_SO4_Zeppelin_clim_std['SO4'],
                fmt='.', color='r', capsize=2, elinewidth=0.7, ms=4, label='SO4')
ax2.yaxis.label.set_color('r')
ax2.tick_params(axis='y', colors='r')
ax2.legend(loc='upper right')

fig.tight_layout()

if SAVE_FIG:
    if WITH_ERRORBARS:
        fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                    + 'MAIA/nwvf_aerosol_climatology_daily_with_errorbars.png',
                    dpi=600, bbox_inches='tight')
    else:
        fig.savefig('/Users/au728490/Documents/PhD_AU/PhD_AU_material/Figures/'
                    + 'MAIA/nwvf_aerosol_climatology_daily.png',
                    dpi=600, bbox_inches='tight')

plt.show()
