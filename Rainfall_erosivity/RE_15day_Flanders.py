# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 08:47:38 2024

Script that reads rainfall data and calculates various metrics 
by grouping based on time intervals and station records. It also
visualise the results

@author: Francis Matthews (franci.matthews@kuleuven.be)
"""
# Import modules
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Define data directory
root = "path to main data directory"
# Read in the data and attach some attributes
# Define data types for efficient memory usage
dtypes = {'erosivity' : np.float32, 'event_rain_cum': np.float32, 
          'max_30min_intensity': np.float32, 'event_energy': np.float32}
# Define path
file_path = os.path.join(root, 'Rainfall_erosivity/erosivity_per_rainfall_event.csv')
# Read the rainfall erosivity data from a CSV file
df_all = pd.read_csv(file_path, dtype = dtypes)
# Extract day of year (DOY) and year from datetime
df_all['DOY'] = pd.to_datetime(df_all['datetime']).dt.dayofyear
df_all['Year'] = pd.to_datetime(df_all['datetime']).dt.year

# Count unique years for each station and prepare to categorize by record length
sy_count = df_all.groupby('station', as_index = False).nunique()[['station', 'Year']].rename(columns = {'Year': 'N_years'})
# # Define bins for record length categorie
record_bins = [0, 10, 20, 30, 105]
bins = pd.cut(sy_count['N_years'], record_bins, right = True)
# Assign record length category to each station
sy_count['Record_length'] = bins
sy_count['Station_group'] = np.where(sy_count['N_years'] == 105, 'Ukkel', 'All other stations')
# Merge station attributes with the main dataframe
df_all = df_all.merge(sy_count, on = 'station', how = 'left')

# Create bins for the 15-day periods in RUSLE
# Divide events into bins based on their day of occurence in the year
month_array = [0,15,31,46,59,74,90,105,120,135,151,166,181,196,212,227,243,258,273,288,304,319,334,349,366]
labels = np.arange(len(month_array) - 1)
bins = pd.cut(df_all['DOY'], month_array, right = True, labels = labels)
# Assign interval based on day of year
df_all['interval'] = bins

# Group by station, year, and interval. Creates a time series record of all 15-day averages
df_g_syi = df_all.groupby(['station', 'Year', 'interval'], as_index = False).sum()[['station', 'Year', 'interval',
                                                                                'event_rain_cum','erosivity']]
# Filter out rows where erosivity is zero
df_g_syi = df_g_syi[df_g_syi['erosivity'] > 0]

# Group by station and interval. This averages over time.
df_g_si = df_g_syi.groupby(['station', 'interval'], as_index = False).mean()[['station', 'interval',
                                                                                'event_rain_cum','erosivity']]
# Merge auxillary station info
df_g_si = df_g_si.merge(sy_count, how = 'left', on = 'station')
df_g_syi = df_g_syi.merge(sy_count, how = 'left', on = 'station')

# Get the mean and median per 15-days for all stations 
df_mean_15day = df_g_syi.groupby('interval').mean()
df_median_15day = df_g_syi.groupby('interval').median()

# Get the mean and median per 15-days for Ukkel
df_mean_15day_ukkel = df_g_syi[df_g_syi["Station_group"] == 'Ukkel'].groupby('interval').mean()
df_median_15day_ukkel = df_g_syi[df_g_syi["Station_group"] == 'Ukkel'].groupby('interval').median()


# Create boxplot for erosivity data per 15-day interval across all stations
fig, ax = plt.subplots(figsize=(15,8))
sns.boxplot(data = df_g_si, x = 'interval', y = 'erosivity', color = 'c', ax = ax)
ax.set_ylabel('15-day mean rainfall erosivity per station')
ax.set_xlabel('15-day interval')

# Create boxplot comparing erosivity by station group with logarithmic scale on the y-axis
fig, ax = plt.subplots(figsize=(15,8))
sns.boxplot(data = df_g_syi, x = 'interval', y = 'erosivity', hue = 'Station_group', ax = ax)
ax.set_ylabel('15-day rainfall erosivity per station')
ax.set_xlabel('15-day interval')
ax.set_yscale('log')
leg = plt.legend(frameon=False)

# Create histogram to visualize the number of years for each gauge station record
fig, ax = plt.subplots(figsize=(15,8))
sns.histplot(data = sy_count, x = 'N_years', ax = ax)
ax.set_ylabel('Count')
ax.set_xlabel('Number of years per gauge station record')


# Save the median 15-day values for Ukkel to CSV in the same format as the example
path = os.path.join(root,'Rainfall_erosivity/Ukkel_15day_median.csv')
df_median_15day_ukkel[['event_rain_cum', 'erosivity']].reset_index().to_csv(path, index=False)
