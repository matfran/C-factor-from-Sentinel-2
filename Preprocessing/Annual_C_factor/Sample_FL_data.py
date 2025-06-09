# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 15:24:02 2024

@author: Francis Matthew & Arno Kasprzak

Script to sample parcels from the Flemish parcel dataset for C-factor calculations
"""
# Import modules
import geopandas as gpd
import pandas as pd
import os

# Define main data directory
root = "Define path to main data directory"
# File paths
fd_path = os.path.join(root,"Landbouwgebruikspercelen_2023_-_Definitief_(extractie_28-03-2024).shp") # Parcel data
eurocrops_ref= os.path.join(root,"BE_VLG_2021_EC21.shp") # Needed to extract english crop names (do not need to match with the year)
# Reference file that indicates which crop should be sampled based on the EuroCrops naming
fd_ref_path = os.path.join(root,"CROPS_TO_SAMPLE_EC_trans_n_ref.csv")

# Load the data
fd = gpd.read_file(fd_path)
fd_euro_ref = gpd.read_file(eurocrops_ref)
fd['Area m2'] = fd.area  # Calculate area in square meters
fd['Area ha'] = fd['Area m2']/10000  # Convert area to hectares
fd = fd.to_crs('EPSG:4326')  # Reproject to WGS84 (EPSG:4326)
fd_ref = pd.read_csv(fd_ref_path, encoding='latin')  # Load reference crop data

# Create a new column Name_trans which contains the translation of the dutch crop name based on EUROCROPS
# This dictionary will map the values from 'GWSNAM_H' in Flemish data to 'EC_trans_n'
mapping_dict = fd_euro_ref.set_index('GWSNAM_H')['EC_trans_n'].to_dict()
# Use the mapping dictionary to assign values from 'EC_trans_n' based on 'GWSNAM_H'
fd['Name_trans'] = fd['GWSNAM_H'].map(mapping_dict)

# Count the number of parcels for each crop type (GWSNAM_H)
fd_count = fd.groupby('GWSNAM_H', as_index=False).agg({'geometry': 'count'}).rename(columns={'geometry': 'PARCEL_COUNT'})

# Get additional information about each crop type
fd_info = fd.groupby('GWSNAM_H', as_index=False).first()[['REF_ID','GRAF_OPP','GWSCOD_V', 'GWSNAM_V', 'GWSCOD_H',
       'GWSNAM_H', 'GWSGRPH_LB', 'GWSCOD_N', 'GWSNAM_N',
       'GWSCOD_N2', 'GWSNAM_N2', 'GESP_PM', 'GESP_PM_LB', 'ERO_NAM',
       'STAT_BGV', 'LANDBSTR', 'STAT_AAR', 'PCT_EKBG', 'PRC_GEM', 'PRC_NIS','Name_trans']]

# Merge parcel count data with additional crop information
fd_count = fd_count.merge(fd_info, on='GWSNAM_H', how='left')

# Save the parcel count data to CSV
save_path = os.path.join(root,"CROPS_TO_SAMPLE_fd_GWSNAM_H.csv")
fd_count.to_csv(save_path,
                encoding='latin')

# Filter the reference crops to only those with SAMPLE == 1 (i.e., the crops to sample)
fd_ref_sample = fd_ref[fd_ref['SAMPLE'] == 1]
fd_sample = fd[fd['GWSNAM_H'].isin(fd_ref_sample['GWSNAM_H'])]

# Create a copy of the filtered DataFrame to avoid the SettingWithCopyWarning
fd_sample = fd_sample.copy()

# Filter parcels with an area greater than 0.5 hectares
fd_sample = fd_sample[fd_sample['Area ha'] > 0.5]

# Select relevant columns (GWSNAM_H, Job_index, and geometry)
fd_sample = fd_sample[['REF_ID','GWSNAM_H', 'Name_trans', 'geometry']]

# Sample 25% of the parcels
fd_sample = fd_sample.sample(frac=0.25)

# Save the sampled parcels as a shapefile
sample_path = os.path.join(root,"BE_VLG_2023_fd23_sample.shp")
fd_sample.to_file(sample_path)