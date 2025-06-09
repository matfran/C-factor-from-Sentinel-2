# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:06:49 2023

@author: Francis Matthews fmatthews1381@gmail.com

Main module for calculating the C-factor for parcels
"""
# Import modules
import geopandas as gpd
import pandas as pd
import os
import sys
sys.path.insert(0,"path to modules")
from C_factor_module_v2 import get_C_factor 
from remap_crops import remap_crops
from C_factor_functions import plot_ts
import pickle


# Export final results
export = True

# Load data
dir_ = "path to main directory (input and interim results based on the goal of the run (annual/multi-year)"
dir_ndvi = os.path.join(dir_,"path to folder with NDVI timeseries")
dir_ndti = os.path.join(dir_,"path to folder with NDTI timeseries")
# Load parcel data
parcels = gpd.read_file(os.path.join(dir_, "BE_VLG_2023_fd23_sample.shp"))

# Input files for C-factor calculation (excluding REDES)
files_to_read = {}
files_to_read['crop_models_file_path'] = os.path.join(dir_,'Canopy_cover_NDVI_relationships_corrected.csv') # Relationships between CC and NDVI
files_to_read['nuts_path'] = os.path.join(dir_,'NUTS1_and_NUTS2_FINAL.shp') # NUTS data
files_to_read['Ukkel_data_path'] = os.path.join(dir_, 'Rainfall_erosivity/Ukkel_15day_median.csv') # Rainfall erosivities

# Input files for the crop residue component
files_to_read['rf_model_path'] = os.path.join(dir_,'v2_crop_res_class_73.pkl') # Tillage evaluation model
files_to_read['soil_path'] = os.path.join(dir_, 'Soil_Properties_Parcels.csv') # Soil properties of the parcels
files_to_read['tillage_practices'] = os.path.join(dir_,'NUTS2_2010_tillage_practices.csv') # Tillage practices data

# 1. Merging NDVI CSV files of the NDVI directory into one DataFrame
def merge_ndvi_csvs(ndvi_directory, output_file):
    """
    Function to read, merge all NDVI CSV files in the given directory,
    and save the merged DataFrame to a CSV file.
    
    ndvi_directory: Path to the directory containing NDVI CSV files.
    output_file: Path to save the merged NDVI data as CSV.
    
    return: Path to the merged CSV file.
    """
    # Load all NDVI timeseries of the NDVI time series directory
    ndvi_files = [f for f in os.listdir(ndvi_directory) if f.endswith('.csv')]
    
    # Initialize an empty list to store each NDVI DataFrame
    ndvi_dataframes = []
    
    # Loop through all CSV files and append to the list
    for file in ndvi_files:
        file_path = os.path.join(ndvi_directory, file)
        df = pd.read_csv(file_path)
        ndvi_dataframes.append(df)
    
    # Concatenate all DataFrames into one
    merged_ndvi_df = pd.concat(ndvi_dataframes, ignore_index=True)
    
    # Save the merged DataFrame to a CSV file
    merged_ndvi_df.to_csv(output_file, index=False)
    
    return output_file

# Call the function to merge NDVI data and save to a file
ndvi_output_file = os.path.join(dir_ndvi, 'merged_ndvi_data.csv')
ndvi_merged_path = merge_ndvi_csvs(dir_ndvi, ndvi_output_file)

# Call the function to merge NDTI data and save to a file
ndti_output_file = os.path.join(dir_ndti, 'merged_ndti_data.csv')  
ndti_merged_path = merge_ndvi_csvs(dir_ndti, ndti_output_file)

# Add the path of the merged NDVI & NDTI data to the files_to_read dictionary
files_to_read['ndvi_ts_path'] = ndvi_merged_path
files_to_read['ndti_ts_path'] = ndti_merged_path

# Load the merged NDVI CSV file into a DataFrame
ndvi_merged_df = pd.read_csv(ndvi_merged_path)
ndvi_merged_df = ndvi_merged_df.drop_duplicates(subset='id', keep='first')

# Take id from parcel data and NDVI time series data
parcels['REF_ID'] = parcels['REF_ID'].astype('int64')
ndvi_merged_df['id'] = ndvi_merged_df['id'].astype('int64')
# You may need to adjust this column name based on your NDVI CSV structure
# Filter the parcel data on id so that you only have parcels for which NDVI time series are available
filtered_parcels = parcels[parcels['REF_ID'].isin(ndvi_merged_df['id'])]

# Renaming columns in parcel data and remapping crops
# Remap the parcel crop data based on external references => enrich dataset by adding beta values
filtered_parcels.rename(columns={'GWSNAM_H': 'crop', 'REF_ID': 'OBJECTID'}, inplace=True)
sampled_crops_ref_path = os.path.join(dir_, 'GSAA_crops_to_sample_belgium_flanders.csv')
reclassification_ref_path = os.path.join(dir_, 'Crop_name_reclassification.csv')
parcels = remap_crops(filtered_parcels, sampled_crops_ref_path, reclassification_ref_path)

# Run the C-factor calculation function
output_dir = dir_

results = get_C_factor(parcels, files_to_read,
                       cf_year=2023, parcel_merge_type='left', 
                       incorporate_crop_res=True, plot=False, 
                       SLR_uncertainty_analysis=False)

# Retrieve resulting slr and fvc time series
fvc_ts = results["GPR FVC"]
slr_ts = results["SLR ts full"]

# Plot timeseries data
plot_ts(fvc_ts,use_clusters=False, quantile_bounds=0.8)
plot_ts(slr_ts,use_clusters=False, quantile_bounds=0.8)

# Export results if needed
if export:
    p_name = 'Result.pickle'
    pickle_p = os.path.join(output_dir, p_name)
    pickle.dump(results, open(pickle_p, 'wb'))
    results["C-factor results"].to_csv(os.path.join(output_dir, "C_factor.csv"), index=False)
    results["SLR ts full"].to_csv(os.path.join(output_dir, "SLR_timeseries.csv"), index=False)
    results["IACS with NDVI ts"].to_csv(os.path.join(output_dir,"IACS_with_NDVI_ts.csv"),index=False)



