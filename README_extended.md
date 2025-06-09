# C-factor via remote sensing python workflow

This document contains all the information regarding the Python/Google Earth Engine scripts used to calculate parcel-specific (multi-year/annual) C-factors based on Sentinel-2 remote sensing imagery for Flanders. Additionally, it includes a necessary Python script for statistically analyzing the results.

The main module for the actual modelling of the C-factor for parcels in Flanders is the Flanders_Cfactor_sample.py script, located in the C_factor_calculation folder. The Common_modules folder, also present in this directory, contains essential functions that are used by the main module.

The entire Python workflow is based on EUROCROPS data and can therefore also be used or adapted for other regions in Europe. The scripts outlined here are, as previously mentioned, an application focused on Flanders, but this does not mean they are exclusively applicable to Flanders. They can be regarded as a template to be used when calculating C-factors for a specific region within Europe.

More information about EUROCROPS data can be found via the link below:
https://www.eurocrops.tum.de/index.html

## Input data 

The following input data are required to run the Python workflow:
- Spatial parcel shapefile/geopackage data (https://landbouwcijfers.vlaanderen.be/open-geodata-landbouwgebruikspercelen)
- Spatial parcel Eurocrops shapefile data (https://www.eurocrops.tum.de/index.html)
- Shapefile containing agricultural regions of Flanders (https://www.vlaanderen.be/datavindplaats/catalogus/landbouwstreken-belgie-toestand-1974-02-15)
- CSV file containing information on which crop to sample (CROPS_TO_SAMPLE_EC_trans_n_ref.csv)
- CSV file containing crop specific canopy cover – NDVI relationships (Canopy_cover_NDVI_relationships_corrected.csv)
- NUTS shapefile data (NUTS1_and_NUTS2_FINAL.shp)
- Machine learning classifier for detecting crop residues (v2_crop_res_class_73.pkl)
- CSV file containing field tillage information (NUTS2_2010_tillage_practices.csv)
- CSV file containing information for crops to sample (GSAA_crops_to_sample_belgium_flanders.csv)
- CSV file containing crop-specific information for the C-factor calculations (Crop_name_reclassification.csv)
- CSV file containing information about erosivity per rainfall event (erosivity_per_rainfall_event.csv)
- CSV file containing legend for crop rotations (RotID_legend.csv)

## Script information

### Preprocessing

Before using the main modules for C-factor calculations, a few preparatory steps need to be completed. The scripts for this part of the workflow can be found in the "Preprocessing" folder.

#### Parcel selection

##### Annual C-factor

For annual C-factor calculations, first, a random selection of the parcel shapefile data needs to be made (in this case, 25%),on which further calculations and analysis will be performed. The script in the "Annual_C_factor" folder should be used for this step.

-Sample_FL_data.py: Script to sample parcels from the Flemish parcel dataset for C-factor calculations
    - Input: 
        - Spatial parcel shapefile data
        - Eurocrops data
        - CROPS_TO_SAMPLE_EC_trans_n_ref
    - Output: 
        - Parcel count data (CROPS_TO_SAMPLE_fd_GWSNAM_H.csv)
        - Shapefile containing the sampled parcels (BE_VLG_2023_fd23_sample.shp)

##### Multi-year C-factor

If multi-year C-factors are to be calculated, a subsample of parcels must first be selected for the analysis. The code for this can be found in the "Crop_rotation_preanalysis" folder. The following scripts should be followed in this folder:

- crop_rotations_analysis.py: Script comparing geometries from three shapefiles of agricultural parcels across three years, identifying unchanged parcels, analysing combinations of crops, and saving the results in shapefiles and a CSV file. The script then analyzes the occurrence of different crop rotations 
for main crops based on that result.
    - Input: 
        - Spatial parcel geopackage data for different years (Landbouwgebruikspercelen_2021.gpkg)
    - Output: :
        - Geopackages containing unchanged parcels for the three years (unchanged_parcelsx.gpkg)
        - CSV file storing, for the different parcels, the combinations of main crops over three consecutive years(parcel_combinations_analysis.csv)
        - CSV file containing the occurrence of each unique combination (crop_combinations.csv)
        - Filtered CSV file containing the occurrence of relevant combinations (crop_combinations_filtered.csv)
        - Excel file containing the occurence of relevant combinations ignoring the crop order(Crop_rotations_unique_combinations.xlsx): This file can then be used to determine the most common crop rotations, which in turn can be used to select the rotations to be included in the final calculations
- sample_crop_rotations_parcels.py: Script to sample 1000 parcels per rotation
    - Input:
        - Geopackages containing unchanged parcels for the three years (unchanged_parcelsx.gpkg)
        - Spatial parcel geopackage data for different years (Landbouwgebruikspercelen_2021.gpkg)
        - Define which rotations you want to sample in the target_crops_list
    - Ouput:
        - Shapefile with sampled parcels for 3 years per rotation (sampled_parcels_set_{i}.shp)
        - Shapefile/Geopackage with sampled parcels per year (Sampled_parcels_year{i+1}.shp|gpkg)
        - Geopackage with sampled parcels per year containing a link_id used to link the parcels over 3 years (Rotation_data_y*.gpkg)

#### Remote sensing/Google Earth Engine data

Next, NDVI/NDTI values from Sentinel-2 imagery need to be extracted for the sampled plots. Additionally, several soil properties need to be retrieved. All of this will be done using Google Earth Engine, and the scripts for this are located in the 'Remote_sensing_extractions' folder.

- NDVI_extraction_GEE.txt: Script to extract NDVI values
    - Input:
        - Shapefile containing the sampled parcels
        - Shapefile containing agricultural regions of Flanders
    - Output: 
        - A CSV file per batch (1000 plots per batch) containing NDVI values per parcel
- NDTI_extraction_GEE.txt: Script to extract NDTI values
    - Input:
        - Shapefile containing the sampled parcels
        - Shapefile containing agricultural regions of Flanders
    - Output: 
        - A CSV file per batch (1000 plots per batch) containing 15 daily NDTI values per parcel
- Soil_properties_extraction_GEE.txt:
    - Input: 
        - Shapefile containing the sampled parcels
        - Shapefile containing agricultural regions of Flanders
    - Output:
        - CSV file containing soil properties per parcel

### Rainfall erosivity

The scripts for calculating rainfall erosivity can be found in the 'Rainfall_erosivity' folder.

- RE_15day_Flanders.py: Script that reads rainfall data and calculates various metrics by grouping based on time intervals and station records. It also
visualise the results
    - Input: CSV file containing information about erosivity per rainfall event (erosivity_per_rainfall_event.csv)
    - CSV file containing time series of rainfall erosivities (Ukkel_15day_median.csv)

### C-factor calculation module

Everything is now ready to calculate the C-factors for the sampled plots. The main script and the modules used in it can be found in the folder "C_factor_calculation".

The main script for calculating the C-factors is the following:
- Flanders_Cfactor_sample.py: Main module for calculating the C-factor for parcels
    - Input:
        - Shapefile containing the sampled parcels
        - Folder containing NDVI timeseries CSV files
        - Folder containing NDTI timeseries CSV files
        - CSV file containing soil properties
        - CSV file containing crop specific canopy cover – NDVI relationships (Canopy_cover_NDVI_relationships_corrected.csv)
        - NUTS shapefile data (NUTS1_and_NUTS2_FINAL.shp)
        - Machine learning classifier for detecting crop residues (v2_crop_res_class_73.pkl)
        - CSV file containing field tillage information (NUTS2_2010_tillage_practices.csv)
        - CSV file containing information for crops to sample (GSAA_crops_to_sample_belgium_flanders.csv)
        - CSV file containing crop-specific information for the C-factor calculations (Crop_name_reclassification.csv)
        - CSV file containing time series of rainfall erosivities (Ukkel_15day_median.csv)
    - Output: 
        - The results are saved in a pickle file containing dataframes with the C-factor per parcel, Soil Loss Ratio time series per parcel, NDVI time series per parcel, SLR × RE time series, number of minimum NDVI acquisitions, error logger, C-factor year, Gaussian Process Regression outputs, and fractional vegetative cover per parcel (Result.pickle)
        - CSV files containing C-factor results, Soil Loss Ratio timeseries and NDVI timeseries per parcel

#### Modules used for C-factor calculations

Modules containing the necessary functions used in the main module for calculating the C-factors can be found in the folder 'Common_modules'. 

The following modules are available:
- Add_LUCAS_geometry.py: A module defining a function to enrich a geometry dataset by adding information
about tillage practices based on NUTS2 regions from a CSV file.
    - Functions:
        - add_NUTS_info
- C_factor_functions.py: This module defines a set of functions for analysing and visualising data,
particularly focused on FVC, SLR and RE.
    - Functions:
        - count_annual_observations
        - format_merge
        - add_harvest_cropres
        - FVC_to_SLR
        - C_factor
        - Calc_risk_period
        - plot_ts
        - C_factor_box_plot
        - C_factor_box_plot_multiy
        - plot_cumulative
        - C_factor_comparison_box_plot
        - load_data
        - load_data_with_name
        - load_data_pkl
        - load_data_gpkg
        - filter_crops
        - perform_kruskal_test
        - compute_phenology_metrics
        - plot_compare_cfactor
        - plot_population_outliers
- C_factor_module_v2.py: A module defining functions to calculate a C-factor for field parcels using a time series data
of ndvi acquisitons.
    - Functions:
        - get_df
        - get_C_factor
- GPR_analysis_v2.py: Module for Implementing Gaussian Process Regression and Spline Interpolation for Remote Sensing phenological Time Series, enabling predictions of Fractional Vegetation Cover.
    - Functions: 
        - format_arrays
        - filter_by_n
        - evaluate_gpr_int
        - GPR_interpolate
- Harvest_delineation.py: Module with functions to identify inflection points (harvest periods) in vegetation cover time series data using differential analysis and generates a summary of NDVI and NDTI values surrounding those inflection points.
    - Functions:
        - identify_inflexions
        - identify_multiple_harvests
- remap_crops.py: This module defines a function for remapping crop data based on external references. By using this function one can enrich the crop data by adding additional information (e.g. beta values).
    - Functions: 
        - remap_crops

#### Script for calculating multi-year C-factor

Based on the annual outputs over multiple years from the main C-factor modules, a multi-year C-factor can be calculated. The script for this can be found in the folder 'Multi_year_analysis'.

- Multiyear_Cfactor_Flanders.py: This script adds C-factor values from different years belonging to the same parcels together and calculates average C-factors per parcel over multiple years.
    - Input:
        - Geopackages with sampled parcels per year (Sampled_parcels_year{i+1}.gpkg)
        - CSV files with C-factor results per year
    - Output:
        - CSV file with multi-year C-factor per parcel (C_factor_results_rot_combined.csv)

### Statistical analysis of the results

Finally, a script for statistically analyzing and visualizing the results can also be found in the 'Statistical_analysis' folder.

- C-factor_feature_analysis.py: Statistical analysis of the C-factor outputs
    - Input: 
        - Resulting CSV files with C-factor results, NDVI timeseries, rotation ID legend, C-factor results per year for the multi-year C-factor
        - Resulting Pickle files
        - Geopackages of the parcels and rotation info per year (Rotation_data_y*.gpkg)
    - Output:
        - Visualisations
        - Statistical outputs