# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 14:06:49 2023

@author: Francis Matthews fmatthews1381@gmail.com
"""

import geopandas as gpd
import os
from C_factor_module_v2 import get_C_factor 
from remap_crops import remap_crops
import pickle

export = False

dir_ = 'C:/Users/u0133999/OneDrive - KU Leuven/PhD/WaTEM_SEDEM_preprocessing/C_factor/Flanders_sample'

iacs = gpd.read_file(os.path.join(dir_, 'belgium_flanders.shp'))

files_to_read = {}
files_to_read['ndvi_ts_path'] = os.path.join(dir_, 'GSAA_NDVI_Belgium_Flanders_2018.csv')
files_to_read['crop_models_file_path'] = os.path.join(dir_,'Canopy_cover_NDVI_relationships_corrected.csv')
files_to_read['REDES_gauges_path'] = 'Monthly_avg_RE_all.csv' #not open access
files_to_read['EnS_file_path'] = os.path.join(dir_,'ens_v8.shp')
files_to_read['nuts_path'] = os.path.join(dir_,'NUTS1_and_NUTS2_FINAL.shp')
files_to_read['all_REDES_path'] = 'Detailed_Events_All_scaled_monthly.csv' #not open access
files_to_read['REDES_reference_path'] = 'R_factor_20150622_REF_Ens.csv' #not open access


name = 'Flanders'
sensor_name = 'Sentinel-2'
output_dir = dir_
sampled_crops_ref_path = os.path.join(dir_, 'GSAA_crops_to_sample_belgium_flanders.csv')
reclassification_ref_path = os.path.join(dir_, 'Crop_name_reclassification.csv')

iacs.rename(columns = {'Name_trans' : 'crop', '_uid_' : 'OBJECTID', 'CULT_NAME' : 'crop'}, inplace = True)
iacs = remap_crops(iacs, sampled_crops_ref_path, reclassification_ref_path)

#run function with a right type merge to only run parcels with a timeseries
results = get_C_factor(iacs, files_to_read, name, sensor_name,
                       cf_year = 2018, iacs_merge_type = 'right', 
                       run_all_parcels = False, plot = True)

if export == True:
    p_name = 'results_all_flanders.pickle'
    pickle_p = os.path.join(output_dir, p_name)
    pickle.dump(results, open(pickle_p, 'wb')) 


