# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:50:56 2021

This module defines a function for remapping crop data based on external references.
By using this function one can enrich the crop data by adding additional information (e.g. beta values)

@author: Francis Matthews
"""
# Import modules
import pandas as pd

# Define the remap_crops function
def remap_crops(crop_dataset, sampled_crops_ref_path, reclassification_ref_path):
    """
    Enriches crop data with names, models, and beta values from external references.

    Parameters
    ----------
    crop_dataset : pd.DataFrame
        The main dataset containing crop data.
    sampled_crops_ref_path : str
        Path to CSV with crop names (Dutch and English).
    reclassification_ref_path : str
        Path to CSV with crop model and beta values.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with additional crop information.
    """
    sampled_crops_ref = pd.read_csv(sampled_crops_ref_path, engine = 'python', encoding='utf8')
    reclassification_ref = pd.read_csv(reclassification_ref_path)
    sampled_crops_ref.rename(columns = {'Name' : 'crop'}, inplace = True) 
    df = crop_dataset.merge(sampled_crops_ref, on = ['crop'], how = 'left')
    df.index = df['OBJECTID'].values

    # Create 2 remappers to get crop model references and crop name references
    remapper_1 = dict(zip(reclassification_ref['Crop_ref'], reclassification_ref['Crop_model']))
    remapper_2 = dict(zip(reclassification_ref['Crop_ref'], reclassification_ref['Crop_name']))
    try:
        remapper_3 = dict(zip(reclassification_ref['Crop_ref'], reclassification_ref['Beta_value']))
        df.insert(1, 'beta_value', df['crop_ref'].replace(remapper_3))
    except:
        print('No crop-specific Beta values given. Using average value for all fields (-0.04)')
        df.insert(1, 'beta_value', -0.04)
    
    try:
        remapper_4 = dict(zip(reclassification_ref['Crop_ref'], reclassification_ref['Beta_cat']))
        df.insert(1, 'beta_cat', df['crop_ref'].replace(remapper_4))
    except:
        print('No categorical Beta values given')  
    
    # Remappers for harvest and sowing date
    remapper_s_date = dict(zip(reclassification_ref['Crop_ref'], reclassification_ref['S_date']))
    df.insert(1, 'S_date', df['crop_ref'].replace(remapper_s_date).fillna(pd.NaT))
    remapper_h_date = dict(zip(reclassification_ref['Crop_ref'], reclassification_ref['H_date']))
    df.insert(1, 'H_date', df['crop_ref'].replace(remapper_h_date).fillna(pd.NaT))
    
    
    df.insert(1, 'crop_model', df['crop_ref'].replace(remapper_1))
    df.insert(1, 'crop_name', df['crop_ref'].replace(remapper_2))
    
    
    try:
        del(df['Name_x'])
    except:
        pass
    
    try:
        del(df['Name_y'])
    except:
        pass
        
    del(df['n'])
    del(df['crop_ref']) 
    print (df.iloc[0])
    return df

