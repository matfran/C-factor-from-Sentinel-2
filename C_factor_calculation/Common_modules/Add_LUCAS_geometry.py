# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 15:05:35 2021

A module defining a function to enrich a geometry dataset by adding information
about tillage practices based on NUTS2 regions from a CSV file.

@author: Francis Matthews
"""
# Import modules
import geopandas as gpd
import pandas as pd

# Defining the add_NUTS_info
def add_NUTS_info(geometry_outer, geometry_inner, files_to_read):
    """
    This function enriches a given geometry_inner dataset by adding information
    about tillage practices based on NUTS2 regions from an external csv file
    
    Parameters
    ----------
    geometry_outer: GEODATAFRAME
        This is the dataset that holds the NUTS2 data including tillage practices
    geometry_inner: GEODATAFRAME
        This is the dataset you want to ernich with the NUTS2 tillage data
        
    """
    # Read the NUTS2 tillage data from the csv file
    nuts2_tillage = pd.read_csv(files_to_read['tillage_practices'])
    # Merge the geometry_outer geodataframe with the NUTS2 dataframe
    geometry_outer = geometry_outer.merge(nuts2_tillage, on = 'NUTS2___de')
    # Select only necessary columns
    geometry_outer = geometry_outer[['NUTS2___de', 'Conventional',
    'Conservation', 'No-till', 'geometry']]
    # Rename columns
    geometry_outer.rename(columns = {'Conventional' : 'NUTS2_conventional', 'Conservation': 'NUTS2_conservation',
                                'No-till': 'NUTS2_notill'}, inplace = True)
    geometry_outer = geometry_outer.to_crs(geometry_inner.crs)
    # Perform a spatial join between geometry_inner and geometry_outer
    subset = gpd.sjoin(geometry_inner, geometry_outer, how='left', op='within')
    # Delete index_right
    del(subset['index_right'])
    
    # Fill missing values
    subset['NUTS2_conventional'] = subset['NUTS2_conventional'].fillna(subset['NUTS2_conventional'].mean())
    subset['NUTS2_conservation'] = subset['NUTS2_conservation'].fillna(subset['NUTS2_conservation'].mean())
    subset['NUTS2_notill'] = subset['NUTS2_notill'].fillna(subset['NUTS2_notill'].mean())
    subset['NUTS2___de'] = subset['NUTS2___de'].fillna(method = 'ffill')
   
    return subset