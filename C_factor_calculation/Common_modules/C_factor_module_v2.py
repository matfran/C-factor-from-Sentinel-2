# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:48:41 2021

A module defining functions to calculate a C-factor for field parcels using a time series data
of ndvi acquisitons.

@author: Franncis Matthews & Arno Kasprzak
"""
# Import modules
import pandas as pd 
import geopandas as gpd
import numpy as np
from GPR_analysis_v2 import GPR_interpolate
from pickle import load
from Harvest_delineation import identify_multiple_harvests
from Add_LUCAS_geometry import add_NUTS_info
from C_factor_functions import count_annual_observations, format_merge
from C_factor_functions import C_factor, Calc_risk_period, add_harvest_cropres, FVC_to_SLR

"""
Defining the get_df function
"""
def get_df(sl_ei_all, obj_id_list, crop_list, EnS_list):
    """
    This function compiles multiple lists into a single DataFrame.
    It arranges the columns in a specific order so that the key reference columns 
    (in this case obeject_id and crop) are placed at
    the beginning of the DataFrame.
    
    Parameters
    ----------
    sl_ei_all : LIST
        A list of lists or nested data representing various observations or measurements
    obj_id_list : LIST
        A list of object IDs corresponding to the entries in `sl_ei_all`
    crop_list : LIST
        A list of crop types corresponding to the entries in `sl_ei_all` 

    Returns
    -------
    df_sl_ei_all : DATAFRAME
        A compilled dataframe of all lists

    """

    df_sl_ei_all = pd.DataFrame(sl_ei_all)
    df_sl_ei_all['object_id'], df_sl_ei_all['crop'], df_sl_ei_all['EnS'] = [obj_id_list, crop_list, EnS_list]
    # Rearrange the columns to put reference columns first
    new_col_ord = df_sl_ei_all.columns[-3:].to_list() + df_sl_ei_all.columns[:-3].to_list()
    df_sl_ei_all = df_sl_ei_all[new_col_ord]
    return df_sl_ei_all

"""
Defining the get_C_factor function
"""
def get_C_factor(parcels, files_to_read, cf_year = None,
                 incorporate_crop_res = True, parcel_merge_type = 'left', 
                 slr_exponent = - 0.035, min_ndvi_y = None, 
                 SLR_uncertainty_analysis = False, plot = False, 
                 beta_column = 'beta_cat'):
    '''
    A function to calculate the C-factor for a series of parcels. The 
    function runs a series of processes in order to return a final C-factor value
    and SLR time series, broadly speaking these are:
        1) Add relevant information to each parcel which is neccessary 
        to make an evaluation of the C-factor. The crucial ones are the NDVI/NDTI
        time series from GEE and a rainfall erosivity time series. 
        2) For all parcels in the geodataframe, interpolate the discrete 
        satellite observations into a 15-day time series using GPR to obtain 
        a smooth time series record without outliers. 
        3) Calculate the C-factor for each parcel and return all the relevant 
        information
        

    Parameters
    ----------
    parcels : GEODATAFRAME
        Geodataframe with field parcels and attributes
    files to read : DICTIONARY
        A dictionary containing all file path names to read in
    cfyear: INTEGER
        The year in the time series for which the C-factor is calculated
    incorporate_crop_res : BOOLEAN, optional, 
        The default is True. Define whether to include crop 
        residue in the C-factor calculation. If True, a path needs to be 
        provided to NDVI time series data and soil property data.
    parcel_merge_type : STRING, default='left'
        Defines the merge strategy for joining additional data to the parcels dataframe.
        Can be 'left', 'right', 'inner', or 'outer'.
    slr_exponent : FLOAT, default=-0.035
        Default exponent value used in the SLR (Soil Loss Ratio) function for modelling the relationship
        between NDVI and erosion potential (when no crop beta is specified in the additional files)
    min_ndvi_y : FLOAT or None, optional
        Minimum NDVI threshold. NDVI values needed for the calculations per parcel.  
    SLR_uncertainty_analysis : BOOLEAN, default=False
        If True, perform uncertainty analysis on the SLR calculation
    plot : BOOLEAN, default=False
        If True, generate plots of time series.
    beta_column : STRING, default='beta_cat'
        Name of the column in the parcels dataframe containing crop/vegetation category beta
        used in the C-factor calculation.
    
    Returns
    
    output : DICTIONARY
    A dictionary with the multiple compiled results. The standard outputs are:
        output['Parcels with NDVI ts']
        output['C-factor results']
        output['SLR RE ts (one year)'] 
        output['SLR ts full']
        output['N minimum NDVI acquisitions']
        output['Error logger']
        output['C-factor year']
    
    '''
    # Initilise the relevant variables-------------------------------------
    # The key operation here is to merge sentinel-2 or Landsat observations
    # With the parcel dataframe
    
    # Unpack the crop_models_file and NUTS file path
    crop_models_file_path = files_to_read['crop_models_file_path']
    nuts_path = files_to_read['nuts_path']

    # Set the method for converting NDVI to FVC
    NDVI_to_fvc_method = 'tenriero_et_al' 
    # Determine whether to evaluate the fit of GPR model
    evaluate_gpr_fit = False
    
    # Set minimum observations 
    if min_ndvi_y == None:
        min_ndvi_y = 12        
    
    # Read the filename path with the ndvi time series 
    parcels_timeseries = pd.read_csv(files_to_read['ndvi_ts_path'])
    # If a crop residue assessment is required then we need to read a classifier
    # Model and provide an NDTI time series and soil properties for each parcel
    if incorporate_crop_res == True:
        rf_model_path = files_to_read['rf_model_path']
        rf_model = load(open(rf_model_path, 'rb'))
        NDTI_timeseries = pd.read_csv(files_to_read['ndti_ts_path'])
        soil_properties = pd.read_csv(files_to_read['soil_path'])
    
    # Define the path to the nuts2 info and add it to the IACS data
    # This way we can know some regional statistics
    nuts_polygons = gpd.read_file(nuts_path)
    parcels = add_NUTS_info(nuts_polygons, parcels, files_to_read)
    # Delete after the merge
    del(nuts_polygons)
        
    # Now merge the formated time series with the shapefile 
    parcels_all = format_merge(parcels_timeseries, parcels, parcel_merge_type).sort_index()
    print('The parcel dataframe length is: ', str(len(parcels)))
    print('The NDVI dataframe length is: ', str(len(parcels_timeseries)))
    print('The merged dataframe length is: ', str(len(parcels_all)))
    # Get an annual count of the satellite acquisitions. This is crucial 
    # because we can't know the crop growth time series with insufficient 
    # acquisitions 
    annual_count = count_annual_observations(parcels_all)
    
    # If a crop residue is required then we format the ndti and soil data
    if incorporate_crop_res == True:
        NDTI_timeseries = format_merge(NDTI_timeseries, parcels, parcel_merge_type)
        soil_properties = format_merge(soil_properties, parcels, parcel_merge_type)
    
    # Read Ukkel rainfall erosivity file
    rain_erosiv_all = pd.read_csv(files_to_read['Ukkel_data_path'])
         
    # Initiate a series of lists to store calculated info
    c_factor_all = []
    slr_ei_all = pd.DataFrame()
    C_factor_uncertainty = []
    harvest_inflexes_all = pd.DataFrame()
    obj_id_list = []
    crop_list = []
    rsme_all = []
    rsme_landsat = []
    all_gpr = {}
    fvc_all = {}
    logger = []
    
    # Initiate a loop through the field parcels in the parcel database---------
    # Each parcel is processed individually within the loop
    for i in np.arange(len(parcels_all)):
        obj_id = parcels_all['OBJECTID'].iloc[i]
        # Take the descriptive crop name
        try:
            crop = parcels_all['crop_name'].iloc[i]
        except:
            crop = 'Not available'
        try:
            # Take the specific beta parameter (exponent) for soil loss ratio
            slr_exponent = parcels_all[beta_column].iloc[i]
        except:
            # Else use default
            slr_exponent = slr_exponent
        
        # Extract geometry from the dataframe
        geom = parcels_all['geometry'].iloc[i]
        # Extract some info from the dataframe
        nuts2_cons = parcels_all['NUTS2_conservation'].iloc[i]
        nuts2_notill = parcels_all['NUTS2_notill'].iloc[i]
        nuts2_reduced = nuts2_cons + nuts2_notill
        
        # Assign the rain_erosiv_all DataFrame to rain_erosiv and set the 'interval' column as its index
        rain_erosiv = rain_erosiv_all
        rain_erosiv = rain_erosiv.set_index('interval', drop = True)
        
        # Generate a day array for plotting purposes
        rain_erosiv = rain_erosiv.iloc[:,-1]
        rain_erosiv = rain_erosiv.to_frame()
        rain_erosiv['Day'] = rain_erosiv.index.astype(float) * 15
        rain_erosiv.columns = ['erosivity', 'Day']

        # Get the sum of the 15-day values (R-factor - annual average)
        R_factor = rain_erosiv['erosivity'].sum().astype('int')
        
        # Now run the interpolation process to get smoothened time series of the NDVI
        output = GPR_interpolate(parcels_all, data_format = 'LPIS', 
                                 column = i, plot = plot,
                                 min_annual_obs = min_ndvi_y, rain_erosiv = rain_erosiv,
                                 crop_models_file_path = crop_models_file_path,
                                 evaluate_gpr_fit = evaluate_gpr_fit, method = 'GPR',
                                 index = 'NDVI')
        # Check if the output is None; if true, set fvc_gpr_error to True
        if output == None:
            fvc_gpr_error = True
            print('skipped 1 - fvc')
            logger.append([obj_id, '1 or 2- - insufficient observations or unresolved GPR'])
            continue
        # If output is not None, extract predicted values and statistics
        else:
            x_days = output['x_days']
            fvc_gpr = output['y_pred_FVC']
            fvc_gpr_low = output['y_pred_FVC_low']
            fvc_gpr_high = output['y_pred_FVC_high']
            ndvi_gpr = output['y_pred_VI']
            obj_id1 = output['obj_id']
            count_per_year_avg = output['count_per_year_avg']
            rsme_eval = output['rsme']
            pcnt_in_sigma_ndvi = None
            rsme_ndvi = None
        
        # Correct FVC during periods with min vegetative cover
        # Replace values in fvc_gpr with fvc_gpr_low where fvc_gpr is below 0.2
        fvc_gpr['gpr_pred'] = fvc_gpr['gpr_pred'].where(fvc_gpr['gpr_pred'] >= 0.2, fvc_gpr_low['gpr_pred'])
        # Set all values below 0.3 to 0 for category 1
        # List of crops to check
        cat1_crops = ['Wheat', 'Barley', 'Triticale', 'Spelt', 'Spinach', 'Rye'] # Winter crops
        cat1s_crops = ['Barley_Summer','Wheat_Summer','Fiber_flax']  # Summer crops
        # Check info of the specific parcel in the loop
        row = parcels_all.iloc[i]
        # Ensure the crop is one of cat 1
        # Winter crops
        if row['crop_name'] in cat1_crops:
            # Ensure H_date (harvest date) is not NaN
            if pd.notna(row["H_date"]):
                # Convert H_date to a datetime object and adjust the year
                h_date = pd.to_datetime(row["H_date"], format="%d-%m").replace(year=cf_year)
                # Create a mask for all values from h_date onward
                mask = fvc_gpr.index >= h_date
                # Replace values with fvc_gpr_low where mask is True
                fvc_gpr.loc[mask, 'gpr_pred'] = fvc_gpr_low.loc[mask, 'gpr_pred']
                # Select the first four indices after h_date
                indices_after_h_date = fvc_gpr.loc[fvc_gpr.index > h_date].index[:4]
                # Set the first four indices explicitly to 0
                fvc_gpr.loc[indices_after_h_date, 'gpr_pred'] = 0
                # Do the same for low  timeseries
                mask = fvc_gpr_low.index >= h_date
                # Select the first four indices after h_date
                indices_after_h_date = fvc_gpr_low.loc[fvc_gpr.index > h_date].index[:4]
                # Set the first four indices explicitly to 0
                fvc_gpr_low.loc[indices_after_h_date, 'gpr_pred'] = 0
        # Summer crops
        if row['crop_name'] in cat1s_crops:
            # Ensure H_date and S_date are not NaN
            if pd.notna(row["H_date"]) and pd.notna(row["S_date"]):
                # Convert H_date and S_date to datetime objects and adjust the year
                h_date = pd.to_datetime(row["H_date"], format="%d-%m").replace(year=cf_year)
                s_date = pd.to_datetime(row["S_date"], format="%d-%m").replace(year=cf_year)
                # Replace all values outside S_date - H_date with fvc_gpr_low (outside growth season)
                mask_before_s = fvc_gpr.index <= s_date  # Values before S_date
                mask_after_h = fvc_gpr.index >= h_date   # Values after H_date
                fvc_gpr.loc[mask_before_s, 'gpr_pred'] = fvc_gpr_low.loc[mask_before_s, 'gpr_pred']
                fvc_gpr.loc[mask_after_h, 'gpr_pred'] = fvc_gpr_low.loc[mask_after_h, 'gpr_pred']
                # Set the first 2 indices after H_date to 0
                indices_after_h_date = fvc_gpr.loc[fvc_gpr.index > h_date].index[:2]
                fvc_gpr.loc[indices_after_h_date, 'gpr_pred'] = 0
                # Check if s_date exists exactly in the index of fvc_gpr
                if s_date in fvc_gpr.index:
                    # If s_date is an exact match, get its position in the index
                    indices_around_s_date = fvc_gpr.index.get_loc(s_date, method='nearest')
                else:
                    # If s_date is not in the index, find the nearest index position
                    indices_around_s_date = fvc_gpr.index.get_indexer([s_date], method='nearest')[0]
                # Determine the first indices before and two indices after s_date
                index_before_s_date_1 = max(0, indices_around_s_date - 1)
                indices_after_s_date = fvc_gpr.index[min(len(fvc_gpr) - 1, indices_around_s_date + 1): min(len(fvc_gpr), indices_around_s_date + 3)]
                # Set the values to 0
                fvc_gpr.loc[fvc_gpr.index[[index_before_s_date_1]], 'gpr_pred'] = 0  # One index before S_date
                fvc_gpr.loc[indices_after_s_date, 'gpr_pred'] = 0  # Two indices after S_date
                # Do the same for fvc_gpr_low timeseries
                indices_after_h_date = fvc_gpr_low.loc[fvc_gpr_low.index > h_date].index[:2]
                fvc_gpr_low.loc[indices_after_h_date, 'gpr_pred'] = 0
                if s_date in fvc_gpr_low.index:
                    indices_around_s_date = fvc_gpr_low.index.get_loc(s_date, method='nearest')
                else:
                    indices_around_s_date = fvc_gpr_low.index.get_indexer([s_date], method='nearest')[0]
                # Determine the last index before and two indices after s_date
                index_before_s_date_1 = max(0, indices_around_s_date - 1)
                indices_after_s_date = fvc_gpr_low.index[min(len(fvc_gpr_low) - 1, indices_around_s_date + 1): min(len(fvc_gpr_low), indices_around_s_date + 3)]
                # Set the values to 0
                fvc_gpr_low.loc[fvc_gpr_low.index[[index_before_s_date_1]], 'gpr_pred'] = 0  # One index before S_date
                fvc_gpr_low.loc[indices_after_s_date, 'gpr_pred'] = 0  # Two indices after S_date
                
        # Run the process for ndti if crop residue is desired
        if incorporate_crop_res == True:
            # Use here NDTI time series
            output2 = GPR_interpolate(NDTI_timeseries, data_format = 'LPIS', 
                                     column = i, plot = plot, 
                                     min_annual_obs = min_ndvi_y, rain_erosiv = rain_erosiv,
                                     crop_models_file_path = crop_models_file_path,
                                     evaluate_gpr_fit = evaluate_gpr_fit, method = 'GPR',
                                     index = 'NDTI') 
            # Again check if output is None
            if output2 == None:
                npvc_gpr_error = True
                print('skipped 2 - npvc')
                logger.append([obj_id, '3 - unresolved ntdi time series'])
            # Again if it is not None, extract predicted values and statistics
            else:
                x_days = output2['x_days']
                npvc_gpr = output2['y_pred_FVC']
                npvc_gpr_low = output2['y_pred_FVC_low']
                npvc_gpr_high = output2['y_pred_FVC_high']
                ndti_gpr = output2['y_pred_VI']
        # If the dataframe exists but all values are zero, skip data series
        if max(fvc_gpr['gpr_pred']) <= 0 or fvc_gpr['gpr_pred'].isnull().all() == True:
            logger.append([obj_id, '5 - irregular interpolation returned'])
            print('skipped 4')
            continue        
        # If the gpr fit was evaluated, append the results to a list
        if evaluate_gpr_fit == True:
            rsme_all.append([obj_id1, rsme_ndvi, pcnt_in_sigma_ndvi, count_per_year_avg])
        
        # If crop residue is desired, run the process to detect inflexes in 
        # the crop growth time series and evaluate the ndti during these time periods
        if incorporate_crop_res == True:
            # Call harvest identification function  
            harvest_inflexes = identify_multiple_harvests(ndvi_gpr, npvc_gpr, obj_id)
            if harvest_inflexes is not None:
                harvests_identified = True
                harvest_inflexes.columns = ['OBJECTID', 'NDVI_inflex', 'NDVI_inflex_average', 'NDVI_minima', 
                                                               'NDTI_inflex', 'NDTI_inflex_average', 'NDTI_minima', 
                                                               'Harvest_identified','Estimated_harvest_date']
                harvest_inflexes =  harvest_inflexes.merge(soil_properties[soil_properties['OBJECTID'] == obj_id], on = 'OBJECTID')
                harvest_inflexes.rename(columns = {'0_b0' : 'clay_pcnt', '1_b0' : 'soil_text_class', '2_b0': 'soil_bulk_density'}, inplace = True)
                harvest_inflexes['NUTS2_reduced'] = nuts2_reduced
        
                x = harvest_inflexes[['NDVI_inflex', 'NDVI_inflex_average', 'NDVI_minima',
                   'NDTI_inflex', 'NDTI_inflex_average', 'NDTI_minima', 'clay_pcnt', 
                   'soil_text_class','soil_bulk_density', 'NUTS2_reduced']]
                
                # Add the crop residue estimation component
                # The Random Forest model is imported outside the loop for efficiency, predictions will be made inside the loop
                # Evaluate the normalized potential vegetation cover (NPVC) at each inflection point 
                # to assess changes in vegetation dynamics
                # Control for any errors during prediction
                try:
                    res_pred = rf_model.predict(x)
                except:
                    # Skip field if input array is incomplete and error is returned
                    print('skipped 5')
                    logger.append([obj_id, '5'])
                    continue
                harvest_inflexes['Crop_res'] = res_pred
                harvest_inflexes_all = harvest_inflexes_all.append(harvest_inflexes)            
            else:
                harvests_identified = False
        
        # Process rainfall erosivity-------------------------------------
        # Creates a dataframe for ONE SINGULAR YEAR (first year) if a specific year is not given
        # Add in code to account for timeseries with multiple years - replicate, change year, append
        if cf_year is not None:
            ts_year_start = int(fvc_gpr.index[0].year)
            ts_year_end = int(fvc_gpr.index[-1].year)
            if cf_year < ts_year_start or cf_year > ts_year_end:
                print('C-factor year not in time series range. Using start year.')
                cf_year = ts_year_start
            day_start = pd.Timestamp(str(cf_year) + '-01-15')
        else: 
            day_start = pd.Timestamp(str(fvc_gpr.index[0].year) + '-01-15')
            cf_year = str(fvc_gpr.index[0].year)
        # Each date in the range will be spaced 15 days apart, generating a series of dates   
        RE_dates = pd.date_range(day_start, periods = len(rain_erosiv), freq = '15d')
        rain_erosiv.index = RE_dates
        # Merge arrays to get a dataframe of vegetation cover and rainfall erosivity.
        # Merge to temporal structure of rainfall erosivity array 
        fvc_RE_merged = pd.merge_asof(rain_erosiv, fvc_gpr, left_index = True, right_index = True)
        fvc_RE_merged_low = pd.merge_asof(rain_erosiv, fvc_gpr_low, left_index = True, right_index = True)
        fvc_RE_merged_high = pd.merge_asof(rain_erosiv, fvc_gpr_high, left_index = True, right_index = True)
               
        # Flatten array and add to dictionary
        fvc_all[obj_id] = fvc_gpr.values.flatten()
        
        # Uncertainty analysis for derived SLR-----------------------------
        # Uncertainty analysis does not include residues 
        if SLR_uncertainty_analysis == True:
            # Simulate the C-factor across the exponent range
            c_range = []
            exponent_array = np.arange(-0.0168, -0.0816, -0.004)
            # Iterate through each exponent in the exponent_array
            for i in exponent_array:
                # Calculate the C-factor estimate using the specified exponent
                c_estimate = C_factor(fvc_RE_merged, i, return_slr_ts = False)
                c_range.append(c_estimate)
            C_factor_uncertainty.append(c_range)
            
        # Incorporate crop residue------------------------------------------
        # Add the crop residue component onto the FVC arrays to get fractional cover
        # Only add if harvest periods were identified 
        if incorporate_crop_res == True and harvests_identified == True:
            fvc_RE_merged, harvest_period_total, harvest_period_w_res = add_harvest_cropres(fvc_RE_merged, harvest_inflexes)
            fvc_RE_merged_low, harvest_period_total, harvest_period_w_res = add_harvest_cropres(fvc_RE_merged_low, harvest_inflexes)
            fvc_RE_merged_high, harvest_period_total, harvest_period_w_res = add_harvest_cropres(fvc_RE_merged_high, harvest_inflexes)
        else:
            # Set both to zero, no harvest identified
            harvest_period_total = harvest_period_w_res = 0
        
        # SLR time series----------------------------------------------------
        # Extract the column names from the existing DataFrame
        date_columns = fvc_gpr['gpr_pred']
        date_columns = date_columns.index
        # Initialize the result array as a DataFrame
        # Convert date_columns to datetime format
        date_datetime = pd.to_datetime(date_columns, errors="coerce")
        # Initialize the result DataFrame with default value -0.035
        beta_array = np.full(date_datetime.shape, slr_exponent)
        # Extract info about the parcel
        row = parcels_all.iloc[i]
        if pd.notna(row["S_date"]) and pd.notna(row["H_date"]):
            # Convert S_date and H_date to datetime objects
            s_date = pd.to_datetime(row["S_date"], format="%d-%m").replace(year=cf_year)
            h_date = pd.to_datetime(row["H_date"], format="%d-%m").replace(year=cf_year)
            # Create masks for all S_date < H_date and S_date >= H_date cases
            # Use only crop related beta value in growth season, outside use the default one
            if s_date < h_date:
                s_date_earlier = s_date < h_date
                mask_Searlier = (date_datetime >= s_date) & (date_datetime <= h_date)
                beta_array[mask_Searlier] = slr_exponent
            if h_date < s_date:
                h_date_earlier = h_date < s_date
                mask_Hearlier = (date_datetime <= h_date)
                # Apply values based on masks
                beta_array[mask_Hearlier] = slr_exponent
        # Calculate slr time series
        slr_gpr = pd.DataFrame(FVC_to_SLR(fvc_gpr['gpr_pred'].values, beta_array),columns = [int(obj_id)],index=fvc_gpr.index) 
        # Correct for senesence
        # Only for the following crops (based on expert knowledge)
        if crop in ["Maize", "Wheat", "Barley", "Triticale", "Fiber_flax", "Spelt", "Peas", "Rye", "Field_beans"]:
            # Handle summer and winter crops separately; otherwise, the correction will be applied incorrectly
            if s_date < h_date:
                min_slr_date = slr_gpr.loc[s_date:h_date].idxmin().values[0]
                slr_gpr.loc[min_slr_date:h_date] = slr_gpr.loc[min_slr_date].values[0]
            if h_date < s_date:
                min_slr_date = slr_gpr.loc[:h_date].idxmin().values[0]
                slr_gpr.loc[min_slr_date:h_date] = slr_gpr.loc[min_slr_date].values[0]
        
        # If first iteration, create template to merge on
        if not 'fvc_std' in locals():
            # If first iteration, create template to merge on
            ds = pd.Timestamp(str(slr_gpr.index[0].year) + '-01-01')
            de = pd.Timestamp(str(slr_gpr.index[-1].year) + '-12-31')
            dates = pd.date_range(ds, de , freq = '15d')
            template = pd.DataFrame(data = (np.arange(len(dates))), index = dates, columns = ['Period'])
            fvc_std = pd.merge_asof(template, slr_gpr, left_index = True, 
                                    right_index = True, direction = 'nearest', tolerance = pd.Timedelta('15d'))
        # Iteratively merge the time series
        elif 'fvc_std' in locals():
            fvc_std = pd.merge_asof(fvc_std, slr_gpr, left_index = True, 
                                    right_index = True, direction = 'nearest', 
                                    tolerance =  pd.Timedelta('15d'))
       
        # Computation and Aggregation of C-Factor Results------------------
        # Take the mean annual fvc value
        fvc_mean = float(fvc_gpr.mean())
        # Collect information on processed parcels
        obj_id_list.append(obj_id)
        crop_list.append(crop)
        # Get the C-factor estimations based on the SLR and rainfall erosivity time series 
        # This method follows the origninal RUSLE procedure 
        c_estimate, slr_ei_merged = C_factor(fvc_RE_merged, slr_exponent, 
                                             sc_uncertainty = False, return_slr_ts = True)
        c_estimate_low, slr_ei_merged = C_factor(fvc_RE_merged_low, slr_exponent, 
                                                 sc_uncertainty = False, return_slr_ts = True)
        c_estimate_high, slr_ei_merged = C_factor(fvc_RE_merged_high, slr_exponent, 
                                                  sc_uncertainty = False, return_slr_ts = True)

        # Calculate some summary indices 
        try:
            month_max_risk, EI30_risk_period, RE_pcnt_low_veg = Calc_risk_period(fvc_RE_merged)
        except Exception as e:
            print(f"Skipping instance due to error in Calc_risk_period: {e}")
            continue
        # Extract rotation id if you are calculating C-factors for rotations
        rotation_value = parcels_all.iloc[i].get("rotations", None) 
        # Append the C-factor results to a master dataframe
        c_factor_all.append([obj_id, crop, c_estimate, c_estimate_low, c_estimate_high,
                             month_max_risk, EI30_risk_period, harvest_period_total, harvest_period_w_res, fvc_mean, RE_pcnt_low_veg, 
                             geom,rotation_value])
        
        # Append the log information 
        logger.append([obj_id, 'Completed'])
        
        # Append sl_ei timeseries 
        slr_ei_merged.insert(0, 'crop', crop)
        slr_ei_merged.insert(0, 'object_id', obj_id, )
        slr_ei_merged['Datetime'] = slr_ei_merged.index
        slr_ei_merged.reset_index()
        slr_ei_all = pd.concat([slr_ei_all, slr_ei_merged], axis = 0, ignore_index = True)

        # Add gpr output to dictionary 
        all_gpr[obj_id] = output
    
        # If any C-factor values are irregular, plot it
        if c_estimate > 1 or c_estimate < 0.001: 
            print('plotting irregular C-factor result')
            output = GPR_interpolate(parcels_all, data_format = 'LPIS', 
                                          column = i, plot = True, rain_erosiv = rain_erosiv,
                                          crop_models_file_path = crop_models_file_path)
        
    # Create a dataframe to store information    
    df_c_factor_all = pd.DataFrame(c_factor_all, columns = ['object_id', 'crop', 'C_factor', 'C_factor_upper', 
                                                            'C_factor_lower', 'Month_highest_risk', 'EI30_risk_period', 'harvest_period_total', 
                                                            'harvest_period_total_with_residues', 'mean_annual_fvc', 'RE_%_low_veg', 'geometry','rotations'])

    # Call function to get dataframes
    df_fvc_all = pd.DataFrame(data = fvc_all, index = fvc_gpr.index).transpose()
    df_fvc_all.insert(0, column = 'object_id', value = obj_id_list)

    
    # Plot a final histogram of C-factor results
    #df_c_factor_all.hist('C_factor', by = 'crop', bins = 20, figsize = (20,15))
    # Create a DataFrame from the RSME values
    if evaluate_gpr_fit == True:
        df_rsme_all = pd.DataFrame(rsme_all, columns = ['object_id', 'RSME', '%_in_gpr_sigma', 'n_obs'])
    
    # Initialize an empty dictionary to store all results
    output_all = {}
    
    # Create a DataFrame for the C-factor uncertainty values
    if SLR_uncertainty_analysis == True:
        c_factor_uncertainty_df = pd.DataFrame(C_factor_uncertainty, columns = ["%.4f" % x for x in exponent_array])
        c_factor_uncertainty_df.index = obj_id_list
        c_factor_uncertainty_df['Crop'] = crop_list
        output_all['SLR uncertainty analysis'] = c_factor_uncertainty_df
    
    # Create a dictionary to store and return all of the output          
    output_all['IACS with NDVI ts'] = parcels_all
    output_all['C-factor results'] = df_c_factor_all
    output_all['SLR RE ts (one year)'] = slr_ei_all
    output_all['SLR ts full'] = fvc_std.drop('Period', axis = 1).transpose().reset_index().rename(columns = {'index': 'object_id'})
    output_all['N minimum NDVI acquisitions'] = min_ndvi_y
    output_all['Error logger'] = logger
    output_all['C-factor year'] = cf_year
    output_all['GPR outputs'] = all_gpr
    output_all['GPR FVC'] = df_fvc_all
    
    # Add C-factor uncertainty DataFrame if uncertainty analysis is enabled
    if SLR_uncertainty_analysis == True:
        output_all['SLR uncertainty analysis'] = c_factor_uncertainty_df
            
    return output_all


