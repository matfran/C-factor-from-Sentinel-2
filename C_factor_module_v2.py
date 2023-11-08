# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:48:41 2021

A module to calculate a C-factor in IACS field parcels using a time series data
of ndvi acquisitons.
@author: FRANCIS MATTHEWS fmatthews1381@gmail.com
"""

import sys
import pandas as pd 
import geopandas as gpd
import numpy as np
from GPR_analysis_v2 import GPR_interpolate
from pickle import load
from REDES_15day_average import REDES_EnS_15day_average
from Harvest_delineation import identify_multiple_harvests
from Add_LUCAS_geometry import sample_by_climate_region, add_NUTS_info
from C_factor_functions import count_annual_observations, format_merge, reformat_LS_cols
from C_factor_functions import C_factor, Calc_risk_period, add_harvest_cropres, FVC_to_SLR

#create a dataframe with arrays 
def get_df(sl_ei_all, obj_id_list, crop_list, EnS_list):
    """
    Parameters
    ----------
    sl_ei_all : LIST
        DESCRIPTION.
    obj_id_list : LIST
        DESCRIPTION.
    crop_list : LIST
        DESCRIPTION.
    EnS_list : LIST
        DESCRIPTION.

    Returns
    -------
    df_sl_ei_all : DATAFRAME
        A compilled dataframe of all lists

    """

    df_sl_ei_all = pd.DataFrame(sl_ei_all)
    df_sl_ei_all['object_id'], df_sl_ei_all['crop'], df_sl_ei_all['EnS'] = [obj_id_list, crop_list, EnS_list]
    #rearrange the columns to put reference cols first
    new_col_ord = df_sl_ei_all.columns[-3:].to_list() + df_sl_ei_all.columns[:-3].to_list()
    df_sl_ei_all = df_sl_ei_all[new_col_ord]
    return df_sl_ei_all

def get_C_factor(iacs, files_to_read, name, sensor_name, cf_year = None,
                 incorporate_crop_res = False, iacs_merge_type = 'left', 
                 run_all_parcels = True, slr_exponent = - 0.04, min_ndvi_y = None, 
                 SLR_uncertainty_analysis = False, plot = False):
    '''
    A function to calculate the C-factor for a series of IACS parcels. The 
    function runs a series of processes in order to return a final C-factor value
    and SLR time series, broadly speaking these are:
        1) Add relevant information to each IACS parcel which is neccessary 
        to make an evaluation of the C-factor. The crucial ones are the NDVI/NDTI
        time series from GEE and a rainfall erosivity time series. 
        2) For all parcels in the geodataframe, interpolate the discrete 
        satellite observations into a 15-day time series using GPR to obtain 
        a smooth time series record without outliers. 
        3) Calculate the C-factor for each parcel and return all the relevant 
        information
        

    Parameters
    ----------
    iacs : GEODATAFRAME
        Geodataframe with field parcels and attributes
    ndvi_ts_path : STRING
        Path to ndvi timeseries for each parcel
    files to read : DICTIONARY
        A dictionary containing all file path names to read in
    sensor_name : STRING
        The sensor name. 'Sentinel-2' or 'Landsat'
    output_folder_name : STRING
        The name of the folder to put the output files in
    cfyear: INTEGER
        The year in the time series for which the C-factor is calculated
    incorporate_crop_res : TYPE, optional, BOOLEAN
        DESCRIPTION. The default is False. Define whether to include crop 
        residue in the C-factor calculation. If True, a path needs to be 
        provided to NDVI time series data and soil property data. 
    run_all_parcels : TYPE, optional, BOOLEAN
        DESCRIPTION. The default is True. If False, this will only run a 
        few parcels for testing purposes

    Returns
    
    output : DICTIONARY
    A dictionary with the multiple compiled results. The standard outputs are:
        output['IACS with NDVI ts']
        output['C-factor results']
        output['SLR RE ts (one year)'] 
        output['SLR ts full']
        output['N minimum NDVI acquisitions']
        output['Error logger']
        output['C-factor year']
    
    -------
    TYPE
        DESCRIPTION.

    '''
    #Initilise the relevant variables-------------------------------------
    #The key operation here is to merge sentinel-2 or Landsat observations
    #With the IACS dataframe

    
    #unpack the paths to the relevant files 
    crop_models_file_path = files_to_read['crop_models_file_path']
    REDES_gauges_path = files_to_read['REDES_gauges_path']
    EnS_file_path = files_to_read['EnS_file_path']
    nuts_path = files_to_read['nuts_path']

    #these are defined within the function currently
    NDVI_to_fvc_method = 'tenriero_et_al' 
    evaluate_gpr_fit = False
    
    #set minimum observations depending on the sensor
    if sensor_name == 'Landsat':
        if min_ndvi_y == None:
            min_ndvi_y = 6
    elif sensor_name == 'Sentinel-2':
        if min_ndvi_y == None:
            min_ndvi_y = 12
    else:
        print('Define sensor: Landsat or Sentinel-2')
        sys.exit()
        
    
    #read the filename path with the ndvi time series 
    iacs_timeseries = pd.read_csv(files_to_read['ndvi_ts_path'])
    #reformat the columns of the landsat timeseries 
    if sensor_name == 'Landsat':
        iacs_timeseries = reformat_LS_cols(iacs_timeseries)
    #if a crop residue assessment is required then we need to read a classifier
    #model and provide an NDTI time series and soil properties for each parcel
    if incorporate_crop_res == True:
        rf_model_path = files_to_read['rf_model_path']
        rf_model = load(open(rf_model_path, 'rb'))
        NDTI_timeseries = pd.read_csv(files_to_read['ndti_ts_path'])
        soil_properties = pd.read_csv(files_to_read['soil_path'])
        
    #This is set to use the EnS (climate region) r-factor time series 
    #the closest gauge is not recommended because it could be distant
    RE_EnS = True
    RE_closest = False
    
    #define the path to the nuts2 info and add it to the IACS data
    #this way we can know some regional statistics
    nuts_polygons = gpd.read_file(nuts_path)
    iacs = add_NUTS_info(nuts_polygons, iacs)
    #delete after the merge
    del(nuts_polygons)
    
    
    #now merge the formated time series with the shapefile 
    IACS_all = format_merge(iacs_timeseries, iacs, iacs_merge_type).sort_index()
    print('The IACS dataframe length is: ', str(len(iacs)))
    print('The NDVI dataframe length is: ', str(len(iacs_timeseries)))
    print('The merged dataframe length is: ', str(len(IACS_all)))
    #get an annual count of the satellite acquisitions. This is crucial 
    #because we can't know the crop growth time series with insufficient 
    #acquisitions 
    annual_count = count_annual_observations(IACS_all)
    

    #if a crop residue is required then we format the ndti and soil data
    if incorporate_crop_res == True:
        NDTI_timeseries = format_merge(NDTI_timeseries, iacs)
        soil_properties = format_merge(soil_properties, iacs)
    
    rain_erosiv_all = REDES_EnS_15day_average(files_to_read['all_REDES_path'], files_to_read['REDES_reference_path'])
    EnS_polygons = gpd.read_file(EnS_file_path)
    LPIS_EnS_matched = sample_by_climate_region(EnS_polygons, iacs)
        
    #initiate a series of lists to store calculated info
    c_factor_all = []
    sl_ei_all = []
    sl_ei_pcnt_all = []
    C_factor_uncertainty = []
    harvest_inflexes_all = pd.DataFrame()
    obj_id_list = []
    crop_list = []
    EnS_list = []
    rsme_all = []
    rsme_landsat = []
    logger = []
    
    #here we can specify a limit for n parcels to analyse
    if run_all_parcels == True:
        n = len(IACS_all)
    elif run_all_parcels == False:
        n = 10
        plot = True
        
    #initiate a loop through the field parcels in the IACS database---------
    #each parcel is processed individually within the loop
    for i in np.arange(n):
        obj_id = IACS_all['OBJECTID'].iloc[i]
        #take the descriptive crop name
        try:
            crop = IACS_all['crop_name'].iloc[i]
        except:
            crop = 'Not available'
        #extract some info from the dataframe
        geom = IACS_all['geometry'].iloc[i]
        nuts2_cons = IACS_all['NUTS2_conservation'].iloc[i]
        nuts2_notill = IACS_all['NUTS2_notill'].iloc[i]
        nuts2_reduced = nuts2_cons + nuts2_notill
        
        #extract the EnS corresponding to the parcel
        try:
            EnS_of_parcel = LPIS_EnS_matched.loc[LPIS_EnS_matched['OBJECTID'] == obj_id]['EnS_name'].iloc[0]
            EnS_prev = EnS_of_parcel
        except:
            #take the EnS of the previous (close) parcel
            #EnS_of_parcel = EnS_prev
            #otherwise if there is nuts3 info, take the EnS based on a closeby parcel
            try:
                nuts3_code = IACS_all[IACS_all['OBJECTID'] == obj_id]['nuts3_code'].iloc[0]
                EnS_of_parcel = LPIS_EnS_matched.loc[LPIS_EnS_matched['nuts3_code'] == nuts3_code].iloc[0]['EnS_name']
            except:
                EnS_of_parcel = EnS_prev
                logger.append([obj_id, 'Used previous EnS of parcel'])
        
        #format the rainfall erosovity timeseries 
        if EnS_of_parcel == 'MDM9':
            EnS_of_parcel = 'MDM8'
        if EnS_of_parcel == 'MDS8':
            EnS_of_parcel = 'MDS7'            

        rain_erosiv = pd.DataFrame(rain_erosiv_all.loc[EnS_of_parcel].T)
        
        #generate a day array for plotting purposes
        rain_erosiv['Day'] = rain_erosiv.index.astype(float) * 15
        rain_erosiv.columns = ['EI30', 'Day']
        #get the sum of the 15-day values (R-factor - annual average)
        R_factor = rain_erosiv['EI30'].sum().astype('int')
        
        #now run the interpolation process to get smoothened time series of the NDVI
        output = GPR_interpolate(IACS_all, data_format = 'LPIS', 
                                 column = i, plot = plot, sensor = sensor_name, 
                                 min_annual_obs = min_ndvi_y, rain_erosiv = rain_erosiv,
                                 crop_models_file_path = crop_models_file_path,
                                 evaluate_gpr_fit = evaluate_gpr_fit)   
        if output == None:
            fvc_gpr_error = True
            print('skipped 1 - fvc')
            logger.append([obj_id, '1 or 2- - insufficient observations or unresolved GPR'])
            continue
        else:
            x_days = output['x_days']
            fvc_gpr = output['y_pred_FVC']
            fvc_gpr_low = output['y_pred_FVC_low']
            fvc_gpr_high = output['y_pred_FVC_high']
            ndvi_gpr = output['y_pred_VI']
            obj_id1 = output['obj_id']
            count_per_year_avg = output['count_per_year_avg']
            rsme_eval = output['rsme']
            if sensor_name == 'Landsat':
                pcnt_in_sigma_ndvi = output['ndvi_replacement_evaluation']['pcnt in sigma']
                rsme_ndvi = output['ndvi_replacement_evaluation']['rsme']
            else:
                pcnt_in_sigma_ndvi = None
                rsme_ndvi = None
               
        #run the process for ndti if crop residue is desired
        if incorporate_crop_res == True:
            #npvc is the ndti
            output2 = GPR_interpolate(NDTI_timeseries, data_format = 'LPIS', 
                                         column = i, plot = False, index = 'NDTI')
            if output2 == None:
                npvc_gpr_error = True
                print('skipped 2 - npvc')
                logger.append([obj_id, '3 - unresolved ntdi time series'])

            else:
                x_days = output2['x_days']
                npvc_gpr = output2['y_pred_FVC']
                npvc_gpr_low = output2['y_pred_FVC_low']
                npvc_gpr_high = output2['y_pred_FVC_high']
                ndti_gpr = output2['y_pred_VI']
    

        #if the dataframe exists but all values are zero, skip data series
        if max(fvc_gpr['gpr_pred']) <= 0 or fvc_gpr['gpr_pred'].isnull().all() == True:
            logger.append([obj_id, '5 - irregular interpolation returned'])
            print('skipped 4')
            continue
        
        
        #if the gpr fit was evaluated, append the results to a list
        if evaluate_gpr_fit == True:
            rsme_all.append([obj_id1, rsme_ndvi, pcnt_in_sigma_ndvi, count_per_year_avg])
        if sensor_name == 'Landsat':
            rsme_landsat.append([obj_id1, rsme_ndvi, pcnt_in_sigma_ndvi])
        
        #if crop residue is desired, run the process to detect inflexes in 
        #the crop growth time series and evaluate the ndti during these time periods
        if incorporate_crop_res == True:
            #call harvest identification function 
            harvest_inflexes = identify_multiple_harvests(ndvi_gpr, npvc_gpr, obj_id)
            if harvest_inflexes is not None:
                harvests_identified = True
                harvest_inflexes.columns = ['OBJECTID', 'NDVI_inflex', 'NDVI_inflex_average', 'NDVI_minima', 
                                                               'NDTI_inflex', 'NDTI_inflex_average', 'NDTI_minima', 
                                                               'SMAP', 'Harvest_identified','Estimated_harvest_date']
                
                harvest_inflexes =  harvest_inflexes.merge(soil_properties[soil_properties['OBJECTID'] == obj_id], on = 'OBJECTID')
                harvest_inflexes.rename(columns = {'0' : 'clay_pcnt', '1' : 'soil_text_class', '2': 'soil_bulk_density'}, inplace = True)
                harvest_inflexes['NUTS2_reduced'] = nuts2_reduced
    
    
                x = harvest_inflexes[['NDVI_inflex', 'NDVI_inflex_average', 'NDVI_minima',
                   'NDTI_inflex', 'NDTI_inflex_average', 'NDTI_minima', 'clay_pcnt', 
                   'soil_text_class','soil_bulk_density','NUTS2_reduced']]
                
                #add in crop residue estimation component 
                #random forest model import outside loop, predict inside 
                #evaluate npvc at each inflexion point 
                #control any errors
                try:
                    res_pred = rf_model.predict(x)
                except:
                    #skip field if input array is incomplete and error is returned
                    print('skipped 5')
                    logger.append([obj_id, '5'])
                    continue
                
                harvest_inflexes['Crop_res'] = res_pred
            
                harvest_inflexes_all = harvest_inflexes_all.append(harvest_inflexes)            
            else:
                harvests_identified = False
        
                   

        #Process rainfall erosivity-------------------------------------
        #creates a dataframe for ONE SINGULAR YEAR (first year) if a specific year is not given
        #add in code to account for timeseries with multiple years - replicate, change year, append
            
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
            
        RE_dates = pd.date_range(day_start, periods = len(rain_erosiv), freq = '15d')
        rain_erosiv.index = RE_dates
        #merge arrays to get a dataframe of vegetation cover and rainfall erosivity.
        #merge to temporal structure of rainfall erosivity array 
        fvc_RE_merged = pd.merge_asof(rain_erosiv, fvc_gpr, left_index = True, right_index = True)
        fvc_RE_merged_low = pd.merge_asof(rain_erosiv, fvc_gpr_low, left_index = True, right_index = True)
        fvc_RE_merged_high = pd.merge_asof(rain_erosiv, fvc_gpr_high, left_index = True, right_index = True)
        

        #Uncertainty analysis for derived SLR-----------------------------
        #uncertainty analysis does not include residues 
        if SLR_uncertainty_analysis == True:
            #simulate the C-factor across the exponent range
            c_range = []
            exponent_array = np.arange(-0.0168, -0.0816, -0.004)
            
            for i in exponent_array:
                c_estimate = C_factor(fvc_RE_merged, i)
                c_range.append(c_estimate)
            
            C_factor_uncertainty.append(c_range)
        

        #Incorporate crop residue------------------------------------------
        #add the crop residue component onto the FVC arrays to get fractional cover
        #only add if harvest periods were identified 
        if incorporate_crop_res == True and harvests_identified == True:
            fvc_RE_merged, harvest_period_total, harvest_period_w_res = add_harvest_cropres(fvc_RE_merged, harvest_inflexes)
            fvc_RE_merged_low, harvest_period_total, harvest_period_w_res = add_harvest_cropres(fvc_RE_merged_low, harvest_inflexes)
            fvc_RE_merged_high, harvest_period_total, harvest_period_w_res = add_harvest_cropres(fvc_RE_merged_high, harvest_inflexes)
        else:
            #set both to zero, no harvest identified
            harvest_period_total = harvest_period_w_res = 0
        
        
        #convert fvc directly to slr to get the whole time series (outside of C-factor method)
        slr_gpr = pd.DataFrame(FVC_to_SLR(fvc_gpr['gpr_pred'].values, slr_exponent), columns = [int(obj_id)], 
                               index = fvc_gpr.index)
        
        #Process FVC into standardised SLR dataframe-----------------------
        if not 'fvc_std' in locals():
            #if first iteration, create template to merge on
            ds = pd.Timestamp(str(slr_gpr.index[0].year) + '-01-01')
            de = pd.Timestamp(str(slr_gpr.index[-1].year) + '-12-31')
            dates = pd.date_range(ds, de , freq = '15d')
            template = pd.DataFrame(data = (np.arange(len(dates))), index = dates, columns = ['Period'])
            fvc_std = pd.merge_asof(template, slr_gpr, left_index = True, 
                                    right_index = True, direction = 'nearest', tolerance = pd.Timedelta('15d'))
        elif 'fvc_std' in locals():
            #iteratively merge the time series
            fvc_std = pd.merge_asof(fvc_std, slr_gpr, left_index = True, 
                                    right_index = True, direction = 'nearest', 
                                    tolerance =  pd.Timedelta('15d'))

        
        
        #take the mean annual fvc value
        fvc_mean = float(fvc_gpr.mean())
        #collect information on processed parcels
        obj_id_list.append(obj_id)
        crop_list.append(crop)
        EnS_list.append(EnS_of_parcel)
        
        
        #get the C-factor estimations based on the SLR and rainfall erosivity time series 
        #this method follows the origninal RUSLE procedure 
        c_estimate, sl_ei_ts, sl_ei_pcnt_ts = C_factor(fvc_RE_merged, slr_exponent, return_slr_ts = True)
        c_estimate_low, sl_ei_ts_low, sl_ei_pcnt_ts_low = C_factor(fvc_RE_merged_low, slr_exponent, return_slr_ts = True)
        c_estimate_high, sl_ei_ts_high, sl_ei_pcnt_ts_high = C_factor(fvc_RE_merged_high, slr_exponent, return_slr_ts = True)
        

        #calculate some summary indices 
        month_max_risk, EI30_risk_period, RE_pcnt_low_veg = Calc_risk_period(fvc_RE_merged)
        
        #append the C-factor results to a master dataframe
        c_factor_all.append([obj_id, EnS_of_parcel, count_per_year_avg, crop, c_estimate, c_estimate_low, c_estimate_high,
                             month_max_risk, EI30_risk_period, harvest_period_total, harvest_period_w_res, fvc_mean, RE_pcnt_low_veg, 
                             geom])
        
        #append the log information 
        logger.append([obj_id, 'Completed'])
        
        #append sl_ei timeseries 
        sl_ei_all.append(sl_ei_ts)
        sl_ei_pcnt_all.append(sl_ei_pcnt_ts)
    
        #if any C-factor values are irregular, plot it
        if c_estimate > 1 or c_estimate < 0.001: 
            print('plotting irregular C-factor result')
            x_days, fvc_gpr, obj_id, count_per_year_avg = GPR_interpolate(IACS_all, data_format = 'LPIS', 
                                          column = i, plot = True, rain_erosiv = rain_erosiv,
                                          crop_models_file_path = crop_models_file_path)
        
    #create a dataframe to store information    
    df_c_factor_all = pd.DataFrame(c_factor_all, columns = ['object_id', 'EnS', 'avg_annual_obs', 'crop', 'C_factor', 'C_factor_upper', 
                                                            'C_factor_lower', 'Month_highest_risk', 'EI30_risk_period', 'harvest_period_total', 
                                                            'harvest_period_total_with_residues', 'mean_annual_fvc', 'RE_%_low_veg', 'geometry'])
    
    #call function to get dataframes
    df_sl_ei_all = get_df(sl_ei_all, obj_id_list, crop_list, EnS_list)
    df_sl_ei_pcnt_all = get_df(sl_ei_pcnt_all, obj_id_list, crop_list, EnS_list)


    
    #plot a final histogram of C-factor results
    #df_c_factor_all.hist('C_factor', by = 'crop', bins = 20, figsize = (20,15))
    if evaluate_gpr_fit == True:
        df_rsme_all = pd.DataFrame(rsme_all, columns = ['object_id', 'RSME', '%_in_gpr_sigma', 'n_obs'])
    if sensor_name == 'Landsat':
        df_rsme_landsat = pd.DataFrame(rsme_landsat, columns = ['object_id', 'RSME', '%_in_gpr_sigma'])
    
    mean_per_crop = df_c_factor_all.groupby('crop').mean()

    if SLR_uncertainty_analysis == True:
        c_factor_uncertainty_df = pd.DataFrame(C_factor_uncertainty, columns = ["%.4f" % x for x in exponent_array])
        c_factor_uncertainty_df.index = obj_id_list
        c_factor_uncertainty_df['Crop'] = crop_list
    
    
    #create a dictionary to store and return all of the output        
    output = {}
    output['IACS with NDVI ts'] = IACS_all
    output['C-factor results'] = df_c_factor_all
    output['SLR RE ts (one year)'] = df_sl_ei_all
    output['SLR ts full'] = fvc_std.drop('Period', axis = 1).transpose().reset_index().rename(columns = {'index': 'object_id'})
    output['N minimum NDVI acquisitions'] = min_ndvi_y
    output['Error logger'] = logger
    output['C-factor year'] = cf_year
    if sensor_name == 'Landsat':
        output['Landsat replacement eval'] = df_rsme_landsat
    if SLR_uncertainty_analysis == True:
        output['SLR uncertainty analysis'] = c_factor_uncertainty_df
            
    return output


