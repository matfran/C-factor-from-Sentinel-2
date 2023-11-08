# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:13:11 2021

@author: fmatt
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
#RBF (Radial-basis function) is also termed the squared exponential kernel
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error
from scipy import interpolate 
import warnings

def format_arrays(Crop, method):    
#embed function to process arrays for GPR
    if method == 'GPR':
        Y = Crop.iloc[:,1][Crop.iloc[:,1].notnull()].values.reshape(-1,1)
        X = Crop['Days_from_start'][Crop.iloc[:,1].notnull()].values.reshape(-1,1)
    else:
        Y = Crop.iloc[:,1][Crop.iloc[:,1].notnull()].values
        X = Crop['Days_from_start'][Crop.iloc[:,1].notnull()].values
    
    return X, Y

def filter_by_n(timeseries, n_observations, modify_ts = False, replace = False):
    '''
    Filter the timeseries by the number of remotely sensed scene observations 
    per year. Produces a filtered timeseries with only the years having sufficient
    observations. This function also replaces the years without a critical number 
    of observations with the time series of the year with an optimum number. This 
    means the time series is no longer indicative of the actual conditions but 
    representative of the general land surface characteristics. 

    Parameters
    ----------
    timseries : PANDAS SERIES
        Dataframe with timeseries per field 
    n_observations : INT
        The critical number of observations required. 
    replace : optionally replace years with insufficient observations 
    (n < n_observations) with the time series of the year with the most observations.

    Returns
    -------
    timeseries_new : DATAFRAME
        A new dataframe with the years exceeding the threshold number of
        observations

    '''
    
    if replace == True:
        modify_ts = True
        
    #get all unique years
    year_start = timeseries.index.year.unique()[0]
    year_end = timeseries.index.year.unique()[-1] + 1
    years = np.arange(year_start, year_end, 1)
    #initiate lists
    years_to_include = []
    years_to_exclude = []
    
    #start a count
    i = 1
    #initialise n_prev so n is always larger
    n_prev = 0
    count_per_year = []
    
    for year in years:
        
        #set year of maximum observations at year 1
        if i == 1:
            year_max_obs = year
        
        #count n observations in year
        n = int(timeseries[timeseries.index.year == year].count())
        #print('n observations: ', n)
        #append the count of observations if it exceeds specified threshold
        count_per_year.append(n)
        
        #update year of maximum observations if current year exceeds previous
        if i > 1:

            if n > n_prev:
                #find year with max observations
                year_max_obs = year
                #update n_prev for the next loop
                n_prev = n
        
        if n >= n_observations:
            years_to_include.append(year)

        else:
            years_to_exclude.append(year)
        
        #increase iteration counter
        i = i + 1
        
    if replace == False and modify_ts == True:
        #only take years with sufficient observations 
        timeseries_new = timeseries[timeseries.index.year.isin(years_to_include)]
    elif replace == True and modify_ts == True:
        #only take years with sufficient observations
        timeseries_new = timeseries[timeseries.index.year.isin(years_to_include)]
        for year in years_to_exclude:
            fill_year = timeseries[timeseries.index.year == year_max_obs]
            #calculate the year difference 
            diff = year - year_max_obs 
            fill_year.index = fill_year.index + pd.offsets.DateOffset(years=diff)
            #timeseries_new = timeseries_new.append(fill_year)
            #pd append is depreciating, use concat instead
            timeseries_new = pd.concat([timeseries_new, fill_year])
        #evaluate the difference between the original NDVI points and the
        #new replacement values 
        #print(timeseries)
    else:
        timeseries_new = timeseries
            
    timeseries_new = pd.DataFrame(timeseries_new.sort_index())
    #return the average number of observations per year
    count_per_year_avg = np.mean(count_per_year)
    return timeseries_new, count_per_year_avg

def eval_ndvi_replacement(ndvi_ts_orig, gpr_pred, gpr_pred_low, gpr_pred_high):
    ndvi_ts_orig = ndvi_ts_orig.sort_index()
    
    gpr_pred['gpr_low'] = gpr_pred_low.iloc[:,0].values
    gpr_pred['gpr_high'] = gpr_pred_high.iloc[:,0].values
    gpr_pred = gpr_pred.sort_index()
    
    merged = pd.merge_asof(ndvi_ts_orig, gpr_pred, right_index = True, 
                           left_index = True, direction = 'nearest', 
                           tolerance = pd.Timedelta('7d'))
    
    merged.columns = ['Original NDVI', 'GPR post-replacement', 'GPR low', 'GPR high']
    #get all corresponding non-null values
    Y_orig = merged.iloc[:,0][merged.iloc[:,0].notnull() & merged.iloc[:,1].notnull()].values
    Y_gpr = merged.iloc[:,1][merged.iloc[:,0].notnull() & merged.iloc[:,1].notnull()].values
    Y_gpr_low = merged.iloc[:,2][merged.iloc[:,0].notnull() & merged.iloc[:,1].notnull()].values
    Y_gpr_high = merged.iloc[:,3][merged.iloc[:,0].notnull() & merged.iloc[:,1].notnull()].values
    
    rsme = np.sqrt(mean_squared_error(Y_orig, Y_gpr))
    
    count = 0
    for i in np.arange(len(Y_orig)):
        #count if inside the uncertainty boundaries 
        if Y_orig[i] > Y_gpr_low[i] and Y_orig[i] < Y_gpr_high[i]:
            count = count + 1
    pcnt_in_sigma = (count/len(Y_orig)) * 100        
    
    output = {}
    output['rsme'] = rsme
    output['pcnt in sigma'] = pcnt_in_sigma
    
    return output

def evaluate_gpr_int(Crop, x_days, method, kernel):
    #drop na values so they aren't contained in validation dataset 
    Crop_val = Crop.dropna()
    Crop_80 = Crop_val.sample(frac = 0.8, random_state = 1)
    Crop_20 = Crop_val.drop(Crop_80.index)
    
    X_t, Y_t = format_arrays(Crop_80, method)
    
    #retrain model with training dataset
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, normalize_y=True).fit(X_t,Y_t)
    y_pred_VI_val, sigma_val = gp.predict(x_days[:, np.newaxis], return_cov=True)

    X_v, Y_v = format_arrays(Crop_20, method)
    
    #calculate RSME of the model fit
    y_pred_val, sigma_val = gp.predict(X_v, return_cov=True)
    #calculate % of test data in the GPR uncertainty boundaries 
    
    y_pred_val_l = (y_pred_val.flatten() - np.sqrt(np.diag(sigma_val)))
    y_pred_val_u = (y_pred_val.flatten() + np.sqrt(np.diag(sigma_val)))
    y_pred_val = y_pred_val.flatten()
    

    count = 0
    for i in np.arange(len(y_pred_val)):
        #count if inside the uncertainty boundaries 
        if y_pred_val[i] > y_pred_val_l[i] and y_pred_val[i] < y_pred_val_u[i]:
            count = count + 1
    pcnt_in_sigma = (count/len(X_v)) * 100        
    
    rsme = np.sqrt(mean_squared_error(Y_v, y_pred_val))
    

    return rsme, pcnt_in_sigma




def GPR_interpolate(timeseries, data_format, column, sensor, plot, method = 'GPR', 
                    min_annual_obs = 10, start_date = None, rain_erosiv = None, 
                    index = 'NDVI', crop_models_file_path = None, 
                    LUCAS_date = None, evaluate_gpr_fit = False):
    '''
    Takes a number of remotely sensed observations and creates a smooth vector 
    representing the field's phenology timeseries. If the index is NDVI, a 
    relative abundance algorithm is applied to obtain the fractional vegetation 
    cover using the representative minimum and maximum NDVI values (taken from
    the GPR curve). If the index is NDTI, a GPR interpolation of the spectral
    indices is instead returned.

    Parameters
    ----------
    timeseries : DATAFRAME
        A dataframe with timseries of optical observations from a satellite 
        sensor. If the data is LPIS, the function will format and transpose 
        the data from the GEE output format. 
    data_format : STRING
        Specify the data format. 'LPIS' or 'LUCAS'
    column : INTEGER
        The column number (representing a singular field) of the dataframe to analyse.
        If iterating function, begin with first column.
    plot: Boolean
         Optionally plot timeseries 
    method: String
        Default 'GPR'. Specify 'Spline' for spline interpolation.
    min_annual_obs: Integer
        Define the minimum number of satelite observations required in a year for 
        it to be included. Default is 15 per year.
    start_date : PANDAS TIMESTAMP
        Optionally add a start date outside of the observation range to define
        the start of the timestamp series. This begins the interpolation from 
        a specific date instead of the first observation. Default is none, the series start date is used.
    rain_erosiv : DATAFRAME
        Dataframe with 15-day average rainfall erosivity (default None).
    index: STRING
        A string stating the spectral index to analyse. 'NDVI' (default)
        or 'NDTI'
    crop_models_file_path:
        file pathway for the 'tenriero_et_al' crop models 
        
    Returns
    -------
    output: DICTIONARY
        The multiple compiled outputs from the function. The dictionary keys are:
            output['x_days']
            output['y_pred_FVC']
            output['y_pred_FVC_low']
            output['y_pred_FVC_high']
            output['y_pred_VI']
            output['obj_id']
            output['count_per_year_avg']
            output['rsme']

    '''
    output = {}
    
    Crop = timeseries

    try:
        #extract the name corresponding to the crop model 
        cropname = Crop.iloc[column]['crop_model']
        #extract the descriptive crop name
        cropname2 = Crop.iloc[column]['crop_name']
    except:
        #if no cropname can be extracted, treat crop as not available
        cropname = None

    #find the first column containing numerical dates
    indexer = []
    for x in Crop.columns:
        num = x.isnumeric()
        indexer.append(num)
    
    Crop = Crop.iloc[:,indexer]
    Crop = Crop.transpose()
    #convert to a datetime
    Crop.index = pd.to_datetime(Crop.index)

    field_obs = Crop.iloc[:, column]
    obj_id = field_obs.name
    
    if sensor == 'Sentinel-2':
        Crop, count_per_year_avg = filter_by_n(field_obs, min_annual_obs, replace = False)
    if sensor == 'Landsat':
        output['Crop NDVI original'] = field_obs
        Crop, count_per_year_avg = filter_by_n(field_obs, min_annual_obs, replace = True)
        output['Crop NDVI modified'] = Crop
    
    #filter by n returns empty df if the array is empty (ie no years with meeting criteria)
    if Crop.empty:
        print('skipping due to not enough observations')
        #return none and end function if no years can be analysed
        #print('crop is empty')
        return None

    Crop.insert(0, 'Days_from_start', (Crop.index - Crop.index[0]).days + 1)

    
    #get arrays in scikitlearn format
    X, Y = format_arrays(Crop, method)

    try:
        crop_models = pd.read_csv(crop_models_file_path)
    except:
        print('Could not load crop models file. Using general relationship for all.')
            
        
    #check cropname is in models list
    if not cropname in list(crop_models['Model']):
        try:
            print('Crop name ' + cropname +' does not match those present in tenriero_et_al ')
        #if cropname is not in the models list, set cropname to none
        except: 
            print('Crop name not in crop models list')
        cropname = None

        
    if cropname is not None:
        #extract the specific model corresponding to the crop type
        model_params = crop_models.loc[crop_models['Model'] == cropname]
        #print('using crop-specific model')
        model_type = model_params['Type'].iloc[0]
        sat_thresh = model_params['Saturation_threshold'].iloc[0]
        a = model_params['a'].iloc[0]
        b = model_params['b'].iloc[0]
        c = model_params['c'].iloc[0]
        
    else:
        #take a generic model if no cropname is present
        #print('using generic model')
        model_params = crop_models.loc[crop_models['Model'] == 'General']
        model_type = model_params['Type'].iloc[0]
        sat_thresh = model_params['Saturation_threshold'].iloc[0]
        a = model_params['a'].iloc[0]
        b = model_params['b'].iloc[0]
        c = model_params['c'].iloc[0]
            
        
    X, Y = format_arrays(Crop, method)
    #generate uniform x array for prediction
    x_days = np.arange(Crop['Days_from_start'].min() , Crop['Days_from_start'].max(), 15, dtype = int)
    #If GPR is selected
    if method == 'GPR':
        
        kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 2e2)) \
        + WhiteKernel(noise_level=1, noise_level_bounds=(1e-12, 1e+1))
        
        #fit GPR but skip if a convergence warning is raised 
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, normalize_y=True).fit(X,Y)
                y_pred_VI, sigma = gp.predict(x_days[:, np.newaxis], return_cov=True)
            except Warning:
                print('SKlearn convergence warning: skipping data series')
                return None
        
    elif method == 'Spline':
        interp = interpolate.splrep(X, Y)
        y_pred_VI = interpolate.splev(x_days, interp)
    else:
        print('select valid interpolation method')
    
    #calculate the VI range in the phenological cycle 
    diff = np.amax(y_pred_VI) - np.amin(y_pred_VI)
    #calculate arrays with upper and lower uncertainty boundaries
    y_pred_VI_low = (y_pred_VI.flatten() - np.sqrt(np.diag(sigma)))
    y_pred_VI_high = (y_pred_VI.flatten() + np.sqrt(np.diag(sigma)))


    #if gpr validation is implemented, take 80% of data to fit and keep 20% for validation
    #run gpr model to generate an RSME for validation points
    if evaluate_gpr_fit == True:
        rsme, pcnt_in_sigma = evaluate_gpr_int(Crop, x_days, method, kernel)
    else:
        rsme = None
        pcnt_in_sigma = None 

    #predict FVC based on the VI extents from the smoothened GPR model
    #the VI range to define an arable crop is currently arbitrary - needs changing/justifying

    if data_format != 'LUCAS' and index == 'NDVI':
        if sat_thresh < 1:
            #account for NDVI saturation threshold, where NDVI is over threshols, set to threshold
            y_pred_VI = np.where(y_pred_VI < sat_thresh, y_pred_VI, sat_thresh)
        if model_type == 'linear':
            #implement models and convert from percentage to decimal
            y_pred_FVC = (a * y_pred_VI + b)/100
            y_pred_FVC_low = (a * y_pred_VI_low + b)/100
            y_pred_FVC_high = (a * y_pred_VI_high + b)/100
            
            y_pred_FVC = np.where(y_pred_FVC > 0, y_pred_FVC, 0)
            y_pred_FVC_low = np.where(y_pred_FVC_low > 0, y_pred_FVC_low, 0)
            y_pred_FVC_high = np.where(y_pred_FVC_high > 0, y_pred_FVC_high, 0)
            
        elif model_type == 'quadratic':
            #implement models and convert from percentage to decimal            
            y_pred_FVC = (a * y_pred_VI ** 2 + b * y_pred_VI + c)/100
            y_pred_FVC_low = (a * y_pred_VI_low ** 2 + b * y_pred_VI_low + c)/100
            y_pred_FVC_high = (a * y_pred_VI_high ** 2 + b * y_pred_VI_high + c)/100
            
            y_pred_FVC = np.where(y_pred_FVC > 0, y_pred_FVC, 0)
            y_pred_FVC_low = np.where(y_pred_FVC_low > 0, y_pred_FVC_low, 0)            
            y_pred_FVC_high = np.where(y_pred_FVC_high > 0, y_pred_FVC_high, 0)            
            
    else:
        #perform no operation on the spectral indices timeseries
        y_pred_FVC = y_pred_VI
        y_pred_FVC_low = (y_pred_VI.flatten() - np.sqrt(np.diag(sigma)))
        y_pred_FVC_high = (y_pred_VI.flatten() + np.sqrt(np.diag(sigma)))
        
        
    if plot == True:
        plt.style.use('seaborn-dark')   
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 18}

        plt.rc('font', **font)
        plt.rcParams["axes.labelweight"] = "bold"
        fig, ax = plt.subplots(figsize = (17,6))
        
    
        ax2 = ax.twinx()
        ax.scatter(X,Y, color = 'r', label= index +' observations')
        ax.plot(x_days, y_pred_VI, 'k--', label='GPR ' + index + ' interpolation')
        if index == 'NDVI':
            ax.plot(x_days, y_pred_FVC, 'g-', linewidth = 3, label='GPR CC interpolation')
        if method == 'GPR':    
            ax.fill_between(x_days, y_pred_VI_low, y_pred_VI_high,
                             alpha=0.2, color='blue')
            ax.fill_between(x_days, y_pred_FVC_low, y_pred_FVC_high,
                             alpha=0.4, color='green')
           
        if index == 'NDVI':        
            ax.set_ylabel('NDVI/CC-fraction')
        else:
            ax.set_ylabel(index)
            
        if rain_erosiv is not None:
            ax2.bar(rain_erosiv['Day'], rain_erosiv['EI30'], width = 5, label = '15-day average rainfall erosivity', fill = 'k', alpha = 0.5, edgecolor = 'blue')
            ax2.set_ylabel('Rainfall erosivity (MJ $\mathregular{ha^{−1} h^{−1} m^{−1}}$)')
        if LUCAS_date is not None:
            ax.bar(LUCAS_date, 0.8, color = 'black', width = 3, label = 'LUCAS survey date')
        ax.set_xlabel('Day of year')
        #plt.xlabel('days')
        #plt.ylabel('NDVI')
        #plt.ylim(0, 1)
        fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if cropname is not None:
            ax.set_title(cropname2 + ' ' + str(obj_id))
    #conditionally insert the correct start dates when creating a date series 
    if start_date is not None:
        date_series = pd.date_range(start_date, periods = len(x_days), freq = '15d')
    else:
        date_series = pd.date_range(Crop.index[0], periods = len(x_days), freq = '15d')
    
    
    #convert arrays to pandas dataframes with a date index     
    y_pred_VI = pd.DataFrame(y_pred_VI, index = date_series, columns = ['gpr_pred'])
    y_pred_VI_low = pd.DataFrame(y_pred_VI_low, index = date_series, columns = ['gpr_pred'])
    y_pred_VI_high = pd.DataFrame(y_pred_VI_high, index = date_series, columns = ['gpr_pred'])    
    
    
    y_pred_FVC = pd.DataFrame(y_pred_FVC, index = date_series, columns = ['gpr_pred'])
    y_pred_FVC_low = pd.DataFrame(y_pred_FVC_low, index = date_series, columns = ['gpr_pred'])
    y_pred_FVC_high = pd.DataFrame(y_pred_FVC_high, index = date_series, columns = ['gpr_pred'])
    
    
    #delete column so it can be recreated in next iteration
    del(Crop['Days_from_start'])
    
    if sensor == 'Landsat':
        #evaluate the replacement of ndvi values between years 
         ndvi_eval = eval_ndvi_replacement(output['Crop NDVI original'], 
                                 y_pred_VI, y_pred_VI_low, y_pred_VI_high)
         output['ndvi_replacement_evaluation'] = ndvi_eval
        
    #pack variables into output dictionary
    output['x_days'] = x_days
    output['y_pred_FVC'] = y_pred_FVC
    output['y_pred_FVC_low'] = y_pred_FVC_low
    output['y_pred_FVC_high'] = y_pred_FVC_high
    output['y_pred_VI'] = y_pred_VI
    output['obj_id'] = obj_id
    output['count_per_year_avg'] = count_per_year_avg
    output['rsme'] = rsme
    
    return output

    
    
    
    
    
