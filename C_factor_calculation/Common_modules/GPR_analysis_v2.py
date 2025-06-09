# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 17:13:11 2021

Module for Implementing Gaussian Process Regression and 
Spline Interpolation for Remote Sensing Phenological Time Series, 
enabling predictions of Fractional Vegetation Cover.

@author: Francis Matthews & Arno Kasprzak
"""
# Import modules
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel,Matern
from sklearn.metrics import mean_squared_error
from scipy import interpolate 
import warnings

"""
Defining the format_arrays function
"""
def format_arrays(Crop, method):   
    """
    Function to reformat arrays for use in GPR or other methods

    Parameters
    ----------
    Crop : DATAFRAME
        The dataframe containing the phenological time series.
    method : STRING
        The method to be used for formatting the arrays..
        For Gaussian Process Regression => 'GPR'

    Returns
    -------
    TUPLE
        - X : ARRAY
          The array of days from the start, formatted according to the selected method.
        - Y : ARRAY
          The array of observed values, formatted according to the selected method.

    """
    # Check if the method is Gaussian Process Regression
    if method == 'GPR':
        # Filter the second column for non-null values and reshape it to a 2D array
        Y = Crop.iloc[:,1][Crop.iloc[:,1].notnull()].values.reshape(-1,1)
        # Filter the 'Days_from_start' for non-null values and reshape it to a 2D array
        X = Crop['Days_from_start'][Crop.iloc[:,1].notnull()].values.reshape(-1,1)
    else:
        # For other methods no reshaping
        Y = Crop.iloc[:,1][Crop.iloc[:,1].notnull()].values
        X = Crop['Days_from_start'][Crop.iloc[:,1].notnull()].values
    return X, Y

"""
Defining filter_by_n function
"""
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
    n_observations : INTEGER
        The critical number of observations required. 
    modify_ts : BOOLEAN, optional
        Whether to modify the timeseries based on the filtering and replacement criteria.
        The default is False
    replace :BOOLEAN, optional
        optionally replace years with insufficient observations 
        (n < n_observations) with the time series of the year with the most observations.
        The default is False.

    Returns
    -------
    timeseries_new : DATAFRAME
        A new dataframe with the years exceeding the threshold number of
        observations
    count_per_year_avg : FLOAT
        The average number of observations per year in the time series.

    '''
    # Set modify_ts to True if replace is True
    if replace == True:
        modify_ts = True
        
    # Get all unique years
    year_start = timeseries.index.year.unique()[0]
    year_end = timeseries.index.year.unique()[-1] + 1
    years = np.arange(year_start, year_end, 1)
    # Initiate lists
    years_to_include = []
    years_to_exclude = []
    
    # Start a count
    i = 1
    # Initialise n_prev so n is always larger
    n_prev = 0
    count_per_year = []
    
    # Iterate through each year to check observation counts
    for year in years:
        # Set year of maximum observations at year 1
        if i == 1:
            year_max_obs = year
        # Count n observations in year
        n = int(timeseries[timeseries.index.year == year].count())
        # print('n observations: ', n)
        # Append the count of observations if it exceeds specified threshold
        count_per_year.append(n)
        
        # Update year of maximum observations if current year exceeds previous
        if i > 1:
            if n > n_prev:
                # Find year with max observations
                year_max_obs = year
                # Update n_prev for the next loop
                n_prev = n
        
        # Determine if the current year meets the observation threshold
        if n >= n_observations:
            years_to_include.append(year)
        else:
            years_to_exclude.append(year)
        
        # Increase iteration counter
        i = i + 1
    
    # Filter or modify the time series based on 'modify_ts' and 'replace' settings
    if replace == False and modify_ts == True:
        # Only take years with sufficient observations 
        timeseries_new = timeseries[timeseries.index.year.isin(years_to_include)]
    elif replace == True and modify_ts == True:
        # Only take years with sufficient observations
        timeseries_new = timeseries[timeseries.index.year.isin(years_to_include)]
        # Replace each excluded year with data from the year with max observations
        for year in years_to_exclude:
            fill_year = timeseries[timeseries.index.year == year_max_obs]
            # Calculate the year difference to shift the data to the excluded year 
            diff = year - year_max_obs 
            fill_year.index = fill_year.index + pd.offsets.DateOffset(years=diff)
            # Concatenate the shifted data for the excluded year into the new time series
            timeseries_new = pd.concat([timeseries_new, fill_year])
        # Evaluate the difference between the original NDVI points and the
        # new replacement values 
        # print(timeseries)
    else:
        timeseries_new = timeseries
    
    # Convert the filtered or modified time series to a DataFrame and sort by date
    timeseries_new = pd.DataFrame(timeseries_new.sort_index())
    # Return the average number of observations per year and the timeseries_new
    count_per_year_avg = np.mean(count_per_year)
    return timeseries_new, count_per_year_avg

"""
Defining the evaluate_gpr_int function
"""
def evaluate_gpr_int(Crop, x_days, method, kernel):
    """
    Function to evaluate the performance of a GPR model on 
    a given crop time series data by training and validating the model 
    on subsets of data. It calculates both the RMSE and the percentage 
    of validation points that fall within the model's uncertainty bounds.

    Parameters
    ----------
    Crop : DATAFRAME
        Time series data of NDVI for a given crop.
    x_days : ARRAY
        Array of day values.
    method : STRING
        Interpolation method to format data for GPR training and validation.
    kernel : SKLEARN KERNEL
        GPR kernel.

    Returns
    -------
    rsme : FLOAT
        RMSE of GPR predictions on the validation set.
    pcnt_in_sigma : FLOAT
        Percentage of validation points within the GPR uncertainty bounds.

    """
    # Drop na values so they aren't contained in validation dataset 
    Crop_val = Crop.dropna()
    # Split data in training and validation subsets
    Crop_80 = Crop_val.sample(frac = 0.8, random_state = 1)
    Crop_20 = Crop_val.drop(Crop_80.index)
    
    # Format the training data arrays
    X_t, Y_t = format_arrays(Crop_80, method)
    
    # Train model with training dataset
    gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, normalize_y=True).fit(X_t,Y_t)
    # Predict values over the specified days with GPR
    y_pred_VI_val, sigma_val = gp.predict(x_days[:, np.newaxis], return_cov=True)

    # Format validation data arrays
    X_v, Y_v = format_arrays(Crop_20, method)
    
    # Predict validation set outputs
    y_pred_val, sigma_val = gp.predict(X_v, return_cov=True)
    # Lower and upper bound
    y_pred_val_l = (y_pred_val.flatten() - np.sqrt(np.diag(sigma_val)))
    y_pred_val_u = (y_pred_val.flatten() + np.sqrt(np.diag(sigma_val)))
    y_pred_val = y_pred_val.flatten()
    
    # Calculate the percentage of validation data points within uncertainty bounds
    count = 0
    for i in np.arange(len(y_pred_val)):
        # Count if inside the uncertainty boundaries 
        if y_pred_val[i] > y_pred_val_l[i] and y_pred_val[i] < y_pred_val_u[i]:
            count = count + 1
    pcnt_in_sigma = (count/len(X_v)) * 100        
    
    # Calculate rmse
    rsme = np.sqrt(mean_squared_error(Y_v, y_pred_val))
    
    return rsme, pcnt_in_sigma

"""
Defining GPR_interpolate function
"""
def GPR_interpolate(timeseries, data_format, column, plot, method = 'GPR', 
                    min_annual_obs = 10, start_date = None, rain_erosiv = None, 
                    index = 'NDVI', crop_models_file_path = None, 
                    LUCAS_date = None, evaluate_gpr_fit = False):
    '''
    Takes a number of remotely sensed observations and creates a smooth vector 
    representing the field's phenology timeseries. If the vegetation index is NDVI, a 
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
    plot: BOOLEAN
         Optionally plot timeseries 
    method: STRING
        Default 'GPR'. Specify 'Spline' for spline interpolation.
    min_annual_obs: INTEGER
        Define the minimum number of satelite observations required in a year for 
        it to be included. Default is 15 per year.
    start_date : PANDAS TIMESTAMP OR NONE
        Optionally add a start date outside of the observation range to define
        the start of the timestamp series. This begins the interpolation from 
        a specific date instead of the first observation. Default is none, the series start date is used.
    rain_erosiv : DATAFRAME OR NONE
        Dataframe with 15-day average rainfall erosivity (default None).
    index: STRING
        A string stating the spectral index to analyse. 'NDVI' (default)
        or 'NDTI'
    crop_models_file_path: STRING OR NONE
        file pathway for the 'tenriero_et_al' crop models 
    evaluate_gpr_fit : BOOLEAN
        If True, the function evaluates the GPR. The default is False.
        
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
    # Initialize output dictionary
    output = {}
    # Copy input timeseries to avoid modifying original data
    Crops = timeseries.copy()
    try:
        # Extract the name corresponding to the crop model 
        cropname = Crops.iloc[column]['crop_model']
        # Extract the descriptive crop name
        cropname2 = Crops.iloc[column]['crop_name']
    except:
        # If no cropname can be extracted, treat crop as not available
        cropname = None
    
    # Find the first column containing numerical dates
    indexer = []
    for x in Crops.columns:
        num = x.isnumeric()
        indexer.append(num)
    # Transpose the relevant data, convert index to datetime for easier date handling
    Crop = Crops.iloc[:,indexer]
    Crop = Crop.transpose()
    # Convert to a datetime
    Crop.index = pd.to_datetime(Crop.index)
    # Get specific field observation data and identifier
    field_obs = Crop.iloc[:, column]
    obj_id = field_obs.name
    
    # Apply observation filter
    Crop, count_per_year_avg = filter_by_n(field_obs, min_annual_obs, replace = False)
    
    # Filter by n returns empty df if the array is empty (ie no years with meeting criteria)
    if Crop.empty:
        print('skipping due to not enough observations')
        # Return none and end function if no years can be analysed
        #print('crop is empty')
        return None
    # Add column for days since start of series
    Crop.insert(0, 'Days_from_start', (Crop.index - Crop.index[0]).days + 1)
    
    # Get arrays in scikitlearn format
    X, Y = format_arrays(Crop, method)
    
    # Load crop models for NDVI to FVC conversion if specified
    try:
        crop_models = pd.read_csv(crop_models_file_path)
        # Check cropname is in models list
        if not cropname in list(crop_models['Model']):
            try:
                print('Crop name ' + cropname +' does not match those present in tenriero_et_al ')
            # If cropname is not in the models list, set cropname to none
            except: 
                print('Crop name not in crop models list')
            cropname = None
    except:
        print('Could not load crop models file. Using general relationship for all.')

    # Assign model parameters based on crop-specific or general model  
    if cropname is not None:
        # Extract the specific model corresponding to the crop type
        model_params = crop_models.loc[crop_models['Model'] == cropname]
        #print('using crop-specific model')
        model_type = model_params['Type'].iloc[0]
        sat_thresh = model_params['Saturation_threshold'].iloc[0]
        a = model_params['a'].iloc[0]
        b = model_params['b'].iloc[0]
        c = model_params['c'].iloc[0]  
    else:
        # Take a generic model if no cropname is present
        #print('using generic model')
        model_params = crop_models.loc[crop_models['Model'] == 'General']
        model_type = model_params['Type'].iloc[0]
        sat_thresh = model_params['Saturation_threshold'].iloc[0]
        a = model_params['a'].iloc[0]
        b = model_params['b'].iloc[0]
        c = model_params['c'].iloc[0]
            
    # Get arrays in scikitlearn format 
    X, Y = format_arrays(Crop, method)
    # Generate uniform x array for prediction
    x_days = np.arange(Crop['Days_from_start'].min() , Crop['Days_from_start'].max() + 30, 15, dtype = int)
    # If GPR is selected
    if method == 'GPR':
        # Define kernel
        kernel = 1.0 * Matern(length_scale=100.0, length_scale_bounds=(10, 300), nu=1.5) \
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-12, 1e+1))
        
        # Fit GPR but skip if a convergence warning is raised 
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                gp = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10, alpha=0.5, normalize_y=True).fit(X,Y)
                y_pred_VI, sigma = gp.predict(x_days[:, np.newaxis], return_cov=True)
            except Warning:
                print('SKlearn convergence warning: skipping data series')
                return None
    elif method == 'Spline':
        # Perform spline interpolation on the data
        interp = interpolate.splrep(X, Y)
        y_pred_VI = interpolate.splev(x_days, interp)
        print(y_pred_VI)
        sigma = 0
    else:
        print('select valid interpolation method')
        
    if method == 'GPR':
        # Calculate arrays with upper and lower uncertainty boundaries
        y_pred_VI_low = (y_pred_VI.flatten() - np.sqrt(np.diag(sigma)))
        y_pred_VI_high = (y_pred_VI.flatten() + np.sqrt(np.diag(sigma)))
    else:
        # For spline interpolation, uncertainty boundaries are equal to predictions
        y_pred_VI_low = y_pred_VI
        y_pred_VI_high = y_pred_VI

    # If gpr validation is implemented, take 80% of data to fit and keep 20% for validation
    # Run gpr model to generate an RSME for validation points
    if evaluate_gpr_fit == True:
        rsme, pcnt_in_sigma = evaluate_gpr_int(Crop, x_days, method, kernel)
    else:
        rsme = None
        pcnt_in_sigma = None 

    # Predict FVC based on the VI extents from the smoothened GPR model
    # The VI range to define an arable crop is currently arbitrary - needs changing/justifying

    if data_format != 'LUCAS' and index == 'NDVI':
        if sat_thresh < 1:
            # Account for NDVI saturation threshold, where NDVI is over threshols, set to threshold
            # Since saturation occured, predictions outsie range are invalid
            y_pred_VI = np.where(y_pred_VI < sat_thresh, y_pred_VI, sat_thresh)
        if model_type == 'linear':
            # Implement models and convert from percentage to decimal
            y_pred_FVC = (a * y_pred_VI + b)/100
            y_pred_FVC_low = (a * y_pred_VI_low + b)/100
            y_pred_FVC_high = (a * y_pred_VI_high + b)/100
            # Ensure predictions are not negative
            y_pred_FVC = np.where(y_pred_FVC > 0, y_pred_FVC, 0)
            y_pred_FVC_low = np.where(y_pred_FVC_low > 0, y_pred_FVC_low, 0)
            y_pred_FVC_high = np.where(y_pred_FVC_high > 0, y_pred_FVC_high, 0)
            
        elif model_type == 'quadratic':
            # Implement models and convert from percentage to decimal            
            y_pred_FVC = (a * y_pred_VI ** 2 + b * y_pred_VI + c)/100
            y_pred_FVC_low = (a * y_pred_VI_low ** 2 + b * y_pred_VI_low + c)/100
            y_pred_FVC_high = (a * y_pred_VI_high ** 2 + b * y_pred_VI_high + c)/100
            # Ensure predictions are not negative
            y_pred_FVC = np.where(y_pred_FVC > 0, y_pred_FVC, 0)
            y_pred_FVC_low = np.where(y_pred_FVC_low > 0, y_pred_FVC_low, 0)            
            y_pred_FVC_high = np.where(y_pred_FVC_high > 0, y_pred_FVC_high, 0)            
            
    else:
        # Perform no operation on the spectral indices timeseries
        y_pred_FVC = y_pred_VI
        y_pred_FVC_low = (y_pred_VI.flatten() - np.sqrt(np.diag(sigma)))
        y_pred_FVC_high = (y_pred_VI.flatten() + np.sqrt(np.diag(sigma)))
    
    # Set up the plot
    if plot == True:
        #plt.style.use('seaborn-dark')   
        font = {'family' : 'normal',
                'weight' : 'bold',
                'size'   : 18}

        plt.rc('font', **font)
        plt.rcParams["axes.labelweight"] = "bold"
        fig, ax = plt.subplots(figsize = (17,6))
        # Determine color and style based on the index
        if index == 'NDVI':
            # point colour
            c_p = 'r'
            #line style
            ls = 'k--'
            # envelope colour for VI
            e_c = 'blue'
        elif index == 'NDTI':
            c_p = 'orange'
            ls = 'k--'
            e_c = 'yellow'            
        
        # Create a second y-axis
        ax2 = ax.twinx()
        # Plot observations and predictions
        ax.scatter(X,Y, color = c_p, label= index +' observations')
        ax.plot(x_days, y_pred_VI, ls, label='GPR ' + index + ' interpolation')
        if index == 'NDVI':
            ax.plot(x_days, y_pred_FVC, 'g-', linewidth = 3, label='GPR CC interpolation')
        # Add uncertainty boundaries to the plot
        if method == 'GPR':    
            ax.fill_between(x_days, y_pred_VI_low, y_pred_VI_high,
                             alpha=0.2, color= e_c)
            if index == 'NDVI':
                ax.fill_between(x_days, y_pred_FVC_low, y_pred_FVC_high,
                                 alpha=0.4, color='green')
        
        # Set y-labels based on the index 
        if index == 'NDVI':        
            ax.set_ylabel('NDVI/CC-fraction')
        else:
            ax.set_ylabel(index)
        
        # Add rainfall erosivity data if available
        if rain_erosiv is not None:
            ax2.bar(rain_erosiv['Day'], rain_erosiv['erosivity'], width = 5, label = 'median 15-day rainfall erosivity', fill = 'k', alpha = 0.5, edgecolor = 'blue')
            ax2.set_ylabel('Rainfall erosivity (MJ $\mathregular{ha^{−1} h^{−1} m^{−1}}$)')
        # Set labels
        ax.set_xlabel('Day of year')
        #plt.xlabel('days')
        #plt.ylabel('NDVI')
        #plt.ylim(0, 1)
        # Add legend
        fig.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Add a title if cropname is available
        if cropname is not None:
            ax.set_title(cropname2 + ' ' + str(obj_id))
    # Conditionally insert the correct start dates when creating a date series 
    if start_date is not None:
        date_series = pd.date_range(start_date, periods = len(x_days), freq = '15d')
    else:
        date_series = pd.date_range(Crop.index[0], periods = len(x_days), freq = '15d')
    
    # Convert arrays to pandas dataframes with a date index     
    y_pred_VI = pd.DataFrame(y_pred_VI, index = date_series, columns = ['gpr_pred'])
    y_pred_VI_low = pd.DataFrame(y_pred_VI_low, index = date_series, columns = ['gpr_pred'])
    y_pred_VI_high = pd.DataFrame(y_pred_VI_high, index = date_series, columns = ['gpr_pred'])    
    
    
    y_pred_FVC = pd.DataFrame(y_pred_FVC, index = date_series, columns = ['gpr_pred'])
    y_pred_FVC_low = pd.DataFrame(y_pred_FVC_low, index = date_series, columns = ['gpr_pred'])
    y_pred_FVC_high = pd.DataFrame(y_pred_FVC_high, index = date_series, columns = ['gpr_pred'])
    
    # Delete column so it can be recreated in next iteration
    del(Crop['Days_from_start'])
            
    # Pack variables into output dictionary
    output['x_days'] = x_days
    output['y_pred_FVC'] = y_pred_FVC
    output['y_pred_FVC_low'] = y_pred_FVC_low
    output['y_pred_FVC_high'] = y_pred_FVC_high
    output['y_pred_VI'] = y_pred_VI
    output['obj_id'] = obj_id
    output['count_per_year_avg'] = count_per_year_avg
    output['rsme'] = rsme
    
    return output

    
    
    
    
    
