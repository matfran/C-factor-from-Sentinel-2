# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 18:37:16 2022

This module defines a set of functions for analysing and visualising data,
particularly focused on FVC, SLR and RE.

@author: Francis Matthews & Arno Kasprzak
"""
# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import geopandas as gpd
import scipy.stats as stats

"""
Defining the count_anual_observations function
"""
def count_annual_observations(LPIS_all):
    '''
    Counts the number of image observations per year, per field parcel. Returns
    a dataframe with the annual counts per parcel ID. 

    Parameters
    ----------
    LPIS_all : DATAFRAME
        A DataFrame containing the image observation data for multiple field parcels

    Returns
    -------
    lpis_n_count : DATAFRAME
        A DataFrame containing the annual count of observations for each field parcel.

    '''
    # Find the column with the relevant id in parcel data
    for col in LPIS_all.columns:
        if 'id' in col:
            id_col = col
        elif 'ID' in col:
            id_col = col 
            
    # Function to count the number of annual observations per field parcel
    # Extract column names and convert them to string
    dates = pd.Series(LPIS_all.columns.astype('str'))
    # Filter columns to retain only those with numeric values (time series)
    ref = dates.str.isnumeric()
    dates2 = dates[ref]
    # Get a dataframe with only the timeseries 
    lpis = LPIS_all[dates2]
    # Convert column strings to datetime
    lpis.columns = pd.to_datetime(lpis.columns)
    # Resample data annually ('Y') and count the number of observations per year
    lpis_n_count = lpis.resample('Y', axis = 1, origin = 'epoch').count()
    # Modify column names to reflect the count of observations per year (e.g., '2020_obs_count')
    cols = list(lpis_n_count.columns.astype('str'))
    new_cols = []
    for i in cols:
        i = i[:4] + '_obs_count'
        new_cols.append(i)
    # Rename the columns with the new observation count names
    lpis_n_count.columns = new_cols
    # Add in the object id column for recognition 
    lpis_n_count[id_col] = LPIS_all[id_col]
    return lpis_n_count

"""
Defining the format_merge function
"""
def format_merge(df_ts, parcels_shp, parcels_merge_type, parcel_id = None, parcel_ts_id = None, 
                 merge_only = False):
    """
    This function merges a time series DataFrame (df_ts) with a shapefile containing 
    parcel information (parcels.shp)

    Parameters
    ----------
    df_ts : DATAFRAME
        A DataFrame containing time series data
    parcels_shp : DATAFRAME
        A DataFrame representation of the shapefile containing parcel information
    parcels_merge_type : String
        The type of merge to perform
    parcel_id : STRING, optional
        The name of the column containing parcel IDs in the shapefile. The default is None.
    parcel_ts_id : STRING, optional
        The name of the column containing parcel IDs in the time series DataFrame. The default is None.
    merge_only : STRING, optional
        If False, performs additional cleaning before merging. The default is False.

    Returns
    -------
    df_ts : DATAFRAME
        The merged DataFrame, containing both time series and parcel information.

    """
    # If id strings are not specified they can be automatically found
    if parcel_id is not None:
        # If the parcel ID is provided, assign it directly to the variable id_col
        id_col = parcel_id 
    else:
        id_list = []
        # Find  the column with the relevant id in parcel data
        for col in parcels_shp.columns:
            if 'id' in col:
                id_col = col
                id_list.append(id_col)
            elif 'ID' in col:
                id_col = col 
                id_list.append(id_col)
    # Now do it for the time series DataFrame in a similar manner 
    if parcel_id is not None:
        id_ts = parcel_ts_id 
    else:       
        for col in df_ts.columns:
            if 'id' in col:
                id_ts = col
                id_list.append(id_col)
            elif 'ID' in col:
                id_ts = col 
                id_list.append(id_col)
    # Error if there are more than 2 ID columns
    if len(id_list) > 2:
        sys.exit('Too many id columns in the parcels shapefile and/or timeseries')
    # Rename the ID columns to OBECTID in both dataframes
    id_col_new = 'OBJECTID'
    parcels_shp.rename(columns = {id_col : id_col_new}, inplace = True)
    df_ts.rename(columns = {id_ts : id_col_new}, inplace = True)

    # Merge based on the id column
    if merge_only == False:
        # Remove unnecessary columns
        del(df_ts['.geo'])
        del(df_ts['system:index'])
        # Replace placeholder values with NaN
        df_ts = df_ts.replace(-9999, np.nan)
        # Merge shapefile with time series dataframe using the ID column
        df_ts = pd.DataFrame(parcels_shp.merge(df_ts, on = [id_col_new], how = parcels_merge_type))
        # Set object id as index
        df_ts.index = df_ts[id_col_new].values
    # If only merging is required (no additional cleaning), simply perform the merge
    else:
        df_ts = pd.DataFrame(parcels_shp.merge(df_ts, on = [id_col_new], how = parcels_merge_type))
        # Set object id as index
        df_ts.index = df_ts[id_col_new].values
    return df_ts

"""
Defining the add_harvest_cropres function
"""
def add_harvest_cropres(df_fvc_re, harvest_inflexes):
    '''
    Add crop residue to the fractional vegetation cover array based on the
    identification of crop residues. Returns an array with the combined FVC
    and crop residue. 

    Parameters
    ----------
    df_fvc_re : DATAFRAME
        A DataFrame containing fractional vegetation cover data with an index 
        representing dates
    harvest_inflexes : DATAFRAME
        A DataFrame containing estimated harvest dates and corresponding crop 
        residue status

    Returns
    -------
    df_fvc_re : DATAFRAME
        The updated DataFrame with adjusted fractional cover including crop 
        residue
    harvest_period_total : INT
        The total number of days categorized as low vegetation periods during 
        which crop residues were identified
    harvest_period_w_res : INT
        The total number of days during which crop residues were present in 
        the low vegetation periods
    
    '''
    # Select relevant columns
    harvest_inflexes_new = harvest_inflexes[['Estimated_harvest_date', 'Crop_res']]
    # Create a copy of the original dataframe
    df_fvc_re_new = df_fvc_re.copy()
    # Add a column for the next merging stage
    df_fvc_re_new['Estimated_harvest_date'] = df_fvc_re_new.index 
    # Merge the FVC DataFrame with the harvest inflection data based on harvest dates
    fc_merged = pd.merge_asof(df_fvc_re_new, harvest_inflexes_new, on = 'Estimated_harvest_date',
                              direction = 'nearest', tolerance = pd.Timedelta('8 days'))
    # Identify periods of low vegetation (canopy cover < 0.3)
    fc_merged['Low_veg_period'] = np.where(fc_merged['gpr_pred'] < 0.3, 1, 0)
    # Count number of time increments and multiply by 15 (days)
    harvest_period_total = fc_merged[fc_merged['Low_veg_period'] == 1]['Low_veg_period'].count() * 15
    # Forward fill array with crop residue status (1 or 0), then fill any values preceeding with 0
    fc_merged['Crop_res_cont'] = fc_merged['Crop_res'].fillna(method = 'ffill').fillna(0)
    # Assign crop residue (0.3) during low veg periods where crop residues were identified
    # Otherwise if high veg period or no crop res, assign 0 cover
    fc_merged['Crop_res_cov'] = np.where(np.logical_and(fc_merged['Crop_res_cont'] == 1, fc_merged['Low_veg_period'] == 1), 0.3, 0)
    # Fill low veg-crop res periods with 0.3, otherwise assign gpr pred value
    fc_merged['Fractional_cov'] = np.where(np.logical_and(fc_merged['Crop_res_cont'] == 1, fc_merged['Low_veg_period'] == 1), 0.3, fc_merged['gpr_pred'])    
    # Count number of days with crop residue
    harvest_period_w_res = fc_merged[fc_merged['Crop_res_cov'] == 0.3]['Crop_res_cov'].count() * 15
    
    # Update the fractional cover variable to include the crop residue
    df_fvc_re['gpr_pred'] = fc_merged['Fractional_cov'].values
    
    return df_fvc_re, harvest_period_total, harvest_period_w_res

"""
Defining the FVC_to_SLR function
"""
def FVC_to_SLR(fvc_ts, multiplier): 
    """
    Function that converts FVC time series to SLR time series.
    
    Parameters
    ----------
    fvc_ts: pd.Series
        A time series of fractional vegetation cover values
    multiplier: ARRAY
        A array containing the beta values to convert 
        fvc timeseries to slr timeseries
    
    Returns
    -------
    slr: pd.Series
        A slr time series calculated from the FVC values
    
    """
    # Convert  to %
    cc = fvc_ts * 100
    # Convert fvc to slr using an exponential relationship
    slr = np.exp(cc * multiplier)
    return slr

"""
Defining the C_factor function
"""
def C_factor(df_fvc_re, b_exponent, sc_uncertainty = False, return_slr_ts = False):
    '''
    Returns the C-factor from an annual series of vegetation cover and rainfall
    erosivity at 15-day intervals. The function follows the standard RUSLE 
    procedure.

    Parameters
    ----------
    df_fvc_re : DATAFRAME
        Dataframe containing columns 'gpr_pred' with vegetation cover and
        'erosivity' with rainfall erosivity at 15 day intervals.
    b_exponent : FLOAT, optional
        Beta value for computing the SLR.
    sc_uncertainty : BOOL, optional
        Whether to incorporate uncertainty into the SC factor. The default is False.
    return_slr_ts: BOOL, optional
        If True, returns the modified DataFrame including intermediate calculations.
        The default is False.

    Returns
    -------
    C_fact : FLOAT
        C-factor (0-1)

    '''
    # Calculate rainfall erosivity as a percent of itself, creating a 100% benchmark for normalization
    df_fvc_re['rainfall_EI_pcnt'] = (df_fvc_re['erosivity']/df_fvc_re['erosivity']) * 100
    # Compute SLR based on FVC using the FVC_to_SLR function and b_exponent
    df_fvc_re['SLR'] = FVC_to_SLR(df_fvc_re['gpr_pred'], b_exponent)
    
    """
    # Optional application of the Incorporate_RUSLE_sc function
    # if one decides to incorporate uncertainty into the SC-factor
    # Update SLR to reflect median SC values based on bounds
    if sc_uncertainty == True:
        df_fvc_re = Incorporate_RUSLE_sc(df_fvc_re, sc_pcnt_bounds = (0, 20), 
                                         b_bounds = (0.024, 0.07), Ru_inch_bounds = (0,5))
        df_fvc_re['SLR'] = df_fvc_re['CC'] * df_fvc_re['sc_median']
    else:
        df_fvc_re['SLR'] = df_fvc_re['SLR']
    """
    # Calculate the C-factor using RUSLE handbook equation (page 187)
    # https://www.ars.usda.gov/arsuserfiles/64080530/rusle/ah_703.pdf
    C_fact = (df_fvc_re['SLR']  * df_fvc_re['rainfall_EI_pcnt']).sum()/df_fvc_re['rainfall_EI_pcnt'].sum()
    # Calculate the sl * ei timeseries
    df_fvc_re['SLR_EI_ts'] = df_fvc_re['SLR'] * df_fvc_re['erosivity']
    # Calculate the sl * ei% timeseries (normalised)
    df_fvc_re['SLR_EI_pcnt_ts'] = df_fvc_re['SLR'] * df_fvc_re['rainfall_EI_pcnt']
    
    # Optionally return the modified DataFrame with SLR series
    if return_slr_ts == True:  
        return C_fact, df_fvc_re
    else:
        return C_fact

"""
Defining the Calc_risk_period
"""
def Calc_risk_period(df_fvc_re, index = 'month'):
    '''
    Retruns the time period of the highest erosion risk from a dataframe 
    containing a series of vegetation cover and rainfall erosivity at 15-day
    intervals. 

    Parameters
    ----------
    df_fvc_re : DATAFRAME
        Dataframe containing columns 'gpr_pred' with vegetation cover and
        'erosivity' with rainfall erosivity at 15 day intervals. 
    index : STRING, optional
        Specify whether to return month of highest risk (default) or '15-day' period (0-360)

    Returns
    -------
    Day_of_max_risk : INTEGER
        15-day interval containing highest erosivity risk.
    Month_of_year : INTEGER
        Month of the year (1-12) with highest erosivity risk.

    '''
    # Calculate the percentage contribution of fvc and rainfall erosivity to their total values
    df_fvc_re['fvc_pcnt'] = (df_fvc_re['gpr_pred']/df_fvc_re['gpr_pred'].sum()) * 100
    df_fvc_re['RE_pcnt'] = (df_fvc_re['erosivity']/df_fvc_re['erosivity'].sum()) * 100
    # Calculate a risk index based on the difference between the scaled RE and FVC
    df_fvc_re['risk_index'] = df_fvc_re['RE_pcnt'] - df_fvc_re['fvc_pcnt']
    # Extract the day corresponding to the maximum risk 
    Day_of_max_risk = df_fvc_re['Day'][ df_fvc_re['risk_index'].idxmax()]
    # Extract the rainfall erosivity value corresponding to the maximum risk index
    EI30_risk = df_fvc_re['erosivity'][ df_fvc_re['risk_index'].idxmax()]
    # Determine the month corresponding to the maximum risk index
    Month_of_year = df_fvc_re['risk_index'].idxmax().month
    
    # Calculate the total rainfall erosivity % during low vegetation periods
    RE_pcnt_low_veg = float(df_fvc_re[df_fvc_re['gpr_pred'] <= 0.3]['RE_pcnt'].sum())
    
    # Return results based on the specified index type
    if index == 'month':
        return Month_of_year, EI30_risk, RE_pcnt_low_veg
    elif index == '15-day':
        return int(Day_of_max_risk), EI30_risk, RE_pcnt_low_veg

"""
Defining the plot_ts function
"""
def plot_ts(slr_ts, clusters=None, use_clusters=True, ndvi_count=None, n_fields=0, id_col='object_id', title=None, 
            slice_ts=None, year_start=None, year_end=None, name=None,
            quantile_bounds=0.95, sowing_dates=None, harvest_dates=None):
    """
    Function to plot slr time series data for different clusters. Optionally
    harvest and sowing dates can be included on the plot. 
    
    Parameters
    ----------
    slr_ts : DATAFRAME
        Dataframe containing the slr time series 
    clusters : ARRAY
        Array indicating cluster membership for each slr time series
    ndvi_count : ARRAY, optional
        The count of NDVI measurements . The default is None.
    n_fields : INTEGER, optional
        Max number of fields to plot for each cluster. The default is 20.
    id_col : STRING, optional
        Column name to be used as the index of slr_ts DataFrame. The default is 'object_id'.
    title : STRING, optional
        Title of the plot. The default is None.
    slice_ts : BOOLEAN, optional
        If True, slices the time series between year_start and year_end. The default is False.
    year_start : INTEGER, optional
        Starting year for slicing the time series. The default is None.
    year_end : INTEGER, optional
        Ending year for slicing the time series. The default is None.
    name : STRING, optional
        Label for the y-axis. The default is None.
    quantile_bounds : FLOAT, optional
        Quantile bounds for calculating the upper and lower bounds of the 
        shaded area in the plot. The default is 0.95.
    sowing_dates : ARRAY, optional
        Sowing dates. The default is None.
    harvest_dates : ARRAY, optional
        Harvest dates. The default is None.

    Returns
    -------
    None.

    """
    # Set index of the Dataframe
    slr_ts = slr_ts.set_index(id_col)
    # Convert columns to datetime format
    cols_2 = pd.to_datetime(slr_ts.columns, yearfirst=True).date
    slr_ts.columns = cols_2
    # If slicing the time series, filter dataframe to include only specified date range
    if slice_ts == True:
        date_start = pd.to_datetime(str(year_start) + '/01/01', yearfirst=True)
        date_end = pd.to_datetime(str(year_end) + '/12/31', yearfirst=True)
        slr_ts = slr_ts.loc[:, (cols_2 >= date_start) & (cols_2 <= date_end)]
    
    # Set plot style and size
    #plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    # Create a new figure for the plot
    fig, ax = plt.subplots(figsize=(22, 6))

    # Different color palette for different clusters
    palette = sns.color_palette("tab10")
    cluster_colors = {}
    
    # Check if clusters should be used
    if use_clusters and clusters is not None:
        u_c = np.unique(clusters)
        for c in u_c:
            # Select the subset of time series that belong to the current cluster
            slr_c = slr_ts[clusters == c]
            # Check if slr_c is not empty
            if not slr_c.empty:
                # Calculate mean, upper, and lower quantiles for the cluster
                mean = slr_c.mean()
                upper = slr_c.quantile(quantile_bounds)
                lower = slr_c.quantile(1 - quantile_bounds)
                i = 0
                # Plot individual time series within the cluster, but limit to `n_fields`
                for row in np.arange(len(slr_c)):
                    if i >= n_fields:  # Limit the number of fields plotted
                        break
                    ts = slr_c.iloc[row]  # Access the row
                    ax.plot(ts, alpha=0.5, linewidth=2, color=palette[c])
                    i += 1

                # Plot the mean time series for the cluster
                ax.plot(mean, linewidth=7, color=palette[c], label=f'Cluster {c} - Median', alpha=0.5)

                # Plot the quantile bounds as a shaded region
                ax.fill_between(slr_ts.columns, lower.values, upper.values, alpha=0.2, color=palette[c], label=f'Cluster {c} - 5th and 95th percentiles')
                # Store color
                cluster_colors[c] = palette[c]
            else:
                print(f"No data available for cluster {c}")  # Inform if cluster has no data
    else:
        # If clusters are not used, calculate and plot the overall mean, upper, and lower quantiles
        mean = slr_ts.mean()
        upper = slr_ts.quantile(quantile_bounds)
        lower = slr_ts.quantile(1 - quantile_bounds)
        # Choose a pastel color palette
        colors = sns.color_palette("pastel", n_fields)
        if n_fields == 0:        
            # Plot the mean time series as a thick pastel green line
            ax.plot(mean, linewidth=4, label='Mean', color='mediumseagreen', alpha=0.9)
            # Plot the quantile range as a light green shaded area
            ax.fill_between(slr_ts.columns, lower.values, upper.values, 
                    alpha=0.2, color='lightgreen', label=' 5th and 95th quantiles')
        else:
            # Select only the first `n_fields` time series
            slr_ts_subset = slr_ts.iloc[:n_fields]
            
            # Plot each time series with a unique pastel color
            for idx, row in enumerate(slr_ts_subset.index):
                ax.plot(slr_ts_subset.columns, slr_ts_subset.loc[row], alpha=0.5, 
                        linewidth=2, color=colors[idx], label=f'Field {idx+1}' if idx < 10 else "_nolegend_")  
                # Plot the mean time series as a thick pastel green line
                ax.plot(mean, linewidth=4, label='Mean', color='mediumseagreen', alpha=0.9)
                # Plot the quantile range as a light green shaded area
                ax.fill_between(slr_ts.columns, lower.values, upper.values, 
                                alpha=0.2, color='lightgreen', label='5th and 95th quantile')

    # Optionally, plot sowing dates as green bars
    if sowing_dates is not None:
        for d in sowing_dates:
            ax.bar(d, 0.8, color='green', width=3, label='Surveyed sowing dates')
    
    # Optionally, plot harvest dates as yellow bars
    if harvest_dates is not None:
        for d in harvest_dates:
            ax.bar(d, 0.8, color='yellow', width=3, label='Surveyed harvest dates')

    # Set plot labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel(name)
    ax.set_title(title)
    ax.set_ylim([0, 1])

    # Annotate the number of NDVI data points if provided
    if ndvi_count is not None:
        x = 0
        for n in ndvi_count.values:
            ax.annotate('n ndvi = ' + str(n), (x, 0.1))
            x = x + 365

    # Show legend and display the plot
    #ax.legend()
    plt.show()

    #fig.legend()
    #ax.set_title(crop)
    #plt.savefig('path')
    # Return color mapping for each cluster to use in the histogram
    return cluster_colors

"""
Defining the C_factor_box_plot
"""
def C_factor_box_plot(Cfactor, hue = None):
    """
    Function to create a box plot of C-factor values across different crops.

    Parameters
    ----------
    Cfactor : DATAFRAME
        Dataframe containing C-factor values.
    hue : STRING, optional
        Column name in the dataframe to use for color-coding the box plots. The default is None.

    Returns
    -------
    None.

    """
    # Styling
    sns.set_theme(style="ticks", palette="pastel", font_scale=1.5)
    plt.rcParams.update({'font.size': 30})
    Cfactor['C_factor'] = Cfactor['C_factor'].clip(0, 1)
    # Create figure and axis
    fig, ax = plt.subplots(figsize = (20,6))
    # Create box plot 
    if hue == None:
        sns.boxplot(x="crop", y="C_factor",
                    data=Cfactor, ax = ax)
    else:
        sns.boxplot(x="crop", y="C_factor",
                    data=Cfactor, hue = hue, palette = 'tab10', ax = ax)
    # Remove the top and right spines from the plot
    sns.despine(offset=10, trim=True)
    # Set labels
    ax.set_xlabel('Crop name')
    ax.set_ylabel('C-factor value')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, fontsize=20)
    
def C_factor_box_plot_multiy(Cfactor, hue = None):
    """
    Function to create a box plot of multi-year C-factor values across different rotations.

    Parameters
    ----------
    Cfactor : DATAFRAME
        Dataframe containing C-factor values.
    hue : STRING, optional
        Column name in the dataframe to use for color-coding the box plots. The default is None.

    Returns
    -------
    None.

    """
    # Styling
    sns.set_theme(style="ticks", palette="pastel", font_scale=1.5)
    plt.rcParams.update({'font.size': 30})
    # Extract C-factors
    Cfactor['C_factor_average'] = Cfactor['C_factor_avg'].clip(0, 1)
    # Create figure and axis
    fig, ax = plt.subplots(figsize = (20,6))
    # Create box plot 
    if hue == None:
        sns.boxplot(x="rotations_x", y="C_factor_average",
                    data=Cfactor, ax = ax)
    else:
        sns.boxplot(x="rotations_x", y="C_factor_average",
                    data=Cfactor, hue = hue, palette = 'tab10', ax = ax)
    # Remove the top and right spines from the plot
    sns.despine(offset=10, trim=True)
    # Set labels
    ax.set_xlabel('Rotation')
    ax.set_ylabel('C-factor value')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, fontsize=20)   
    
"""
Defining the plot_cumulative function
"""
def plot_cumulative(slr_rf_ts):
    """
    Function to create a cumulative plot of slr multiplied by rainfall erosivity over time

    Parameters
    ----------
    slr_rf_ts : DATAFRAME
        Dataframe containing slr and rainfall erosivity time series data

    Returns
    -------
    None.

    """
    # Styling
    sns.set_theme(style="ticks", palette="pastel", font_scale=1.5)
    # Create figure and axis    
    fig, ax = plt.subplots(figsize = (15,6))
    # Draw line plot to show cumulative SLR * erosivity over time
    sns.lineplot(x='Datetime', y='SLR_EI_ts_cum', hue = 'crop',
                data=slr_rf_ts, errorbar = 'ci', ax = ax)
    # Set labels
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative SLR * rainfall erosivity')
        
"""
Defining the C_factor_comparison_box_plot function
"""
def C_factor_comparison_box_plot(dfs, scenario_names):
    """
    Creates a box plot comparing C-factor values across different crops and scenarios.
    
    Parameters:
    ----------
    dfs : LIST OF DATAFRAMES  
    List of DataFrames, each containing 'crop' and 'C_factor' columns.  
        scenario_names : LIST  
    List of strings representing the names corresponding to each scenario DataFrame.
    
    Returns
    -------
    None.
    
    """
    # Combine all dataframes into one, adding a 'Scenario' column
    combined_df = pd.concat(
        [df.assign(Scenario=scenario) for df, scenario in zip(dfs, scenario_names)],
        ignore_index=True  # Ensures proper merging without index mismatches
    )

    # Clip C-factor values to ensure they remain within the valid range [0,1]
    combined_df['C_factor'] = combined_df['C_factor'].clip(0, 1)

    # Set Seaborn theme and font settings
    sns.set_theme(style="ticks", palette="pastel", font_scale=1.5)
    plt.rcParams.update({'font.size': 30})

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(20, 6))

    # Generate boxplot with multiple scenarios per crop
    sns.boxplot(x="crop_x", y="C_factor", hue="Scenario",
                data=combined_df, palette="pastel", ax=ax)

    # Format plot
    sns.despine(offset=10, trim=True)
    ax.set_xlabel('Crop Name')
    ax.set_ylabel('C-factor Value')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, fontsize=20)
    plt.legend()
    
    # Show the plot
    plt.show()
    
"""
Defining load_data function
"""
def load_data(file_paths):
    """
    Loads a list of CSV files into a list of pandas DataFrames.

    Parameters:
    -----------
    file_paths : list of str
        List of file paths to CSV files.

    Returns:
    --------
    dataframes : list of pandas.DataFrame
        List containing a DataFrame for each loaded CSV file.
    """
    dataframes = [pd.read_csv(file) for file in file_paths]
    return dataframes

"""
Defining load_data_with_name function
"""
def load_data_with_name(file_paths):
    """
    Loads a list of CSV files into a list of pandas DataFrames and adds metadata.

    Each DataFrame will include:
    - 'file_name': the original file path.
    - 'ROT_i': rotation index inferred from the filename.

    Parameters:
    -----------
    file_paths : list of str
        List of file paths to CSV files.

    Returns:
    --------
    dataframes : list of pandas.DataFrame
        List containing a DataFrame for each loaded CSV file,
        with added metadata columns.
    """
    dataframes = []
    for file in file_paths:
        df = pd.read_csv(file)
        df['file_name'] = file  
        df['ROT_i'] = file[-5:-4]
        dataframes.append(df)
    return dataframes

"""
Defining load_data_pkl function
"""
def load_data_pkl(file_paths):
    """
    Loads a list of pickle files into a list of pandas DataFrames.

    Parameters:
    -----------
    file_paths : list of str
        List of file paths to .pkl files (pickled DataFrames).

    Returns:
    --------
    dataframes : list of pandas.DataFrame
        List containing a DataFrame for each loaded pickle file.
    """
    dataframes = [pd.read_pickle(file) for file in file_paths]
    return dataframes

"""
Defining load_data_gpkg function
"""
def load_data_gpkg(file_paths):
    """
    Loads a list of GeoPackage (.gpkg) files into a list of GeoDataFrames and adds metadata.

    Each GeoDataFrame will include:
        - 'file_name': the original file path.
        - 'ROT_i': rotation index extracted from the file name (single character, 6th last position).

    Parameters:
    -----------
    file_paths : list of str
        List of file paths to .gpkg files.

    Returns:
    --------
    dataframes : list of geopandas.GeoDataFrame
        List containing a GeoDataFrame for each loaded GeoPackage file,
        with added metadata columns.
"""
    dataframes = []
    for file in file_paths:
        df = gpd.read_file(file)
        df['file_name'] = file  
        df['ROT_i'] = file[-6:-5]
        dataframes.append(df)
        
    return dataframes

"""
Defining filter_crops function
"""
def filter_crops(dataframes, min_count=100):
    """
    Filters crops that appear in all given dataframes with a minimum count of observations.

    The function counts the occurrences of each crop in each of the dataframes and retains only those crops
    that have an occurrence count greater than or equal to the specified threshold (min_count) in all dataframes.
    The result is a list of crops that meet this condition across all dataframes.

    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
        List of DataFrames in which the crop is analyzed. Each DataFrame must contain a 'crop' column
        that holds the crop names.

    min_count : int, optional
        The minimum threshold for the number of observations per crop in each dataframe. The default value is 100.

    Returns:
    --------
    valid_crops : list of str
        List of crops that appear in all dataframes with at least the specified number of observations (min_count).
    """
    crop_counts = {}
    
    for df in dataframes:
        crop_counts_df = df['crop'].value_counts().to_dict()
        for crop, count in crop_counts_df.items():
            if crop in crop_counts:
                crop_counts[crop].append(count)
            else:
                crop_counts[crop] = [count]
    
    valid_crops = [crop for crop, counts in crop_counts.items() if all(c >= min_count for c in counts)]
    return valid_crops

"""
Defining perform_kruskal_test function
"""
def perform_kruskal_test(dataframes, valid_crops):
    """
    Performs the Kruskal-Wallis H-test on the 'C_factor' values of valid crops across multiple dataframes.

    The function compares the distribution of the 'C_factor' values for each valid crop across all provided dataframes.
    The Kruskal-Wallis H-test is a non-parametric method used to determine if there are statistically significant differences
    between two or more independent groups.

    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
        List of DataFrames, each containing a 'crop' column and a 'C_factor' column. The 'crop' column is used to filter
        data for each valid crop, and 'C_factor' values are tested for statistical differences.

    valid_crops : list of str
        List of crops that are considered valid for testing. The function performs the Kruskal-Wallis test only on these crops.

    Returns:
    --------
    kruskal_results : dict
        A dictionary where the keys are the valid crops and the values are dictionaries containing:
        - 'H-statistic': the Kruskal-Wallis H-statistic.
        - 'p-value': the corresponding p-value for the test.
    """
    kruskal_results = {}
    
    for crop in valid_crops:
        crop_data = [df[df['crop'] == crop]['C_factor'].dropna().values for df in dataframes]
        
        if sum(len(data) > 0 for data in crop_data) > 1:
            h_stat, p_value = stats.kruskal(*crop_data)
            kruskal_results[crop] = {'H-statistic': h_stat, 'p-value': p_value}
    
    return kruskal_results

"""
Defining compute_phenology_metrics function
"""
def compute_phenology_metrics(group):
    """
    Computes key phenology metrics from vegetation time-series data.

    This function calculates the start of season (SOS), peak of season (POS), end of season (EOS), 
    length of season (LOS), amplitude, mean vegetation cover, and green-up rate (GUR) from a 
    time-series of vegetation values. The calculations are based on the 'value' column, which 
    represents the vegetation cover at each timestamp.

    Parameters:
    -----------
    group : pandas.DataFrame
        A DataFrame containing time-series data with at least two columns:
        - 'timestamp': the time or date of each observation.
        - 'value': the vegetation cover value at each timestamp.

    Returns:
    --------
    pd.Series
        A pandas Series with the following phenology metrics:
        - 'SOS': the start of the season (timestamp when the value first exceeds a threshold).
        - 'POS': the peak of season (timestamp of maximum vegetation cover).
        - 'EOS': the end of the season (timestamp when the value falls below a threshold).
        - 'LOS': the length of season (number of days between SOS and EOS).
        - 'Amplitude': the difference between the maximum and minimum vegetation cover.
        - 'Mean_Vegetation_Cover': the average vegetation cover across the time series.
        - 'Green_Up_Rate': the rate of green-up from SOS to POS (change in value per day).
    """
    group = group.sort_values('timestamp')
    
    max_value = group['value'].max()
    min_value = group['value'].min()
    amplitude = max_value - min_value
    mean_value = group['value'].mean()
    
    # Define SOS & EOS thresholds (e.g., 20% of max value)
    threshold = min_value + 0.2 * amplitude

    # Find SOS (first time value exceeds threshold)
    sos_row = group[group['value'] >= threshold].iloc[:1]
    sos = sos_row['timestamp'].min()
    sos_value = sos_row['value'].min()

    # Find POS (timestamp of maximum vegetation cover)
    pos_row = group.loc[group['value'].idxmax()]
    pos = pos_row['timestamp']
    
    # Find EOS (last time value falls below threshold)
    eos = group[group['value'] >= threshold]['timestamp'].max()
    
    # Compute Length of Season (LOS)
    los = (eos - sos).days if pd.notna(eos) and pd.notna(sos) else np.nan
    
    # Compute Green-Up Rate (GUR)
    gur = (max_value - sos_value) / (pos - sos).days if pd.notna(pos) and pd.notna(sos) and (pos - sos).days > 0 else np.nan
    
    return pd.Series({
        'SOS': sos,
        'POS': pos,
        'EOS': eos,
        'LOS': los,
        'Amplitude': amplitude,
        'Mean_Vegetation_Cover': mean_value,
        'Green_Up_Rate': gur
    })

"""
Defining the plot_compare_cfactor function
"""
def plot_compare_cfactor(df1, df2, df1_name='DF1', df2_name='DF2'):
    """
    Compares the mean C-factor between two datasets across different crop rotation types.

    This function calculates the mean C-factor for each crop rotation type in both input datasets
    and visualizes the comparison using a bar plot. The plot allows for easy comparison of the C-factor
    across rotations and between datasets, with the bars color-coded by dataset source.

    Parameters:
    -----------
    df1 : pandas.DataFrame
        The first DataFrame containing the 'rotation_crops' and 'C_factor' columns.
        
    df2 : pandas.DataFrame
        The second DataFrame containing the 'rotation_crops' and 'C_factor' columns.
        
    df1_name : str, optional, default='DF1'
        The name to label the first dataset in the plot legend.

    df2_name : str, optional, default='DF2'
        The name to label the second dataset in the plot legend.

    Returns:
    --------
    None
    """
    # Prepare data
    df1_means = df1.groupby('rotation_crops')['C_factor'].mean().reset_index()
    df1_means['source'] = df1_name
    
    df2_means = df2.groupby('rotation_crops')['C_factor'].mean().reset_index()
    df2_means['source'] = df2_name
    
    combined = pd.concat([df1_means, df2_means])
    
    # Create plot
    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='rotation_crops', y='C_factor', hue='source', 
                    data=combined, palette=['skyblue', 'salmon'])
    
    # Customize plot
    plt.xlabel('Rotation type')
    plt.ylabel('Standard deviation of C-factor')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

"""
Defining plot_population_outliers function
"""
def plot_population_outliers(df, value_col, Cfactor_col, time_col, object_id_col, crop_col):
    """
    Visualizes time series outliers in canopy cover per crop type using line plots.

    This function detects outliers in C-factor values within each crop population using the 
    interquartile range (IQR) method. It then visualizes the time series of both all parcels 
    and identified outlier parcels in separate subplots (up to 6 crops). Outliers are colored 
    differently and overlaid on the crop’s full dataset for comparison.

    Parameters:
    -----------
    df : pandas.DataFrame
        The dataset containing time series values, crop identifiers, and C-factor values.

    value_col : str
        The column name representing the variable to plot over time (e.g., canopy cover).

    Cfactor_col : str
        The column used to detect outliers (e.g., C-factor values).

    time_col : str
        The column representing the time dimension (e.g., day of year or date).

    object_id_col : str
        The column representing unique parcel or object IDs.

    crop_col : str
        The column representing crop type or population group.

    Returns:
    --------
    None
    """
    # Calculate outliers per crop population
    # Initialize empty DataFrame
    outliers = pd.DataFrame()
    
    # First pass to calculate  IQR-based bounds for each crop
    crop_bounds = {}
    for crop, group in df.groupby(crop_col):
        Q1 = group[Cfactor_col].quantile(0.25)
        Q3 = group[Cfactor_col].quantile(0.75)
        IQR = Q3 - Q1
        # Define lower and upper bounds for outlier detection (standard IQR rule)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        crop_bounds[crop] = (lower_bound, upper_bound)
    
    # Second pass to identify outliers and their type
    for crop, group in df.groupby(crop_col):
        lower_bound, upper_bound = crop_bounds[crop]
        # Identify outliers and their type
        crop_outliers = group[(group[Cfactor_col] < lower_bound) | (group[Cfactor_col] > upper_bound)].copy()
        crop_outliers['outlier_type'] = np.where(
            crop_outliers[Cfactor_col] < lower_bound, 
            'lower', 
            'upper'
        )
        # Append to outliers dataframe
        outliers = pd.concat([outliers, crop_outliers])
    # If no outliers were detected, exit early with message
    if outliers.empty:
        print("No outliers found in any population.")
        return
    
    # Plotting
    # Create figure with 6 subplots (2 rows, 3 columns)
    fig, axes = plt.subplots(3, 2, figsize=(22, 12))
    axes = axes.flatten()  # Flatten for easier iteration
    
    # Get unique crops (assuming there are exactly 6)
    unique_crops = df[crop_col].unique()
    if len(unique_crops) != 6:
        print(f"Warning: Expected 6 crops, found {len(unique_crops)}")
    
    # Plot each crop in its own subplot
    for i, crop in enumerate(unique_crops[:6]):  # Ensure we only plot 6 even if more exist
        ax = axes[i]
        # Get outlier and full data for current crop
        crop_outliers = outliers[outliers[crop_col] == crop]
        all_vals = df[df[crop_col] == crop]
        # Skip plotting outliers if none exist for this crop
        if crop_outliers.empty:
            ax.set_title(f"{crop}\n(No outliers)")
            continue
        # Plot time series of outlier parcels
        sns.lineplot(data = crop_outliers, x = time_col, y = value_col,
                     ax = axes[i], errorbar=('pi', 95), color = "blue",
                     label = "outliers C-factors")
        # Overlay time series of all parcels for context
        sns.lineplot(data = all_vals, x = time_col, y = value_col,
                     ax = axes[i], errorbar=('pi', 95), color = "green",
                     label = "all parcels")
        # Format subplot        
        ax.set_title(f"{crop}")
        ax.set_xlabel("time")
        ax.set_ylabel("Canopy cover")
        ax.grid(True)
        
        # Add legend if there are outliers
        if not crop_outliers.empty:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()