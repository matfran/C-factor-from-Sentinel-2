# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 18:37:16 2022

@author: u0133999
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn import metrics
from scipy import stats
import sys


def count_annual_observations(LPIS_all):
    '''
    Counts the number of image observations per year, per field parcel. Returns
    a dataframe with the annual counts oer parcel ID. 

    Parameters
    ----------
    LPIS_all : TYPE
        DESCRIPTION.

    Returns
    -------
    lpis_n_count : TYPE
        DESCRIPTION.

    '''
    #find  the column with the relevant id in iacs
    for col in LPIS_all.columns:
        if 'id' in col:
            id_col = col
        elif 'ID' in col:
            id_col = col 
            
    #function to count the number of annual observations per field parcel
    #take the columns with the dates
    dates = pd.Series(LPIS_all.columns.astype('str'))
    ref = dates.str.isnumeric()
    dates2 = dates[ref]
    #get a dataframe with only the timeseries 
    lpis = LPIS_all[dates2]
    #convert column strings to datetime
    lpis.columns = pd.to_datetime(lpis.columns)
    lpis_n_count = lpis.resample('Y', axis = 1, origin = 'epoch').count()
    cols = list(lpis_n_count.columns.astype('str'))
    new_cols = []
    for i in cols:
        i = i[:4] + '_obs_count'
        new_cols.append(i)
    lpis_n_count.columns = new_cols
    #add in the object id column for recognition 
    lpis_n_count[id_col] = LPIS_all[id_col]
    return lpis_n_count


def format_merge(df_ts, IACS_shp, iacs_merge_type, iacs_id = None, iacs_ts_id = None, 
                 merge_only = False):
    #if id strings are not specified they can be automatically found
    if iacs_id is not None:
        id_col = iacs_id 
    else:
        id_list = []
        #find  the column with the relevant id in iacs
        for col in IACS_shp.columns:
            if 'id' in col:
                id_col = col
                id_list.append(id_col)
            elif 'ID' in col:
                id_col = col 
                id_list.append(id_col)
    if iacs_id is not None:
        id_ts = iacs_ts_id 
    else:       
        for col in df_ts.columns:
            if 'id' in col:
                id_ts = col
                id_list.append(id_col)
            elif 'ID' in col:
                id_ts = col 
                id_list.append(id_col)
    if len(id_list) > 2:
        sys.exit('Too many id columns in the iacs shapefile and/or timeseries')
    id_col_new = 'OBJECTID'
    IACS_shp.rename(columns = {id_col : id_col_new}, inplace = True)
    df_ts.rename(columns = {id_ts : id_col_new}, inplace = True)

    #merge based on the id column
    if merge_only == False:
        
        del(df_ts['.geo'])
        del(df_ts['system:index'])
        df_ts = df_ts.replace(-9999, np.nan)
        df_ts = pd.DataFrame(IACS_shp.merge(df_ts, on = [id_col_new], how = iacs_merge_type))
        #set object id as index
        df_ts.index = df_ts[id_col_new].values
    else:
        df_ts = pd.DataFrame(IACS_shp.merge(df_ts, on = [id_col_new], how = iacs_merge_type))
        #set object id as index
        df_ts.index = df_ts[id_col_new].values
    return df_ts



def reformat_LS_cols(iacs):
    cols = iacs.columns 
    new_cols = []
    
    test_list = ['LT', 'LE']
    counter = 0
    for i in cols:
        counter = counter + 1
        res = any(ele in i for ele in test_list)
        if res == True:
            #take only the date component of landsat tile id
            new_cols.append(i[-8:])
        else:
            new_cols.append(i)
        
    iacs.columns = new_cols
        
    return iacs   

def add_harvest_cropres(df_fvc_re, harvest_inflexes):
    '''
    Add crop residue to the fractional vegetation cover array based on the
    identification of crop residues. Returns an array with the combined FVC
    and crop residue. 

    Parameters
    ----------
    df_fvc_re : Dataframe
        DESCRIPTION.
    harvest_inflexes : Dataframe
        DESCRIPTION.

    Returns
    -------
    df_fvc_re : Dataframe
        DESCRIPTION.

    '''
    
    harvest_inflexes_new = harvest_inflexes[['Estimated_harvest_date', 'Crop_res']]
    df_fvc_re_new = df_fvc_re.copy()
    #add a column for the next merging stage
    df_fvc_re_new['Estimated_harvest_date'] = df_fvc_re_new.index 
    #merge dataframes to give columns with crop res status at inflex point
    fc_merged = pd.merge_asof(df_fvc_re_new, harvest_inflexes_new, on = 'Estimated_harvest_date',
                              direction = 'nearest', tolerance = pd.Timedelta('8 days'))
    #identify periods of low vegetation (canopy cover < 0.3)
    fc_merged['Low_veg_period'] = np.where(fc_merged['gpr_pred'] < 0.3, 1, 0)
    #count number of time increments and multiply by 15 (days)
    harvest_period_total = fc_merged[fc_merged['Low_veg_period'] == 1]['Low_veg_period'].count() * 15
    #forward fill array with crop residue status (1 or 0), then fill any values preceeding inflex with 0
    fc_merged['Crop_res_cont'] = fc_merged['Crop_res'].fillna(method = 'ffill').fillna(0)
    #assign crop residue (0.3) during low veg periods where crop residues were identified
    #otherwise if high veg period or no crop res, assign 0 cover
    fc_merged['Crop_res_cov'] = np.where(np.logical_and(fc_merged['Crop_res_cont'] == 1, fc_merged['Low_veg_period'] == 1), 0.3, 0)
    #fill low veg-crop res periods with 0.3, otherwise assign gpr pred value
    fc_merged['Fractional_cov'] = np.where(np.logical_and(fc_merged['Crop_res_cont'] == 1, fc_merged['Low_veg_period'] == 1), 0.3, fc_merged['gpr_pred'])    
    #count number of days w crop residue
    harvest_period_w_res = fc_merged[fc_merged['Crop_res_cov'] == 0.3]['Crop_res_cov'].count() * 15
    
    #update the fractional cover variable to include the crop residue
    df_fvc_re['gpr_pred'] = fc_merged['Fractional_cov'].values
    
    return df_fvc_re, harvest_period_total, harvest_period_w_res

def FVC_to_SLR(fvc_ts, exponent): 
    #convert from fractional to %
    cc = fvc_ts * 100
    sl = np.exp(exponent * cc)
    return sl
    

def C_factor(df_fvc_re, exponent = -0.0492, return_slr_ts = False):
    '''
    Returns the C-factor from an annual series of vegetation cover and rainfall
    erosivity at 15-day intervals. The function follows the standard RUSLE 
    procedure.

    Parameters
    ----------
    df_fvc_re : DATAFRAME
        Dataframe containing columns 'gpr_pred' with vegetation cover and
        'EI30' with rainfall erosivity at 15 day intervals.

    Returns
    -------
    C_fact : FLOAT
        C-factor (0-1)

    '''
    vegetation_ts = df_fvc_re['gpr_pred']
    rainfall_EI = df_fvc_re['EI30']
    rainfall_EI_pcnt = (rainfall_EI / rainfall_EI.sum()) * 100
    sl = FVC_to_SLR(vegetation_ts, exponent)
    #page 187 of the RUSLE handbook
    #https://www.ars.usda.gov/arsuserfiles/64080530/rusle/ah_703.pdf
    C_fact = (sl * rainfall_EI_pcnt).sum()/rainfall_EI_pcnt.sum()
    #calculate the sl * ei timeseries
    sl_ei_ts = sl * rainfall_EI
    #calculate the sl * ei% timeseries (normalised)
    sl_ei_pcnt_ts = sl * rainfall_EI_pcnt
    
    if return_slr_ts == True:  
        return C_fact, sl_ei_ts, sl_ei_pcnt_ts
    else:
        return C_fact
    
def Calc_risk_period(df_fvc_re, index = 'month'):
    '''
    Retruns the time period of the highest erosion risk from a dataframe 
    containing a series of vegetation cover and rainfall erosivity at 15-day
    intervals. 

    Parameters
    ----------
    df_fvc_re : DATAFRAME
        Dataframe containing columns 'gpr_pred' with vegetation cover and
        'EI30' with rainfall erosivity at 15 day intervals. 
    index : STRING
        Specify whether to return month of highest risk (default) or '15-day' period (0-360)

    Returns
    -------
    Day_of_max_risk : INTEGER
        15-day interval containing highest erosivity risk.
    Month_of_year : INTEGER
        Month of the year (1-12) with highest erosivity risk.

    '''
    df_fvc_re['fvc_pcnt'] = (df_fvc_re['gpr_pred']/df_fvc_re['gpr_pred'].sum()) * 100
    df_fvc_re['RE_pcnt'] = (df_fvc_re['EI30']/df_fvc_re['EI30'].sum()) * 100
    #calculate a risk index based on the diff between the scaled RE and FVC
    df_fvc_re['risk_index'] = df_fvc_re['RE_pcnt'] - df_fvc_re['fvc_pcnt']
    #extract the day corresponding to the maximum risk 
    Day_of_max_risk = df_fvc_re['Day'][ df_fvc_re['risk_index'].idxmax()]
    EI30_risk = df_fvc_re['EI30'][ df_fvc_re['risk_index'].idxmax()]
    Month_of_year = df_fvc_re['risk_index'].idxmax().month
    
    #sum all the % of rainfall erosivity during the log vegetation period
    RE_pcnt_low_veg = float(df_fvc_re[df_fvc_re['gpr_pred'] <= 0.3]['RE_pcnt'].sum())
    
    if index == 'month':
        return Month_of_year, EI30_risk, RE_pcnt_low_veg
    elif index == '15-day':
        return int(Day_of_max_risk), EI30_risk, RE_pcnt_low_veg
        
        
def plot_slr(slr_ts, ndvi_count = None, n_fields = 20, id_col = 'object_id', 
             year_start = None, year_end = None, name = None):

    #slr_ts.columns = np.arange(1,(len(slr_ts.columns) * 15), 15)
    cols_1 = np.arange(1,(len(slr_ts.columns) * 15), 15)
    cols_2 = pd.to_datetime(slr_ts.columns, yearfirst = True).date
    slr_ts.columns = cols_2
    
    date_start = pd.to_datetime(str(year_start) + '/01/01', yearfirst = True)
    date_end = pd.to_datetime(str(year_end) + '/12/31', yearfirst = True)
    slr_ts = slr_ts.loc[:, (cols_2 >= date_start)&(cols_2 <= date_end)]
    
    
    median = slr_ts.median()
    upper = slr_ts.quantile(.95)
    lower = slr_ts.quantile(.05)

    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize = (22,6))

    i = 0
    for row in np.arange(len(slr_ts)):
        if i > n_fields:
            break
        #slr_ts.iloc[row].plot()
        ts = slr_ts.iloc[row]
        
        ax.plot(ts, alpha = 0.5, linewidth = 2)
        
        '''
        ax.fill_between(np.array(EnS_sample_grouped_low.index.values, dtype = float), EnS_sample_grouped_low[crop].values, EnS_sample_grouped_high[crop].values,
                         alpha=0.2, color= colours_dict[EnS])
        '''
        i = i + 1

    ax.plot(median, linewidth = 7, color = 'green', label = 'Median catchment condition', 
            alpha = 0.5)
    ax.fill_between(slr_ts.columns, lower.values, upper.values,
                     alpha=0.2, color = 'green', label = '5th and 95th percentiles')
    ax.set_xlabel('Date')
    ax.set_ylabel('Soil loss ratio (SLR)')

    if ndvi_count is not None:
        x = 0
        for n in ndvi_count.values:
            ax.annotate('n ndvi = ' + str(n), (x, 0.1))
            x = x + 365
    #fig.legend()
    #ax.set_title(crop)
    plt.savefig('C:/Users/u0133999/OneDrive - KU Leuven/PhD/Work_package_3/FIGURES/FIG4_'+ name +'.svg')
    
    
def plot_slr_kind(slr_ts, ndvi_count = None, n_fields = 20, id_col = 'object_id', 
                  n_bare_observations = None, year_start = None, year_end = None):

    
    cols_2 = pd.to_datetime(slr_ts.columns, yearfirst = True).date
    slr_ts.columns = cols_2
    
    date_start = pd.to_datetime(str(year_start) + '/01/01', yearfirst = True)
    date_end = pd.to_datetime(str(year_end) + '/12/31', yearfirst = True)
    slr_ts = slr_ts.loc[:, (cols_2 >= date_start)&(cols_2 <= date_end)]
    cols_1 = np.arange(1,(len(slr_ts.columns) * 15), 15)
    
    median = slr_ts.median()
    mean = slr_ts.mean()    
    upper = slr_ts.quantile(.95)
    lower = slr_ts.quantile(.05)

    plt.style.use('seaborn-darkgrid')
    plt.rcParams.update({'font.size': 20})
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"
    fig, ax = plt.subplots(figsize = (22,6))

    i = 0
    for row in np.arange(len(slr_ts)):
        if i > n_fields:
            break
        #slr_ts.iloc[row].plot()
        ts = slr_ts.iloc[row]
        
        ax.plot(ts, alpha = 0.5, linewidth = 2)
        
        '''
        ax.fill_between(np.array(EnS_sample_grouped_low.index.values, dtype = float), EnS_sample_grouped_low[crop].values, EnS_sample_grouped_high[crop].values,
                         alpha=0.2, color= colours_dict[EnS])
        '''
        i = i + 1

    ax.plot(median, linewidth = 7, color = 'green', label = 'Median catchment condition', 
            alpha = 0.5)
    ax.fill_between(slr_ts.columns, lower.values, upper.values,
                     alpha=0.2, color = 'green', label = '5th and 95th percentiles')
    
    if n_bare_observations is None:
        sys.exit('Provide Kinderveld observations')
    ax2 = ax.twinx()
    ax2.bar(pd.to_datetime(n_bare_observations.index), n_bare_observations['N_bare'], 
            width = 10, label = 'N bare field observations', fill = 'brown', 
            alpha = 0.5, edgecolor = 'k')
    ax2.grid(False)
    ax2.set_ylabel('N bare field observations')
    
    stats_ = pd.DataFrame({'mean': mean, 'median':median, '5th pcntl': lower, '95th pcntl':upper})
    stats_['n_days'] = cols_1.astype('int64')
    n_bare_observations['n_days'] = n_bare_observations['n_days'].astype('int64')
    stats_ = pd.merge_asof(n_bare_observations, stats_, on = 'n_days', direction = 'nearest')

    r2_median = stats.pearsonr(stats_['median'], stats_['N_bare'])
    r2_mean = stats.pearsonr(stats_['mean'], stats_['N_bare'])
    r2_95 = stats.pearsonr(stats_['95th pcntl'], stats_['N_bare'])
    print('R2 value for median slr: ' + str(r2_median))
    print('R2 value for mean slr: ' + str(r2_mean))
    print('R2 value for 95th pcntl slr: ' + str(r2_95))
    ax.set_xlabel('Day')
    ax.set_ylabel('Soil loss ratio')

    if ndvi_count is not None:
        x = 0
        for n in ndvi_count.values:
            ax.annotate('n ndvi = ' + str(n), (x, 0.1))
            x = x + 365
            
    fig.legend(bbox_to_anchor = [0.9, 0.9], frameon = False, framealpha = 0.9)
    plt.savefig('C:/Users/u0133999/OneDrive - KU Leuven/PhD/Work_package_3/FIGURES/FIG4_kinderveld.svg')
    
    f, ax = plt.subplots(figsize=(15, 7))
    sns.despine(f, left=True, bottom=True)
    sns.scatterplot(data = stats_, x = 'median', y = 'N_bare', ax=ax, palette="light:m_r")
    ax.set_xlabel('Median predicted catchment slr value')
    ax.set_ylabel('N observed bare field parcels')  


    
    #ax.set_title(crop)
    
def plot_histogram(df, col, x_label, out_f, min_ndvi_y):
    sns.set(font_scale = 3)
    f, ax = plt.subplots(figsize=(25, 15))
    sns.despine(f, left=True, bottom=True)
    sns.histplot(data = df, x= col, ax=ax, palette="light:m_r")
    ax.set_title('') 
    ax.set_xlabel(x_label)
    ax.set_ylabel('n field parcels')   
    outpath = os.path.join(out_f, col + '_' + min_ndvi_y + 'ndvi_per_year.png')
    plt.savefig(outpath)


