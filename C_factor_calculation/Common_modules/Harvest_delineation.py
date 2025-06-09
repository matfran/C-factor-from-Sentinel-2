# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 12:47:20 2022

Module with functions to identify inflection points (harvest periods) 
in vegetation cover time series data using differential analysis 
and generates a summary of NDVI and NDTI values surrounding those 
inflection points.

@author: Francis Matthews
"""
# Import modules
import sys
import math
import pandas as pd

"""
Defining identify_inflexions function
"""
def identify_inflexions(fvc_gpr, diff_array, LUCAS_specific = False, LUCAS_date = None, row_start = 0):
    '''
    Identify inflexion points (harvest periods) from vegetation cover timeseries.

    Parameters
    ----------
    fvc_gpr : PANDAS SERIES
        Vegetation timeseries (15-day) with timestamp indices.
    diff_array : NUMPY ND ARRAY
        Differential array (n1-n0) of vegetation timeseries.
    LUCAS_specific : BOOLEAN, optional
        Specify if timeseries are LUCAS and should consider the LUCAS survey date.
        The default is False.
    LUCAS_date : TIMESTAMP, optional
        The timestamp of the LUCAS observation. The default is None.
    row_start : INTEGER, optional
        Specify the index position from which inflexion points are sought. 
        The default is 0 ie starts from beginning of array.

    Returns
    -------
    inflex_date : TIMESTAMP or None
        Identified inflexion date (or None).
    harvest_identified : BOOLEAN
        True if a harvest point was detected, else False.
    row : INTEGER or None
        Index of the inflexion point in the original series (or None).
    inflex_end : INTEGER or None
        Index immediately after the inflexion point for next search (or None).
    '''
    
    # Check if LUCAS-specific processing is required without a LUCAS date
    if LUCAS_specific == True and LUCAS_date == None:
        sys.exit('A LUCAS survey date is required')

    # Start at index 1 to be able to evaluate position 0
    count = 1
    # If not starting from beginning of array ie finding second harvest, slice array 
    if row_start != 0:
        # Slice both arrays from the start row onwards
        fvc_gpr = fvc_gpr[row_start:]
        diff_array = diff_array[row_start:]
        # eg last harvest was the end of the array
        if len(diff_array) == 0:
            # Set row to None in advance of entering for loop
            row = None
            
    # Loop through the differential array to find inflexion points
    for x in diff_array:
        # Evaluate element, if nan, don't evaluate if it's an inflex point
        if math.isnan(x) == False:
            try:
                # If the last timesteps had a decreasing NDVI value eg before inflex point
                if diff_array[count - 1] < 0 and diff_array[count + 1] > 0:
                    # If inflexion point found, record the row index
                    row = count - 1
                    harvest_identified = True
                    inflex_date = fvc_gpr.index[row]
                    # If the inflex point is close to the LUCAS observation
                    # Otherwise keep searching
                    if LUCAS_specific == True and float(fvc_gpr.iloc[row]) <= 0.4:
                        if (inflex_date - LUCAS_date) > pd.Timedelta('-90 days +00:00:00'):
                            break
                    elif LUCAS_specific == False and float(fvc_gpr.iloc[row]) <= 0.4:
                        # If LUCAS survey date is not important, break loop and keep date 
                        break
                    # Increase the count to keep iterating 
                    count = count + 1
                else: 
                    # If no inflexion point, just increment count
                    count = count + 1
                    row = None
            except:
                # If error (eg nan value) skip to next array element 
                count = count + 1
                row = None 
                continue
        else:
            # If the current value is NaN, increment count and reset row
            count = count + 1
            row = None 
            
    # If all elements of array were nan, row will be None 
    if row == None:
        harvest_identified = False
        inflex_date = None
        inflex_end = None
    else:
        # Track the end position of the identified inflexion for the entire array
        inflex_end = row_start + count + 1
        # Adjust the row index if we started from a non-zero position
        row = row_start + row 
    
    return inflex_date , harvest_identified, row, inflex_end

"""
Defining the identify_multiple_harvests function
"""
def identify_multiple_harvests(fvc_gpr, npvc_gpr, obj_id):
    """
    Identifies multiple harvest periods from vegetation cover time series data.

    Parameters
    ----------
    fvc_gpr : PANDAS SERIES
        Time series of fractional vegetation cover data.
    npvc_gpr : PANDAS SERIES
        Time series of non photosynthetic vegetation cover data.
    obj_id : TYPE
        Unique identifier for the object being analyzed.

    Returns
    -------
    DATAFRAME
        DataFrame containing identified harvest inflexion points 
        and associated NDVI and NDTI metrics.        

    """
    # Calculate the difference between consecutive values in the FVC time series
    fvc_gpr_diff = fvc_gpr.diff()
    diff_array = fvc_gpr_diff.values.flatten()

    # List to store harvest inflexion data
    harvest_inflexes = []
    # Identify the first inflexion point in the time series
    inflex_date , harvest_identified, row, inflex_end = identify_inflexions(fvc_gpr, diff_array)
    # If harvest was identified, more could exist in ts
    if harvest_identified == True:
        # Take ndvi at inflexion point identified by the row variable
        ndvi_1 = float(fvc_gpr.iloc[row].values)
        # Take mean of ndvi 15 days either side of inflexion point 
        ndvi_2 = float(fvc_gpr.iloc[[row - 1,row,row + 1]].mean())
        # Take mean of all values below a certain ndvi cover
        ndvi_3 = float(fvc_gpr[:inflex_end].loc[fvc_gpr['gpr_pred'] < 0.4].mean())
        
        # Capture NDTI at the identified inflexion point
        ndti_1 = float(npvc_gpr.values[row])
        # Take ndti 15 days either side of ndvi inflexion point 
        ndti_2 = float(npvc_gpr.iloc[[row - 1,row,row + 1]].mean())
        # Take mean of ndti values where ndvi is low
        ndti_3 = float(npvc_gpr.loc[fvc_gpr['gpr_pred'] < 0.4].mean())                               

        # Append the data for the identified harvest to the list
        harvest_inflexes.append([obj_id, ndvi_1, ndvi_2, ndvi_3, ndti_1, ndti_2, ndti_3, harvest_identified, inflex_date])
        # Only iterate if harvest was found before and more potentially exist
        while harvest_identified == True:
            # Store index of where the array slice begins
            array_start = inflex_end + 1
            inflex_date , harvest_identified, row, inflex_end = identify_inflexions(fvc_gpr, diff_array, row_start = array_start)
            if harvest_identified == True:
                # Take ndvi at inflexion point identified by the row variable
                ndvi_1 = float(fvc_gpr.iloc[row].values)
                # Take mean of ndvi 15 days either side of inflexion point 
                ndvi_2 = float(fvc_gpr.iloc[[row - 1,row,row + 1]].mean())
                # Take mean of all values below a certain ndvi cover
                ndvi_3 = float(fvc_gpr[array_start:inflex_end].loc[fvc_gpr['gpr_pred'] < 0.4].mean())
                # Capture NDTI at the identified inflexion point
                ndti_1 = float(npvc_gpr.values[row])
                # Take ndti 15 days either side of ndvi inflexion point 
                ndti_2 = float(npvc_gpr.iloc[[row - 1,row,row + 1]].mean())
                # Take mean of ndti values where ndvi is low
                ndti_3 = float(npvc_gpr.loc[fvc_gpr['gpr_pred'] < 0.4].mean())    
                # Append the data for the newly identified harvest to the list
                harvest_inflexes.append([obj_id, ndvi_1, ndvi_2, ndvi_3, ndti_1, ndti_2, ndti_3, harvest_identified, inflex_date])
                    
        return pd.DataFrame(harvest_inflexes)

            
    
