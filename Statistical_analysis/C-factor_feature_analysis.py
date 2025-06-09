# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:35:38 2025

Statistical analysis of the C-factor outputs

@author: Francis Matthews & Arno kasprzak
"""
# Import modules
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import wkt
import os
import glob
import sys
sys.path.insert(0,r"Final_scripts\C_factor_calculation\Common_modules")
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from matplotlib.lines import Line2D  # For creating custom legend handles
from C_factor_functions import load_data, load_data_with_name, load_data_pkl,load_data_gpkg, filter_crops, perform_kruskal_test, compute_phenology_metrics, C_factor_box_plot
from C_factor_functions import plot_compare_cfactor, plot_population_outliers, C_factor_comparison_box_plot, plot_ts
import matplotlib.ticker as mticker

"""
Prepare Data
"""
# Choose if you want to export the visualisations
export = True

# Define root directory
root_ = "path to results"

# Define path to results annual C-factors
file_paths = glob.glob(os.path.join(root_, "C_factor_*.csv")) #C_factor csv files
parcel_info_paths = glob.glob(os.path.join(root_,"IACS_with_NDVI_*.csv")) #IACS parcel info csv files
file_paths_pkl = glob.glob(os.path.join(root_, "Result_*.pickle")) #Result pickle files
# Define path to results multi-year C-factors
file_paths_rot = glob.glob(os.path.join(root_, "C_factor_roty*.csv")) # CSV C-factor file per year 
file_paths_rot_p = glob.glob(os.path.join(root_, "Rotation_data_y*.gpkg")) # Geopackage of the parcels and rotation info per year
file_paths_rot2 = glob.glob(os.path.join(root_, "C_factor_roty*_extra.csv")) # CSV C-factor file per year for 3 extra rotations
# Load csv file with legend to couple rotation IDs to rotation names
rot_ref_p = os.path.join(root_, "RotID_legend.csv")

# Load the data
dataframes = load_data(file_paths)
iacs_dataframes = load_data(parcel_info_paths)
dataframes_pkls = load_data_pkl(file_paths_pkl)
dataframes_rot = load_data_with_name(file_paths_rot + file_paths_rot2)
geodataframes_rot = load_data_gpkg(file_paths_rot_p)
rot_ref = pd.read_csv(rot_ref_p)
rot_ref.columns = ['rotations', 'rotation_crops']

# Only do analysis on common crops: apply filter_crops function
valid_crops = filter_crops(dataframes)

# Process single-year C-factor data
# Merge data of the different regions
all_data = pd.concat(dataframes).reset_index()
# Remove parcels where C-factor = 0
all_data = all_data[all_data['C_factor'] != 0]
# Convert WKT geometry strings to shapely geometry objects
all_data['geometry'] = all_data['geometry'].apply(wkt.loads)
# Convert the DataFrame into a GeoDataFrame with the specified coordinate reference system (CRS)
all_data = gpd.GeoDataFrame(all_data, geometry='geometry', crs="EPSG:3035")

# IACS parcel info data with NDVI ts
iacs = pd.concat(iacs_dataframes).reset_index()

# Calculate metrics
all_data['area'] = all_data.geometry.area
all_data['perimeter'] = all_data.geometry.length
# Compactness (ratio of area to the area of a circle with the same perimeter)
all_data['compactness'] = (4 * np.pi * all_data['area']) / (all_data['perimeter'] ** 2)

# Process crop rotation data
all_data_rot = pd.concat(dataframes_rot).reset_index().sort_values('object_id')
# Concatenate spatial GeoDataFrames and select relevant columns
rot_gdf = pd.concat(geodataframes_rot)[['REF_ID','link_id', 'ROT_i', 'geometry']].reset_index()
# Set object_id for spatial join consistency
rot_gdf['object_id'] = rot_gdf['REF_ID']
# Merge crop data with geometry using ROT_i and object_id as keys
all_data_rot = all_data_rot.merge(rot_gdf, on = ['ROT_i', 'object_id'], how = 'left')
# Filter to keep only links with exactly 3 associated crop records
all_data_rot = all_data_rot[
    all_data_rot.groupby('link_id')['link_id'].transform('size') == 3
]

# Merge with rotation reference table to add descriptive rotation info
all_data_rot = all_data_rot.merge(rot_ref, on = 'rotations', how = 'left')

# Crop mapping: this way crops are plotted without _
crop_mapping = {
    'Maize_silage': 'Silage maize',
    'Maize': 'Grain maize',
    'Potatoes_late': 'Potatoes (not early)',
    'Barley': 'Winter barley',
    'Wheat': 'Winter wheat'
}


"""
CALCULATE STAISTICS OF SINGLE YEAR C-FACTORS
STUDY THE EFFECTS OF CROP RESIDUE BASED ON MEAN VEGETATION CLASSES
"""
# Add a column with the quartiles that the C-factor belongs to on a per-crop basis (4)
all_data['C_factor_quantile'] = all_data.groupby('crop')['C_factor'].transform(
    lambda x: pd.qcut(x, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
)

# Add a column with the deciles that the C-factor belongs to on a per-crop basis (10)
all_data['C_factor_quantile_10'] = all_data.groupby('crop')['C_factor'].transform(
    lambda x: pd.qcut(x, 10, labels=['Q' + str(i) for i in np.arange(1,11)])
)

# Add residue presence indicator and quantile-based classifications for mean annual FVC
all_data["Crop_Res"] = np.where(all_data['harvest_period_total_with_residues'] > 0, 1, 0)
# Compute crop-specific quartile classification of mean annual FVC
all_data['Mean_FVC_quantile'] = all_data.groupby('crop')['mean_annual_fvc'].transform(
    lambda x: pd.qcut(x, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
)
# Compute crop-specific decile classification of mean annual FVC
all_data['Mean_FVC_quantile_10'] = all_data.groupby('crop')['mean_annual_fvc'].transform(
    lambda x: pd.qcut(x, 10, labels=['Q' + str(i) for i in np.arange(1,11)])
)

# Compute the mean of all numeric columns for each crop type
mean_crops_all = all_data.groupby('crop').mean()
# Compute the median of all numeric columns for each crop type
median_crops_all = all_data.groupby('crop').median()
# Compute the mean of all numeric columns, grouped by crop, vegetation quantile, and residue presence
mean_crops_res = all_data.groupby(['crop', 'Mean_FVC_quantile', "Crop_Res"]).mean()

# Calculate the percentage of data for each 'Crop_Res' class per crop
percentages = all_data.groupby(['crop', 'Crop_Res']).size().unstack(fill_value=0)
percentages = percentages.div(percentages.sum(axis=1), axis=0) * 100  # Convert to percentages

"""
CALCULATE STAISTICS OF MULTI YEAR C-FACTORS
"""
# Function to check for repeated characters (case-sensitive)
def has_repeated_characters(s):
    """
    Checks whether a string represents a monoculture by verifying if all characters are the same.

    Parameters:
    -----------
    s : str
        A string representing the crop rotation (e.g., 'AAAA', 'ABAB').

    Returns:
    --------
    int
        Returns 1 if the string represents a monoculture (i.e., all characters are the same),
        otherwise returns 0.
    """
    # Return 1 if monoculture
    return int(len(set(s)) == 1)  # 1 if repeated, 0 otherwise

# Apply the function to the string column
all_data_rot['monoculture'] = all_data_rot['rotation_crops'].apply(has_repeated_characters)

# Compute overall rotation-level statistics (mean, median, standard deviation)
mean_rot_all = all_data_rot.groupby('rotation_crops').mean()
median_rot_all = all_data_rot.groupby('rotation_crops').median()
std_rot_all = all_data_rot.groupby('rotation_crops').std()

# Compute parcel-level statistics: mean and std for each rotation-parcel combination
mean_rot_all_p = all_data_rot.groupby(['rotation_crops', 'link_id'], 
                                      as_index = False).mean()

std_rot_all_p = all_data_rot.groupby(['rotation_crops', 'link_id'], 
                                      as_index = False).std()

# Calculate variability within rotations: mean of standard deviations across parcels
mean_rot_std = std_rot_all_p.groupby('rotation_crops', 
                                      as_index = False).std()

# Calculate variability between parcels: std of parcel means within each rotation
std_rot_mean = mean_rot_all_p.groupby('rotation_crops', 
                                      as_index = False).std()

# Compare different standard deviations
plot_compare_cfactor(mean_rot_std, std_rot_mean, df1_name='Mean std within parcel rotation', df2_name='Mean std between parcel rotations')

# Violin Plot: Monoculture vs. Crop Type
# Create FVC Quantiles within Rotations
mean_rot_all_p['Mean FVC quantile'] = mean_rot_all_p.groupby('rotation_crops')['mean_annual_fvc'].transform(
    lambda x: pd.qcut(x, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
)

# # Calculate median values of variables per crop and monoculture status
mean_culture_type = all_data_rot.groupby(['crop', 'monoculture']).median()

# Fist violin plot (Monocultures)
fig, ax = plt.subplots(figsize = (15,6))
sns.violinplot(all_data_rot, x = 'crop', y = 'C_factor', hue = 'monoculture', ax = ax,
               palette = "Set3")
ax.set_ylabel('C-factor value (single year)')
ax.set_xlabel('Crop name')
# Save figure if export is enabled
if export == True:
    plt.savefig(os.path.join(root_,r"C:\Users\arnok\Downloads\fig1.png"), dpi = 300)

# Second violin plot (Specific rotations)
rots = ['mmm', 'pmm', 'mmw', 'fff','pmw']
fig, ax = plt.subplots(2, 1, figsize = (15,12))
# Violin plot of C-factor per selected rotation type, colored by FVC quantile
sns.violinplot(mean_rot_all_p[mean_rot_all_p['rotation_crops'].isin(rots)], x = 'rotation_crops', y = 'C_factor', 
               hue = 'Mean FVC quantile', ax = ax[1],
               palette = "Set2")
ax[1].set_ylabel('C-factor value (3-year rotation)')
ax[1].set_xlabel('Crop name')

# Boxplot of C-factor per selected rotation type
sns.boxplot(mean_rot_all_p[mean_rot_all_p['rotation_crops'].isin(rots)], x = 'rotation_crops', y = 'C_factor', 
               ax = ax[0],
               palette = "Set2")
ax[0].set_ylabel('C-factor value (3-year rotation)')
ax[0].set_xlabel('Crop name')

# Save figure if export is enabled
if export == True:
    plt.savefig(os.path.join(root_,r"C:\Users\arnok\Downloads\fig2.png"), dpi = 300)

"""
COMBINE TIME SERIES INFORMATION
"""

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the data
all_data['crop_code'] = label_encoder.fit_transform(all_data['crop'])

# Concatenate 'IACS with NDVI ts' DataFrames from the list of dictionaries
concatenated_df = pd.concat([d['IACS with NDVI ts'] for d in dataframes_pkls], ignore_index=True)
# Create a standard column name for merging later
concatenated_df['object_id'] = concatenated_df['OBJECTID']
# Identify columns with purely numeric names 
numeric_cols = [col for col in concatenated_df.columns if col.isdigit()]
# Count number of non-NaN values across time steps for each row
concatenated_df['Non-NaN Count'] = concatenated_df[numeric_cols].notna().sum(axis=1)

# Concatenate time series data for GPR FVC from all dataframes
concatenated_ts = pd.concat([d['GPR FVC'] for d in dataframes_pkls], ignore_index=True)
# Concatenate SLR RE time series (single year) data 
concatenated_slr_re = pd.concat([d['SLR RE ts (one year)'] for d in dataframes_pkls], ignore_index=True)

# Merge GPR FVC time series data into main dataset using object_id as key
all_data = all_data.merge(concatenated_ts, how = 'left', on = 'object_id')
# Merge additional NDVI-related and NUTS2 feature columns into main dataset
all_data = all_data.merge(concatenated_df[['object_id','Non-NaN Count',  'NUTS2_conventional',
                                           'NUTS2_conservation', 'NUTS2_notill']], how = 'left', on = 'object_id')



"""
PLOT FIGURES
"""
#-----Boxplot per crop---------
Cfactor_filtered = all_data[all_data['crop'].isin(valid_crops)]
Cfactor_filtered['crop'] = Cfactor_filtered['crop'].replace(crop_mapping)

# Creat boxplot
C_factor_box_plot(Cfactor_filtered)

#-----Plot relationships-------
# Plot 1: C-factor vs variable
# Define variables
# y variable
y_var = 'C_factor'  
# Different x variables
x_vars = ['Non-NaN Count', 'mean_annual_fvc', 'harvest_period_total', 'Month_highest_risk',
          'EI30_risk_period', 'RE_%_low_veg']
# Names related to the x variables
x_names = ['N Sentinel-2 observations', 'Mean canopy cover (CC) in the year', 'Days with low canopy cover (< 30 %)', 
           'Month with highest erosion risk', 'Rainfall erosivity in 15-day period with highest exposure', 
           '% of annual rainfall erosivity in low vegetation period']

# Colors for the different plots
colours = ["slategrey", 'lightgreen', 'sienna', 'red', 'blue', 'blue']
# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

count = 0
# Loop through variables and plot each
for ax, x_var in zip(axes.flat, x_vars):
    sns.histplot(data=all_data, x=x_var, y=y_var, bins=50, cbar=True, ax=ax, color = colours[count])
    ax.set_xlabel(x_names[count],fontsize=14)
    ax.set_ylabel("C-factor",fontsize=14)    
    count += 1

# Adjust layout
plt.tight_layout()
if export == True:
    plt.savefig(os.path.join(root_,r"C:\Users\arnok\Downloads\fig3.png"), dpi = 300)
plt.show()

# Plot 2: Crop specific statistics 
# Define variables
# x variable
x_var = 'crop'  
# y variable
y_vars = ['Non-NaN Count', 'mean_annual_fvc', 'harvest_period_total', 'Month_highest_risk',
          'EI30_risk_period', 'RE_%_low_veg']
# y variable names
y_names = ['N Sentinel-2 observations', 'Mean canopy cover (CC) in the year', 'Days with low canopy cover (< 30 %)', 
           'Month with highest erosion risk', 'Rainfall erosivity in 15-day period with highest exposure', 
           '% of annual rainfall erosivity in low vegetation period']

# Get the six most common crop types
top_crops = all_data['crop'].value_counts().nlargest(6).index

# Filter the dataframe to only include those crops
df_filtered = all_data[all_data['crop'].isin(top_crops)]
df_filtered1 = df_filtered.copy()
df_filtered1['crop'] = df_filtered1['crop'].replace(crop_mapping)

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2 rows, 3 columns

# Loop through variables and plot each
count = 0
for ax, y_var in zip(axes.flat, y_vars):
    sns.violinplot(data=df_filtered1, x=x_var, y=y_var, ax=ax, inner="box", palette="Set3",scale='width')
    ax.set_ylabel(y_names[count], fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate labels for better readability
    ax.tick_params(axis='x', labelsize=12)
    count += 1

# Adjust layout
plt.tight_layout()
if export == True:
    plt.savefig(os.path.join(root_, r"C:\Users\arnok\Downloads\fig4.png"), dpi = 300)
plt.show()

# Plot 3: RE vs C-factor per crop
## Define a color palette, e.g., ['r', 'g', 'b', 'c', 'm', 'y']
colors = sns.color_palette('husl', len(valid_crops))  # Or define your own colors

# Loop through each crop and create a separate figure
for crop, color in zip(valid_crops, colors):
    crop_data = all_data[all_data['crop'] == crop]
    display_name = crop_mapping.get(crop, crop)
    # Create a new figure
    plt.figure(figsize=(7, 5))
    # Scatter plot with a unique color for each crop
    sns.scatterplot(x='RE_%_low_veg', 
                    y='C_factor', 
                    data=crop_data, 
                    color=color)  # Assign a unique color to each crop
    # Set title and labels
    plt.title(f'{display_name}', fontsize=14)
    plt.xlabel('High RE during Low Vegetation Periods (%)', fontsize=12)
    plt.ylabel('C-factor', fontsize=12)
    # Set y-axis limits between 0 and 0.8
    plt.ylim(0, 0.8)
    # Enable grid
    plt.grid(True)
    # Hide the legend
    plt.legend([],[], frameon=False)  
    # Show the plot
    plt.show()


"""
CROP RESIDUES PREDICTED 
"""
# Detect regions from file paths, as before
region_labels = []
for path in file_paths:
    for region in ['Campine', 'Sand', 'SandyLoam', 'Loess', 'Polders']:
        if region.lower() in path.lower():
            region_labels.append(region)
            break
# Prepare one combined DataFrame with region info
residue_data = []

# Loop through each region: merge C-factor and IACS data, assign region label
for df, iacs_df, region in zip(dataframes, iacs_dataframes, region_labels):
    # Merge the C-factor data with the corresponding IACS data using the parcel ID as key
    merged_df = pd.merge(df, iacs_df, left_on="object_id", right_on="OBJECTID", how="inner")
    
    # Only keep relevant crops
    merged_df = merged_df[merged_df["crop_x"].isin(valid_crops)].copy()

    # Ensure residue column exists and fill missing values
    if 'harvest_period_total_with_residues' not in merged_df.columns:
        continue
    merged_df['harvest_period_total_with_residues'] = merged_df['harvest_period_total_with_residues'].fillna(0)
    
    # Add region and binary indicator for residue presence
    merged_df["Residues_detected"] = (merged_df["harvest_period_total_with_residues"] > 15).astype(int)
    merged_df["Region"] = region
    merged_df["survey_name"] = merged_df["crop_x"].map(crop_mapping)

    residue_data.append(merged_df)

# Combine all regional residue data
residue_df = pd.concat(residue_data, ignore_index=True)

# Count residue presence per crop and region
residue_counts = residue_df.groupby(['survey_name', 'Region'])['Residues_detected'].sum().reset_index()
total_counts = residue_df.groupby(['survey_name', 'Region']).size().reset_index(name='total')
residue_percentages = pd.merge(residue_counts, total_counts, on=['survey_name', 'Region'])
residue_percentages['percentage'] = (residue_percentages['Residues_detected'] / residue_percentages['total']) * 100

# Plot percentage of parcels with residue presence by crop and region
plt.figure(figsize=(10, 6))
sns.barplot(
    data=residue_percentages,
    x='survey_name',
    y='percentage',
    hue='Region',
    palette='pastel'
)

plt.xlabel("Crop")
plt.ylabel("Parcels with residues detected (%)")
plt.xticks(rotation=45)
plt.ylim(0, 100)
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
if export == True:
    plt.savefig(os.path.join(root_, r"C:\Users\arnok\Downloads\fig5.png"), dpi = 300)
plt.show()

"""
ANALYSE THE IMPACT OF CROP RESIDUES PER FRACTIONAL VEGETATION QUANTILE
"""
# Get the unique crops
crops = df_filtered1['crop'].unique()
num_crops = len(crops)

# Create a figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # 2 rows, 3 columns for 6 crops
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

# Define a color palette for FVC_quantile
palette = ['firebrick', 'orange', 'gold', 'forestgreen']

# Create a dictionary to store legend handles and labels
legend_handles = []
legend_labels = []

# Loop through each crop and plot its empirical cummulative distribution function (ECDF) in a separate subplot
for i, crop in enumerate(crops):
    subset = df_filtered1[df_filtered1['crop'] == crop]
    ax = axes[i]
    
    # Iterate through each FVC_quantile and Crop_Res combination
    for j, fvc_q in enumerate(sorted(subset['Mean_FVC_quantile'].unique())):
        for k, crop_res in enumerate(sorted(subset['Crop_Res'].unique())):
            # Filter the data for the current FVC_quantile and Crop_Res
            data = subset[(subset['Mean_FVC_quantile'] == fvc_q) & (subset['Crop_Res'] == crop_res)]
            
            # Plot the ECDF for this combination
            line = sns.ecdfplot(
                data=data,
                x='C_factor',  # The variable to plot on the x-axis
                color=palette[j],  # Use the color corresponding to FVC_quantile
                linestyle='--' if crop_res == 1 else '-',  # Use dashed lines for Crop_Res=1, solid for Crop_Res=0
                lw=4,  # Line width
                ax=ax  # Plot in the corresponding subplot
            )
            
            # Store legend handles and labels (only once)
            if i == 0:  # Only add legend entries for the first subplot
                # Create a custom legend handle with explicit color and line style
                handle = Line2D(
                    [], [],
                    color=palette[j],  # Explicitly set the color
                    linestyle='--' if crop_res == 1 else '-',  # Explicitly set the line style
                    lw=2  # Line width
                )
                legend_handles.append(handle)
                legend_labels.append(f'FVC_q={fvc_q}, Crop_Res={crop_res}')
    
    name = crop.replace("_", "")
    ax.set_title(f'{crop}')  # Set title for the subplot
    ax.set_xlabel('C factor value per parcel')  # Set x-axis label
    ax.set_ylabel('Cumulative Probability')  # Set y-axis label

# Create a single legend for the entire figure
fig.legend(
    handles=legend_handles,
    labels=legend_labels,
    title='CC quantile & Residue',
    bbox_to_anchor=(1.05, 1),  # Place the legend outside the plot
    loc='upper left'
)

# Adjust layout and show the plot
plt.tight_layout()
if export == True:
    plt.savefig(os.path.join(root_, r"C:\Users\arnok\Downloads\fig4.png"),dpi=300, bbox_inches='tight')
plt.show()


"""
SHOW TIME SERIES RISK PER CROP
"""
# Filter dataset to include only the top crops
slr_re_top_crops = concatenated_slr_re[concatenated_slr_re["crop"].isin(top_crops)]
slr_re_top_crops['crop'] = slr_re_top_crops['crop'].replace(crop_mapping)
# Ensure the 'Day' column is in integer format
slr_re_top_crops['Day'] = slr_re_top_crops['Day'].astype(int)
fig, axes = plt.subplots(3,2, figsize=(20, 18))  # 2 rows, 3 columns
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
# Increase overall text size using rcParams
plt.rcParams.update({
    'font.size': 14,  # Default font size for text
    'axes.titlesize': 16,  # Font size for subplot titles
    'axes.labelsize': 15,  # Font size for x and y axis labels
    'xtick.labelsize': 13,  # Font size for x-axis tick labels
    'ytick.labelsize': 13,  # Font size for y-axis tick labels
    'legend.fontsize': 12,  # Font size for legend
})

# Loop through each crop and plot its time series in a separate subplot
for i, crop in enumerate(crops):
    subset = slr_re_top_crops[slr_re_top_crops['crop'] == crop]
    sns.boxplot(data = subset, y = 'SLR_EI_ts', x = "Day", 
                ax=axes[i], saturation = 0.1)
    axes[i].set_title(f'Crop: {crop}')  # Set title for the subplot
    axes[i].set_xlabel('Day of the year')  # Set x-axis label
    axes[i].set_ylabel('Distrbution of erosion exposure (EI30 (15-day) x SLR)', fontsize = 12)  # Set y-axis label
    axes[i].tick_params(axis='y', labelsize=14)
    axes[i].tick_params(axis='x', labelsize=12)
# Adjust layout to prevent overlaps
plt.tight_layout()
if export == True:
    plt.savefig(os.path.join(root_, r"C:\Users\arnok\Downloads\fig6.png"),dpi=300)
plt.show()

"""
PRODUCE TIME SERIES PLOTS OF THE 6 MOST COMMON CROPS, SPLIT BY THEIR C-FACTOR 
QUANTILE GROUP. DIVIDE INTO 4 QUANTILES PER CROP.
"""
# Function to check if a column name can be parsed as a datetime
def is_timestamp_column(column_name):
    """
    Checks whether a given value or string can be interpreted as a timestamp.

    This function attempts to convert the input to a pandas datetime object. 
    If the conversion is successful, it returns True, indicating that the 
    input is likely a timestamp or date-like string. If not, it catches the 
    exception and returns False.

    Parameters:
    -----------
    column_name : str
        The input value (typically a string) that is checked to see if it 
        represents a valid datetime or timestamp.

    Returns:
    --------
    bool
        True if the input can be parsed as a timestamp, otherwise False.
    """
    try:
        pd.to_datetime(column_name)
        return True
    except ValueError:
        return False

# Identify timestamp columns
timestamp_cols = [col for col in df_filtered.columns if is_timestamp_column(col)]

# Convert timestamp columns to datetime format (for plotting)
timestamps = pd.to_datetime(timestamp_cols)

# Reshape the dataframe from wide to long format
df_long = pd.melt(df_filtered, id_vars=['object_id', 'crop', 'C_factor_quantile', 'C_factor'], value_vars=timestamp_cols, 
                  var_name='timestamp', value_name='value')
df_long['timestamp'] = pd.to_datetime(df_long['timestamp'])

# Resample to monthly frequency
df_long.set_index('timestamp', inplace=True)
df_resampled = df_long.groupby(['crop', 'C_factor_quantile']).resample('15D').agg({'value': ['mean', 'std']}).reset_index()

# Flatten the multi-level columns
df_resampled.columns = ['crop', 'C_factor_quantile', 'timestamp', 'mean', 'std']

# Get unique crops
crops = df_long['crop'].unique()

# Create a figure with 6 subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 15))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration
colours = ['green', 'yellow', 'orange', 'red']

# Loop through each crop and plot
for i, crop in enumerate(crops):
    ax = axes[i]
    # Filter data for the current crop
    crop_data = df_resampled[df_resampled['crop'] == crop]
    display_name = crop_mapping.get(crop, crop)
    # Plot each quantile group
    for j, quantile in enumerate(crop_data['C_factor_quantile'].unique()):
        c = colours[j]
        quantile_data = crop_data[crop_data['C_factor_quantile'] == quantile]
        ax.plot(quantile_data['timestamp'], quantile_data['mean'], label=f'{quantile} Mean',
                color = c)
        ax.fill_between(quantile_data['timestamp'], 
                        quantile_data['mean'] - quantile_data['std'], 
                        quantile_data['mean'] + quantile_data['std'], 
                        alpha=0.2, label=f'{quantile} Std Dev', color = c)
    # Set title and labels
    ax.set_title(f'Canopy Cover Time Series for {display_name} per C-factor Quantile')
    ax.set_xlabel('Time')
    ax.set_ylabel('Canopy cover')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=10)

# Adjust layout and show plot
plt.tight_layout()
if export == True:
    plt.savefig(os.path.join(root_, r"C:\Users\arnok\Downloads\fig7.png"),dpi=300)
plt.show()

"""
PLOT OUTLIER VALUES - i.e. very low and very high C-factors
"""
# Resample the time series data to 15-day intervals for each parcel, crop, and C-factor
df_resampled_parcel = df_long.groupby(['object_id', 'crop',  'C_factor']).resample('15D').agg({'value': 'mean'}).reset_index()
df_resampled_parcel=df_resampled_parcel[df_resampled_parcel["crop"].isin(top_crops)]
df_resampled_parcel['crop'] = df_resampled_parcel['crop'].replace(crop_mapping)

# Compute phenology metrics per object_id
phenology_metrics = df_resampled_parcel.groupby('object_id').apply(compute_phenology_metrics).reset_index()

# Plot outliers in the resampled data
plot_population_outliers(
    df_resampled_parcel, 
    "value", 
    "C_factor", 
    "timestamp", 
    "object_id", 
    "crop"
)

"""
STATISTICAL ANALYSIS OF THE REGIONAL DIFFERENCE IN C-FACTOR
"""
# ----------Statistical test-----------------
# Perform Kruskal-Wallis test => check if regional differences are significant
kruskal_results = perform_kruskal_test(dataframes, valid_crops)

# Print output for the most common crops
for crop, result in kruskal_results.items():
    print(f"Crop: {crop}, H-statistic: {result['H-statistic']:.4f}, p-value: {result['p-value']:.2e}")
    if result['p-value'] < 0.05:
        print(f"=> Significant difference in C-factor across all datasets for {crop}!")

# -----------Plot result per region-------------
# Get region names from filenames
region_labels = []
for path in file_paths:
    filename = os.path.basename(path)
    matched = False
    for region in ['Campine', 'SandyLoam', 'Sand', 'Loess', 'Polders']:
        if f"C_factor_{region}".lower() in filename.lower():
            region_labels.append(region)
            matched = True
            break
# Prepare filtered C-factor DataFrames for each region
Cfactor_filtered_list = []
region_name_list = []

# Loop over each C-factor dataframe, matching IACS data and the corresponding region name
for df, iacs_df, region in zip(dataframes, iacs_dataframes, region_labels):
    # Merge the C-factor data with the IACS data based on the parcel ID
    merged_df = pd.merge(df, iacs_df, left_on="object_id", right_on="OBJECTID", how="inner")
    # Filter the merged data to only keep valid crops used in the analysis
    filtered = merged_df[merged_df['crop_x'].isin(valid_crops)]
    filtered['crop_x'] = filtered['crop_x'].replace(crop_mapping)
    # Add region column
    filtered["Region"] = region
    Cfactor_filtered_list.append(filtered)
    region_name_list.append(region)
    
# Plot regional comparison
C_factor_comparison_box_plot(Cfactor_filtered_list, [f"{r} region" for r in region_name_list])

#--------plot fvc timeseries per region ------------
# Choose crop which you want to investigate
crop = 'Maize_silage'
# List to store merged DataFrames for each region
Cfactor_with_fvc = []
# Loop over each region and corresponding data
for df, region, pkl_data in zip(dataframes, region_labels, dataframes_pkls):
    # Extract GPR FVC time series from the pickle data
    gpr_fvc_df = pkl_data["GPR FVC"]
    # Merge C-factor data with GPR FVC on 'object_id'
    merged = pd.merge(df, gpr_fvc_df, on="object_id", how="left")
    # Add region label
    merged["Region"] = region
    merged = merged[merged["crop"]==crop]
    merged['crop'] = merged['crop'].replace(crop_mapping)
    # Store the merged DataFrame
    Cfactor_with_fvc.append(merged)
    merged = merged.drop(merged.columns[1:15].tolist() + [merged.columns[-1]], axis=1)
    plot_ts(merged, n_fields=0, title= f"{region}", name="Fractional Vegetation Cover")

# Plot: Histogram showing the percentage of silage maize fields belonging to a specific % RE category
# Choose crop
crop = 'Maize_silage'
# Filter based on selected crop
data = all_data[all_data['crop'] == crop]
min_value = data['RE_%_low_veg'].min()
max_value = data['RE_%_low_veg'].max()
q1 = data['RE_%_low_veg'].quantile(0.33)  
q3 = data['RE_%_low_veg'].quantile(0.66)  
# Categories
categories = ["Low", "Moderate", "High"]
# Dictionary to store category counts per region
region_category_counts = {}
# Loop over each file and process it
for df, region in zip(dataframes, region_labels):
    df = df[df['crop'] == crop]
    # Function to assign categories based on thresholds
    def categorize(value):
        if value < q1:
            return "Low"
        elif value < q3:
            return "Moderate"
        else:
            return "High"
    # Apply the categorization function to a new column
    df['RE_category'] = df['RE_%_low_veg'].apply(categorize)
    # Count occurrences of each category
    category_counts = df['RE_category'].value_counts(normalize=True)  # Relative frequency
    region_category_counts[region] = [category_counts.get(cat, 0) for cat in categories]

# Convert dictionary to DataFrame for easier plotting
df_hist = pd.DataFrame(region_category_counts, index=categories)

# Plot grouped bar chart with relative frequencies as percentages
ax = df_hist.T.mul(100).plot(kind="bar", stacked=False, figsize=(10, 6), colormap="viridis")  # Convert to percentage

# Labels and formatting
plt.xlabel("Region")
plt.ylabel("Relative Frequency (%)")  # Display percentage on y-axis
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.7)
# Move the legend outside the plot
plt.legend(title="REI during low veg periods", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14)  
# Adjust layout to fit everything
plt.tight_layout()
if export == True:
    plt.savefig(r"C:\Users\arnok\Downloads\fig8.png", dpi = 300)
# Show plot
plt.show()

"""
ANALYSIS OF THE METHODOLOGICAL IMPACTS
"""
# === LOAD DATASETS FOR DIFFERENT METHODOLOGICAL SCENARIOS ===
methods_paths = glob.glob(os.path.join(root_, r"OneDrive - KU Leuven\Bestanden van Francis Matthews - C_project\Results\ARNO_output_26022025\Modifications_impacts_Campine\C_factor_*.csv"))
parcel_info_paths_m = glob.glob(os.path.join(root_,r"OneDrive - KU Leuven\Bestanden van Francis Matthews - C_project\Results\ARNO_output_26022025\Modifications_impacts_Campine\IACS_with_NDVI_*.csv")) 

# Load the CSV data into dataframes
dataframes_m = load_data(methods_paths)
dataframes_iacs_m = load_data(parcel_info_paths_m)

# Remapping the method names
method_mapping = {
    "final": "final",
    "without_fvc": "No FVC correction",
    "without_residues": "No crop residues",
    "without_s": "No senescence"
}


# Get region names from filenames
method_labels = []
for path in methods_paths:
    filename = os.path.basename(path)
    matched = False
    for method in ['final', 'without_fvc', 'without_residues', 'without_s']:
        if f"C_factor_Campine_{method}".lower() in filename.lower():
            method_labels.append(method_mapping[method]) 
            matched = True
            break
# Prepare filtered C-factor DataFrames for each region
Cfactor_filtered_list = []
method_name_list = []


for df, iacs_df, method_label in zip(dataframes_m, dataframes_iacs_m, method_labels):
    # Merge C-factor data with IACS data on parcel ID
    merged_df = pd.merge(df, iacs_df, left_on="object_id", right_on="OBJECTID", how="inner")
    # Filter valid crops (defined earlier in script)
    filtered = merged_df[merged_df['crop_x'].isin(valid_crops)].copy()
    filtered['crop_x'] = filtered['crop_x'].replace(crop_mapping)
    # Add method label
    filtered["Method"] = method_label
    # Add to list
    Cfactor_filtered_list.append(filtered)

# === PLOT USING EXISTING FUNCTION ===
C_factor_comparison_box_plot(Cfactor_filtered_list, method_labels)

"""
ANALYSIS OF THE REMOTE SENSING OBSERVATIONS
"""
# 1: Scatterplot of observations for a specific region (eg SandyLoam) vs C-factor
# Choose region on which you want to do the analysis
region_label ="SandyLoam"
filtered_file_path = None

for path in file_paths:
    filename = os.path.basename(path)
    if "SandyLoam".lower() in filename.lower():  # Case-insensitive match
        filtered_file_path = path
        break  # Assuming there's only 1 file
        
# If the file is found, process it
if filtered_file_path:
    # Get the corresponding dataframe and IACS dataframe
    idx = file_paths.index(filtered_file_path)
    df = dataframes[idx]
    iacs_df = iacs_dataframes[idx]
    # Filter the data to keep only valid crops
    iacs_df = iacs_df[iacs_df['crop_name'].isin(valid_crops)]
    # Extract date columns (assuming last columns represent NDVI values per date)
    date_columns = iacs_df.columns[16:]
    # Convert date format from YYYYMMDD to datetime
    new_date_columns = pd.to_datetime(date_columns, format='%Y%m%d', errors='coerce')
    date_map = dict(zip(date_columns, new_date_columns))
    iacs_df.rename(columns=date_map, inplace=True)
    # Convert columns to numeric format
    iacs_df[new_date_columns] = iacs_df[new_date_columns].apply(pd.to_numeric, errors='coerce')
    colors = sns.color_palette('husl', len(valid_crops)) 
    # Loop through each valid crop and plot it
    for c,color in zip(valid_crops,colors):
        crop_data = iacs_df[iacs_df['crop_name'] == c]
        display_name = crop_mapping.get(c, crop)
        # Define two time intervals
        interval_1_start = pd.to_datetime('2023-01-01')
        interval_1_end = pd.to_datetime('2023-12-31')
        interval_1_columns = [col for col in new_date_columns if interval_1_start <= col <= interval_1_end]
        # Count the number of non-NaN NDVI observations per plot in both intervals
        ndvi_counts = []
        for index, row in crop_data.set_index('OBJECTID').iterrows():
            total_count = row[interval_1_columns].notna().sum()
            ndvi_counts.append([index, total_count])
        # Convert to DataFrame
        ndvi_counts_df = pd.DataFrame(ndvi_counts, columns=['OBJECTID', 'ndvi_count_total'])
        # Merge with C-factor dataset
        df_merged = ndvi_counts_df.merge(df[['object_id', 'C_factor']], left_on='OBJECTID', right_on='object_id', how='left')
        # Calculate mean and standard deviation per NDVI observation count
        grouped_stats = df_merged.groupby('ndvi_count_total')['C_factor'].agg(['mean', 'std']).reset_index()
        # Plot the results
        plt.figure(figsize=(10, 5))
        plt.scatter(df_merged['ndvi_count_total'], df_merged['C_factor'], color=color, alpha=0.5)
        plt.xlabel('Number of NDVI Observations')
        plt.ylabel('C-factor')
        plt.title(f'{display_name}')
        plt.grid()
        plt.legend()
        plt.show() 

        # 2: Boxplot for a low number of observations during period of high rainfall erosivity vs the C-factor
        # Define the time interval
        interval_1_start = pd.to_datetime('2023-06-01')
        interval_1_end = pd.to_datetime('2023-09-30')
        interval_1_columns = [col for col in new_date_columns if interval_1_start <= col <= interval_1_end]
        # Count the number of non-NaN NDVI observations per plot
        ndvi_counts = []
        for index, row in crop_data.set_index('OBJECTID').iterrows():
            total_count = row[interval_1_columns].notna().sum()
            ndvi_counts.append([index, total_count])
        # Convert to DataFrame
        ndvi_counts_df = pd.DataFrame(ndvi_counts, columns=['OBJECTID', 'ndvi_count_total'])
        # Merge with C-factor dataset
        df_merged = ndvi_counts_df.merge(df[['object_id', 'C_factor']], left_on='OBJECTID', right_on='object_id', how='left')
        # Define NDVI observation categories
        bins = [0, 10, float('inf')]
        labels = ['< 10', '>10']
        df_merged['ndvi_category'] = pd.cut(df_merged['ndvi_count_total'], bins=bins, labels=labels, right=False)
        # Create the boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='ndvi_category', y='C_factor', data=df_merged, palette='Blues')
        # Labels and title
        plt.xlabel('Number of NDVI Observations', fontsize=16)
        plt.ylabel('C-factor',fontsize=16)
        plt.title(f'{display_name}',fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Show the plot
        plt.show()

"""
ANALYSIS OF THE TIMING OF THE PLANTING
"""
# Choose region on which you want to do the analysis
region_label ="SandyLoam"
filtered_file_path = None

for path in file_paths:
    filename = os.path.basename(path)
    if "SandyLoam".lower() in filename.lower():  # Case-insensitive match
        filtered_file_path = path
        break  # Assuming there's only 1 file
        
# If the file is found, process it
if filtered_file_path:
    # Get the corresponding dataframe and IACS dataframe
    idx = file_paths.index(filtered_file_path)
    df = dataframes[idx]
    iacs_df = iacs_dataframes[idx]
    # Filter the data to keep only valid crops
    iacs_df = iacs_df[iacs_df['crop_name'].isin(valid_crops)]
    # Extract date columns (assuming last columns represent NDVI values per date)
    date_columns = iacs_df.columns[16:]
    # Convert date format from YYYYMMDD to datetime
    new_date_columns = pd.to_datetime(date_columns, format='%Y%m%d', errors='coerce')
    date_map = dict(zip(date_columns, new_date_columns))
    iacs_df.rename(columns=date_map, inplace=True)        
    # Convert columns to numeric format
    iacs_df[new_date_columns] = iacs_df[new_date_columns].apply(pd.to_numeric, errors='coerce')
    # Loop through each valid crop and plot it
    for c in valid_crops:
        crop_data = iacs_df[iacs_df['crop_name'] == c]
        winter_crops = ['Wheat', 'Barley', 'Triticale', 'Canola','Rye','Clover','Oats']
        if c in winter_crops:
            display_name = crop_mapping.get(c, crop)
            # Define the time interval
            start_date = pd.to_datetime('2023-01-01')
            end_date = pd.to_datetime('2023-04-30')
            date_columns = [col for col in new_date_columns if start_date <= col <= end_date]
            
            # Find the first occurrence where NDVI > 0.80 per plot
            ndvi_threshold = 0.80
        else:
            # Define the time interval
            display_name = crop_mapping.get(c, crop)
            start_date = pd.to_datetime('2023-05-01')
            end_date = pd.to_datetime('2023-09-01')
            date_columns = [col for col in new_date_columns if start_date <= col <= end_date]
            # Find the first occurrence where NDVI > 0.80 per plot
            ndvi_threshold = 0.50
        # Initialize an empty list to store the first valid NDVI date per parcel
        first_valid_dates = []
        # Loop over each row in the NDVI columns of the IACS dataframe        
        for index, row in crop_data.set_index('OBJECTID')[date_columns].iterrows():
            # Create a boolean Series where values are True if NDVI > threshold
            above_threshold = row > ndvi_threshold
            # Get a list of dates (column names) where the condition is True
            valid_dates = above_threshold[above_threshold].index.tolist()
            first_valid_dates.append([index, valid_dates[0] if valid_dates else pd.NaT])
        # Convert the results into a new DataFrame with OBJECTID and first valid NDVI date
        first_dates = pd.DataFrame(first_valid_dates, columns=['OBJECTID', 'first_date'])
        # Remove NaT values before merging
        first_dates = first_dates.dropna(subset=["first_date"])
        # Merge with C-factor dataset on 'OBJECTID' from NDVI data and 'object_id' from C-factor data
        df_merged = first_dates.merge(df[['object_id', 'C_factor']], left_on='OBJECTID', right_on='object_id', how='left')
        # Convert dates to monthly format
        df_merged["month"] = df_merged["first_date"].dt.strftime("%Y-%m")  # Format as 'YYYY-MM'
        # Sort months in ascending order
        sorted_months = sorted(df_merged["month"].unique())
        # Plot the violin plot using Seaborn (with green color palette)
        plt.figure(figsize=(10, 5))
        sns.violinplot(x="month", y="C_factor", data=df_merged, palette="Greens", inner="box", order=sorted_months)
        # Customize the plot
        plt.xlabel("Month")
        plt.ylabel("C-factor")
        plt.title(f"{display_name}")
        plt.xticks(rotation=45)
        plt.grid(axis="y")
        # Show the plot
        plt.show()
        