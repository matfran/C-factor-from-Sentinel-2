"""
This script adds C-factor values from different years 
belonging to the same parcels together and calculates average C-factors
per parcel over multiple years

@author: Arno Kasprzak
"""

# Import modules
import geopandas as gpd
import pandas as pd
import os

# Define main data directory
root = "path to main data directory"
# Load the three geopackage files
parcel_year1 = gpd.read_file(os.path.join(root,"Rotation_data_y1.gpkg"))
parcel_year2 = gpd.read_file(os.path.join(root,"Rotation_data_y2.gpkg"))
parcel_year3 = gpd.read_file(os.path.join(root,"Rotation_data_y3.gpkg"))

# Merge parcels based on exact geometry matches
# With up to data Rotation_data_{year} it is also possible to merge just on a link_id
merged_1_2 = parcel_year1.merge(parcel_year2, on="geometry", how="inner")
merged_all = merged_1_2.merge(parcel_year3, on="geometry", how="inner")

# Select relevant columns for matching
matched_parcels = merged_all[["REF_ID_x", "REF_ID_y", "REF_ID", "rotations"]]
matched_parcels.columns = ["REF_ID_1", "REF_ID_2", "REF_ID_3", "rotations"]

# Load C_factor CSV files
c_factor1 = pd.read_csv("C_factor_roty1.csv")
c_factor2 = pd.read_csv("C_factor_roty2.csv")
c_factor3 = pd.read_csv("C_factor_roty3.csv")

# Merge C_factor values based on object_id
matched_parcels = matched_parcels.merge(c_factor1, left_on="REF_ID_1", right_on="object_id", how="left")
matched_parcels = matched_parcels.merge(c_factor2, left_on="REF_ID_2", right_on="object_id", how="left", suffixes=("_y1", "_y2"))
matched_parcels = matched_parcels.merge(c_factor3, left_on="REF_ID_3", right_on="object_id", how="left")

# Rename C_factor and related columns from year 3
matched_parcels = matched_parcels.rename(columns={
    "C_factor": "C_factor_y3",
    "C_factor_upper": "C_factor_upper_y3",
    "C_factor_lower": "C_factor_lower_y3"
})


# Compute average C_factor
matched_parcels["C_factor_avg"] = matched_parcels[["C_factor_y1", "C_factor_y2", "C_factor_y3"]].mean(axis=1)

# Compute average C_factor_upper and C_factor_lower
matched_parcels["C_factor_upper_avg"] = matched_parcels[["C_factor_upper_y1", "C_factor_upper_y2", "C_factor_upper_y3"]].mean(axis=1)
matched_parcels["C_factor_lower_avg"] = matched_parcels[["C_factor_lower_y1", "C_factor_lower_y2", "C_factor_lower_y3"]].mean(axis=1)

# Drop rows with NaN values in any of the C_factor-related columns
matched_parcels = matched_parcels.dropna(subset=[
    "C_factor_y1", "C_factor_y2", "C_factor_y3",
    "C_factor_upper_y1", "C_factor_upper_y2", "C_factor_upper_y3",
    "C_factor_lower_y1", "C_factor_lower_y2", "C_factor_lower_y3"
])

# Select final columns
output_df = matched_parcels[[
    "REF_ID_1", "REF_ID_2", "REF_ID_3",
    "C_factor_y1", "C_factor_y2", "C_factor_y3",
    "C_factor_upper_y1", "C_factor_upper_y2", "C_factor_upper_y3",
    "C_factor_lower_y1", "C_factor_lower_y2", "C_factor_lower_y3",
    "C_factor_avg", "C_factor_upper_avg", "C_factor_lower_avg",
    "rotations_x"
]]
output_df = output_df.iloc[:, :-1]
# Save to CSV
output_df.to_csv(os.path.join(root,"C_factor_results_rot_combined.csv"), index=False)
