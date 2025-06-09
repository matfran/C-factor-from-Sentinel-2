# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:28:41 2025

@author: Arno Kasprzak

Script to sample 1000 parcels per rotation
"""
# Import modules
import geopandas as gpd
from collections import Counter
import os
import pandas as pd
import glob

# Define main data directory
root = "path to main module"

"""
Sample specific parcels based on crop rotation patterns 
"""
# Input geopackage paths
input_gpkg_1 = os.path.join(root,"Unchanged_parcels21.gpkg")
input_gpkg_2 = os.path.join(root,"Unchanged_parcels22.gpkg")
input_gpkg_3 = os.path.join(root,"Unchanged_parcels23.gpkg")

# Output directory for shapefiles
output_directory = os.path.join(root,"Sampled_Parcels")
os.makedirs(output_directory, exist_ok=True)

# Target crops (in any order)
target_crops_list = [
    ["Korrelmaïs", "Korrelmaïs", "Korrelmaïs"]
    ]

def sample_common_parcels_with_crops_direct_links(
    gpkg1, gpkg2, gpkg3, output_dir, target_crops_list
):
    """
    This function processes geospatial data from three GeoPackages (representing agricultural parcels for the years 2021, 2022, and 2023),
    filters parcels based on specified crop rotation patterns, and saves the results as sampled shapefiles.

    Args:
        gpkg1 (str): Path to the GeoPackage file for the 2021 parcel data.
        gpkg2 (str): Path to the GeoPackage file for the 2022 parcel data.
        gpkg3 (str): Path to the GeoPackage file for the 2023 parcel data.
        output_dir (str): Directory where the output shapefiles will be saved.
        target_crops_list (list of list of str): A list of target crop rotation patterns to filter parcels. Each inner list represents 
                                                  a specific crop rotation pattern (e.g., ["Wintertarwe", "Suikerbieten", "Aardappelen"]).
    """
    print("Loading geopackages...")
    gdf1 = gpd.read_file(gpkg1)
    gdf2 = gpd.read_file(gpkg2)
    gdf3 = gpd.read_file(gpkg3)

    # Check if the CRS (Coordinate Reference System) of all datasets is consistent
    if not (gdf1.crs == gdf2.crs == gdf3.crs):
        print("Reprojecting CRS to match...")
        gdf2 = gdf2.to_crs(gdf1.crs)
        gdf3 = gdf3.to_crs(gdf1.crs)

    # Temporarily drop the geometry column to perform merging
    print("Dropping geometry columns temporarily...")
    gdf1_no_geom = gdf1.drop(columns=['geometry'])
    gdf2_no_geom = gdf2.drop(columns=['geometry'])
    gdf3_no_geom = gdf3.drop(columns=['geometry'])

    # Merge the datasets based on the 'id_link' field (a unique identifier for each parcel)
    print("Merging datasets based on 'id_link'...")
    merged = gdf1_no_geom.merge(gdf2_no_geom, on="id_link", suffixes=("_2021", "_2022"))
    merged = merged.merge(gdf3_no_geom, on="id_link")
    merged = merged.rename(columns={'GWSNAM_H': 'GWSNAM_H_2023'})

     # Add back the geometry column from gdf1 (this will be used for spatial operations)
    print("Adding geometry back to merged dataset...")
    merged = merged.merge(gdf1[['id_link', 'geometry']], on="id_link")

    # Convert the merged DataFrame back to a GeoDataFrame
    merged = gpd.GeoDataFrame(merged, geometry='geometry', crs=gdf1.crs)

    # Process each set of target crops and filter parcels based on the specified crop rotation
    for i, target_crops in enumerate(target_crops_list, 1):
        print(f"Processing target crop set {i}: {target_crops}")

        # Filter the merged dataset based on the target crop rotation (comparing crops from 2021, 2022, and 2023)
        filtered_matches = merged[
            merged.apply(lambda row: Counter([
                row['GWSNAM_H_2021'], row['GWSNAM_H_2022'], row['GWSNAM_H_2023']
            ]) == Counter(target_crops), axis=1)
        ]

        # Save filtered parcels with only required attributes
        if not filtered_matches.empty:
            filtered_matches = filtered_matches[[
                'geometry', 'id_link', 'REF_ID_2021', 'GWSNAM_H_2021',
                'REF_ID_2022', 'GWSNAM_H_2022',
                'REF_ID', 'GWSNAM_H_2023'
            ]]
            filtered_matches = filtered_matches.rename(columns={"geometry": "geometry"})
            # If more than 1000 parcels match, randomly sample 1000 of them
            if len(filtered_matches) > 1000:
                filtered_matches = filtered_matches.sample(n=1000, random_state=42)
            # Save the filtered and sampled parcels to a shapefile in the specified output directory
            output_file = os.path.join(output_dir, f"sampled_parcels_set_{i}.shp")
            print(f"Saving filtered parcels for crop set {i} to {output_file}...")
            filtered_matches.to_file(output_file, driver='ESRI Shapefile')
        else:
            print(f"No parcels found for target crop set {i}.")

    print("All crop sets processed successfully!")


# Call the function
sample_common_parcels_with_crops_direct_links(
    input_gpkg_1, input_gpkg_2, input_gpkg_3, output_directory, target_crops_list
)

"""
Now filter these sampled parcels from the parcel geodata so that
we end up with only 3 geodata files containing the sampled parcels
"""
def filter_files(parcel_files, filter_shapefiles):
    """
    This function filters agricultural parcels based on reference IDs from filter shapefiles
    and saves the filtered parcels into new shapefiles for each year (2021, 2022, 2023).

    Args:
        parcel_files (list): A list of file paths to the agricultural parcel shapefiles for the years 2021, 2022, and 2023.
        filter_shapefiles (list): A list of file paths to shapefiles containing the sampled parcels to filter by.

    The function iterates over the parcel shapefiles, filters them based on the reference IDs
    in the filter shapefiles, and saves the results as new shapefiles for each year.
    """
    # Specify the column names for each parcel file that will be used for filtering
    filter_columns = ['REF_ID_202', 'REF_ID_2_1', 'REF_ID']
    
    # Loop over all rotation samples
    for i, parcel_file in enumerate(parcel_files):
        # Read the parcel shapefile
        parcels = gpd.read_file(parcel_file)

        # Determine the correct filter column name for this parcel file
        filter_column = filter_columns[i]

        # Initialize a list to store all filtered results for this parcel file
        all_filtered_results = []

        # Iterate through all filter shapefiles
        for j, filter_shapefile in enumerate(filter_shapefiles):
            # Read the filter shapefile
            filter_data = gpd.read_file(filter_shapefile)

            # Filter parcels where the REF_ID matches values in the filter shapefile
            filtered = parcels[parcels["REF_ID"].isin(filter_data[filter_column])]

            # Add a 'rotations' column to indicate the filter shapefile used (1-based index)
            filtered['rotations'] = j + 1

            # Append the filtered results to the list
            all_filtered_results.append(filtered)

        # Combine all filtered results for this parcel file
        combined_filtered = gpd.GeoDataFrame(pd.concat(all_filtered_results, ignore_index=True))

        # Save the combined filtered results to a new shapefile
        output_file = os.path.join(root,f'Sampled_parcels_year{i+1}.shp')
        combined_filtered.to_file(output_file)
        gpkg_file= os.path.join(root,f'Sampled_parcels_year{i+1}.gpkg')
        combined_filtered.to_file(gpkg_file, driver='GPKG')


# Example usage
parcel_files = [
    os.path.join(root,"Landbouwgebruikspercelen_2021.gpkg"),
    os.path.join(root,"Landbouwgebruikspercelen_2022.gpkg"),
    os.path.join(root,"Landbouwgebruikspercelen_2023.gpkg"),
]

filter_shapefiles = glob.glob(f"{output_directory}sampled_parcels_set_*.shp")

filter_files(parcel_files, filter_shapefiles)

def update_geopackage_with_links(geopackage_path, shape_files, output_path, ref_col='REF_ID', search_col='REF_ID', link_col='id_link'):
    """
    Updates a GeoPackage by adding a 'link_id' column based on external shapefiles.
    
    Parameters:
        geopackage_path (str): Path to the main GeoPackage to update.
        shape_files (list): List of paths to shapefiles that contain the mapping info.
        ref_col (str): Column name in the GeoPackage to match (e.g., 'REF_ID').
        search_col (str): Column name in the shapefiles to match against (e.g., 'REF_ID').
        link_col (str): Column name in the shapefiles containing the link value (e.g., 'id_link').
        output_path (str): Path to save the updated GeoPackage.
    """
    # Load the main layer from the geopackage
    gdf_main = gpd.read_file(geopackage_path)
    
    # Create an empty dictionary to map REF_IDs to link_id values
    ref_to_link = {}
    
    # Iterate through all shapefiles to map REF_ID_202 to id_link
    for shp in shape_files:
        if os.path.exists(shp):
            gdf = gpd.read_file(shp)
            for _, row in gdf.iterrows():
                ref_value = row.get(search_col)
                link_value = row.get(link_col)
                if ref_value and link_value:
                    ref_to_link[ref_value] = link_value
    
    # Add the new column 'link_id' based on the found mappings
    gdf_main['link_id'] = gdf_main[ref_col].map(ref_to_link)
    
    # Save to a new geopackage
    gdf_main.to_file(output_path, driver='GPKG')
    print(f'Updated geopackage saved to {output_path}')

# Example usage
geopackage_path1 = os.path.join(root,"Sampled_parcels_year1.gpkg")  # Path to the geopackage
geopackage_path2 = os.path.join(root,"Sampled_parcels_year2.gpkg")  # Path to the geopackage
geopackage_path3 = os.path.join(root,"Sampled_parcels_year3.gpkg")  # Path to the geopackage
shape_files = [os.path.join(root,f"Sampled_Parcels/sampled_parcels_set_{i}.shp") for i in range(1, 21)]  # List of 20 shapefiles
output_path1 = os.path.join(root,'Rotation_data_y1.gpkg')
output_path2 = os.path.join(root,'Rotation_data_y2.gpkg')
output_path3 = os.path.join(root,'Rotation_data_y3.gpkg')

update_geopackage_with_links(geopackage_path1, shape_files, output_path1)
update_geopackage_with_links(geopackage_path2, shape_files, output_path2)
update_geopackage_with_links(geopackage_path3, shape_files, output_path3)