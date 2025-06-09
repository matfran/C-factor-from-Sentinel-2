"""
Script comparing geometries from three shapefiles of agricultural parcels 
across three years, identifying unchanged parcels, analysing combinations 
of crops, and saving the results in shapefiles and a CSV file.
The script then analyzes the occurrence of different crop rotations 
for main crops based on that result.

@author: Arno Kasprzak
"""
# Import modules
import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry
import os

# Define main data directory
root = "path to main directory"

"""
Find for unchanged parcels crop rotation combination per parcels
"""
# Define the buffer distance in meters for comparison
tolerance_distance = 5

# Input shapefiles
data_1 = os.path.join(root,"Landbouwgebruikspercelen_2021.gpkg")
data_2 = os.path.join(root,"Landbouwgebruikspercelen_2022.gpkg")
data_3 = os.path.join(root,"Landbouwgebruikspercelen_2023.gpkg")

# Output geopackage files
output_gpkg_1 = os.path.join(root,"unchanged_parcels21.gpkg")
output_gpkg_2 = os.path.join(root,"unchanged_parcels22.gpkg")
output_gpkg_3 = os.path.join(root,"unchanged_parcels23.gpkg")

# Output CSV file for crop analysis
output_csv_file = os.path.join(root,"parcel_combinations_analysis.csv")

def compare_and_analyze(data1, data2, data3, gpkg1,
                        gpkg2,gpkg3,output_csv, tolerance):
    """
    Compares three geopackages, writes unchanged geometries to separate shapefiles, 
    and analyses combinations of crops across years.

    Parameters:
        shapefile1 (str): Path to the first geodata.
        shapefile2 (str): Path to the second geodata.
        shapefile3 (str): Path to the third geodata.
        output_file_1 (str): Path to save the output shapefile with unchanged geometries from the first dataset.
        output_file_2 (str): Path to save the output shapefile with unchanged geometries from the second dataset.
        output_file_3 (str): Path to save the output shapefile with unchanged geometries from the third dataset.
        gpkg1 (str): Path to save the output geopackage with unchanged geometries from the first dataset.
        gpkg2 (str): Path to save the output geopackage with unchanged geometries from the second dataset.
        gpkg3 (str): Path to save the output geopackage with unchanged geometries from the third dataset.
        output_csv (str): Path to save the CSV file with GWS combination analysis.
        tolerance (float): Distance tolerance in meters for geometry comparison.
    """
    print("Loading data...")
    # Read input shapefiles
    gdf1 = gpd.read_file(data1)
    gdf2 = gpd.read_file(data2)
    gdf3 = gpd.read_file(data3)

    # Ensure all shapefiles use the same CRS
    if not (gdf1.crs == gdf2.crs == gdf3.crs):
        print("CRS mismatch detected. Reprojecting shapefiles...")
        gdf2 = gdf2.to_crs(gdf1.crs)
        gdf3 = gdf3.to_crs(gdf1.crs)

    # Normalize and fix geometries
    gdf1['geometry'] = gdf1['geometry'].apply(lambda geom: geom.buffer(0).simplify(tolerance) if geom.is_valid else None)
    gdf2['geometry'] = gdf2['geometry'].apply(lambda geom: geom.buffer(0).simplify(tolerance) if geom.is_valid else None)
    gdf3['geometry'] = gdf3['geometry'].apply(lambda geom: geom.buffer(0).simplify(tolerance) if geom.is_valid else None)

    # Drop rows with invalid geometries
    gdf1 = gdf1.dropna(subset=['geometry'])
    gdf2 = gdf2.dropna(subset=['geometry'])
    gdf3 = gdf3.dropna(subset=['geometry'])

    # Create spatial indexes for efficient comparison
    print("Building spatial indexes...")
    spatial_index_2 = gdf2.sindex
    spatial_index_3 = gdf3.sindex

    # Lists to store unchanged geometries for each dataset and combinations
    unchanged_geometries_1 = []
    unchanged_geometries_2 = []
    unchanged_geometries_3 = []
    combinations = []

    print("Comparing geometries...")
    id_counter = 1  # Counter for generating unique ID links
    for idx, geom1 in enumerate(gdf1.geometry):
        if not isinstance(geom1, BaseGeometry):
            print(f"Skipping invalid geometry at index {idx}.")
            continue

        # Find potential matches in gdf2 within the buffer distance
        possible_matches_index_2 = list(spatial_index_2.intersection(geom1.buffer(tolerance).bounds))
        possible_matches_2 = gdf2.iloc[possible_matches_index_2]

        match_found_2 = False
        matched_row_2 = None
        for idx2, geom2 in possible_matches_2.geometry.iteritems():
            if geom1.equals(geom2):
                match_found_2 = True
                matched_row_2 = gdf2.loc[idx2]
                break

        if not match_found_2:
            continue  # No match in the second dataset, skip to next geometry

        # Find potential matches in gdf3 within the buffer distance
        possible_matches_index_3 = list(spatial_index_3.intersection(geom1.buffer(tolerance).bounds))
        possible_matches_3 = gdf3.iloc[possible_matches_index_3]

        match_found_3 = False
        matched_row_3 = None
        for idx3, geom3 in possible_matches_3.geometry.iteritems():
            if geom1.equals(geom3):
                match_found_3 = True
                matched_row_3 = gdf3.loc[idx3]
                break

        if match_found_3:
            # Generate a unique ID link for these matched geometries
            id_link = f"ID_{id_counter}"
            id_counter += 1

            # Append geometries and assign the ID link
            gdf1_row = gdf1.iloc[idx].copy()
            matched_row_2 = matched_row_2.copy()
            matched_row_3 = matched_row_3.copy()

            gdf1_row["id_link"] = id_link
            matched_row_2["id_link"] = id_link
            matched_row_3["id_link"] = id_link

            unchanged_geometries_1.append(gdf1_row)
            unchanged_geometries_2.append(matched_row_2)
            unchanged_geometries_3.append(matched_row_3)

            # Add crop names to combinations
            combination = {
                "GWS_H_year1": gdf1.loc[idx, "GWSNAM_H"],
                "GWS_H_year2": matched_row_2["GWSNAM_H"],
                "GWS_H_year3": matched_row_3["GWSNAM_H"],
                "GRAF_OPP_year1": gdf1.loc[idx, "GRAF_OPP"],
                "tot_area": gdf1.loc[idx, "GRAF_OPP"],  # Use only the first year's area for `tot_area`
                "id_link": id_link
            }
            combinations.append(combination)

        # Print progress every 10,000 iterations
        if idx % 10000 == 0:
            print(f"Processed {idx} features...")

    print("Geometry comparison complete.")
    print(f"Found {len(unchanged_geometries_1)} geometries unchanged across all three years.")

    # Save unchanged geometries
    if unchanged_geometries_1:
        unchanged_gdf_1 = gpd.GeoDataFrame(unchanged_geometries_1, columns=list(gdf1.columns) + ["id_link"], crs=gdf1.crs)
        unchanged_gdf_2 = gpd.GeoDataFrame(unchanged_geometries_2, columns=list(gdf2.columns) + ["id_link"], crs=gdf2.crs)
        unchanged_gdf_3 = gpd.GeoDataFrame(unchanged_geometries_3, columns=list(gdf3.columns) + ["id_link"], crs=gdf3.crs)

        unchanged_gdf_1.to_file(gpkg1, driver="GPKG")
        unchanged_gdf_2.to_file(gpkg2, driver="GPKG")
        unchanged_gdf_3.to_file(gpkg3, driver="GPKG")
    else:
        print("No unchanged geometries found. Output shapefiles will not be created.")
     
    # Analyze combinations and save as CSV
    print("Analyzing GWS combinations...")
    combination_df = pd.DataFrame(combinations).fillna("NULL")

    # Aggregate counts of combinations
    combination_counts = combination_df.groupby(list(combination_df.columns[:-2])).agg(
        total_area=("tot_area", "sum"),
        count=("tot_area", "size")
    ).reset_index()

    print(f"Writing crop combination analysis to {output_csv}...")
    combination_counts.to_csv(output_csv, index=False)
    print("Crop combination analysis saved.")
    
compare_and_analyze(
        data_1,
        data_2,
        data_3,
        output_gpkg_1,
        output_gpkg_2,
        output_gpkg_3,
        output_csv_file,
        tolerance_distance
    )
    
"""
Analyze the occurrence of each rotation
"""

# Input CSV file containing the combination data
input_csv_file = os.path.join(root,"parcel_combinations_analysis.csv")

# Output CSV file with aggregated results
output_csv_file = os.path.join(root,"crop_combinations.csv")

# Function to group combinations and perform aggregations
def aggregate_combinations(input_csv, output_csv):
    """
    Group rows based on unique combinations of `GWS_H_year1`, `GWS_H_year2`, `GWS_H_year3`.
    Sum `counts` and `total_area` for these combinations and save the result to a new CSV.

    Parameters:
        input_csv (str): Path to the input CSV.
        output_csv (str): Path to the output CSV.
    """
    print("Loading data from CSV...")
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_csv)

    print("Aggregating combinations...")
    # Group by the specified columns and sum the values
    aggregated_df = df.groupby([
        'GWS_H_year1', 'GWS_H_year2', 'GWS_H_year3'
    ], as_index=False).agg(
        {
            'count': 'sum',       # Sum of the counts
            'total_area': 'sum'  # Sum of the total_area
        }
    )

    print("Saving aggregated results to CSV...")
    # Save the result to a new CSV
    aggregated_df.to_csv(output_csv, index=False)
    print(f"Aggregated results saved to {output_csv}")

aggregate_combinations(input_csv_file, output_csv_file)
    
"""
Filter the rotation data  keeping only the usefull rotations
"""

# Path to the input Excel file
input_csv = os.path.join(root,"crop_combinations.csv")
output_csv = os.path.join(root,"crop_combinations_filtered.csv")

# List of crops for which rows should be removed if GWSNAM_H appears exactly three times in consecutive years
crop_list = ["Grasland", "Stal", "Loods (bv. voor machines, opslag,?)", "Woonhuis"
             ,"Poelen <= 0,1 ha", "Gebouw i.k.v. verbreding", "Grasklaver"
             ,"Houtkanten en houtwallen <= 10 m breed", "Faunamengsel",
             "Niet nader omschreven gebouw","Mengsel van gras en vlinderbloemigen (andere dan grasklaver of grasluzerne)",
             "Ander gebouw", "Bloemenmengsel", "Bloemenmengsel voor EAG Braak",
             "Braakliggend land zonder minimale activiteit", "Graskruiden mengsel"
             ,"Niet ingezaaid akkerland", "Mengsel van niet-vlinderbloemige groenbedekkers",
             "Braakliggend land met minimale activiteit met EAG", "Mengsel van vlinderbloemigen",
             "Braakliggend land met minimale activiteit zonder EAG", "Bomenrijen",
             "Andere vlinderbloemige groenbedekker","Natuurlijk grasland met minimumactiviteit",
             "Begraasde niet-landbouwgrond","Laanbomen","Onverharde landingsbaan of veiligheidszones op vliegvelden",
             "Natuurlijk grasland zonder minimumactiviteit", "Heide in natuurbeheer",
             "Andere niet-vlinderbloemige groenbedekker", "Andere bedekking","Solitaire bomen",
             "Bebossing (korte omlooptijd)", "Rode klaver", "Bebossing loofbomen-economisch",
             "Eenjarige klaver", "Volkstuinpark", "Bebossing loofbomen-ecologisch", 
             "Bebossing met contract voor 2008", "Bebossing populieren", "Bos",
             "Biologische niet-landbouwgrond"]  # Add crop names here

def filter_rows_by_crop(input_file, output_file, crops):
    """
    Filter rows in a csv file based on the condition that GWSNAM_H appears in the crop list
    for all three years (GWS_H_year1, GWS_H_year2, and GWS_H_year3).

    Parameters:
        input_file (str): Path to the input csv file.
        output_file (str): Path to save the filtered csv file.
        crops (list): List of crops to filter rows for.
    """
    # Load the Excel file
    print("Loading csv file...")
    df = pd.read_csv(input_file)

    # Filter the rows where all three years have crops from the list
    print("Filtering rows...")
    df_filtered = df[~(
        (df['GWS_H_year1'].isin(crops)) &
        (df['GWS_H_year2'].isin(crops)) &
        (df['GWS_H_year3'].isin(crops))
    )]

    # Save the filtered DataFrame to a new Excel file
    print(f"Writing filtered data to {output_file}...")
    df_filtered.to_csv(output_file, index=False)
    print("Filtered data saved.")

filter_rows_by_crop(input_csv, output_csv, crop_list)
    
"""
Ignore the order of the crops in the rotation: do a final count on
the occurrence of each rotation
"""
    
def unique_combinations(input_file, output_file):
    """
    Processes crop rotation data, identifies unique crop combinations, 
    and calculates the total area for each combination across three years.

    Parameters:
        input_file (str): Path to the input CSV file containing crop rotation data.
        output_file (str): Path to save the output Excel file with unique crop combinations and total area.
    """
    # Load input data
    df = pd.read_csv(input_file)

    # Ensure columns are correctly named
    columns_to_combine = ['GWS_H_year1', 'GWS_H_year2', 'GWS_H_year3']

    # Create a column for unique combinations of crops (order-independent)
    df['unique_combination'] = df[columns_to_combine].apply(lambda row: tuple(sorted(row)), axis=1)

    # Group by the unique combinations and sum the percentage_area
    grouped = df.groupby('unique_combination', as_index=False).agg({
        'total_area': 'sum'
    })

    # Split the combinations back into separate columns
    grouped[columns_to_combine] = pd.DataFrame(grouped['unique_combination'].tolist(), index=grouped.index)

    # Remove the temporary unique_combination column
    grouped = grouped.drop(columns=['unique_combination'])

    # Write the output to a new Excel file
    grouped.to_excel(output_file, index=False)
    print(f"Output written to: {output_file}")

input_file = os.path.join(root,"crop_combinations_filtered.csv")
output_file = os.path.join(root,"Crop_rotations_unique_combinations.xlsx")  
unique_combinations(input_file, output_file)



