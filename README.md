<img src="https://github.com/user-attachments/assets/a9e140fb-5ec0-4ed2-9db3-de6ce1a3973e" alt="C-factor-from-Sentinel-2" width="300"/>


# **C-factor-from-Sentinel-2**

**General description**

This repository contains all the information regarding the Python/Google Earth Engine scripts used to calculate parcel-specific (multi-year/annual) crop cover and manageent factors (C-factors) based on Sentinel-2 remote sensing imagery for Flanders. 
The code analyses the temporal evolution of the ground cover within the year and calculated the erosion risk and C-factor based on the interations with the 15-day rainfall erosivity. Additionally, it includes a necessary Python script for statistically analyzing the results.

The code permits the processing of the C-factor for a large-sample or all field parcels in a region based on the methodology outlined in Matthews et al., (2023): https://www.sciencedirect.com/science/article/pii/S2095633922000788

The user can apply default parameters or make refinements where desired to calibrate the C-factor values based on regional management or C-factor information.

The outputs of the workflow give the following insights into the soil erosion risk associated with crop cultivations:
1) C-factor values for each field parcel which can be compared between and within main crop cultivations.
2) Information of the timing of the erosion risk within the year for a given main crop cultivation.

**Workflow**

<img src="https://github.com/user-attachments/assets/005a2b45-840f-46fb-a900-5e69e7f0cfb5" width="300"/>


The workflow composes of 3 general steps:
1) Sampling the desired field parcels to analyse based on a GSA dataset of field parcel geometries and crop cultivation declarations.
2) Uploading and processing the parcels in Google Earth Engine (GEE) to freely extract and download all the available Sentinel-2 timeseries data for the sample of parcels.
3) Calculating the C-factor for the sampled field parcels using the C-factor Python module and statistically analysing the results.

Note that the C-factor is calculated based on the average pixel value per field parcel. This gives robust values per parcel but may miss variability within field parcels which may be neccessary for use-cases requiring high spatial detail in a given year. 

This repository gives and end-to-end example for a sample of field parcels in Flanders, Belgium, which calculated the annual C-factor for a large sample of 25 % of the field parcels in Flanders, as well as the multi-year C-factor values for a smaller sample of crop rotations.

**Expanding the workflow across regions and years**

The entire Python workflow is based on EUROCROPS data and can therefore also be used or adapted for other regions in Europe. The scripts provided give an application in Flanders, Belgium, but with data inputs which are available in multiple European countries (IACS data compiled in EUROCROPS) and Sentinel-2. The workflow is therefore replicable in other European countries with minimal modifications.

The main module for the actual modelling of the C-factor for parcels in Flanders is the Flanders_Cfactor_sample.py script, located in the C_factor_calculation folder. The Common_modules folder, also present in this directory, contains essential functions that are used by the main module. Replication in other European regions or countries requires the data to be downloaded from EUROCROPS, sampled, uploaded in GEE to sample Sentinel-2 imagery, and run through the general C-factor module which runs the method for the desired field parcels. By leveraging Google Earth Engine, the workflow can be freely repeated in areas and years covered by EUROCROPS.

More information about EUROCROPS data can be found via the link below:
https://www.eurocrops.tum.de/index.html

**Funding**

This work was funded over multiple projects:
1) The Collaborative Doctoral Partnerships (CDP) initiative of the Joint Research Centre (JRC) grant number 35332 and the Fonds Wetenschappelijk Onderzoek Vlaanderen (Research Foundation Flanders- application S003017N). - focussed on the general methodological developments
2) The C-project issued to KU Leuven by the Departement Omgeving of the Flemish Government. - developed an in-depth regional application in Flanders, Belgium.

For questions, contributions, or commercial use inquiries, please contact fmatthews1381@gmail.com
