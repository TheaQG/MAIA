# MAIA

The MAIA project is focused on examining the correlations between water vapour transport into the arctic and aerosol concentrations in the region. The project is carried out by a team of researchers (U. Imre, P. L. Langen, and T. Quistgaard) at the Department of Environmental Science, Aarhus University.

The project is funded by ... and is part of theResearch Project ...

The project examines different aspects of the water vapour transport and aerosol concentrations in the arctic through ground-based measurements, reanalysis data, and model simulations.


## OBSERVATIONS

## MODELS

## MIXED



### Contents of the repository

- #### **'Data\'**: contains the data used in the project
    - *NCEP_MERRA_Arctic_Slice_MoistureFlux.csv*: Contains timeseries of modelled (AMIP Non-Interactive Nudged) moisture fluxes - nudged either with NCEP or MERRA reanalysis data.
    - *TS_AerosolBurden_ArcticSlice.txt*: Contains timeseries of modelled aerosol concentrations (Black Carbon, Dust, Organic Carbon, Sulphate, SS and PM25) in the section 45W-45E and 70N-90N.
    - *TS_AerosolBurden_ArcticSlice_small.txt*: Contains timeseries of modelled aerosol concentrations (Black Carbon, Dust, Organic Carbon, Sulphate, SS and PM25) in the section 45W-45E and 70N-75N. 
- #### **'Figures\'**: contains the figures produced in the project
    - **'ModResults\'**: contains the figures produced from the model results
    - **'ObsResults\'**: contains the figures produced from the observations
    - **'MixedResults\'**: contains the figures produced from the mixed model and observation
- #### **'Scripts\'**: contains the scripts used in the project for analysis and plotting
    - **'Analysis\'**: contains the scripts used for the analysis
    - **'Examinations\'**: contains the scripts used for the examination of the data and testing of the scripts
    - **'Plotting\'**: contains the scripts used for the plotting
    - **'Preprocessing\'**: contains the scripts used for loading and preprocessing of the data
    - **'Results\'**: contains the results produced in the project


- 'aerosol_meas.py':
- 'cdsAPI_all_MAIA_download.py':
- 'climatology.py':
- 'comp_6hr_mean.sh':
- 'event_analysis_class.py':
- 'event_analysis.py':
- 'MAIA__NWVF_DataProcessing.py':
- 'MAIA_Temp_SeaIceCover_DataProcessing.py':
- 'MAIA_examination.py':
- 'MERRA_aerosol_event_analysis.py':
- 'MERRA_aerosol_output.py':
- 'model_aerosol_climatology.py':
- 'model_aerosol_event_analysis.py':
- 'monthly_events.py: