# 2024PIDDRN
Supplemental codes for "A Physics-Informed, Deep Double Reservoir Network for Forecasting Boundary Layer Velocity" by Matthew Bonas, David H. Richter and Stefano Castruccio

## Data
Folder containing the .RData file with the 8 Vicous Burgers' Equations simulated datasets used throughout the associated manuscript. The data used for the application are openly available and can be obtained through request [here](https://eprints.soton.ac.uk/416120/).

## Code
Folder containing the .R scripts used for implementing the DESN, DDRN, PIDESN, and PIDDRN models. We include files showing an example forecasting scenario for each method for both the Burgers' equation simulated data and the water field application data. To use the water field forecasting script, one should first request the data using the link above.

## Workflow
To reproduce results for the simulated Burgers’ equation data from the simulation study, one should first download the two .RData files from the “Data” folder. The next step is to then download the following .R files from the “Code” folder: “data_processing.R”, “deep_functions_physics_BurgerSim.R”, and “forecasting_BurgerSim.R”. A user should save all of the aforementioned file in the same directory, then open the file “forecasting_BurgerSim.R” and run it line by line to reproduce results for each of the methods. This file has lines of code that will load any data or functions from the other downloaded files. 

To reproduce the results for the water field application, one should first request the data as detailed above and preprocess it into a 2D matrix (Time by Space) and subsample the spatial locations as detailed in the manuscript. A user should then download the following two files from the “Code” folder: “deep_functions_physics_WaterApplication.R” and “forecasting_WaterApplication.R”, then open the file “forecasting_WaterApplication.R” and run it line by line to reproduce results for each of the methods. This file has lines of code that will load any functions from the other downloaded files but a user will have to add lines themselves to load their version of the preprocessed application data.


