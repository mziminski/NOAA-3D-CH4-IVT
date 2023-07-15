# NOAA-3D-CH4-IVT 

The *NOAA 3-D Methane Interactive Visualization Tool* was completed for my Master's Project Option project, and is informally published on UIC's Electronic Visualization Labratory's (EVL) website, link: https://www.evl.uic.edu/pubs/2718.

# Repository Structure
This repository consists of three main sub-directories:

1. ```./dash-app``` consists of code for a dash-app implementation of the IVT.
2. ```./flask-app``` consists of a Python Flask app as the backend and a PlotlyJS frontend implementation of the IVT. 
3. ```./py-plotly2html``` concosts of PlotlyPy code used during the prototyping phase to create the volumetric and sphereic Plotly plots separately.
4. ```./data``` is MISSING (i.e. hidden by .gitignore). Please create this directory after cloning this repository, and go to https://gml.noaa.gov/ccgg/carbontracker-ch4/ to download the current version of the the CH4 Mole Fraction dataset to run the code found within the above sub-directories. However, any 3-D climate/atmospheric dataset should work with "minor" changes to the code found in the above sub-directories.

# Dashboard Screenshot
![image](https://github.com/mziminski/NOAA-3D-CH4-IVT/assets/26189380/ec06f97b-2845-4f0d-a7dc-458eb23d0f6b)


# TODO:
1. Optimizations/refactorings have yet to be imlemented to speed up the plot loading times
