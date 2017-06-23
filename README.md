# SkyModelling
This git repo contains code to:
1) Create a dataset of sky flux from BOSS data. This dataset includes each sky fiber for each plate/image in the BOSS data on nersc.
2) Create file of meta data for all sky spectra
3) Fit sky spectra and separate the airglow lines from the continuum.

## Included in this repo:
* spframe_flux.py: Run to collect sky flux for all BOSS data. Turns spframe electrons into flux. Generates pkl files for flux, wavelength, and raw metadata.
* fit_spectra.py: Takes spframe spectra and fits them using least squares. Returns the model, with the airglow lines, continuum, and residuals returned individually
* get_meta_rich.py: script that takes raw metadata in file meta_raw.npy and calculates additional meta data. output is file (quite large) called meta_rich.npy 
* good_plates.txt: Includes QUALITY data for every plate-day from platelist.fits for BOSS data. This is used by MetaDataEval.ipynb.
* sky_fibers.npy: Includes each sky fiber identified for each BOSS plate. This is used by spframe_flux.py.
* MetaDataEval.ipynb: looks at the distribution of the metadata from meta_rich.npy and serves as a way to cross check the data
* AirglowLines: Atlas of airglow lines identified by UVES. Using files in cosby.
* bitmask.py: This code is used by spframe_flux.py to decode the pixel mask.
* spframeflux1.slurm: sbatch code to run on nersc.
* fitspectra1.slurm: sbatch code to run on nersc.

## Code Dependencies
All code is in python. You will need the following packages:
* numpy, scipy, pandas, matplotlib
* astropy
* ephem (pip install ephem)
* multiprocessing
* statsmodel (conda install statsmodels)

## How to get your own sky flux dataset
* Clone this repo to nersc
* Identify a location to save your data and modify spframe_flux.py: `SAVE_DIR` with that location
* Run the batch script "spframeflux1.slurm" by typing `sbatch spframeflux1.slurm`. To download all files you'll have to run this 5 times.
* It should take ~2.5 hours to run this. This will output a 2 .npy files for each plate: "plate_calibrated_sky.npy" and "raw_meta/plate_raw_meta.npy"
* When complete, modify get_rich_meta_data.py: `RAW_META` with the location of your saved meta data (`SAVE_DIR+/raw_meta/`)
* run get_rich_meta_data.py. This will give you a single meta data file for your full dataset. This should taken ~1hour.

## How to fit your sky spectra
* Identify a location for your fit/split data
* Modify fit_spectra.py: `SAVE_DIR` with this location and `SPECTRA_DIR` with the location of the saved data above
* Run the batch script "fitspectra1.slurm" by typing `sbatch fitspectra1.slurm`
* This will take ~x hours. You will need to run the slurm script ~x times.
