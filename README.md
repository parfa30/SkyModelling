# SkyModelling
This git repo contains code to analyze the sky spectrum measured by BOSS and others. The primary goal is to create a model for the sky continuum and OH line emissions.

This repo is a collaboration between Parker Fagrelius, Anze Slocum, and David Kirkby and could grow to include more people.

Included in this repo:
* spframe_flux.py: Run to collect sky flux for all BOSS data. Turns spframe electrons into flux. Generates pkl files for flux, wavelength, and raw metadata.
* good_plates.txt: Includes QUALITY data for every plate-day from platelist.fits for BOSS data
* sky_fibers.npy: Includes each sky fiber identified for each BOSS plate
* no_spc_match.pkl: file that contains all observation skipped because there was no matching spc file.
* meta_raw.npy: file generated when running spframe_flux.py. 
* get_meta_rich.py: script that takes raw metadata in file meta_raw.npy and calculates additional meta data. output is file (quite large) called meta_rich.npy 
* MetaDataEval.ipynb: looks at the distribution of the metadata from meta_rich.npy and serves as a way to cross check the data
* fit_spectra.py: Takes spframe spectra and fits them using least squares. Returns the model, with the airglow lines, continuum, and residuals returned individually
* AirglowLines: Atlas of airglow lines identified by UVES. Using files in cosby.

