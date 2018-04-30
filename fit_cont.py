#!/usr/bin/env python
"""
This program essentially takes the place of 'fit_spectra.py'.

This fits SpFrame spectra with airglow lines and a simple model for the continuum.
The fit is done with a nonlinear fit of a handful of airglow lines with a profile of
two gaussians and a lorentzian for the scattered light from the VPH grating.

INPUT: * SpFrame flux files as .npy files with wavelength and sky spectra

OUTPUT: fits files arranged by plate number: '####_split_flux.fits'.
        The fits file will be organized as follows:
        * HDU[1] includes the meta data for each observation in the fits file
        * The remaining HDUs are named by their associated SPECNO that is in the MetaData
          files and identifies each spectrum with their spframe flux counter part
        * Each remaining HDU includes the following: WAVE, CONT, LINES, FLUX


Title: SpFrame Flux Continuum Fit
Author: P. Fagrelius
Date: March, 21018

export OMP_NUM_THREADS=1 (32/#) multiprocessing)

MPI - multiprocessing across nodes. not great for python data parallelization. steep learnign curve
Quequ do. 

"""

import os, sys
import glob
import numpy as np
from datetime import datetime
import astropy.table
from astropy.io import fits
from scipy.optimize import least_squares, curve_fit
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num", type=int,
                    help="group number of plates to be run")
args = parser.parse_args()

#Set up directories
SAVE_DIR = '/global/cscratch1/sd/parkerf/split_flux/dark_blue_split/'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

SPDATA_DIR = '/global/cscratch1/sd/parkerf/sky_flux/' #'/Volumes/PFagrelius_Backup/sky_data/sky_flux/' #Spframe flux files


def scatter_profile(x, amplitude, center, N_eff):
    w = center/N_eff * (1/(np.sqrt(2)*np.pi))
    top = w**2.
    bot = ((x-center)**2+w**2)
    l = amplitude*top/bot
    return l

def my_profile(x, *args):
    amp1, amp2, a, center, wave1, wave2, sig1, sig2, N = list(args[0])
    gauss1 = amp1*np.exp(-(x-wave1)**2/(2*sig1**2.))
    gauss2 = amp2*np.exp(-(x-wave2)**2/(2*sig2**2.))
    core = gauss1 + gauss2

    scatt = scatter_profile(x, a, center, N)
    return core + scatt

def my_model(wave, *P):
    model = None
    for i, line in enumerate(blue_vac_lines):
        amp1, amp2, a, center, wave1, wave2, sig1, sig2, N = np.array(P)[i*9:(i+1)*9]
        gauss1 = amp1*np.exp(-(wave-wave1)**2/(2*sig1**2.))
        gauss2 = amp2*np.exp(-(wave-wave2)**2/(2*sig2**2.))
        core = gauss1 + gauss2

        w = center/N * (1/(np.sqrt(2)*np.pi))
        top = w**2.
        bot = ((wave-center)**2+w**2)
        scatt = a*top/bot
        line_model = core+scatt

        if model is None:
            model = line_model
        else:
            model = model + line_model
    return model + P[-1]

def get_cont(wave, data, *P):
    CONT = data.copy()
    LINES = None
    for i, line in enumerate(blue_vac_lines):
        line_model = my_profile(wave, np.array(P)[i*9:(i+1)*9])
        if LINES is None:
            LINES = line_model
        else:
            LINES = LINES+line_model
        remove_idx = np.where(line_model > .0001)[0]
        CONT = CONT - line_model
        try:
            all_pix = np.hstack([CONT[remove_idx[0]-5:remove_idx[0]-2], CONT[remove_idx[-1]+2:remove_idx[-1]+5]])
            mean_area = np.mean(all_pix)
            CONT[remove_idx[0]-2:remove_idx[-1]+2] = mean_area
        except:
            pass
    return CONT, LINES


def fit_sky_spectrum(spectrum, params, low_bounds, high_bounds):
    start = datetime.now()

    # Clean Spectrum
    ok = ((np.isfinite(spectrum['SKY'])) & (spectrum['SKY'] > 0.))

    sky = spectrum['SKY'][ok]
    ivar = spectrum['IVAR'][ok]
    disp = spectrum['DISP'][ok]
    wave = spectrum['WAVE'][ok]
    
    #Fit Spectrum
    popt, popc = curve_fit(my_model, wave, sky, params, sigma=1/np.sqrt(ivar), absolute_sigma = True, bounds = (low_bounds, high_bounds))
    cont, lines = get_cont(wave, sky, *popt) 

    model_fit = np.zeros(len(cont), dtype=[('WAVE', 'f8'), ('LINES', 'f8'), ('CONT', 'f8'), ('FLUX', 'f8')])
    model_fit['WAVE'] = np.array(wave)
    model_fit['LINES'] = np.array(lines)
    model_fit['CONT'] = np.array(cont)
    model_fit['FLUX'] = np.array(sky)
    print('time for spectrum: %.2f' %(datetime.now()-start).seconds)

    return model_fit

def fit_plate_continuum(plate):
  
    start = datetime.now()
    print("Running Fit and Split for Plate %d" % int(plate))

    data = np.load(SPDATA_DIR+'/%s_calibrated_sky.npy' % str(int(plate)))

    PlateMeta = MetaData[(MetaData['PLATE']==int(plate))&((MetaData['CAMERAS']=='b1')|(MetaData['CAMERAS']=='b2'))]

    plate_spectra = data[PlateMeta['SPECNO']]

    print("Splitting continuum of %d spectra" % len(plate_spectra))

    ## Create guesses
    P = []    
    Bounds_low = []
    Bounds_high = []   
    for line in blue_vac_lines:
        if line == 557.89:
            P.append([200,200,10,line,line+.06,line-.09,0.1,0.1,83200])
        else:
            P.append([10,10,10,line,line+.06,line-.09,0.1,0.1,83200])
        Bounds_low.append([0,0,0,line-0.1,line,line-0.1,0,0,500])
        Bounds_high.append([np.inf,np.inf,np.inf,line+0.1,line+0.1,line,1,1,83200])
    Bounds_low.append(0)
    Bounds_high.append(10) 
    P.append(1)

    fit_data = []
    for spectrum in plate_spectra:
        fit_data.append(fit_sky_spectrum(spectrum, np.hstack(P), np.hstack(Bounds_low), np.hstack(Bounds_high)))
         
    np.save('%d_fit_cont', fit_data)
    print('plate time %.2f' % (datetime.now() - start).seconds)

def main():
    global blue_vac_lines
    blue_vac_lines = np.load('/global/homes/p/parkerf/Sky/SkyModelling/files/blue_vac_lines.npy') #List of airglow lines converted to vacuum wavelength

    #Meta data
    global MetaData
    hdu = fits.open('/global/homes/p/parkerf/Sky/SkyModelling/spframe_line_sum.fits')
    MetaData = astropy.table.Table(hdu[1].data)
    hdu.close()

    # Select plates
    filen = '/global/homes/p/parkerf/Sky/SkyModelling/files/dark_data.fits'
    data = astropy.table.Table.read(filen)
    plates = np.unique(data['PLATE'])

    #Check which files have been completed
    print("Using directiory %s" % SAVE_DIR)
    Complete_files = glob.glob(SAVE_DIR+"/*_split_flux.fits")
    Completed_Plates = [int(os.path.split(filen)[1][0:4]) for filen in Complete_files]
    print("Completed Plates: ",Completed_Plates)

    plates_needed = np.array([p for p in plates if p not in Completed_Plates])
    print("Plates still needed: (%d)"%len(plates_needed),plates_needed)

    group_of_plates = np.array_split(plates_needed, 32)
    these_plates = group_of_plates[args.num]
    print(these_plates)

    for plate in these_plates:
       fit_plate_continuum(plate)


        
if __name__ == '__main__':
    main()
    

