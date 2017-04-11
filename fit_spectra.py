#!/usr/bin/env python
"""
This fits SpFrame spectra with airglow lines and a simple model for the continuum, 
including zodiacal light. The fit is done using a simple linear regression. 
After running the least squares fit, the fitted lines are separated from the continuum and 
the lines, continuum and residuals are returned independently.

INPUT: * SpFrame flux files as .npy files with wavelength and sky spectra 
       * Also need a list of airglow lines. these should be on the github repo

OUTPUT: numpy files identified in the same way as the sky_flux files with the plate number 
and file identifier "split_spectra". The numpy arrays have the following fields: WAVE, LINES, CONT, RESIDS

Before running, identify the directory that the spframe flux files are kept and where the airglow lines are
saved. Also identify where you want to save the files generated by this program (SAVE_DIR)

Title: SpFrame Flux Spectra Fit
Author: P. Fagrelius
Date: Mar. 21, 2017

"""
from __future__ import print_function,division
import glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from astropy.io import fits
from scipy import interpolate
import multiprocessing

parallel=True
MPI=False

#Directory to save data
SAVE_DIR = os.getcwd()+'/split_spectra/'
#Directory where all SpFrame flux files reside
SPECTRA_DIR = os.getcwd()+'/sky_flux/'

#Files from UVES that contain all airglow lines downloaded 
AIRGLOW_DIR = '/Users/parkerf/Research/BOSS_Sky/flux_repo/SkyModelling/AirglowSpectra/'

def main():
    # Load spectra data
    spectra_files = glob.glob(SPECTRA_DIR+'*.npy')
    print('Will be analyzing %d plate files' %len(spectra_files))

    #Get meta data
    global MetaData
    MetaData = np.load('meta_rich.npy')
    
    #Get zodi
    global ZodiLookup
    ZodiLookup = np.load('zodi_spectra.npy')
    
    #Get Airglowlines
    AirglowFiles = glob.glob(AIRGLOW_DIR+'/*.dat')
    AirglowLines = []
    for file in AirglowFiles:
        data = np.genfromtxt(file,skip_header=3,dtype=[('Sequence', '<i4'), ('CENTER', '<f8'), ('INT', '<f8'), ('FWHM', '<f8'), ('FLUX', '<f8')])
        AirglowLines.append(np.array(data))
    AirglowLines = np.hstack(AirglowLines)

    #Identify only "significant" lines and djust wavelength to vacuum 
    global VacLines
    sig = np.where(AirglowLines['FLUX']>1.)
    VacLines = air_to_vac(AirglowLines['CENTER'])
    VacLines = VacLines[sig]/10. #nm to A
    print("Number of Airglow Lines we will fit for: ", len(VacLines))

    ############
    ## SCRIPT ##
    ############

    
    if parallel:
        ## implement if MPI
        #multiprocessing speedup
        pool = multiprocessing.Pool(processes=32)
        data = pool.map(fit_and_separate_spectra, spectra_files)
        pool.terminate()
    else:
        data = [fit_and_separate_spectra(p) for p in spectra_files]


def get_zodi(ecl_lat, ecl_lon, wave_range):
    """ This returns the zodiacal spectrum over a specified wavelength range for a 
    given ecl_lat and ecl_lon. It takes the value of the zodiacal light at 500nm 
    for the lookup table "ZodiLookup" and then multiplies by the relative brightness
    at the given wavelength (in a solar spectrum). 
    """
    l = np.argmin([np.abs(ecl_lon-lon) for lon in ZodiLookup[0]])
    b = np.argmin([np.abs(ecl_lat-lat) for lat in ZodiLookup[1]])
    zodi = ZodiLookup[2][l][b]

    flux = interpolate.interp1d(zodi['WAVE'],zodi['ZODI'])
    Zodi = flux(wave_range)
    return Zodi

def clean_spectra(spectrum):
    """Takes out all nan/inf so lstsq will run smoothly
    """
    ok = np.isfinite(spectrum['SKY'])

    wave = spectrum['WAVE'][ok]
    sky = spectrum['SKY'][ok]
    sigma = spectrum['SIGMA'][ok]
    return [wave,sky,sigma]

def air_to_vac(wave):
    """Index of refraction to go from wavelength in air to wavelength in vacuum
    Equation from (Edlen 1966)
    vac_wave = n*air_wave
    """
    #Convert to um
    wave_um = wave*.001
    ohm2 = (1./wave_um)**(2)

    #Calculate index at every wavelength
    nn = []
    for x in ohm2:
        n = 1+10**(-8)*(8342.13 + (2406030/float(130.-x)) + (15997/float(389-x)))
        nn.append(n)
    
    #Get new wavelength by multiplying by index of refraction
    vac_wave = nn*wave
    return vac_wave

def airglow_line_components(airglow_lines, wave_range, sigma_range):
    """ Takes each Airglow line included in the analysis and creates a gaussian profile 
    of the line. 
    INPUT: - List of airglow lines wanted to model
           - Wavelength range of the spectra
           - Sigma for the wavelength range of the spectra
    OUTPUT: 
           Matrix with all lines used for lienar regression. Size[len(wave_range),len(airglow_lines)]
    """
    AA = []
    for line in airglow_lines:
        ss = []
        for i, w in enumerate(wave_range):
            sig = sigma_range[i]
            ss.append(np.exp(-0.5*((w-line)/sig)**2))
        AA.append(ss)
    return np.vstack(AA)

def lin_regress_model(airglow_lines, ecl_lat, ecl_lon, wave_range, sigma_range, sky_spectra):
    AA = airglow_line_components(airglow_lines, wave_range, sigma_range)

    # Continuum model
    A0 = np.ones(len(wave_range))
    A1 = wave_range
    A2 = np.square(wave_range)
    A3 = get_zodi(ecl_lat, ecl_lon, wave_range)

    A = np.stack(np.vstack((A0,A1,A2,A3,AA)),axis=1)
    lsq = lstsq(A,sky_spectra)

    num_comp = A.shape[1] - len(AA)

    model = np.dot(A,lsq[0])
    
    R_1 = np.sum([(i-np.mean(sky_spectra))**2 for i in model])
    R_2 = np.sum([(i-np.mean(sky_spectra))**2 for i in sky_spectra])  
    R = R_1/R_2  
    return [np.dot(A,lsq[0]),A,lsq,num_comp,R]

def split_spectra(model,flux):
    """ Takes model and separates the continuum from the sky lines
    """
    A = model[1]
    lsq = model[2]
    num_comp = model[3]

    cont = np.dot(A[:,0:num_comp],lsq[0][0:num_comp])
    lines = np.dot(A[:,num_comp:],lsq[0][num_comp:])
    res = flux - model[0]

    return [lines, cont, res]

def fit_and_separate_spectra(spectra_file):
    plate_num = spectra_file[-23:-19]
    print("Fitting spectra in plate %s" %plate_num)
    spectra = np.load(spectra_file)
    this_plate = MetaData[MetaData['PLATE'] == int(plate_num)]
    max_num = 50 #len(spectra) Number of spectra in a given plate that you want to run this for. Mostly for debugging
    num = 0
    data = []
   
    for i, spectrum in enumerate(spectra):
        if num < max_num:
            print('splitting spectra %d/%d for plate %s' % (i,len(spectra),plate_num))
            this_obs = this_plate[this_plate['SPECNO'] == i]
            wave,sky,sigma = clean_spectra(spectrum)
            model = lin_regress_model(VacLines, this_obs['ECL_LAT'], this_obs['ECL_LON'], wave, sigma, sky)
    
            #Split model
            split_model = split_spectra(model,sky)
            model_fit = np.zeros(len(sky),dtype=[('WAVE','f8'),('LINES','f8'),('CONT','f8'),('RESIDS','f8'),('R','f8')])
            model_fit['WAVE'] = wave
            model_fit['LINES'] = split_model[0]
            model_fit['CONT'] = split_model[1]
            model_fit['RESIDS'] = split_model[2]
            model_fit['R'] = model[4]
            data.append(model_fit)
            num+=1
        else:
            break

    np.save(SAVE_DIR+plate_num+'_split_fit',data)

        

if __name__=="__main__":
          main()

