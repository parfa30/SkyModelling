"""
Title: Get Mean Spectra for SpFrame Flux files
Author: P. Fagrelius
Date: August, 2018

Description: This script makes mean spectra from the spframe files saved in SKY_FLUX_DIR. 
They are saved by observation (image, camera). Each spectrum is on wavelength grid of 
xx = np.linspace(300, 1040, (1040-300)*100)


==HOW TO RUN==

Run on nersc, with a python distribution that includes numpy, astropy, and scipy
python ./get_line_sum_file.py

"""

import astropy.table
import numpy as np 
import multiprocessing
import pickle
import os, sys, glob
from scipy.interpolate import interp1d

try:
    DATA_DIR = os.environ['SKY_FLUX_DIR']
except KeyError:
    print("Need to set SKY_FLUX_DIR\n export SKY_FLUX_DIR=`Directory to save data'")

MEAN_DIR = DATA_DIR+'/mean_spectra/'
if not os.path.exists(MEAN_DIR):
    os.makedirs(MEAN_DIR)


def main():

    #Get data. Checks is some data has already been collected
    spframe_files = glob.glob(DATA_DIR+"/*_calibrated_sky.npy")
    Complete_Rich_Plus = [d[0:4] for d in os.listdir(MEAN_DIR)]
    All_Rich = [d[0:4] for d in spframe_files]
    rich_plus_needed = [i for i, x in enumerate(All_Rich) if x not in Complete_Rich_Plus]
    these_spframe_files = np.array(spframe_files)[rich_plus_needed]
    print("Getting line sum data for %d files" % len(these_spframe_files))

    pool = multiprocessing.Pool(processes=64)
    pool.map(make_mean_spectrum, these_spframe_files)
    pool.terminate()

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def make_mean_spectrum(spframe_filen):

    plate = int(os.path.split(spframe_filen)[1][0:4])
    data = np.load(spframe_filen)
    xx = np.linspace(300, 1040, (1040-300)*100)

    PLATE_DIR = MEAN_DIR+'/%d' % plate
    if not os.path.exists(PLATE_DIR):
        os.makedirs(PLATE_DIR)

    image_meta = np.load(DATA_DIR+'/raw_meta/%d_raw_meta.npy' % plate)
    for image in np.unique(image_meta['IMG']):
        for cam in np.unique(image_meta['CAMERAS']):
            these_specnos = image_meta[(image_meta['IMG'] == image)&(image_meta['CAMERAS'] == cam)]['SPECNO']
            SKY_SPECTRA = []
            VARS = []
            for specno in these_specnos:
                spectrum = data[specno]
                sky = spectrum['SKY']
                nans, x = nan_helper(sky)
                try:
                    sky[nans]= np.interp(x(nans), x(~nans), sky[~nans])
                    f = interp1d(spectrum['WAVE'], sky, bounds_error=False, fill_value=0)
                    g = interp1d(spectrum['WAVE'], spectrum['IVAR'], bounds_error=False, fill_value=0)
                    SKY_SPECTRA.append(f(xx))
                    VARS.append(g(xx))
                except:
                    print(image, cam)
            mean_spectrum = np.ma.average(np.array(SKY_SPECTRA), axis=0, weights=np.array(VARS))
            mean_var = np.average(np.array(VARS), axis=0)
            mean_spectrum.dump(PLATE_DIR+'/%d_%s_mean_spectrum.npy'% (image, str(cam)[2:4]))
            mean_var.dump(PLATE_DIR+'/%d_%s_mean_var.npy'% (image, str(cam)[2:4]))


if __name__ == '__main__':
    main()





