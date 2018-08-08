import astropy.table
import numpy as np 
import multiprocessing
import pickle
import os, sys, glob
from scipy.interpolate import interp1d

MetaFile = 'data_files/good_clean_data_180704.fits'
DIR = '/global/cscratch1/sd/parkerf/sky_flux_corrected/'

def main():

    MEAN_DIR = DIR+'/mean_spectra/'
    if not os.path.exists(MEAN_DIR):
        os.makedirs(MEAN_DIR)

    global MF
    MF = astropy.table.Table.read(MetaFile)
    needed_images = np.load('needed_images.npy')

    newMF = []
    for image in needed_images:
        newMF.append(MF[MF['IMG']==image])
    newMF = astropy.table.vstack(newMF)
    print(np.unique(newMF['PLATE']))

    PLATES = np.unique(newMF['PLATE'])

    pool = multiprocessing.Pool(processes=64)
    pool.map(make_mean_spectrum, PLATES)
    pool.terminate()

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def make_mean_spectrum(plate):
    data = np.load(DIR+'%d_calibrated_sky.npy' % plate)
    xx = np.linspace(300, 1040, (1040-300)*100)

    PLATE_DIR = DIR+'/mean_spectra/%d' % plate
    if not os.path.exists(PLATE_DIR):
        os.makedirs(PLATE_DIR)

    image_meta = MF[MF['PLATE']==plate]
    for image in np.unique(image_meta['IMG']):
        for cam in ['b1','b2','r1','r2']:
            these_specnos = image_meta[(image_meta['IMG'] == image)&(image_meta['CAMERAS'] == cam)]['SPECNO']
            SKY_SPECTRA = []
            VARS = []
            for specno in these_specnos:
                spectrum = data[specno]
                sky = spectrum['SKY']
                nans, x = nan_helper(sky)
                sky[nans]= np.interp(x(nans), x(~nans), sky[~nans])
                f = interp1d(spectrum['WAVE'], sky, bounds_error=False, fill_value=0)
                g = interp1d(spectrum['WAVE'], spectrum['IVAR'], bounds_error=False, fill_value=0)
                SKY_SPECTRA.append(f(xx))
                VARS.append(g(xx))
            
            mean_spectrum = np.ma.average(np.array(SKY_SPECTRA), axis=0, weights=np.array(VARS))
            mean_var = np.average(np.array(VARS), axis=0)
            mean_spectrum.dump(PLATE_DIR+'/%d_%s_mean_spectrum.npy'% (image, cam))
            mean_var.dump(PLATE_DIR+'/%d_%s_mean_var.npy'% (image, cam))


if __name__ == '__main__':
    main()





