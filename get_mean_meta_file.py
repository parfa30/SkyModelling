"""
Title: Get Mean Meta File for SpFrame Files
Author: P. Fagrelius
Date: August, 2018

Description: This file takes the rich meta files and the mean spectrum files and measures the
line strenghts and mean continuum values of those listed below.

The meta data is resaved by plate under SKY_FLUX_DIR/mean_meta/PLATE_mean_meta.fits

==HOW TO RUN==

Run on nersc, with a python distribution that includes numpy, astropy, and scipy
Run on node that has 64 processes
python get_mean_meta_file.py

"""

import astropy.table
import numpy as np 
import os, sys, glob 
import pickle
import speclite.filters
import multiprocessing
from scipy.interpolate import interp1d
from astropy import units as u

try:
    DATA_DIR = os.environ['SKY_FLUX_DIR']
except KeyError:
    print("Need to set SKY_FLUX_DIR\n export SKY_FLUX_DIR=`Directory to save data'")

MEAN_META = DATA_DIR+'/mean_meta/'
if not os.path.exists(MEAN_META):
    os.makedirs(MEAN_META)

def main():

    #Information to measure broadband magnitudes
    vega = astropy.table.Table.read('ftp://ftp.stsci.edu/cdbs/current_calspec/alpha_lyr_stis_008.fits')
    vegawave, vegaflux = np.array(vega["WAVELENGTH"]), np.array(vega['FLUX'])
    global VEGA
    VEGA = interp1d(vegawave, vegaflux, bounds_error=False, fill_value = 0)
    global ugriz
    ugriz = speclite.filters.load_filters('sdss2010-*')
    global bessell
    bessell = speclite.filters.load_filters('bessell-*')

    #Line list that will be measured in mean spectra
    global Lines
    Lines = pickle.load(open('/global/homes/p/parkerf/Sky/SkyModelling/util/line_file.pkl','rb'))


    #Get data. Checks is some data has already been collected
    rich_files = glob.glob(DATA_DIR+"rich_meta/*_rich_meta.fits")
    Complete_Rich_Plus = [d[0:4] for d in os.listdir(MEAN_META)]
    All_Rich = [d[0:4] for d in os.listdir(DATA_DIR+'rich_meta/')]
    rich_plus_needed = [i for i, x in enumerate(All_Rich) if x not in Complete_Rich_Plus]
    these_rich_files = np.array(rich_files)[rich_plus_needed]
    print("Getting line sum data for %d files" % len(these_rich_files))


    pool = multiprocessing.Pool(processes=64)
    pool.map(make_mean_meta_file, these_rich_files)
    pool.terminate()

def get_vmag(filt, flux, wave_range):
    """Calculates the vega magnitudes
    """
    znum = filt.convolve_with_array(wave_range, VEGA(wave_range))
    zden = filt.convolve_with_function(lambda wlen: u.Quantity(1))
    zp = (znum/zden)
    
    mnum = filt.convolve_with_array(wave_range, flux)
    mden = filt.convolve_with_function(lambda wlen: u.Quantity(1))
    mp = (mnum/mden)
    
    M = -2.5 * np.log10(mp/zp) 
    return M

 
def make_mean_meta_file(rich_filen):

    #Get meta data and just pull the unique observation
    Meta = astropy.table.Table.read(rich_filen)
    Meta.remove_columns(['SPECNO', 'FIB', 'XFOCAL','YFOCAL','FIBER_RA','FIBER_DEC'])
    ThisMeta = astropy.table.unique(MF, keys=['PLATE','IMG','CAMERAS'])
    plate = np.unique(ThisMeta['PLATE'])[0]
    print(plate)

    #add columns to astropy.table for each line
    for camera, lines in Lines.items():
        if (camera == 'b1')|(camera=='r1'): #only want one set
            for name in lines.keys():
                ThisMeta[name] = astropy.table.Column(np.zeros(len(ThisMeta)).astype(np.float32))
    for filt in ugriz:
        ThisMeta[filt.name] = astropy.table.Column(np.zeros(len(ThisMeta)).astype(np.float32))
    for filt in bessell:
        ThisMeta[filt.name] = astropy.table.Column(np.zeros(len(ThisMeta)).astype(np.float32))

    
    mean_spectra_dir = DATADIR+'/mean_spectra/%d/' % plate
    num_pix_lines = 5 #+/- pixels used for sum
    num_pix_cont = 1
    fiber_area = np.pi

    for meta in ThisMeta:
        image = meta['IMG']
        cam = meta['CAMERAS']
        sky = np.load(mean_spectra_dir+'/%d_%s_mean_spectrum.npy'% (image, cam))
        wave = np.linspace(300, 1040, (1040-300)*100)
        lines = Lines[cam]
        for name, info in lines.items():
            Type, line = info
            line = float(line)
            my_pix = np.argmin(np.abs(wave - line))
            if Type == 'cont':
                flux = np.mean(sky[my_pix - num_pix_cont: my_pix + num_pix_cont])
            elif Type == 'line':
                flux = np.sum(sky[my_pix - num_pix_lines: my_pix + num_pix_lines])
            else:
                print("not a good type")

            meta[name] = astropy.table.Column([flux])

        for filt in ugriz:
            flux, wlen = filt.pad_spectrum(1e-17*sky/fiber_area, wave*10)
            mag = filt.get_ab_magnitude(flux,wlen)
            meta[filt.name] = astropy.table.Column([mag])
        for filt in bessell:
            flux, wlen = filt.pad_spectrum(1e-17*sky/fiber_area, wave*10)
            mag = get_vmag(filt, flux, wlen)
            meta[filt.name] = astropy.table.Column([mag])

    #save astropy table as fits file in rich_plus
    rich_mean_filen = MEAN_META+'/%d_mean_meta.fits'%plate
    if os.path.exists(rich_mean_filen):
        os.remove(rich_mean_filen)
    ThisMeta.write(rich_mean_filen,format='fits')


if __name__ == '__main__':
    main()





