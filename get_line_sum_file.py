"""
Title: Get Line Sum for SpFrame Files
Author: P. Fagrelius
Date: August, 2018

Description: This file takes the rich meta files and the spframe files and measures the
line strenghts and mean continuum values of those listed below, and then save all this
with the rich meta data as astropy Tables in rich_plus/####_rich_plus.fits.

The lines are saved in util/line_file.pkl. For the airglow lines, the code takes the center
wavelength and sums +/- 10 pixels around that. For the continuum points (identified as cont_)
the code takes the mean of the +/- 10 pixels around the center wavelength.

==HOW TO RUN==
Identify the folde rwhere the spframe_flux files and rich_meta files are saved as DATA_DIR.

Run on nersc, with a python distribution that includes numpy, astropy, and scipy
srun -n 1 -c 64 python ./get_line_sum_file.py

"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import astropy.table
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy.optimize import curve_fit
import multiprocessing
import pickle
from datetime import datetime
import speclite.filters
import astropy.units as u

#Identify the folder where you data is saved
try:
    DATA_DIR = os.environ['SKY_FLUX_DIR']
except KeyError:
    print("Need to set SKY_FLUX_DIR\n export SKY_FLUX_DIR=`Directory to save data'") 

SAVE_DIR = DATA_DIR+'/rich_plus_meta/'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def main():
    global Lines
    Lines = pickle.load(open(os.getcwd()+'/util/line_file.pkl','rb'))

    vega = astropy.table.Table.read('ftp://ftp.stsci.edu/cdbs/current_calspec/alpha_lyr_stis_008.fits')
    vegawave, vegaflux = np.array(vega["WAVELENGTH"]), np.array(vega['FLUX'])
    global VEGA
    VEGA = interp1d(vegawave, vegaflux, bounds_error=False, fill_value = 0)
    

    #Get data. Checks is some data has already been collected
    rich_files = glob.glob(DATA_DIR+"/rich_meta/*_rich_meta.fits")
    Complete_Rich_Plus = [d[0:4] for d in os.listdir(SAVE_DIR)]
    All_Rich = [d[0:4] for d in os.listdir(DATA_DIR+'/rich_meta/')]
    rich_plus_needed = [i for i, x in enumerate(All_Rich) if x not in Complete_Rich_Plus]
    these_rich_files = np.array(rich_files)[rich_plus_needed]
    print("Getting line sum data for %d files" % len(these_rich_files))

    ## Get LINE SUMS. Multiprocessing pool runs through rich files.
    start = datetime.now()

    pool = multiprocessing.Pool(processes=64)
    pool.map(get_line_sums, these_rich_files)
    pool.close()

    total_time = (datetime.now() - start).seconds
    print('finished %d plates in %.2f seconds' % (len(these_rich_files),total_time))

def get_vmag(filt, flux, wave_range):
    """Calculates vega magnitudes
    """
    znum = filt.convolve_with_array(wave_range, VEGA(wave_range))
    zden = filt.convolve_with_function(lambda wlen: u.Quantity(1))
    zp = (znum/zden)
    
    mnum = filt.convolve_with_array(wave_range, flux)
    mden = filt.convolve_with_function(lambda wlen: u.Quantity(1))
    mp = (mnum/mden)
    
    M = -2.5 * np.log10(mp/zp) 
    return M

def get_line_sums(rich_file):
    print(rich_file)
    Meta = astropy.table.Table.read(rich_file)

    fiber_area = np.pi
    ugriz = speclite.filters.load_filters('sdss2010-*')
    bessell = speclite.filters.load_filters('bessell-*') 

    #add columns to astropy.table for each line
    for camera, lines in Lines.items():
        if (camera == 'b1')|(camera=='r1'): #only want one set
            for name, line_info in lines.items():
                Meta[name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))
    for filt in ugriz:
        Meta[filt.name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))
    for filt in bessell:
        Meta[filt.name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))

    #get spframe flux data
    plate = np.unique(Meta['PLATE'])[0]
    data = np.load(DATA_DIR+'/%d_calibrated_sky.npy'%plate)
    num_pix_lines = 5 #+/- pixels used for sum
    num_pix_cont = 1
    
    for meta in Meta:
        try:
            spectrum = data[meta['SPECNO']]

            #Clean data
            ok = np.isfinite(spectrum['SKY'])
            sky = spectrum['SKY'][ok]
            wave = spectrum['WAVE'][ok]
            lines = Lines[meta['CAMERAS']]
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
        except:
            pass

    #save astropy table as fits file in rich_plus
    rich_id = os.path.split(rich_file)[1][0:6]
    rich_filen = SAVE_DIR+'/%s_meta_with_lines.fits'%rich_id
    if os.path.exists(rich_filen):
        os.remove(rich_filen)
    Meta.write(rich_filen,format='fits')


if __name__ == '__main__':
    main()
