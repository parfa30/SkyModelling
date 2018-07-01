"""
Title: Get Line Sum for SpFrame Files
Author: P. Fagrelius
Date: April, 2018

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

#Identify the folder where you data is saved
DATA_DIR = '/global/cscratch1/sd/parkerf/sky_flux_corrected/' #'/Volumes/PFagrelius_Backup/sky_data/sky_flux/'

def main():
    global Lines
    Lines = pickle.load(open(os.getcwd()+'/util/line_file_updated.pkl','rb'))

    if not os.path.exists(DATA_DIR+'rich_plus/'):
        os.makedirs(DATA_DIR+'rich_plus/')

    #Get data. Checks is some data has already been collected
    rich_files = glob.glob(DATA_DIR+"rich_meta/*_rich_meta.fits")
    Complete_Rich_Plus = [d[0:4] for d in os.listdir(DATA_DIR+'rich_plus/')]
    All_Rich = [d[0:4] for d in os.listdir(DATA_DIR+'rich_meta/')]
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

def get_line_sums(rich_file):
    print(rich_file)
    Meta = astropy.table.Table.read(rich_file)
    #add columns to astropy.table for each line
    for camera, lines in Lines.items():
        if (camera == 'b1')|(camera=='r1'): #only want one set
            for name, line_info in lines.items():
                Meta[name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))

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
                my_pix = np.argmin(np.abs(wave - line))
                if Type == 'cont':
                    flux = np.mean(sky[my_pix - num_pix_cont: my_pix + num_pix_cont]) 
                elif Type == 'line':
                    flux = np.sum(sky[my_pix - num_pix_lines: my_pix + num_pix_lines])
                else: 
                    print("not a good type")

                meta[name] = astropy.table.Column([flux])

    #save astropy table as fits file in rich_plus
    rich_filen = DATA_DIR+'rich_plus/%d_rich_plus.fits'%plate
    if os.path.exists(rich_filen):
        os.remove(rich_filen)
    Meta.write(rich_filen,format='fits')



if __name__ == '__main__':
    main()
