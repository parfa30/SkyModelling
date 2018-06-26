"""
Title: Get Line Means from line sum files
Author: Parker Fagrelius
Date: April, 2018

As an input, this script takes the ####_rich_plus.fits files which contain the line strengths and mean
continuum values for a list of lines in util/line_list.pkl and adds them to the rich meta files.
It then calculates the mean and std for an observation (plate, image, camera). This data can be 
used to identify outliers and also differntials in flux across the focal plate.

==HOW TO RUN==
Identify the data directory where the rich_plus directory sits in DATA_DIR (not the rich_plus directory but one up).

Run on nersc. Python distribution requires: numpy, astropy, scipy

srun -c 1 -n 64 python ./get_line_mean_file.py

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

DATA_DIR = '/global/cscratch1/sd/parkerf/sky_flux/' #'/Volumes/PFagrelius_Backup/sky_data/sky_flux/'

def main():
    if not os.path.exists(DATA_DIR+'rich_mean/'):
        os.makedirs(DATA_DIR+'rich_mean/')

    global Lines
    Lines = pickle.load(open(os.getcwd()+'/util/line_file.pkl','rb'))

   
    #Get data
    rich_plus_files = glob.glob(DATA_DIR+"rich_plus/*_rich_plus.fits")
    Complete_Rich_Mean = [d[0:4] for d in os.listdir(DATA_DIR+'rich_mean/')]
    All_Rich_Plus = [d[0:4] for d in os.listdir(DATA_DIR+'rich_plus/')]
    rich_mean_needed = [i for i, x in enumerate(All_Rich_Plus) if x not in Complete_Rich_Mean]
    these_rich_plus_files = np.array(rich_plus_files)[rich_mean_needed]
    print("Getting line sum data for %d files" % len(these_rich_plus_files))

    #Calculate the mean values. Multiprocessing pool runs one rich_plus file at a time   
    start = datetime.now()
    pool = multiprocessing.Pool(processes=64)
    mean_rich_plus = pool.map(get_image_means, these_rich_plus_files)
    pool.close()
    total_time = (datetime.now()-start).seconds
    print('Total time: %.2f'%total_time)

def get_image_means(rich_plus_file):
    Rich_Plus = astropy.table.Table.read(rich_plus_file)
    plate = np.unique(Rich_Plus['PLATE'])[0]

    #Add columns to astropy.Table
    for camera, lines in Lines.items():
        if (camera == 'b1')|(camera=='r1'): #only want one set
            for name, line_info in lines.items():
                Rich_Plus["mean_"+name] = astropy.table.Column(np.zeros(len(Rich_Plus)).astype(np.float32))
                Rich_Plus['std_'+name] = astropy.table.Column(np.zeros(len(Rich_Plus)).astype(np.float32))

    #Collect meta data in groups of observations (plate, image, camera)
    New_Meta = []    
    for img in np.unique(Rich_Plus[Rich_Plus['PLATE']==plate]['IMG']):
        for cam in np.unique(Rich_Plus[Rich_Plus['IMG'] == img]['CAMERAS']):
            This_Meta = Rich_Plus[(Rich_Plus['IMG'] == img) & (Rich_Plus['CAMERAS'] == cam)]

            lines = Lines[cam]
            for name, line in lines.items():
                This_Meta["mean_"+name] = np.mean(This_Meta[name])
                This_Meta['std_'+name] = np.std(This_Meta[name])
            New_Meta.append(This_Meta)

    #Save new astropy.table as fits file for each input file.
    Rich_Mean_Meta = astropy.table.vstack(New_Meta) 
    Rich_Mean_Meta.write(DATA_DIR+'/rich_mean/%d_rich_plus_mean.fits' % plate,format = 'fits')


if __name__ == '__main__':
    main()
