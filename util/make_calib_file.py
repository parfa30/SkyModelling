"""
This file takes the spFluxcalib file that we want to use and turns it into a more useful format
"""

import numpy as np
from astropy.io import fits
import glob, os
import pickle

calib_folder = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/v5_7_0/3985'
my_calib_files = glob.glob(calib_folder+'/spFluxcalib-*-00114439*')

CV = {}
for filen in my_calib_files:
    cam = os.path.splitext(os.path.splitext(filen)[0])[0][-11:-9]
    hdu = fits.open(filen)
    CV[cam] = hdu[0].data
    hdu.close()

pickle.dump(CV, open('CalibVector.pkl','wb'))
print("Done")
