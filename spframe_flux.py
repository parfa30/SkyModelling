#!/usr/bin/env python

"""
Title: SpFrame flux conversion
Author: P. Fagrelius, A. Slosar

Version:
3.0 Jun, 2017 Modification of flux calculation and removal of flagged pixels (Parker)
2.1 Mar, 2017 Refactoring by A. Slosar
1.0 Apri, 2017 P. Fagrelius


==Inputs==
This code takes the spectra from spFrame BOSS files and converts it into flux similar to that in spCframe files.
Flux output is 10^-17 ergs/s/cm2/A.

This conversion is made only for "sky" fibers as identified in the plug list in spframe files hdu[5]. These have 
OBJTYPE=SKY. A full list of all sky fibers is saved in a pickle file 'sky_fibers.pkl' with associated plate and camera.

The flux is converted with the equation spflux = eflux(spFrame[0])/spCalib.
This is equal to spcframe flux without correction and distortion included. i.e spflux = spcflux/(corr*distort*R)

??We are using only one calibration file for all conversions, which was selected from a good seeing day

==Outputs==
Running this code generates txt files that contain the following information:
Plate, image number, fiber number, TAI_beg, TAI_end, RA, DEC, Camera type, Airmass, Alt, Exptime
for each image with the calculated flux and wavelength solution for each fiber in each image for every plate.

The flux is returned for lambda = 365-635nm for the blue cameras and lambda = 565-1040nm for red cameras, 
along with the corresponding wavelength solution from the spcframe files in nm.

Each text file is identified by its image number and contains the flux and corresponding data for every sky fiber
in each camera for that image. They are saved in folders corresponding to the plate.

When you run this script it will generate these files for every spframe in the /boss/spectro/redux file, or wherever 
you save your data. 


"""
from __future__ import print_function,division
import os, glob, fnmatch
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
import numpy as np
import bitmask as b  
from astropy.io import fits


#######################
#. SETUP DIRECTORIES. #
#######################
# identify directory to save data
SAVE_DIR = '/Volumes/PARKER/boss_files/new_sky_flux' #this is the folder where you want to save the output

# identify spframe directory
BASE_DIR = '/Volumes/PARKER/boss_files/'
FOLDERS = ['raw_files']

# detector 
DETECTORS = {'b1':[[365,635], 'b1'], 'b2':[[365,635], 'b2'], 'r1':[[565,1040], 'b1'], 'r2':[[565,1040], 'b2']}

parallel=True
MPI=False
nplates=0 # use just this many plates

def main():
    
    #Collect directories for each plate for each folder in the spframe director
    PLATE_DIRS = []
    for folder in FOLDERS:
        dir = BASE_DIR+folder
        print("directory: ", dir)
        for p in os.listdir(dir):
            pp = os.path.join(dir, p)
            if os.path.isdir(pp) and p != 'spectra':
                PLATE_DIRS.append(pp)

    #Compare with what has already been done
    COMPLETE_DIRS = os.listdir(SAVE_DIR)
    Complete_Plate_Names = []
    All_Plate_Names = []
    for d in COMPLETE_DIRS:
        Complete_Plate_Names.append(d[-8:-4])
    for d in PLATE_DIRS:
        All_Plate_Names.append(d[-4:])

    plates_needed_idx = [i for i, x in enumerate(All_Plate_Names) if x not in Complete_Plate_Names]
    PLATES = []
    for x in plates_needed_idx:
        PLATES.append(PLATE_DIRS[x])

    if (nplates>0):
        PLATES=PLATES[:nplates]
                
    #Get Sky Fibers. 
    global Sky_fibers
    Sky_fibers = np.load('/Users/parkerf/Research/SkyModel/SkyModelling/sky_fibers.npy')
    print("Sky fibers identified for %i plates, total sky fibers=%i."%(len(np.unique(Sky_fibers['PLATE'])), len(Sky_fibers)))

    #Get Calib File. Using the same calibration file for ALL.
    CAMERAS = ['b1', 'b2', 'r1', 'r2']
    global CalibVector
    CalibVector = {}
    for camera in CAMERAS:
        hdu = fits.open(BASE_DIR+'raw_files/5399/spFluxcalib-%s-00139379.fits.gz' % camera)
        data = hdu[0].data
        CalibVector[camera] = data
    print("calibration data set")

    if parallel:
        ## implement if MPI
        #multiprocessing speedup
        pool = multiprocessing.Pool(processes=32)
        ret = pool.map(calc_flux_for_sky_fibers_for_plate, PLATES)
        pool.terminate()
    else:
        ret=[calc_flux_for_sky_fibers_for_plate(p) for p in PLATES]

    raw_meta=np.hstack(tuple(t for t in ret))
    np.save(SAVE_DIR+'meta_raw',raw_meta)
    print("Done")

def ffe_to_flux(spframe_hdu, calib_data):
    """ Flat fielded electrons from spFrame to flux in 10^-17 ergs/s/cm2/A.
    flux = eflux*calibration, where calib = calibration^-1
    
    INPUTS:  spframe_hdu = hdu list of the spFrame. Use data sky data and super flat
             calib_vect = calibration file name associated with the spframe 
    OUTPUT:  numpy array of spframe converted into flux units
    """
    #eflux  = #6 is sky, 0 is residuals
    eflux = spframe_hdu[6].data + spframe_hdu[0].data 
    spflux = eflux/calib_data
    
    return spflux

def remove_rejects(bitmask, sky_flux):
    """Removes any pixels that are flagged in spcframe as FULLREJECT or
    COMBINEREJ. This gets rid of any pixels in a sky fiber that have cosmic rays
    or such things.
    """
    bad_pix = []
    for pix, bit in enumerate(bitmask):
        flags = b.decode_bitmask(b.SPPIXMASK,bit)
        for flag in flags:
            if flag == 'FULLREJECT' or flag == 'COMBINEREJ':
                bad_pix.append(pix)
    sky_flux[bad_pix] = 0
    return sky_flux

def failsafe_dict(d,key):
    try:
        return d[key]
    except:
        return -1

def calc_flux_for_sky_fibers_for_plate(plate_folder):
    """
    This function is used in the multiprocessing. It will calculate the flux from spFrame 
    for each sky fiber in all cameras in every image for a given plate. 
    INPUT: plate directory
    OUTPUT: folder containing txt file for each image. Each txt file includes:
    flux for each fiber for each observation and header information
    (Plate,image,fiber,TAI beg,TAI end,RA,DEC,Camera,Airmass,Alt,Exptime)
    """
    plate_name = plate_folder[-4:]
    print("Doing extraction for plate ", plate_name)

    #Get all spFrame and spCFrame files
    spCFrame_files = glob.glob(plate_folder+'/spCFrame*')
    spFrame_files = glob.glob(plate_folder+'/spFrame*')

    #Get all image numbers in the plate folder
    spPlate = glob.glob(plate_folder+'/spPlate*')[0]
    plate_hdu = fits.open(spPlate)
    image_ids = [plate_hdu[0].header[str(expid)][0:11] for expid in plate_hdu[0].header['EXPID*']]

    #Some spframes don't have corresponding spcframe images. These are skipped but saved as a txt file
    no_spc_match = []
    raw_meta = []
    data = []
    specno = -1
    
    meta_dtype=[('PLATE', 'i4'),('SPECNO','i4'),('IMG', 'i4'),('FIB', 'i4'), ('XFOCAL','f4'),('YFOCAL','f4'),('FIBER_RA','f4'),
                ('FIBER_DEC','f4'), ('TAI-BEG','f8'),('TAI-END','f8'),('RA','f8'), ('DEC','f8'),('CAMERAS','S2'),('AIRMASS','f4'),
                ('ALT','f8'), ('AZ','f8'),('EXPTIME','f4'), ('SEEING20','f4'),('SEEING50','f4'),('SEEING80','f4'), ('AIRTEMP','f4'), 
                ('DEWPOINT','f4'),('DUSTA','f4'), ('DUSTB','f4'),('WINDD25M','f4'), ('WINDS25M','f4'), ('GUSTD','f4'), 
                ('GUSTS','f4'), ('HUMIDITY','f4'), ('PRESSURE','f4'), ('WINDD','f4'), ('WINDS','f4')]


    for image_id in image_ids:
        print("identifier", image_id)

        #Get sky data and header information for spFrame
        sp_file = fnmatch.filter(spFrame_files, '*%s*' % image_id)[0]
        sp_hdu = fits.open(sp_file)
        hdr = sp_hdu[0].header
        Camera_type = hdr['CAMERAS'] 
        image_num = hdr['EXPOSURE']

        calib_data = CalibVector[Camera_type]
        Sky_flux = ffe_to_flux(sp_hdu, calib_data)

        #get wavelength
        spcframe = fnmatch.filter(spCFrame_files, '*spCFrame-%s*' % image_id)[0]
        spc_hdu = fits.open(spcframe)
        logwaves = spc_hdu[3].data
        disps = spc_hdu[4].data
        sigmas = spc_hdu[1].data

        #get bitmask
        bitmask = spc_hdu[2].data

        #Get fibers
        cam_lims, det_num = DETECTORS[Camera_type]
        fibers = Sky_fibers[(Sky_fibers['PLATE'] == int(plate_name)) & (Sky_fibers['CAMERAS'].astype('<U2') == str(det_num))]['FIB']

        for fiber in fibers:
            if fiber >= 500:
                fiber_id = fiber-500
            else:
                fiber_id = fiber
        
            logwave = logwaves[fiber_id]

            wave = (10**logwave)/10.
            limits = np.where((wave > cam_lims[0]) & (wave < cam_lims[1]))
            wave_to_write = wave[limits]

            sky_clean = remove_rejects(bitmask[fiber_id], Sky_flux[fiber_id])
            sky_to_write = sky_clean[limits]

            sigma_to_write = sigmas[fiber_id][limits] #ivar from spcframes in flux
            disp = 10**(disps[fiber_id]*10**(-4)*logwave)/10.
            disp_to_write = disp[limits] #wavelength dispersion in nm

            spec=np.zeros(len(sky_to_write),dtype=[('WAVE','f8'),('SKY','f8'),('SIGMA','f8'),('DISP','f8')])
            spec['WAVE'] = wave_to_write
            spec['SKY'] = sky_to_write
            spec['SIGMA'] = sigma_to_write
            spec['DISP'] = disp_to_write
            specno+=1
            data.append(spec)

            fiber_meta = sp_hdu[5].data[fiber_id]
            mdata=[hdr['PLATEID'], specno, int(image_num), int(fiber),fiber_meta['XFOCAL'], fiber_meta['YFOCAL'], 
                      fiber_meta['RA'], fiber_meta['DEC']]+[failsafe_dict(hdr,x[0]) for x in meta_dtype[8:]]
            raw_meta.append(tuple(mdata))
                    

    np.save(SAVE_DIR+'/'+plate_name+'_calibrated_sky',data)
    raw_meta=np.array(raw_meta,dtype=meta_dtype)
    return raw_meta

if __name__=="__main__":
          main()

