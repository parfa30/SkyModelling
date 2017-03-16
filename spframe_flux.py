#!/usr/bin/env python

"""
Title: SpFrame flux conversion
Author: P. Fagrelius, A. Slosar
Version: 2.1
Date: Mar. 14, 2017

==Inputs==
This code takes the spectra from spFrame BOSS files and converts it into flux similar to that in spCframe files.
Flux output is 10^-17 ergs/s/cm2/A.

This conversion is made only for "sky" fibers as identified in the plug list in spframe files hdu[5]. These have 
OBJTYPE=SKY. A full list of all sky fibers is saved in a pickle file 'sky_fibers.pkl' with associated plate and camera.

The flux is converted with the equation flux = electrons*calibration, 
where calib = calibration^-1 and electrons = flat fielded electrons * super flat * fiber flat

We are using only one calibration file for all conversions, which was selected from a good seeing day

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

==TO-DO==
* Add MPI

"""
from __future__ import print_function,division
import os, glob, fnmatch
try:
    import cPickle as pickle
except:
    import pickle
import multiprocessing
import numpy as np
from astropy.io import fits


#######################
#. SETUP DIRECTORIES. #
#######################
# identify directory to save data
NEW_DIR = os.getcwd()+'/sky_flux/' #this is the folder where you want to save the output

# identify spframe directory
BASE_DIR = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/'
FOLDERS = ['v5_7_0', 'v5_7_2/']

# detector 
DETECTORS = {'b1':[[365,635], 'b1'], 'b2':[[365,635], 'b2'], 'r1':[[565,1040], 'b1'], 'r2':[[565,1040], 'b2']}

parallel=True
MPI=False
nplates=0 # use just this many plates

def main():

    #Collect directories for each plate for each folder in the spframe director
    plates = []
    for folder in FOLDERS:
        dir = BASE_DIR+folder
        print("directory: ", dir)
        for p in os.listdir(dir):
            pp = os.path.join(dir, p)
            if os.path.isdir(pp) and p != 'spectra':
                plates.append(pp)
    if (nplates>0):
        plates=plates[:nplates]
                
    ##############
    #   SCRIPT  #
    ##############

    #Get Calibration data. This is a preselected observation
    CAMERAS = ['b1', 'b2', 'r1', 'r2']
    ## nasty!
    global CalibVector
    CalibVector = {}
    for camera in CAMERAS:
        hdu = fits.open(BASE_DIR+'/v5_7_0/5399/spFluxcalib-%s-00139379.fits.gz' % camera)
        data = hdu[0].data
        CalibVector[camera] = data
    print("calibration data set")

    #Get Sky Fibers. 
    ## nasty!
    global Sky_fibers
    Sky_fibers = np.load('sky_fibers.npy')
    Sky_fibers['FIBER'] = Sky_fibers['FIBER'] - 1 #for numpy index
    print("Sky fibers identified for %i plates, total sky fibers=%i."%(len(Sky_fibers['PLATE']), len(Sky_fibers)))

    if parallel:
        ## implement if MPI
        #multiprocessing speedup
        pool = multiprocessing.Pool(processes=32)
        ret = pool.map(calc_flux_for_sky_fibers_for_plate, plates)
        pool.terminate()
    else:
        ret=[calc_flux_for_sky_fibers_for_plate(p) for p in plates]

    raw_meta=np.hstack(tuple(t[0] for t in ret))
    no_spc_match=[t[1] for t in ret]
    pickle.dump(no_spc_match, open('no_spc_match.pkl','wb'))
    np.save('meta_raw',raw_meta)
    print("Done")

def ffe_to_flux(spframe_hdu, calib_vect, flat_files):
    """ Flat fielded electrons from spFrame to flux in 10^-17 ergs/s/cm2/A.
    flux = electrons*calibration, where calib = calibration^-1
    electrons = flat fielded electrons * super flat * fiber flat
    
    INPUTS:  spframe_hdu = hdu list of the spFrame. Use data sky data and super flat
             calib_vect = data from the spFluxcalib file associated with the camera.
                        This was preselected based on seeing and same is used 
                        for every observation.
             flat_files = list of fiber flat files taken on day of spframe with same camera.
                        The correct flat file is selected from an ID in the spframe header
    OUTPUT:  numpy array of spframe converted into flux units
    """
    #Get eflux data and superflux data
    hdr = spframe_hdu[0].header
    eflux = spframe_hdu[6].data + spframe_hdu[0].data #6 is sky, 0 is residuals
    super_flat = spframe_hdu[8].data
    
    #get fiber flat data corresponding to the spframe
    flat_name = hdr['FLATFILE'][-15:-4]
    flat_file = fnmatch.filter(flat_files, '*%s*' % str(flat_name))[0]
    flat_hdu = fits.open(flat_file)
    fiber_flat = flat_hdu[0].data
    flat_hdu.close()

    #Calculate flux
    electrons = eflux * super_flat * fiber_flat
    flux = electrons/calib_vect
    
    return flux

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
    spFlat_files = glob.glob(plate_folder+'/spFlat*')

    #Get all image numbers in the plate folder
    image_ids = [os.path.splitext(spfile)[0][-13:-5] for spfile in spFrame_files]
    unique_image_ids = np.unique(np.array(image_ids))

    #Some spframes don't have corresponding spcframe images. These are skipped but saved as a txt file
    no_spc_match = []
    raw_meta = []
    data = []
    specno = -1
    
    meta_dtype=[('PLATE', 'i4'),('SPECNO','i4'),('IMG', 'i4'),('FIB', 'i4'), ('TAI-BEG','f8'),('TAI-END','f8'),
                ('RA','f8'),('DEC','f8'),('CAMERAS','S2'),('AIRMASS','f4'),('ALT','f8'), ('AZ','f8'),('EXPTIME','f4'),
                ('SEEING20','f4'),('SEEING50','f4'),('SEEING80','f4'), ('AIRTEMP','f4'), ('DEWPOINT','f4'),
                ('DUSTA','f4'), ('DUSTB','f4'), ('WINDD25M','f4'), ('WINDS25M','f4'), ('GUSTD','f4'), ('GUSTS','f4'), ('HUMIDITY','f4'), 
                ('PRESSURE','f4'), ('WINDD','f4'), ('WINDS','f4')]


    for image_num in unique_image_ids:
        #Find 4 fits images (b1, b2, r1, r2) associated with this unique image_num
        images = fnmatch.filter(spFrame_files, '*%s*' % image_num)
        for image in images:
            #Match with other files
            identifier = os.path.splitext(image)[0][-16:-5]
            print("identifier", identifier)

            #Get sky data and header information for spFrame
            sp_hdu = fits.open(image)
            hdr = sp_hdu[0].header
            Camera_type = hdr['CAMERAS']
            Sky_flux = ffe_to_flux(sp_hdu, CalibVector[Camera_type], spFlat_files)

            #get wavelength
            spcframes = fnmatch.filter(spCFrame_files, '*%s*' % identifier)
            if len(spcframes) > 0:
                spcframe = spcframes[0]
                spc_hdu = fits.open(spcframe)
                waves = spc_hdu[3].data
 
                #Get fibers
                cam_lims, det_num = DETECTORS[Camera_type]
                fibers = Sky_fibers[(Sky_fibers['PLATE'] == int(plate_name)) & (Sky_fibers['CAMERAS'].astype('<U2') == str(det_num))]['FIBER']

                for fiber in fibers:
                    if fiber >= 500:
                        fiber_id = fiber-500
                    else:
                        fiber_id = fiber
                
                    logwave = waves[fiber_id]
                    wave = (10**logwave)/10.
                    limits = np.where((wave > cam_lims[0]) & (wave < cam_lims[1]))
                    wave_to_write = wave[limits]

                    sky = Sky_flux[fiber_id]
                    sky_to_write = sky[limits]

                    spec=np.zeros(len(sky_to_write),dtype=[('WAVE','f8'),('SKY','f8')])
                    spec['WAVE']=wave_to_write
                    spec['SKY']=sky_to_write
                    specno+=1
                    data.append(spec)
                    mdata=[hdr['PLATEID'], specno, int(image_num), int(fiber)]+[failsafe_dict(hdr,x[0]) for x in meta_dtype[4:]]
                    raw_meta.append(tuple(mdata))
                    
            else:
                no_spc_match.append([identifier,plate_name])
                print("Can't find a match for the spcframe")
    np.save(NEW_DIR+'/'+plate_name+'_calibrated_sky',data)
    raw_meta=np.array(raw_meta,dtype=meta_dtype)
    return (raw_meta,no_spc_match)

if __name__=="__main__":
          main()

