#!/usr/bin/env python

"""
Title: SpFrame flux conversion
Author: P. Fagrelius
Date: Mar. 3, 2017

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
import os, glob, fnmatch
import pickle
import multiprocessing
import numpy as np
from astropy.io import fits

#######################
#. SETUP DIRECTORIES. #
#######################
# identify directory to save data
NEW_DIR = os.getcwd()+'/sky_flux/' #this is the folder where you want to save the output
FILE_IDEN = 'boss_flux' #file identifier, so saved files will be boss_flux_imagenum.txt

# identify spframe directory
BASE_DIR = '/global/projecta/projectdirs/sdss/data/sdss/dr12/boss/spectro/redux/'
FOLDERS = ['v5_7_0', 'v5_7_2/']

#Collect directories for each plate for each folder in the spframe director
PLATES = []
for folder in FOLDERS:
    dir = BASE_DIR+folder
    print("directory: ", dir)
    for p in os.listdir(dir):
        pp = os.path.join(dir, p)
        if os.path.isdir(pp) and p != 'spectra':
            PLATES.append(pp)

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

##############
#   SCRIPT  #
##############

#Get Calibration data. This is a preselected observation
CAMERAS = ['b1', 'b2', 'r1', 'r2']
CalibVector = {}
for camera in CAMERAS:
    hdu = fits.open(BASE_DIR+'/v5_7_0/5399/spFluxcalib-%s-00139379.fits.gz' % camera)
    data = hdu[0].data
    CalibVector[camera] = data
print("calibration data set")

#Get Sky Fibers. 
with open('sky_fibers.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    sf_data = u.load() 
Sky_fibers = {}
for key, values in sf_data.items():
    plate = key[-4:]
    Sky_fibers[plate] = values
print("Sky fibers identified")

            
DETECTORS = {'b1':[[365,635], 'b1'], 'b2':[[365,635], 'b2'], 'r1':[[565,1040], 'b1'], 'r2':[[565,1040], 'b2']}


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

    #make folder to dump txt files into
    plate_dir = NEW_DIR+str(plate_name)
    if not os.path.exists(plate_dir):
        os.makedirs(plate_dir)

    #Get all spFrame and spCFrame files
    spCFrame_files = glob.glob(plate_folder+'/spCFrame*')
    spFrame_files = glob.glob(plate_folder+'/spFrame*')
    spFlat_files = glob.glob(plate_folder+'/spFlat*')

    #Get all image numbers in the plate folder
    image_ids = [os.path.splitext(spfile)[0][-13:-5] for spfile in spFrame_files]
    unique_image_ids = np.unique(np.array(image_ids))

    #Some spframes don't have corresponding spcframe images. These are skipped but saved as a txt file
    no_spc_match = []

    for image_num in unique_image_ids:
        #Start writing file
        sky_flux_file = open(plate_dir+'/'+str(FILE_IDEN)+'_%s.txt' % image_num, 'wt')
        sky_flux_file.write('Plate image fiber TAI_beg TAI_end RA DEC Camera Airmass Alt Exptime flux wave\n')
        
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
                fibers = Sky_fibers[plate_name][det_num]

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

                    #Write the file
                    sky_flux_file.write('%s %d %d %.2f %.2f %.2f %.2f %s %.2f %.2f %.2f ' % (hdr['PLATEID'], int(fiber), int(image_num), hdr['TAI-BEG'], hdr['TAI-END'], hdr['RA'], hdr['DEC'], hdr['CAMERAS'], hdr['AIRMASS'], hdr['ALT'], hdr['EXPTIME']))
                    for x in sky_to_write:
                        sky_flux_file.write('%.2f,'% x)
                    sky_flux_file.write(' ')
                    for y in wave_to_write:
                        sky_flux_file.write('%.2f,'% y)
                    sky_flux_file.write('\n')
            else:
                no_spc_match.append([identifier,plate_name])
                print("Can't find a match for the spcframe")

        sky_flux_file.close
    return no_spc_match

#multiprocessing speedup
pool = multiprocessing.Pool(processes=32)
no_spc_match = pool.map(calc_flux_for_sky_fibers_for_plate, PLATES)
pool.terminate()

pickle.dump(no_spc_match, open('no_spc_match.pkl','wb'))

print("Done")
