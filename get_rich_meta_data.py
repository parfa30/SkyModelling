#!/usr/bin/env python

"""

Version 2 (Apr. 2018): refactored to output fits file and added some items. 

Title: BOSS Sky Spectra Meta Data
Author: P. Fagrelius
Date: Mar. 6, 2017

==INPUT==
This program takes the raw metadata pickle files created in spframe_flux.py and uses that 
fits header information to calculate additional data about the observations. This data 
in the input pickle files must include as a minimum: Plate, image, RA, DEC, Alt, and Airmass

==OUTPUT==
There is a single output file for each input raw file. They are saved in rich_meta as ####_rich_meta.fits.
The meta data is saved as an astropy Table in the fits file. The data can be accessed as astropy.table.Table.read(filen).
This program adds the following meta data to the raw info from teh SpFrame headers.
Most of these meta data are calculated using astropy.
- Moon information: MOOND,MOON_SEP,MOON_AZ,MOON_ALT,MOON_ZENITH,MOON_ILL,MOON_PHASE
- Sun location: SUN_SEP, SUN_ALT, SUN_AZ, MOON_SUN_SEP
- Fiber lat and lon: ECL_LAT, ECL_LON, GAL_LAT, GAL_LON, OBZ_ZENITH
- Solar Flux: SOLARFLUX
- Timing info: MONTH, HOUR 
- cloudiness: PHOTO

Solar Flux is taken from Ottawa/Penticon 2800MHz data

==HOW TO USE==
Best to run on nersc. Identify folder that contains the spframe_flux file (*_calibrated_sky.npy) and teh raw_meta folder under RAW_META_DATA_DIR.

Run the script using 1 node and as many cores as possible (64). Your python distribution needs to include: numpy, pandas, astropy, and scipy.

srun -n 1 -c 64 python ./get_meta_data_rich.py
"""
from __future__ import print_function,division
import math, os, sys, fnmatch,glob 
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import multiprocessing
import pandas as pd
from datetime import datetime
from astroplan import Observer
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u 
import astropy.table
from scipy.interpolate import interp1d


APACHE = EarthLocation.of_site('Apache Point')

##############
#  SET DIRS #
##############
RAW_META_DATA_DIR = '/global/cscratch1/sd/parkerf/sky_flux' 
###############

def main():
    start = datetime.now()
    # Get Solar Flux Data
    solar_data = np.load(os.getcwd()+'/util/solar_flux.npy')
    global solar_flux
    solar_flux = interp1d(solar_data['MJD'], solar_data['fluxobsflux'], bounds_error=False, fill_value = 'interpolate')

    #get cloud data
    global cloud_data
    cloud_data = pd.DataFrame(np.load(os.getcwd()+'/util//phot_rec.npy'))

    #Get raw meta files
    raw_files = glob.glob(RAW_META_DATA_DIR+"/raw_meta/*_raw_meta.npy")
    Complete_Rich = [d[0:4] for d in os.listdir(RAW_META_DATA_DIR+'/rich_meta/')]
    All_Raw = [d[0:4] for d in os.listdir(RAW_META_DATA_DIR+'/raw_meta/')]
    rich_needed = [i for i, x in enumerate(All_Raw) if x not in Complete_Rich]
    these_raw_files = np.array(raw_files)[rich_needed]

    print('got %d raw files. going to start pool' % len(these_raw_files))

    #Get rich data
    pool = multiprocessing.Pool(processes=64)
    pool.map(get_rich_data,these_raw_files)
    pool.close()

    total_time = (datetime.now() - start).seconds
    print("Total time = %.2f for %d files" %(total_time, len(raw_files)))

def get_rich_data(raw_file):
    print(raw_file)
    start = datetime.now()
    try:
        rich_data = astropy.table.Table(np.load(raw_file))

        start_time = Time(rich_data['TAI-BEG']/86400., scale='tai', format='mjd', location=APACHE)
        
        coord = SkyCoord(rich_data['FIBER_RA'], rich_data['FIBER_DEC'], unit='deg', frame = 'fk5') #FK5 frame
        obs_zenith = 90 - rich_data['ALT']
        rich_data['OBS_ZENITH'] = astropy.table.Column(obs_zenith.astype(np.float32), unit='deg')
        rich_data['ECL_LAT'] = astropy.table.Column(coord.geocentrictrueecliptic.lat.value)
        rich_data['ECL_LON'] = astropy.table.Column(coord.geocentrictrueecliptic.lon.value)
        rich_data['GAL_LAT'] = astropy.table.Column(coord.galactic.b.value)
        rich_data['GAL_LON'] = astropy.table.Column(coord.galactic.l.value)

        moon = get_moon(start_time, location=APACHE) #gcrs frame
        moon_altaz = moon.transform_to(AltAz(obstime = start_time, location = APACHE))
        rich_data['MOOND'] = astropy.table.Column(moon.distance.value) #distance from frame origin (geocentric)
        rich_data['MOON_SEP'] = astropy.table.Column(moon.separation(coord).degree) #properly accounts for difference in frames
        rich_data['MOON_ALT'] = astropy.table.Column(moon_altaz.alt.value)
        rich_data['MOON_AZ'] = astropy.table.Column(moon_altaz.az.value)
        moon_zenith = 90-rich_data['MOON_ALT']
        rich_data['MOON_ZENITH'] = astropy.table.Column(moon_zenith.astype(np.float32), unit='deg')

        apache = Observer(APACHE)
        rich_data['MOON_ILL'] = astropy.table.Column(apache.moon_illumination(start_time)) #percentage
        rich_data['MOON_PHASE'] = astropy.table.Column(apache.moon_phase(start_time).value) #radians (2pi = new) actually, phase angle

        sun = get_sun(start_time)
        sun_altaz = sun.transform_to(AltAz(obstime=start_time, location=APACHE))
        rich_data['SUN_SEP'] = astropy.table.Column(sun.separation(coord).degree)
        rich_data['SUN_ALT'] = astropy.table.Column(sun_altaz.alt.value)
        rich_data['SUN_AZ'] = astropy.table.Column(sun_altaz.az.value)
        rich_data['SUN_MOON_SEP'] = astropy.table.Column(moon.separation(sun).degree)

        month = [time.datetime.month for time in start_time]
        hour = [time.datetime.hour for time in start_time]
        minute = [time.datetime.minute/60. for time in start_time]
        rich_data['MONTH'] = astropy.table.Column(month)
        rich_data['HOUR'] = astropy.table.Column(np.array(hour)+np.array(minute))
        
        # Get clouds. This takes a little bit of time
        clouds = [get_cloud_data(line) for line in rich_data]
        rich_data['PHOTO'] = astropy.table.Column(np.hstack(clouds).astype(np.float32))

        #get solar flux from interpolation
        solar = [solar_flux(line['MJD']) for line in rich_data]
        rich_data['SOLARFLUX'] = astropy.table.Column(np.hstack(solar).astype(np.float32))

        plate = np.unique(rich_data['PLATE'])[0]
        rich_filen = RAW_META_DATA_DIR+'/rich_meta/%d_rich_meta.fits' % plate
        print(rich_filen)
        if os.path.exists(rich_filen):
            os.remove(rich_filen)
        rich_data.write(rich_filen, format = 'fits') 
        print('finished plate ',plate)

    except:
        print('some error occurred')

def get_cloud_data(line):
    clouds = cloud_data[(cloud_data['STARTTAI']<= line['TAI-BEG']) & (cloud_data['ENDTAI']>line['TAI-BEG'])]['PHOTOMETRIC'].values
    if len(clouds) == 0:
        clouds = cloud_data[(cloud_data['STARTTAI']< line['TAI-END']) & (cloud_data['ENDTAI']>=line['TAI-END'])]['PHOTOMETRIC'].values
        if len(clouds) == 0:
            clouds = 0.5
    
    return clouds

if __name__=="__main__":
    main()

