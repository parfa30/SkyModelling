#!/usr/bin/env python

"""
Title: BOSS Sky Spectra Meta Data
Author: P. Fagrelius
Date: Mar. 6, 2017

==INPUT==
This program takes the raw metadata pickle files created in spframe_flux.py and uses that 
fits header information to calculate additional data about the observations. This data 
in the input pickle files must include as a minimum: Plate, image, RA, DEC, Alt, and Airmass

==OUTPUT==
The output is one pickle file that includes the metadata for all BOSS spectra. There is one
row of data for each unique image for every plate, and includes:
- image info: plate, image, TAI_beg, TAI_end
- raw meta data: ra, dec, airmass, alt, exposure time
- Moon location:  moon_lat,moon_lon, moon_alt, moon_az
- Sun location: sun_lat, sun_lon, sun_alt, sun_az
- moon phase (days to full moon)
- frame lat and lon in ecliptic and galactic coordinate frames
- Azimuth
- Fractional Lunar Illumination
- Solar Flux 

Most of these are calculated using astropy.

Solar Flux is taken from Ottawa/Penticon 2800MHz data
"""
from __future__ import print_function,division
import math, os, sys, fnmatch,glob 
try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import multiprocessing
import ephem
import pandas as pd
from datetime import datetime
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u 

APACHE = EarthLocation.of_site('Apache Point Observatory')

##############
#  SET DIRS #
##############

RAW_META_DATA_DIR = '/global/homes/p/parkerf/BOSS_Sky/sky_flux/'


def main():
    raw_meta_files = []
    for filename in glob.iglob(RAW_META_DATA_DIR+'/**/raw_meta_**.pkl', recursive=True):
        raw_meta_files.append(filename)
    print("Number plates ", len(raw_meta_files))

    # Get Solar Flux Data
    SOLAR_FILE = 'ftp://ftp.geolab.nrcan.gc.ca/data/solar_flux/daily_flux_values/fluxtable.txt'
    SUN_FLUX_DATA = pd.read_csv(SOLAR_FILE, header=0, delim_whitespace=True)
    Solar_Flux = SUN_FLUX_DATA.drop(SUN_FLUX_DATA.index[0])
    Solar_Flux = pd.DataFrame(Solar_Flux, dtype=float) #underlines

    Solar_Flux['MJD'] = Solar_Flux.fluxdate.apply(get_mjd_from_fluxtime)
    new_solar_flux = Solar_Flux.groupby(['MJD']).mean()
    solar_dict = new_solar_flux.to_dict()
    global solar_flux
    solar_flux = solar_dict['fluxobsflux']
    print("solar table loaded")

    
    pool = multiprocessing.Pool(processes=32)
    meta_results = pool.map(get_meta_data,raw_meta_files)

    pickle.dump(meta_results, open('boss_rich_meta.pkl', 'wb'))

    print("Done")


def get_meta_data(raw_meta_file):
    """ Gets additional data about each image. Input is a pickle file that contains fits 
    header info for every image in a given plate
    """
    print(raw_meta_file)
    raw_data = pickle.load(open(raw_meta_file,'rb'))

    #Drop any duplicates so only run code for unique observations
    raw_df = pd.DataFrame(raw_data, columns = ['PLATE','image','TAI_beg','TAI_end','RA','DEC','Camera','Airmass',
                                                'Alt','Exptime','fibers'])
    raw_df.drop('Camera', axis=1, inplace=True)
    raw_df.drop('fibers', axis=1, inplace=True)
    raw_df = raw_df.drop_duplicates(subset=['PLATE','image'])
    meta_data = raw_df.as_matrix()

    #Calculate additional data for each unique observation
    rich_meta_data = []
    for obs in meta_data:
        plate, image, taibeg, taiend, ra, dec, am, alt, etime = obs
        time = Time(taibeg/86400., scale='tai', format='mjd', location=APACHE)
        moon_lat, moon_lon, sun_lat, sun_lon, moon_alt, moon_az, sun_alt, sun_az = moon_and_sun(time)
        days_to_full = moon_phase(time)
        ecl_lat, ecl_lon, gal_lat, gal_lon = gal_and_ecl(ra,dec)
        az = az_from_radec(dec, ra, time)
        fli = frac_lun_ill(np.deg2rad(moon_lon), np.deg2rad(sun_lon), np.deg2rad(moon_lat))
        this_solar_flux = get_solar_flux(solar_flux,time.value)
        rich_meta_data.append([plate, image, taibeg, taiend, ra, dec, am, alt, etime, moon_lat,moon_lon, sun_lat, sun_lon, moon_alt, moon_az, sun_alt, sun_az, days_to_full, ecl_lat, ecl_lon, gal_lat, gal_lon, az, fli, this_solar_flux])
    
    return rich_meta_data


def moon_and_sun(time):
    """
    Input: astropy.time.Time object
    Uses astropy get_moon and get_sun for lat and lon, then transforms to AltAz frame
    """
    moon = get_moon(time, location=APACHE)
    sun = get_sun(time)

    #geocentric latitude and longitude
    moon_lat, moon_lon = (moon.geocentrictrueecliptic.lat.value,moon.geocentrictrueecliptic.lon.value)
    sun_lat, sun_lon = (sun.geocentrictrueecliptic.lat.value,sun.geocentrictrueecliptic.lon.value)

    #change to altaz frame
    moon_altaz = moon.transform_to(AltAz(obstime = time, location = APACHE))
    sun_altaz = sun.transform_to(AltAz(obstime=time, location=APACHE))
    moon_alt, moon_az = (moon_altaz.alt.value, moon_altaz.az.value)
    sun_alt, sun_az = (sun_altaz.alt.value, sun_altaz.az.value)

    return [moon_lat, moon_lon, sun_lat, sun_lon, moon_alt, moon_az, sun_alt, sun_az]

def moon_phase(time):
    """
    Input: astropy.time.Time object
    calculates number of days until full moon using pyephem
    """
    Apache = ephem.Observer()
    Apache.lat = APACHE.latitude.value
    Apache.lon = APACHE.longitude.value
    Apache.elevation = APACHE.height
    Apache.date = time.iso

    m = ephem.Moon(Apache)

    #days until next full moon
    full = ephem.next_full_moon(time.iso)
    full = Time(full.datetime())
    time_diff = abs(full - time)
    days_to_full = time_diff.sec/86400.
 
    return days_to_full

def gal_and_ecl(ra,dec):
    """
    Calculates the latitude and longtidue of the frame in ecliptic coordinates and galactic coordinates
    given the ra and dec of the frame
    """
    try:
        o = SkyCoord(ra,dec,unit='deg')
        ecl_lat, ecl_lon = (o.geocentrictrueecliptic.lat.value, o.geocentrictrueecliptic.lon.value)
        gal_lat, gal_lon = (o.galactic.b.value, o.galactic.l.value)
    except:
        (ecl_lat, ecl_lon, gal_lat, gal_lon) = (0,0,0,0)
    return [ecl_lat, ecl_lon, gal_lat, gal_lon]


def az_from_radec(dec_obj, ra_obj, time):
    """
    Calculates azimuth of an object from its RA, DEC and time of observation
    """
    plate = SkyCoord(ra_obj, dec_obj, "icrs", unit="deg")
    azaltplate = plate.transform_to(AltAz(obstime = time, location=APACHE))
    return azaltplate.az.value

def frac_lun_ill(moon_long, sun_long, moon_lat):
    """ Calculates the Fractional Lunar Illumination (FLI) as defined here:
    http://www.ing.iac.es/PR/newsletter/news6/tel1.html
    INPUTS: longitude of the moon (radians)
            longitude of the sun (radians)
            galactic latitude of the moon (radians)
    """
    fli = (1/2.)*(1-math.cos(moon_lat)*math.cos(moon_long-sun_long))
    return fli

#Solar Flux
def get_mjd_from_fluxtime(date):
    """
    Compares datestamp in solar flux data to the tai_beg timestamp in the raw metadata
    """
    str_date = str(date)
    t = datetime(int(str_date[0:4]),int(str_date[4:6]),int(str_date[6:8]))
    tt = Time(t)
    return tt.mjd

def get_solar_flux(flux_table, MJD):
    """
    returns solar flux for a given time
    """
    print('solar time: ', MJD)
    try:
        flux = flux_table[int(MJD)]
    except:
        flux = 0
    return flux


if __name__=="__main__":
    main()













