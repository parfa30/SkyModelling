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
import matplotlib
matplotlib.use("Agg")
import numpy as np
import multiprocessing
import ephem
import pandas as pd
#import PyQt4
from datetime import datetime
from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_moon
from astropy import units as u 

APACHE = EarthLocation.of_site('Apache Point')

##############
#  SET DIRS #
##############
RAW_META_DATA_DIR = '/Volumes/PARKER/boss_files/new_sky_flux/' 
SKY_FIBER_DIR = os.getcwd() 
###############

def main():

    #Get raw meta files
    raw_files = glob.glob(RAW_META_DATA_DIR+"/raw_meta/*_raw_meta.npy")
    raw_data = []
    for file in raw_files:
        data = np.load(file)
        if len(data) > 0:
            raw_data.append(np.stack(data))

    raw_data = np.hstack(raw_data)
    print("Got all the raw meta files")
    
    #Send only critical info to fetch rich metadata to save time
    raw_df = pd.DataFrame.from_records(raw_data, columns = raw_data.dtype.names)
    needed_info = raw_df[['PLATE','IMG','TAI-BEG','RA','DEC']]
    needed_info.drop_duplicates(inplace=True)
    raw_array = needed_info.as_matrix()
    print("Observation number: ",len(raw_array))

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

    #Get rich data
    pool = multiprocessing.Pool(processes=32)
    rich_results = pool.map(get_rich_data,raw_array)
    rich_df = pd.DataFrame(np.hstack(rich_results))

    #Get fiber meta data
    sky_fiber_data = np.load(SKY_FIBER_DIR+'/sky_fibers.npy') 
    fiber_df = pd.DataFrame(sky_fiber_data)

    #Get quality
    qhdu = fits.open(os.getcwd()+'/util/platelist.fits')
    qdata = qhdu[1].data
    platelist = pd.DataFrame([qdata['PLATE'], qdata['MJD'], qdata['PLATEQUALITY']]).T
    platelist.columns = ['PLATE', 'MJD', 'QUALITY']
    qhdu.close()
    
    #Combine with previous data and save as new numpy object
    new_df = raw_df.merge(rich_df,on=['PLATE','IMG','TAI-BEG','RA','DEC'],how='left')
    quality_df = new_df.merge(platelist,on=['PLATE','MJD'],how='left')
    full_df = quality_df.merge(fiber_df,on=['PLATE','CAMERAS','FIB'],how='left')
    rich_meta_array = full_df.to_records(index=False)

    np.save(RAW_META_DATA_DIR+'meta_rich',rich_meta_array)
    print("Done")



def get_rich_data(raw_array):
    """ Gets additional data about each image. i
    """
    PLATE, IMG, TAI, RA, DEC = raw_array
    print("Getting rich data for this plate/image ",PLATE,IMG)
    start = datetime.now()
 
    time = Time(TAI/86400., scale='tai', format='mjd', location=APACHE)
    moon_lat, moon_lon, sun_lat, sun_lon, moon_alt, moon_az, sun_alt, sun_az, moon_dist, moon_sep, sun_moon_sep, sun_elong = moon_and_sun(time, RA, DEC)
    days_to_full = moon_phase(time)
    ecl_lat, ecl_lon, gal_lat, gal_lon = gal_and_ecl(RA,DEC)
    az = az_from_radec(DEC, RA, time)
    fli = frac_lun_ill(np.deg2rad(moon_lon), np.deg2rad(sun_lon), np.deg2rad(moon_lat))
    season, hour_start = get_season(time)
    this_solar_flux = get_solar_flux(solar_flux,time.value)


    rich_dtype=[('PLATE', 'i4'),('IMG', 'i4'),('TAI-BEG','f8'),('RA','f8'),('DEC','f8'),('MOON_LAT','f4'),('MOON_LON','f4'), ('SUN_LAT','f4'),('SUN_LON','f4'),('MOON_ALT','f4'),('MOON_AZ','f4'),('SUN_ALT','f4'), ('SUN_AZ','f4'), ('MOON_D','f8'), ('MOON_SEP','f8'), ('SUN_MOON_SEP','f8'),('SUN_ELONG','f8'),('DAYS2FULL','f4'),('ECL_LAT','f4'), ('ECL_LON','f4'), ('GAL_LAT','f4'), ('GAL_LON','f4'), ('AZ_CALC','f4'), ('FLI','f4'), ('SEASON','i4'), ('HOUR', 'i4'),('SOLARFLUX','f4')]

    rich_meta_data = np.array([(PLATE,IMG,TAI,RA,DEC,moon_lat,moon_lon, sun_lat, sun_lon, moon_alt, moon_az, sun_alt, sun_az, moon_dist, moon_sep, sun_moon_sep, sun_elong, days_to_full, ecl_lat, ecl_lon, gal_lat, gal_lon, az, fli, season, hour_start, this_solar_flux)],dtype=rich_dtype)
    
    print("Time: ", (datetime.now()-start).total_seconds())
    return rich_meta_data

def moon_and_sun(time, RA, DEC):
    """
    Input: astropy.time.Time object
    Uses astropy get_moon and get_sun for lat and lon, then transforms to AltAz frame
    """
    moon = get_moon(time, location=APACHE)
    sun = get_sun(time)
    target = SkyCoord(RA, DEC, unit=u.deg, frame = moon.frame)
    
    #separations
    moon_dist = moon.distance.value
    moon_sep = moon.separation(target).degree
    sun_moon_sep = moon.separation(sun.transform_to(moon.frame)).degree
    sun_elong = target.separation(sun.transform_to(moon.frame)).degree
    
    #geocentric latitude and longitude
    moon_lat, moon_lon = (moon.geocentrictrueecliptic.lat.value,moon.geocentrictrueecliptic.lon.value)
    sun_lat, sun_lon = (sun.geocentrictrueecliptic.lat.value,sun.geocentrictrueecliptic.lon.value)

    #change to altaz frame
    moon_altaz = moon.transform_to(AltAz(obstime = time, location = APACHE))
    sun_altaz = sun.transform_to(AltAz(obstime=time, location=APACHE))
    moon_alt, moon_az = (moon_altaz.alt.value, moon_altaz.az.value)
    sun_alt, sun_az = (sun_altaz.alt.value, sun_altaz.az.value)

    return [moon_lat, moon_lon, sun_lat, sun_lon, moon_alt, moon_az, sun_alt, sun_az, moon_dist, moon_sep, sun_moon_sep, sun_elong]

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

def get_season(time):
    month = time.datetime.month
    season = np.rint(month/2.)
    
    tt = time.datetime.hour+(time.datetime.minute/60.)
    hour_start = np.rint(tt)
    
    return [season, hour_start]

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














