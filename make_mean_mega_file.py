import astropy.table
import numpy as np 
import os, sys, glob 
import pickle
import speclite.filters
import multiprocessing
from scipy.interpolate import interp1d
from astropy import units as u


def main():

    DIR = '/global/cscratch1/sd/parkerf/sky_flux_corrected/'
    global MEAN_META
    MEAN_META = DIR+'/mean_meta/'
    if not os.path.exists(MEAN_META):
        os.makedirs(MEAN_META)

    #rich_meta_files = os.listdir(DIR+'/rich_meta/')
    vega = astropy.table.Table.read('ftp://ftp.stsci.edu/cdbs/current_calspec/alpha_lyr_stis_008.fits')
    vegawave, vegaflux = np.array(vega["WAVELENGTH"]), np.array(vega['FLUX'])
    global VEGA
    VEGA = interp1d(vegawave, vegaflux, bounds_error=False, fill_value = 0)

    MF = astropy.table.Table.read('/global/homes/p/parkerf/Sky/SkyModelling/data_files/good_clean_data_180713.fits')
    MF.remove_columns(['SPECNO', 'FIB', 'XFOCAL','YFOCAL','FIBER_RA','FIBER_DEC'])

    global Meta
    Meta = astropy.table.unique(MF, keys=['PLATE','IMG','CAMERAS'])

    global Lines
    Lines = pickle.load(open('/global/homes/p/parkerf/Sky/SkyModelling/util/line_file.pkl','rb'))

    global ugriz
    ugriz = speclite.filters.load_filters('sdss2010-*')

    global bessell
    bessell = speclite.filters.load_filters('bessell-*')
    
    #add columns to astropy.table for each line
    for camera, lines in Lines.items():
        if (camera == 'b1')|(camera=='r1'): #only want one set
            for name in lines.keys():
                Meta[name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))
    for filt in ugriz:
        Meta[filt.name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))
    for filt in bessell:
        Meta[filt.name] = astropy.table.Column(np.zeros(len(Meta)).astype(np.float32))

    PLATES = np.unique(Meta['PLATE'])


    pool = multiprocessing.Pool(processes=64)
    pool.map(make_mean_meta_file, PLATES)
    pool.terminate()

def get_vmag(filt, flux, wave_range):

    znum = filt.convolve_with_array(wave_range, VEGA(wave_range))
    zden = filt.convolve_with_function(lambda wlen: u.Quantity(1))
    zp = (znum/zden)
    
    mnum = filt.convolve_with_array(wave_range, flux)
    mden = filt.convolve_with_function(lambda wlen: u.Quantity(1))
    mp = (mnum/mden)
    
    M = -2.5 * np.log10(mp/zp) 
    return M

 
def make_mean_meta_file(plate):
    
    mean_spectra_dir = '/global/cscratch1/sd/parkerf/sky_flux_corrected/mean_spectra/%d/' % plate
    num_pix_lines = 5 #+/- pixels used for sum
    num_pix_cont = 1
    fiber_area = np.pi

    ThisMeta = Meta[Meta['PLATE'] == plate]
    #print(ThisMeta)
    for meta in ThisMeta:
        image = meta['IMG']
        cam = meta['CAMERAS']
        sky = np.load(mean_spectra_dir+'/%d_%s_mean_spectrum.npy'% (image, cam))
        wave = np.linspace(300, 1040, (1040-300)*100)
        lines = Lines[cam]
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

    #save astropy table as fits file in rich_plus
    rich_mean_filen = MEAN_META+'/%d_mean_meta.fits'%plate
    if os.path.exists(rich_mean_filen):
        os.remove(rich_mean_filen)
    ThisMeta.write(rich_mean_filen,format='fits')


if __name__ == '__main__':
    main()





