import astropy.table
import numpy as np 
import os, sys, glob 
import pickle
import speclite.filters


def main():

	DIR = '/global/cscratch1/sd/parkerf/sky_flux_corrected/'
	MEAN_META = DIR+'/mean_meta/'
    if not os.path.exists(MEAN_META):
        os.makedirs(MEAN_META)

	file_dir = DIR+'/mean_spectra/'
	rich_meta_files = os.listdir(DIR+'/rich_meta/')


	global Lines
	Lines = pickle.load(open('/global/homes/p/parkerf/Sky/SkyModelling/util/line_file.pkl','rb'))



	pool = multiprocessing.Pool(processes=64)
    pool.map(make_mean_meta_file, rich_meta_files)
    pool.terminate()

    
def make_mean_meta_file(rich_meta_file):
	MF = astropy.table.Table.read(rich_meta_file)
	MF.remove_columns(['SPECNO', 'FIB', 'XFOCAL','YFOCAL','FIBER_RA','FIBER_DEC'])
	Meta = astropy.table.unique(MF, keys=['PLATE','IMG','CAMERAS'])
	plate = np.unique(MF['PLATE'])

	ugriz = speclite.filters.load_filters('sdss2010-*')
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

    mean_spectra_dir = file_dir+'%d' % plate

    for meta in Meta:
    	image = meta['IMG']
    	cam = meta['CAMERAS']
    	mean_spectrum = np.load(mean_spectra_dir+'/%d_%s_mean_spectrum.npy'% (image, cam))
    	#Clean data
        ok = np.isfinite(mean_spectrum['SKY'])
        sky = mean_spectrum['SKY'][ok]
        wave = mean_spectrum['WAVE'][ok]
        lines = Lines[cam]
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

        for filt in ugriz:
            flux, wlen = filt.pad_spectrum(1e-17*sky/fiber_area, wave*10)
            mag = filt.get_ab_magnitude(flux,wlen)
            meta[filt.name] = astropy.table.Column([mag])
        for filt in bessell:
            flux, wlen = filt.pad_spectrum(1e-17*sky/fiber_area, wave*10)
            mag = get_vmag(filt, flux, wlen)
            meta[filt.name] = astropy.table.Column([mag])

    #save astropy table as fits file in rich_plus
    rich_mean_filen = DATA_DIR+'mean_meta/%d_mean_meta.fits'%plate
    if os.path.exists(rich_mean_filen):
        os.remove(rich_mean_filen)
    Meta.write(rich_mean_filen,format='fits')


if __name__ == '__main__':
	main()





