import astropy.table
import numpy as np 
import multiprocessing
import pickle

MetaFile = 'good_clean_data_%s.fits'
DIR = '/global/cscratch1/sd/parkerf/sky_flux_corrected/'

def main():

    MEAN_DIR = DIR+'/mean_spectra/'
    if not os.path.exists(MEAN_DIR):
        os.makedirs(MEAN_DIR)

    MF = astropy.table.Table.read(MetaFile)

    # Mean_Image_Names = []
    # for img in np.unique(MF['IMG']):
    #     plate = np.unique(MF[MF['IMG']==img]['PLATE'])
    #     for cam in ['b1','b2','b3','b4']:
    #         Mean_Image_Names.append('%d_%d_%s'%(plate, img, cam))

    global SPECNOS
    SPECNOS = pickle.load(open('image_specnos.pkl','rb'))
    # for plate in np.unique(MF['PLATE']):
    #     SPECNOS[plate] = {}
    #     for img in np.unique(MF[MF['PLATE']==plate]['IMG']):
    #         SPECNOS[plate][img] = {}
    #         for cam in np.unique(MF[MF['IMG'] == img]['CAMERAS']):
    #             specs = MF[(MF['IMG'] == img)&(MF['CAMERAS'] == cam)]
    #             SPECNOS[plate][img][cam] = specs

    # Complete_Images = [os.path.splitext(d)[0][-14:-6] for d in MEAN_DIR]
    # Images_Needed = [i for i, x in enumerate(Mean_Image_Names) if x not in Complete_Images]

    PLATES = np.unique(MF['PLATE'])

    pool = multiprocessing.Pool(processes=64)
    pool.map(make_mean_spectrum, PLATES)
    pool.terminate()

def make_mean_spectrum(plate):
    data = np.load(DIR+'%d_calibrated_sky.npy' % plate)
    xx = np.linspace(300, 1040, (1040-300)*100)

    PLATE_DIR = DIR+'/mean_spectra/%d' % plate
    if not os.path.exists(PLATE_DIR):
        os.makedirs(PLATE_DIR)

    image_info = SPECNOS[plate]
    for image, cams in image_info.items():
        for cam, specnos in cams.items():
            these_specnos = image_info[image][cam]
            SKY_SPECTRA = []
            VARS = []
            for specno in these_specnos:
                spectrum = data[specno]
                f = interp1d(spectrum['WAVE'], spectrum['SKY'], bounds_error=False, fill_value=0)
                g = interp1d(spectrum['WAVE'], spectrum['IVAR'], bounds_error=False, fill_value=0)
                SKY_SPECTRA.append(f(xx))
                VARS.append(g(xx))
            
            mean_spectrum = np.ma.average(np.array(SKY_SPECTRA), axis=0, weights=np.array(VARS))
            mean_var = np.average(np.array(VARS), axis=0)
            mean_spectrum.dump(PLATE_DIR+'/%d_%s_mean_spectrum'% (image, camera))
            mean_var.dump(PLATE_DIR+'/%d_%s_mean_var'% (image, camera))


if __name__ == '__main__':
    main()





