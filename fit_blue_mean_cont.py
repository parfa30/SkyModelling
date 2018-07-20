import glob
import os
import numpy as np 
import multiprocessing
import astropy.table
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d

from lmfit import models, Parameters, Parameter, Model
from lmfit.models import LinearModel, ConstantModel

def main():
    SPECTRA_DIR = '/Volumes/PFagrelius_Backup/sky_data/sky_mean_spectra/'
    mean_blue_files = glob.glob(SPECTRA_DIR + '/*/*_b*_mean_spectrum.npy', recursive=True)

    global SAVE_DIR
    SAVE_DIR = '/Volumes/PFagrelius_Backup/sky_data/sky_mean_spectra/mean_cont_spectra/'

    global mean_wave
    mean_wave = np.linspace(300, 1040, (1040-300)*100)

    # Get Lines
    global blue_airglow_lines
    blue_airglow_lines = np.load('/Users/parkerf/Research/SkyModel/BOSS_Sky/ContFitting/files/blue_airglow_lines.npy')
    print(blue_airglow_lines)

    pool = multiprocessing.Pool(processes=2)
    pool.map(save_new_file, mean_blue_files)
    pool.close()

    # for file in mean_blue_files[1:6]:
    #     save_new_file(file)

def save_new_file(filen):
    names = os.path.split(filen)
    plate = names[0][-4:]
    print(plate)
    name = names[1][0:9]
    print(filen)
    try:
        spectrum = np.load(filen)
        new_spectrum = run_model_for_spectrum(spectrum)
        filtered_spectrum = gaussian_filter1d(new_spectrum, 200)

        #Create files
        spec=np.zeros(len(spectrum),dtype=[('WAVE','f8'),('SKY','f8'),('CONT','f8'),('FILT_CONT','f8')])
        spec['WAVE'] = mean_wave
        spec['SKY'] = spectrum
        spec['CONT'] = new_spectrum
        spec['FILT_CONT'] = filtered_spectrum

        folder = SAVE_DIR+'/'+plate+'/'
        print(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

        np.save(folder+name+'.npy', spec)
    except:
        print("This file doesn't work")


def run_model_for_spectrum(spectrum):
    NewFlux = spectrum.copy()
    for airglow_line in blue_airglow_lines:
        if airglow_line>625:
            line_flux, line_wave, line_weights = window_line(airglow_line, 1, spectrum)
        elif (np.floor(airglow_line) == 568)|(np.floor(airglow_line) == 482)|(np.floor(airglow_line) == 483):
            line_flux, line_wave, line_weights = window_line(airglow_line, 4, spectrum)
        elif (np.floor(airglow_line) == 588)|(np.floor(airglow_line) == 589)|(np.floor(airglow_line) == 498)|(np.floor(airglow_line) == 416):
            line_flux, line_wave, line_weights = window_line(airglow_line, 3, spectrum)
        elif (np.floor(airglow_line) == 466)|(np.floor(airglow_line) == 467)|(np.floor(airglow_line) == 442):
            line_flux, line_wave, line_weights = window_line(airglow_line, 5, spectrum)
        else:
            line_flux, line_wave, line_weights = window_line(airglow_line, 1.5, spectrum)

        mod = Model(my_profile)
        params = mod.make_params()
        params.add('amp', value = 10, min = 0)
        params.add('center', value = airglow_line, min = airglow_line-0.2, max = airglow_line+0.2, vary = True)
        params.add('wave', value = airglow_line, min = airglow_line-0.2, max = airglow_line+0.2, vary = True)
        params.add('a', value = 10, min = 0)
        params.add('N', value = 30000, vary = False)
        params.add('sig', value = 1, min = 0)
        params.add('c', value = 1, min = 0)

        model = mod.fit(line_flux, params, x = line_wave, weights = line_weights)
        idx = np.where((mean_wave>line_wave[0])&(mean_wave<line_wave[-1]))
        NewFlux[idx] = model.params['c'].value

    return NewFlux


def my_profile(x, amp, center, wave, N, a, sig, c):

    gauss = amp*np.exp(-(x-wave)**2/(2*sig**2.)) 
    core = gauss
    
    w = center/N * (1/(np.sqrt(2)*np.pi))
    top = w**2.
    bot = ((x-center)**2+w**2)
    scatt = a*top/bot 
    return np.convolve(core, scatt, 'same') + c

def window_line(line, window_size, flux):

    start, stop = [line-window_size, line+window_size]

    section = np.where((mean_wave>start)&(mean_wave<stop))[0]
    #print(section)
    window_wave = mean_wave[section]

    window_flux= flux[section]
    window_flux[window_flux<0] = 1e-18
    
    window_weights = error_model(window_flux)
    
    return window_flux, window_wave, window_weights

def error_model(flux):
    sig_sq = flux + (0.01*flux)**2.
    ww = 1/np.sqrt(sig_sq)
    ww[ww>1e5] = 1/9
    
    return ww

if __name__ == '__main__':
    main()


