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
    SPECTRA_DIR = '/global/cscratch1/sd/parkerf/sky_flux_corrected/mean_spectra/'
    mean_red_files = glob.glob(SPECTRA_DIR + '/*/*_r*_mean_spectrum.npy', recursive=True)

    global SAVE_DIR
    SAVE_DIR = SPECTRA_DIR + '/mean_cont_spectra/'
    done_red_files = glob.glob(SAVE_DIR + '/*/*_r*.npy', recursive=True)

    Complete_Images = [os.path.split(d)[1][0:9] for d in done_red_files]
    All_Images = [os.path.split(d)[1][0:9] for d in mean_red_files]
    images_needed_idx = [i for i, x in enumerate(All_Images) if x not in Complete_Images]
    IMAGES = [mean_red_files[x] for x in images_needed_idx]
    print("All files",len(mean_red_files))
    print("Done files",len(done_red_files))
    print("Needed files",len(IMAGES))

    global mean_wave
    mean_wave = np.linspace(300, 1040, (1040-300)*100)

    # Get Lines
    global red_airglow_lines
    red_airglow_lines = np.load('util/red_airglow_lines.npy')
    print(red_airglow_lines)

    pool = multiprocessing.Pool(processes=64)
    pool.map(save_new_file, IMAGES)
    pool.close()

    # for file in mean_blue_files[1:6]:
    #     save_new_file(file)

def save_new_file(filen):
    names = os.path.split(filen)
    plate = names[0][-4:]
    print(plate)
    name = names[1][0:9]
    print(filen)

    points = [588,595,602,605,608,613,618,622,632,640,643,647,652,655,658,662,668,676,682,686,691,693,695,697,700,
         703,712,720,721,740,751,755,759,768,775,777,782,791,794,804,806,809.5,812,816,823,827,833,842,
         849,851.5,853,857,872,880,882,888,895.3,901,912,914.5,919.5,921,924,928,935,943,951954.5,958.5,960,
         961,965,977,986,993,999,1005.5,1011.5]
    try:
        spectrum = np.load(filen)
        new_spectrum = run_model_for_spectrum(spectrum)
        data_points = [new_spectrum[np.argmin(np.abs(mean_wave-p))] for p in points]

        f = interp1d(points,data_points, bounds_error=False, fill_value = 'extrapolate')
        filtered_spectrum = gaussian_filter1d(f(mean_wave), 200)

        #Create files
        spec=np.zeros(len(spectrum),dtype=[('WAVE','f8'),('SKY','f8'),('CONT','f8'),('FILT_CONT','f8')])
        spec['WAVE'] = mean_wave
        spec['SKY'] = spectrum
        spec['CONT'] = new_spectrum
        spec['FILT_CONT'] = filtered_spectrum

        folder = SAVE_DIR+'/'+plate+'/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        np.save(folder+name+'.npy', spec)
    except:
        print('file didnt work')


def run_model_for_spectrum(spectrum):
    NewFlux = spectrum.copy()
    for airglow_line in red_airglow_lines:
        try:
            if np.floor(airglow_line) in [588,589,682,683,647,772,862,864,865,866,867,868]:
                window = 2
            elif np.floor(airglow_line) in [836]:
                window =  2.5
            elif np.floor(airglow_line) in [724,725,772,784,933]:
                window =  .25
            elif np.floor(airglow_line) in [593, 629,828, 915]:
                window = 1
            elif np.floor(airglow_line) in [623,625,628,771,888,886,892,896,894,922,940,942,987,991]:
                window = 0.75
            elif np.floor(airglow_line) in [615, 616]:
                window = 4
            elif np.floor(airglow_line) in [698, 697, 695, 694, 692, 691, 690,689,686,655,653,650,620,633,632,655,623,701,727,835,840,850,900]:
                window = 0.5
            else:
                window = 0.4
                
            line_flux, line_wave, line_weights = window_line(airglow_line, window, spectrum)

            mod = Model(my_profile)
            params = mod.make_params()
            params.add('amp', value = 10, min = 0)
            params.add('center', value = airglow_line, min = airglow_line-0.5, max = airglow_line+0.5, vary = True)
            params.add('wave', value = airglow_line, min = airglow_line-0.5, max = airglow_line+0.5, vary = True)
            params.add('a', value = 10, min = 0)
            params.add('N', value = 40000, vary = False)
            params.add('sig', value = 1, min = 0)
            params.add('c', value = 1, min = 0)

            model = mod.fit(line_flux, params, x = line_wave, weights = line_weights)
            idx = np.where((mean_wave>line_wave[0])&(mean_wave<line_wave[-1]))
            NewFlux[idx] = model.params['c'].value
        except:
            pass
            #print('%.2f line didnt work'% airglow_line)

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
    #window_flux[window_flux<0] = 1e-18
    
    window_weights = error_model(window_flux)
    
    return window_flux, window_wave, window_weights

def error_model(flux):
    sig_sq = flux + (0.01*flux)**2.
    ww = 1/np.sqrt(sig_sq)
    ww[ww>1e5] = 1/9
    
    return ww

if __name__ == '__main__':
    main()


