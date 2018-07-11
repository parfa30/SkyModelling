import numpy as np
import pickle

def air_to_vac(wave):
    """Index of refraction to go from wavelength in air to wavelength in vacuum
    Equation from (Edlen 1966)
    vac_wave = n*air_wave
    """

    wave_um = wave*0.0001 # A --> um
    x = (1./wave_um)**(2)

    n = 1+10**(-8)*(8342.13 + (2406030/float(130.-x)) + (15997/float(389-x)))
    
    #Get new wavelength by multiplying by index of refraction
    vac_wave = n*wave
    return vac_wave

def main():
    blue_lines = {'NI': 5200, "OI": 5577, "NaID1": 5890, "NaID2": 5896, 'HgIb': 4358, 'HgIa': 4047, 'NaIa': 4983, 'HgIc': 5461,
    'NaIb':5683, 'NaIc': 5688}
    red_lines = {'OIr1': 6300, "OIr2": 6364,'O2a': 8605, 'O2b': 8695}
    blue_cont = [380, 410, 425, 460, 480, 510, 540, 565, 583, 602, 615]
    red_cont = [642, 675, 710, 720, 740, 825, 833, 873, 920, 977, 1025]

    blue_dict = {}
    for name, l in blue_lines.items():
        wave = air_to_vac(l)
        blue_dict[name] = ['line','%.2f' % wave]
    for cline in blue_cont:
        name = 'cont_b_%s' % str(cline)
        blue_dict[name] = ['cont', cline]

    red_dict = {}
    for name, l in red_lines.items():
        wave = air_to_vac(l)
        red_dict[name] = ['line','%.2f' % wave]
    for cline in red_cont:
        name = 'cont_b_%s' % str(cline)
        red_dict[name] = ['cont', cline]

    line_dict = {}
    line_dict['b1'] = blue_dict
    line_dict['b2'] = blue_dict
    line_dict['r1'] = red_dict
    line_dict['r2'] = red_dict

    pickle.dump(line_dict,open('line_file.pkl','wb'))

if __name__ == '__main__':
    main()

