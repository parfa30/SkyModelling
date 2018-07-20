import numpy as np 
import pandas as pd 

SAVE_DIR = '/Users/parkerf/Research/SkyModel/BOSS_Sky/ContFitting/files/'
LINE_LIST_DIR = '/Users/parkerf/Research/SkyModel/BOSS_Sky/ContFitting/FitTest/'
def main():
	airglow_line_list = pd.read_csv(LINE_LIST_DIR + 'line_list.txt',delim_whitespace=True,skiprows=1)
	Lines = airglow_line_list[['obs_lam','FWHM','peak_int']]
	Lines.drop_duplicates(inplace=True)

	Lines['vacuum'] = air_to_vac(Lines['obs_lam'])
	BlueLines = Lines[(Lines['vacuum']>360)&(Lines['vacuum']<628)&(Lines['peak_int']>150)]['vacuum']
	RedLines = Lines[(Lines['vacuum']>570)&(Lines['vacuum']<1040)&(Lines['peak_int']>150)]['vacuum']

	Artificial = pd.read_csv(SAVE_DIR + '/artificial_lines.csv')
	ArtLines = air_to_vac(np.array(Artificial[' wave'])/10.)
	BlueArt = ArtLines[(ArtLines>360)&(ArtLines<628)]
	RedArt = ArtLines[(ArtLines>570)&(ArtLines<1040)]

	BB = np.hstack([BlueLines,BlueArt])
	RR = np.hstack([RedLines,RedArt])

	NewBlue = remove_close_lines(BB)
	NewRed = remove_close_lines(RR)

	AllLines = np.hstack([NewBlue, NewRed])
	print("All Lines: ", len(np.unique(AllLines)))
	print("Blue Lines: ", len(np.unique(NewBlue)))
	print("Red Lines: ", len(np.unique(NewRed)))
	print("Artificial Lines: ", len(remove_close_lines(ArtLines)))

	np.save(SAVE_DIR + 'blue_airglow_lines.npy', NewBlue)
	np.save(SAVE_DIR + 'red_airglow_lines.npy', NewRed)


def air_to_vac(wave):
    """Index of refraction to go from wavelength in air to wavelength in vacuum
    Equation from (Edlen 1966)
    vac_wave = n*air_wave
    """
    #Convert to um
    wave_um = wave*.001
    ohm2 = (1./wave_um)**(2)

    #Calculate index at every wavelength
    nn = []
    for x in ohm2:
        n = 1+10**(-8)*(8342.13 + (2406030/float(130.-x)) + (15997/float(389-x)))
        nn.append(n)
    
    #Get new wavelength by multiplying by index of refraction
    vac_wave = nn*wave
    return vac_wave

def remove_close_lines(line_list):
	to_remove = []
	for i,line in enumerate(np.array(line_list)):
	    if i == len(line_list)-1:
	        pass
	    else:
	        diff = (line_list[i+1] - line)
	        if diff<0.3:
	            to_remove.append(i+1)
	            
	NewLines = np.delete(line_list,np.array(to_remove))
	return(NewLines)

if __name__ == '__main__':
	main()



