"""
Takes any set of files from raw_meta, rich_meta, rich_plus, or mean_plus
and turns it into one big meta file.

"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import astropy.table
from datetime import datetime

def main():

    file_dir = '/global/cscratch1/sd/parkerf/sky_flux/rich_mean/'

    
    files = glob.glob(file_dir+'/*.fits')
    ("got %d files from %s" % (len(files), file_dir))

    Mega_file = []
    for filen in files:
    	d = astropy.table.Table.read(filen)
    	Mega_file.append(d)
    print("Got all the data")

    MF = astropy.table.vstack(Mega_file)
    print("Made it into one big file")

    # Remove Bad Observations
    REMOVED = []
    pl = astropy.table.Table.read(os.getcwd()+'/util/platelist.fits')
    Pl = pl[['PLATE','MJD','PLATEQUALITY','QUALCOMMENTS']]
    bad = Pl[Pl['PLATEQUALITY'] == 'bad ']

    bad_idx = []
    for line in bad:
        bad_idx.append(np.where((MF['PLATE'] == line['PLATE'])&(MF['MJD'] == line['MJD'])))

    idx = np.hstack(bad_idx)
    REMOVED.append(MF[idx[0]])
    MF.remove_rows(idx[0])

    weird_idx = []
    for plate in [5745, 6138, 7258]:
        weird_idx.append(np.where(MF['PLATE'] == plate))

    weird_idx = np.hstack(weird_idx)
    REMOVED.append(MF[weird_idx[0]])
    MF.remove_rows(weird_idx[0])

    MF.write('good_data_%s.fits'% datetime.now().strftime('%y%m%d'),format='fits')
    print("Removed bad obs")

    # Remove outliers
    Lines = pickle.load(open('/global/homes/p/parkerf/Sky/SkyModelling/util/line_file_updated.pkl','rb'))
    blue_lines = []
    red_lines = []
    for cam, info in lines.items():
        if cam == 'b1':
            for name, val in info.items():
                blue_lines.append(name)
        if cam == 'r1':
            for name, val in info.items():
                red_lines.append(name)

    blue_all_zero = MF[((MF['CAMERAS'] == 'b1')|(MF['CAMERAS'] == 'b2'))&(MF[blue_lines[0]]==0)&(MF[blue_lines[1]]==0)&(MF[blue_lines[2]]==0)&(MF[blue_lines[3]]==0)&(MF[blue_lines[4]]==0)&(MF[blue_lines[5]]==0)&(MF[blue_lines[6]]==0)&(MF[blue_lines[7]]==0)&(MF[blue_lines[8]]==0)&(MF[blue_lines[9]]==0)&(MF[blue_lines[10]]==0)&(MF[blue_lines[11]]==0)&(MF[blue_lines[12]]==0)]
    red_all_zero = MF[((MF['CAMERAS'] == 'r1')|(MF['CAMERAS'] == 'r2'))&(MF[red_lines[0]]==0)&(MF[red_lines[1]]==0)&(MF[red_lines[2]]==0)&(MF[red_lines[3]]==0)&(MF[red_lines[4]]==0)&(MF[red_lines[5]]==0)&(MF[red_lines[6]]==0)&(MF[red_lines[7]]==0)&(MF[red_lines[8]]==0)&(MF[red_lines[9]]==0)&(MF[red_lines[10]]==0)&(MF[red_lines[11]]==0)&(MF[red_lines[12]]==0)&(MF[red_lines[13]]==0)]

    all_zero_idx = []
    for line in blue_all_zero:
        cam = line['CAMERAS']
        fib = line['FIB']
        image = line['IMG']
        all_zero_idx.append(np.where((MF['CAMERAS'] == cam)&(MF['IMG'] == image)&(MF['FIB'] == fib)))
    print("removed blue outliers")

    for line in red_all_zero:
        cam = line['CAMERAS']
        fib = line['FIB']
        image = line['IMG']
        all_zero_idx.append(np.where((MF['CAMERAS'] == cam)&(MF['IMG'] == image)&(MF['FIB'] == fib)))
    print("removed redoutliers")

    all_zero_idx = np.hstack(all_zero_idx)
    REMOVED.append(MF[all_zero_idx[0]])
    MF.remove_rows(all_zero_idx[0])

    MF.write('good_clean_data_%s.fits'% datetime.now().strftime('%y%m%d'),format='fits')
    Removed = astropy.table.vstack(REMOVED)
    Removed.write('removed_observations.fits', format='fits')

    print("Done!")


if __name__ == '__main__':
    main()
