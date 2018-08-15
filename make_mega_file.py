"""
Takes any set of files from raw_meta, rich_meta, rich_plus, or mean_plus
and turns it into one big meta file.

"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import astropy.table
from datetime import datetime
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--full", action='store_true',
                    help="determines that want a merged file for all sky meta data")
parser.add_argument("--mean", action='store_true',
                    help="determines that want a merged file for only mean sky spectra meta data")
args = parser.parse_args()

try:
    DATA_DIR = os.environ['SKY_FLUX_DIR']
except KeyError:
    print("Need to set SKY_FLUX_DIR\n export SKY_FLUX_DIR=`Directory to save data'")

def main():

    if args.full:
        file_dir = DATA_DIR+ '/rich_plus_meta/'
        filen = 'meta_data_%s.fits'% datetime.now().strftime('%y%m%d')
    elif args.mean:
        file_dir = DATA_DIR+ '/mean_meta/'
        filen = 'mean_meta_data_%s.fits'% datetime.now().strftime('%y%m%d')
    else:
        print("No file set was selected. Creating mean file for rich meta")
        file_dir = DATA_DIR+ '/rich_meta/'
        filen = 'rich_meta_data_%s.fits'% datetime.now().strftime('%y%m%d')
    files = glob.glob(file_dir + '/*.fits')

    print("got %d files from %s" % (len(files), file_dir))

    #Load all files
    Mega_file = []
    for ff in files:
        try:
    	    d = astropy.table.Table.read(ff)
    	    Mega_file.append(d)
        except:
            print(ff)
    print("Got all the data")

    MF = astropy.table.vstack(Mega_file)
    print(DATA_DIR+'/all_'+filen)
    MF.write(DATA_DIR+'/all_'+filen,format='fits')
    print("Made it into one big file")

    print("All images")
    print("Total Plates: %d" % len(np.unique(MF['PLATE'])))
    print("Total Obs: %d" % len(np.unique(MF['IMG'])))
    print("Total Days: %d" % len(np.unique(MF['MJD'])))

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

    MF.write(DATA_DIR+'/good_'+filen,format='fits')
    print("Removed bad obs")

    # Remove outliers
    Lines = pickle.load(open(os.getcwd()+'/util/line_file.pkl','rb'))
    blue_lines = []
    red_lines = []
    for cam, info in Lines.items():
        if cam == 'b1':
            for name, val in info.items():
                blue_lines.append(name)
        if cam == 'r1':
            for name, val in info.items():
                red_lines.append(name)

    blue_all_zero = MF[((MF['CAMERAS'] == 'b1')|(MF['CAMERAS'] == 'b2'))&(MF[blue_lines[0]]==0)&(MF[blue_lines[1]]==0)&(MF[blue_lines[2]]==0)&(MF[blue_lines[3]]==0)&(MF[blue_lines[4]]==0)&(MF[blue_lines[5]]==0)&(MF[blue_lines[6]]==0)&(MF[blue_lines[7]]==0)&(MF[blue_lines[8]]==0)&(MF[blue_lines[9]]==0)&(MF[blue_lines[10]]==0)&(MF[blue_lines[11]]==0)&(MF[blue_lines[12]]==0)&(MF[blue_lines[13]]==0)&(MF[blue_lines[14]]==0)&(MF[blue_lines[15]]==0)&(MF[blue_lines[16]]==0)&(MF[blue_lines[17]]==0)&(MF[blue_lines[18]]==0)&(MF[blue_lines[19]]==0)&(MF[blue_lines[20]]==0)]
    red_all_zero = MF[((MF['CAMERAS'] == 'r1')|(MF['CAMERAS'] == 'r2'))&(MF[red_lines[0]]==0)&(MF[red_lines[1]]==0)&(MF[red_lines[2]]==0)&(MF[red_lines[3]]==0)&(MF[red_lines[4]]==0)&(MF[red_lines[5]]==0)&(MF[red_lines[6]]==0)&(MF[red_lines[7]]==0)&(MF[red_lines[8]]==0)&(MF[red_lines[9]]==0)&(MF[red_lines[10]]==0)&(MF[red_lines[11]]==0)&(MF[red_lines[12]]==0)&(MF[red_lines[13]]==0)&(MF[red_lines[14]]==0)]
    print('got lines and all')

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

    try:
        all_zero_idx = np.hstack(all_zero_idx)
        REMOVED.append(MF[all_zero_idx[0]])
        MF.remove_rows(all_zero_idx[0])
    except:
        print('no outliers to remove')
    MF.write(DATA_DIR+'/good_clean_'+filen,format='fits')
    Removed = astropy.table.vstack(REMOVED)
    Removed.write(DATA_DIR+'/removed_observations_%s.fits'% datetime.now().strftime('%y%m%d'), format='fits')

    print("All images after removal")
    print("Total Plates: ", len(np.unique(MF['PLATE'])))
    print("Total Obs: ", len(np.unique(MF['IMG'])))
    print("Total Days: ", len(np.unique(MF['MJD'])))

    print("Done!")


if __name__ == '__main__':
    main()
