"""
Takes any set of files from raw_meta, rich_meta, rich_plus, or mean_plus
and turns it into one big meta file.

"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import astropy.table

file_dir = '/global/cscratch1/sd/parkerf/sky_flux/rich_mean_more/'

files = glob.glob(file_dir+'/*.fits')
("got %d files from %s" % (len(files), file_dir))

Mega_file = []
for filen in files:
	d = astropy.table.Table.read(filen)
	Mega_file.append(d)
print("Got all the data")

MF = astropy.table.vstack(Mega_file)
print("Made it into one big file")

# Now get platequality info
pl = astropy.table.Table.read(os.getcwd()+'/util/platelist.fits')
Pl = pl[['PLATE','MJD','PLATEQUALITY','QUALCOMMENTS']]
bad_plates = np.unique(Pl[Pl['PLATEQUALITY'] == 'bad ']['PLATE'])
all_bad = []
for plate in bad_plates:
    this_data = Pl[Pl['PLATE'] == plate]
    if len(np.unique(this_data['PLATEQUALITY'])) == 1:
        all_bad.append(plate)

for plate in all_bad:
    MF.remove_rows(MF['PLATE'] == plate)

MF.write('spframe_line_mostly_good.fits',format='fits')

some_bad = []
for plate in bad_plates:
    this_data = Pl[Pl['PLATE'] == plate]
    if len(np.unique(this_data['PLATEQUALITY'])) == 2:
        bad_ones = this_data[this_data['PLATEQUALITY'] == 'bad ']
        for bad_one in bad_ones:
            some_bad.append([bad_one['PLATE'],bad_one['MJD']])
np.save('some_bad_list.npy', some_bad)

for bad_one in some_bad:
    try:
        MF.remove_rows((MF['PLATE'] == plate)&(MF['MJD'] == bad_one['MJD']))
    except:
        print(bad_one)
MF.write('spframe_line_good.fits',format='fits')
        
print('finished writing!')
