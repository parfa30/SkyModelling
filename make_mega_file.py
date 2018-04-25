"""
Takes any set of files from raw_meta, rich_meta, rich_plus, or mean_plus
and turns it into one big meta file.

"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import astropy.table

file_dir = '/global/cscratch1/sd/parkerf/sky_flux/rich_mean/'

files = glob.glob(file_dir+'/*.fits')
("got %d files from %s" % (len(files), file_dir))

Mega_file = []
for filen in files:
	d = astropy.table.Table.read(filen)
	Mega_file.append(d)
print("Got all the data")

Mega_file = astropy.table.vstack(Mega_file)
print("Made it into one big file")

# Now get platequality info
pl = astropy.table.Table.read(os.getcwd()+'/util/platelist.fits')
Pl = pl[['PLATE','MJD','PLATEQUALITY','QUALCOMMENTS']]

Mega_file_good = astropy.table.join(Mega_file, Pl, keys=('PLATE','MJD'))
print("Combined with platequality")

Mega_file_good.write('spframe_line_sum.fits',format='fits')
print('finished writing!')
