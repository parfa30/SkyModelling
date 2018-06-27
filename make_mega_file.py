"""
Takes any set of files from raw_meta, rich_meta, rich_plus, or mean_plus
and turns it into one big meta file.

"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import astropy.table

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

    good_MF = apply_platequality(MF)
    good_MF.write('spframe_line_good.fits',format='fits')

    clean_good_MF = data_cuts(good_MF)
    good_MF.write('good_data_%s.fits'% date.strftime('%y%m%d'),format='fits')

def apply_platequality(DataTable):
    # Now get platequality info
    pl = astropy.table.Table.read(os.getcwd()+'/util/platelist.fits')
    Pl = pl[['PLATE','MJD','PLATEQUALITY','QUALCOMMENTS']]
    bad_plates = np.unique(Pl[Pl['PLATEQUALITY'] == 'bad ']['PLATE'])

    bad_days = []
    for plate in bad_plates:
        this_data = Pl[(Pl['PLATE'] == plate)&(Pl['PLATEQUALITY'] == 'bad')]
        for bad_one in this_data:
            bad_days.append([bad_one['PLATE'],bad_one['MJD']])
                
    for day in bad_days:
        try:
            DataTable.remove_rows((MF['PLATE'] == plate)&(MF['MJD'] == bad_one['MJD']))
        except:
            print(bad_one)

    return DataTable

def data_cuts(DataTable):
    #Taken from the MakeTestDataFile.ipynb
    remove_these_plates = [5745, 6138, 7258] 
    #5745 because sky fiber test plate
    #6138 because for no reason given, there isn't data for r2 in all observations and some done't have r1
    #7258 "Forced complete"

    print('outliers in blue')
    out1 = DataTable[(DataTable['CAMERAS'] == 'b1')&(DataTable['std_cont_b_460'] > DataTable['mean_cont_b_460'])]['PLATE','IMG','FIB']
    out2 = DataTable[(DataTable['CAMERAS'] == 'b1')&(DataTable['cont_b_460'] <= 0)]['PLATE','IMG','FIB']

    print('outliers in red')
    out3 = DataTable[(DataTable['CAMERAS'] == 'r1')&((DataTable['std_cont_r_720'] > DataTable['mean_cont_r_720']))]['PLATE','IMG','FIB']
    out4 = DataTable[(DataTable['CAMERAS'] == 'r1')&(DataTable['cont_r_720'] < 0)]['PLATE','IMG','FIB']
    out5 = DataTable[(DataTable['CAMERAS'] == 'r1')&(DataTable['cont_r_720'] == 0)]['PLATE','IMG','FIB']

    print('stacked outliers')
    outs = astropy.table.vstack[out1,out2,out3,out4,out5]

    idx = []
    for out in outs:
        id0 = np.where((DataTable['IMG'] == out["IMG"])&(DataTable['FIB'] == out['FIB']))
        idx.append(id0)
    print('done idx')
    idx = np.hstack(idx)
    DataTable.remove_rows(idx[0])
    print('done removing rows')

    return DataTable

print('finished writing!')

if __name__ == '__main__':
    main()
