#####################################################
# This script is meant to regrid and dealiase the polarimetric spectra that were measured during the tripex-pol Campaign. 
#####################################################

import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import glob as glob
from sys import argv
#from matplotlib import pyplot as plt
#own routines:
import process_Wband_functions as pro
#import plotting_routines as plot
debugging = False

dateStart = pd.to_datetime('20220201'); dateEnd = pd.to_datetime('20220228')
dateList = pd.date_range(dateStart, dateEnd,freq='D')
dataOutPath = '/project/meteo/work/L.Terzi/tripex_pol_scan/output/'


for dayIndex,date2proc in enumerate(dateList):
	dataPath = '/archive/meteo/external-obs/juelich/tripex-pol-scan/wband_scan/l0/{year}/{month}/{day}'.format(year=date2proc.strftime('%Y'),
		                                                                             month=date2proc.strftime('%m'),
		                                                                             day=date2proc.strftime('%d'))
	dataOutputPath = dataOutPath + '{year}/{month}/{day}/'.format(year=date2proc.strftime('%Y'),
				                                              month=date2proc.strftime('%m'),
				                                              day=date2proc.strftime('%d'))
	
	fileId = 'CEL__{dateStr}_??????_P06_CEL.LV0.nc'.format(dateStr=date2proc.strftime('%y%m%d'))
	filePath = os.path.join(dataPath, fileId)
	fileList=sorted(glob.glob(filePath)) #=sorted(glob.glob(filePath2)) #.extend(sorted(glob.glob(filePath2)))
	if fileList: #- only process if files are there
		if debugging==True:
			print(fileList)
			#quit()
		for f in fileList:
			print(f)
			#quit()
			file2save = f.split('/')[-1].split('.nc')[0]+'.dealiasedNew.nc'
			#quit()
			data = xr.open_dataset(f)
			 
			time = data.Time.values + data.Timems.values/1000.
			data = data.assign({'Time':time})
			data.Time.attrs['units']='Seconds since 01.01.2001 00:00:00'
			data = xr.decode_cf(data)
			# get rid of duplicate time values:
			_, index_time = np.unique(data['Time'], return_index=True)
			data = data.isel(Time=index_time)
			
			#-- now merge the chirps into one range, and while we are at it, I am going to recalculate the VSpec (noise doesnt need to be removed, this is already done automatically in he RPG software)
			# (the VSpec that is in this dataset for now is not really the VSpec, but a composite of H and V, and with the ChReVHSpec variable we can calculate the real VSpec.
			if debugging==True:
				print('now merging chirps')
			dataMerged = pro.mergeChirps(data,spec=True)
			#quit()
			dataMerged['sSNR_H'] = 10*np.log10(dataMerged['sSNR_H'])
			dataMerged['sSNR_V'] = 10*np.log10(dataMerged['sSNR_V'])
			dataMerged['HSpec'] = dataMerged['HSpec'].where(dataMerged['sSNR_H']>10)#>10**(-37/10))
			dataMerged['sZDR'] = 10*np.log10(dataMerged['HSpec']) - 10*np.log10(dataMerged['VSpec'])
			
			#- dealize the spectra:
			if debugging==True:
				print('now dealiasing')
				
			#dataDealized = pro.dealiase(dataMerged)
			dataDealized = pro.dealiazeParallel(dataMerged)
			if debugging==True:
				print('now done dealiasing, saving the file again')
			if not os.path.isdir(dataOutputPath):
					os.makedirs(dataOutputPath)
			dataDealized.to_netcdf('{path}{file}'.format(path=dataOutputPath,file=file2save))

