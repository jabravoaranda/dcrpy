import os
import time
import numpy as np
import pandas as pd
import xarray as xr
import glob as glob
from sys import argv
#from netCDF4 import Dataset
import math
import matplotlib.pyplot as plt
from datetime import date
import time
import multiprocessing
import warnings
warnings.filterwarnings("ignore") # TODO don't use normally!!!
def mergeChirps(data,spec=False):
# this function merges the chirps of the dataset. I am doing this, because the non-polarimetric X and Ka-Band don't have a chirps and therefore we only have one fixed range-resolution
# input: data as xarray-dataset, if you set spec=True, the spectral data will be merged.
    try:
    	chirpNum = data.ChirpNum.values[0]
    except:
    	chirpNum = data.ChirpNum.values
    if spec==True:
        #try:
        #print(len(data.DoppLen))
        if len(data.DoppLen)>3:
        	maxVel = data.MaxVel.values[0]; doppLen = data.DoppLen.values[0] 
        else:	
        	maxVel = data.MaxVel.values; doppLen = data.DoppLen.values 
        dv_vec = np.empty(chirpNum)
        maxRange = [max(data.C1Range),max(data.C2Range),max(data.C3Range)]
        for chirp in range(chirpNum):
            ChRange = 'C{chirp}Range'.format(chirp=chirp+1)
            ChHSpec = 'C{chirp}HSpec'.format(chirp=chirp+1)
            ChVSpec = 'C{chirp}VSpec'.format(chirp=chirp+1)
            ChReVHSpec = 'C{chirp}ReVHSpec'.format(chirp=chirp+1)
            ChHNoisePow = 'C{chirp}HNoisePow'.format(chirp=chirp+1)
            ChVNoisePow = 'C{chirp}VNoisePow'.format(chirp=chirp+1)
            ChVel = 'C{chirp}Vel'.format(chirp=chirp+1)
            
            #- calculate noise density:
            NoiseDensV = data[ChVNoisePow]/doppLen[chirp]
            NoiseDensH = data[ChHNoisePow]/doppLen[chirp]
            #- now we need to decompose the VSpec, because this is not actually the vert. Spectrum but saved as a composite of H and V
            data[ChVSpec] = 4*data[ChVSpec] - data[ChHSpec] - 2*data[ChReVHSpec]
            #- there was a software mistake, so the noise of ReHV was not stored correctly. Therefore Alexander suggested to use s SNR threshold of 10dB, otherwise the data will be masked. For this we need to calculate SNR: SNR = signal power/noise power. In order to calculate the correct values, we need to mask all values below -90 dBZ.
            specThreshold = 10**(-90/10)
            data[ChHSpec] = data[ChHSpec].where(data[ChHSpec]>specThreshold,np.NaN)
            data[ChVSpec] = data[ChVSpec].where(data[ChVSpec]>specThreshold,np.NaN)
            NoisePowH = data[ChHSpec].count(dim=ChVel)*NoiseDensH
            NoisePowV = data[ChVSpec].count(dim=ChVel)*NoiseDensV
            SignalPowH = data[ChHSpec].sum(dim=ChVel)
            SignalPowV = data[ChVSpec].sum(dim=ChVel)
            data['SNR_H'] = SignalPowH / NoisePowH
            data['SNR_V'] = SignalPowV / NoisePowV
            #-- now also make it spectral (so spectral ZDR against SNR in respective bin (we see also some high ZDR values at the left (fast) edge, which should not be there)) and then the data should be masked with sSNR > 10dB aswell
            data['sSNR_H'] = data[ChHSpec] / NoiseDensH
            data['sSNR_V'] = data[ChVSpec] / NoiseDensH 
            #-- it is the easiest to calculate Z,ZDR,ZDP at this stage, because once the chirps are merged, it is difficult to sort out the dv needed to integrate over the spectrum...
            data['ZH'] = data[ChHSpec].sum(dim=ChVel)
            data['ZV'] = data[ChVSpec].sum(dim=ChVel)
            data['ZDR'] = 10*np.log10(data['ZH'])-10*np.log10(data['ZV'])
            data['ZDP'] = data['ZH']-data['ZV']
            #- because the different chirps have a different Doppler resolution, we also need to regrid along that axis:
            velData = np.linspace(-maxVel[chirp], maxVel[chirp], doppLen[chirp], dtype=np.float32) # in the dataformat, the dopplervelocity was not assigned yet, but rather it is stored as maxVel and doppLen which you then need to manually assign to the doppler Vel coordinate
            dv_diff = np.diff(velData) # since we regrid along the doppler axis, we need to divide the regridded doppler spectra by dv
            #print(dv_diff)
            dv_vec[chirp] = dv_diff[0]
            data = data.assign({ChVel:velData})
            velRef = np.linspace(-maxVel[0], maxVel[0], doppLen[0], dtype=np.float32)# just use the Doppler velocity from the smallest chirp
            data = data.reindex({ChVel:velRef}, method = 'nearest', tolerance = 0.05) # regrid
            data[ChVSpec] = data[ChVSpec]/dv_vec[chirp]
            data[ChHSpec] = data[ChHSpec]/dv_vec[chirp]
            data[ChReVHSpec] = data[ChReVHSpec]/dv_vec[chirp]
            #- now we can rename the variables to without the chirps and then we merge the datasets along the range coordinate
            dataCh = data[[ChVSpec,ChHSpec,ChHNoisePow,ChVNoisePow,'ZH','ZV','ZDR','ZDP','SNR_H','SNR_V','sSNR_H','sSNR_V']]
            dataCh = dataCh.rename({ChRange:'range',ChVSpec:'VSpec',ChHSpec:'HSpec',ChHNoisePow:'HNoisePow',ChVNoisePow:'VNoisePow',ChVel:'Vel'})
            if chirp==0:
                finalData = dataCh
            else:
                finalData = xr.concat([finalData,dataCh],dim='range')
        dv =  xr.DataArray(dv_vec,dims=('Chirp'))
        temp = xr.Dataset({'Azm':data.Azm,
                       'Elv':data.Elv,
                       'RangeRes':data.RangeRes,
                       #'delRange':delRange,
                       'dv':dv,
                       'maxVel':maxVel,
                       'doppLen':doppLen,
                       'maxRange':maxRange
                       })
    else:
        for chirp in range(chirpNum):
            ChRange = 'C{chirp}Range'.format(chirp=chirp+1)
            ChZDR = 'C{chirp}ZDR'.format(chirp=chirp+1)
            ChPhiDP = 'C{chirp}PhiDP'.format(chirp=chirp+1)
            ChRHV = 'C{chirp}RHV'.format(chirp=chirp+1)
            ChZe = 'C{chirp}ZE'.format(chirp=chirp+1)
            ChSkew = 'C{chirp}Skew'.format(chirp=chirp+1)
            ChWidth = 'C{chirp}SpecWidth'.format(chirp=chirp+1)
            ChKurt = 'C{chirp}Kurt'.format(chirp=chirp+1)
            ChVel = 'C{chirp}MeanVel'.format(chirp=chirp+1)
            # now merge the dataset along the range coordinate
            dataCh = data[[ChZDR,ChPhiDP,ChRHV,ChZe,ChSkew,ChWidth,ChKurt,ChVel]]
            dataCh = dataCh.rename({ChRange:'range',ChZDR:'ZDR',ChPhiDP:'PhiDP',ChRHV:'RHV',ChZe:'DBZ',ChSkew:'SK',ChWidth:'WIDTH',ChKurt:'KURT',ChVel:'VEL'})
            if chirp==0:
                finalData = dataCh    
            else:
                finalData = xr.concat([finalData,dataCh],dim='range')
    
        delRange = np.concatenate((selfDiv(data.C1Range)*data.RangeRes[0],
                           selfDiv(data.C2Range)*data.RangeRes[1],
                           selfDiv(data.C3Range)*data.RangeRes[2],))
        delRange = xr.DataArray(delRange,
                            dims=('range'),
                            coords={'range':finalData.range})
        temp = xr.Dataset({'Azm':data.Azm,
                       'Elv':data.Elv,
                       'RangeRes':data.RangeRes,
                       'delRange':delRange})
    
    finalData = xr.merge([finalData, temp])
    return finalData 


def dealiazeParallel(data):
	
	n_cores = 16#multiprocessing.cpu_count()
	#if n_cores > 1:
	#	n_cores = n_cores - 1 # we have the main function running on one core and our institute does not allow to use all cores
	pool = multiprocessing.Pool(n_cores)
	Nyquvist = data.maxVel.values#[6.350626, 4.981658, 2.61496353]
	RangeEnd = data.maxRange.values#[715.4785, 5902.698, 17994.29]
	doppLen = data.doppLen.values
	vel = data.Vel.values
	dv = np.diff(vel)[0]
	newVel = np.arange(-Nyquvist[0]*2,Nyquvist[0]*2+dv,dv,dtype=np.float32)
			
	data['minVelH'] = xr.DataArray(dims=('range','Time'),
			                				coords={'range':data.range,'Time':data.Time})
	data['maxVelH'] = data['minVelH'].copy()
	data['minVelV'] = data['minVelH'].copy()
	data['maxVelV'] = data['minVelH'].copy()
	data['minVelZDR'] = data['minVelH'].copy()
	data['maxVelZDR'] = data['minVelH'].copy()
	data['HSpec_rot'] = xr.DataArray(dims=('Time','range','vel_rot'),coords={'vel_rot':newVel,'Time':data.Time,'range':data.range})
	data['VSpec_rot'] = data['HSpec_rot'].copy()
	data['sZDR_rot'] = data['HSpec_rot'].copy()
	#- testing dealiasing:
	for tind,t in enumerate(data.Time):
		print(t.values)
		#print(data)
		dataRangeValues = data.range.where(np.isfinite(10*np.log10(data.sel(Time=t).ZH)),drop=True)
		args =  [(r,data.sel(Time=t)) for r in dataRangeValues]
		for r,HSpec,VSpec,sZDR,sSNR_H,sSNR_V,maxVelH,minVelH,maxVelV,minVelV,maxVelZDR,minVelZDR,HSpec_rot,VSpec_rot,sZDR_rot in pool.starmap(dealiazeOneHeight,args):
			data['HSpec'].loc[t,r,:] = HSpec
			data['VSpec'].loc[t,r,:] = VSpec
			data['sZDR'].loc[t,r,:] = sZDR
			data['sSNR_H'].loc[t,r,:] = sSNR_H
			data['sSNR_V'].loc[t,r,:] = sSNR_V
			data['maxVelH'].loc[r,t] = maxVelH
			data['minVelH'].loc[r,t] = minVelH
			data['maxVelV'].loc[r,t] = maxVelV
			data['minVelV'].loc[r,t] = minVelV
			data['maxVelZDR'].loc[r,t] = maxVelZDR
			data['minVelZDR'].loc[r,t] = minVelZDR
			data['HSpec_rot'].loc[t,r,:] = HSpec_rot
			data['VSpec_rot'].loc[t,r,:] = VSpec_rot
			data['sZDR_rot'].loc[t,r,:] = sZDR_rot
		
	return data

def dealiazeOneHeight(r,data):
	Nyquvist = data.maxVel.values#[6.350626, 4.981658, 2.61496353]
	RangeEnd = data.maxRange.values#[715.4785, 5902.698, 17994.29]
	doppLen = data.doppLen.values
	vel = data.Vel.values
	dv = np.diff(vel)[0]
	newVel = np.arange(-Nyquvist[0]*2,Nyquvist[0]*2+dv,dv,dtype=np.float32)
	#print(r)
	datasel = data.sel(range=r)
	#datasel = datasel[['sZDR','HSpec','VSpec','sSNR_H','sSNR_V']]
	sZDR = datasel.sZDR
	HSpec = datasel.HSpec
	VSpec = datasel.VSpec
	sSNR_H = datasel.sSNR_H
	sSNR_V = datasel.sSNR_V
	if r > RangeEnd[1]: # get correct nyquvist vel (I already regridded everything so I need to do it like tha
		maxVel = Nyquvist[2]
		ra = RangeEnd[2]
	elif r <= RangeEnd[0]:
		maxVel = Nyquvist[0]
		ra = RangeEnd[0]
	else:
		maxVel = Nyquvist[1]
		ra = RangeEnd[1]
	
	if (~np.isnan(sZDR.sel(Vel=maxVel,method='nearest')) or ~np.isnan(sZDR.sel(Vel=-maxVel,method='nearest')) or ~np.isnan(sZDR.sel(Vel=-maxVel+dv,method='nearest')) 
		or ~np.isnan(sZDR.sel(Vel=maxVel-dv,method='nearest')) or ~np.isnan(sZDR.sel(Vel=-maxVel+2*dv,method='nearest')) or ~np.isnan(sZDR.sel(Vel=maxVel-2*dv,method='nearest')) 
		or ~np.isnan(sZDR.sel(Vel=-maxVel+3*dv,method='nearest')) or ~np.isnan(sZDR.sel(Vel=maxVel-3*dv,method='nearest'))): 
	# if we have non-noise values at +- Nyquvist range or upto 3 vel bins to the right or left
		datadrop = datasel.where(data.Vel < maxVel,drop=True) #- since we regridded, I need to drop everything that is larger or smaller than +- Ny to actually kit the spectra together (otherwise we would have a lot of nan in between..)
		datadrop = datadrop.where(data.Vel > -maxVel,drop=True)
		q = len(datadrop.Vel.values)-1;
		k = 0;
		
		if np.isnan(datadrop.HSpec.isel(Vel=q)):
			datadrop.HSpec.values[q] = -99999
		for i in range(q):
			if np.isnan(datadrop.HSpec.isel(Vel=[i])) and  ~np.isnan(datadrop.HSpec.isel(Vel=[i+1])):
				datadrop.HSpec.values[i] = -99999
			elif np.isnan(datadrop.HSpec.isel(Vel=[i])) and  ~np.isnan(datadrop.HSpec.isel(Vel=[i+2])):
				datadrop.HSpec.values[i] = -99999
			elif np.isnan(datadrop.HSpec.isel(Vel=[i])) and  ~np.isnan(datadrop.HSpec.isel(Vel=[i+3])):
				datadrop.HSpec.values[i] = -99999
			elif np.isnan(datadrop.HSpec.isel(Vel=[i])) and  ~np.isnan(datadrop.HSpec.isel(Vel=[i+4])):
				datadrop.HSpec.values[i] = -99999
			elif np.isnan(datadrop.HSpec.isel(Vel=[i])) and  ~np.isnan(datadrop.HSpec.isel(Vel=[i+5])):
				datadrop.HSpec.values[i] = -99999
		if datadrop.HSpec.isnull().any(dim='Vel'):
			while ~np.isnan(datadrop.HSpec.isel(Vel=q)): #-- find out how much we need to shift
				k = k + 1;
				q = q - 1;
		# now roll the coordinates:
		datadrop = datadrop.where(datadrop.HSpec!=-99999,np.nan)
		#print(maxVel)
		dataRoll = datadrop.roll(Vel=k+1,roll_coords=False)  # shift everything, the direction doesnt matter
		
		dataRollRegridded = dataRoll.reindex({'Vel':data.Vel.values}, method = 'nearest', tolerance = 0.05) # now I need to go back to original velocity
		HSpec = dataRollRegridded.HSpec
		VSpec = dataRollRegridded.VSpec
		sZDR = dataRollRegridded.sZDR
		sSNR_H = dataRollRegridded.sSNR_H
		sSNR_V = dataRollRegridded.sSNR_V
	
	
	#- now move everything to 0 and name that rot (better for plotting)
	velMatrix = HSpec.values/HSpec.values*vel
	maxVelH = np.nanmax(velMatrix)#,axis=1)
	minVelH = np.nanmin(velMatrix)#,axis=1)
	if np.isnan(maxVelH):
		maxVelH = 0
	if np.isnan(minVelH):
		minVelH = 0
	
	velMatrix = VSpec.values/VSpec.values*vel
	maxVelV = np.nanmax(velMatrix)#,axis=1)
	minVelV = np.nanmin(velMatrix)#,axis=1)
	if np.isnan(maxVelV):
		maxVelV = 0
	if np.isnan(minVelV):
		minVelV = 0
		
	velMatrix = sZDR.values/sZDR.values*vel
	maxVelZDR = np.nanmax(velMatrix)#,axis=1)
	minVelZDR = np.nanmin(velMatrix)#,axis=1)
	if np.isnan(maxVelZDR):
		maxVelZDR = 0
	if np.isnan(minVelZDR):
		minVelZDR = 0
	#- now shift everything to 0, because we might get folding I have to increase Dopplervelocity range to have +-12
	HSpecRe = HSpec.reindex({'Vel':newVel}, method = 'nearest', tolerance = 0.05) 
	vel2rot = int((-maxVelH/dv).round())
	if datasel.maxVelV > 0:
		HSpec_rot = HSpecRe.roll(Vel=vel2rot-1,roll_coords=False)
	else:
		HSpec_rot = HSpecRe.roll(Vel=vel2rot,roll_coords=False)
	
	VSpecRe = VSpec.reindex({'Vel':newVel}, method = 'nearest', tolerance = 0.05) 	
	vel2rot = int((-maxVelV/dv).round())
	if datasel.maxVelV > 0:
		VSpec_rot = VSpecRe.roll(Vel=vel2rot-1,roll_coords=False)
	else:
		VSpec_rot = VSpecRe.roll(Vel=vel2rot,roll_coords=False)
		
	sZDRRe = sZDR.reindex({'Vel':newVel}, method = 'nearest', tolerance = 0.05) 
	vel2rot = int((-maxVelZDR/dv).round())
	if datasel.maxVelZDR > 0:
		sZDR_rot = sZDRRe.roll(Vel=vel2rot-1,roll_coords=False)
	else:
		sZDR_rot = sZDRRe.roll(Vel=vel2rot,roll_coords=False)
	return r,HSpec,VSpec,sZDR,sSNR_H,sSNR_V,maxVelH,minVelH,maxVelV,minVelV,maxVelZDR,minVelZDR,HSpec_rot,VSpec_rot,sZDR_rot

def calcPhiDP(data):
    # change phidp, since it was stored weird
    #PHIDP = data['PhiDP'].values
    
    data['PhiDP'] = np.rad2deg(data['PhiDP']) # convert to deg, add -1 because convention is other way around (now the phase shift gets negative, we want it to get positive with range...) TODO: check with Alexander if that makes sense!!
    #data['PhiDP'] = data['PhiDP'].rolling(range=5, min_periods=1,center=True).mean()
    data['PhiDP'].attrs = {'standard_name':'PhiDP',
                     'long_name': 'Differential phase shift',
                     'units':'deg'}
    return data


def calcKDP(data,timeWindow = 30):
    # time window: timewindow*timeres gives the amount of seconds over which will be averaged 
    # calculate KDP from phidp directly
    delRange = np.diff(data.range)
    if len(data.time)>1: # need to do that because when radar crashed we might have really small files where the time moving mean does not work anymore
      PHIDP = data['PhiDP'].rolling(time=timeWindow,min_periods=1,center=True).mean() # moving window average in time
      PHIDP = PHIDP.rolling(range=5, min_periods=1, center=True).mean() #moving window average in range
      data['KDP'] = PHIDP.diff(dim='range')/(2.*abs(delRange)*1e-3) # in order to get °/km we need to multiply with 1e-3
      data['KDP'].attrs = {'long_name': 'Specific differential phase shift',
                   'units':'deg/km'}
    else:
      data['KDP'] = data['PhiDP'].copy()*np.nan
      data['KDP'].attrs = {'long_name': 'Specific differential phase shift',
                   'units':'deg/km'}  
    return data