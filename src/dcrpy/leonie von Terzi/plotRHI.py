######################################################################################################################################
# this routine automatically reads the binary RPG version 4 files into nc and then resamples them to the common tripex-pol-scan grid
# author: Leonie von Terzi
# last changed: 06.12.2021
######################################################################################################################################

import numpy as np
import xarray as xr
import pandas as pd
import glob
import os
from sys import argv

#import read_RPG_lvl1_v4 as readLV1
import processData as pro
import matplotlib.pyplot as plt

scriptname, date  = argv #, date,dataPath,dataPathOutput,Band #, date, dataPath, dataPathOutput, emptyDataPath
print(date)
#quit()
date = pd.to_datetime(date)
'''
input: 
date: date that you want to have processed
dataPath: path where the X-band data is stored
dataPathOutput: path where to put the plot
emptyDataPath: path to where there is a nc file with empty data in it
'''

pathOut = '/data/optimice/tripex-pol/plots'
filePathW = '/data/obs/campaigns/tripex-pol-scan/wband_scan/l0/'
filePathKa = '/data/obs/site/jue/joyrad35/'
fileExtW = '.LV1.nc'
fileExtKa = '_tripex_pol_II_rhi.znc'
# path to emptyData
emptyDataPath = '/data/optimice/tripex-pol/auxPlotData/noData1.nc'
#inputPath = '/data/obs/campaigns/tripex-pol-scan/wband_scan/l0/'
# This block defines the range reference grid.
# The reference grid is the same for all radars
#
beginRangeRef = 0 # starting height of the ref grid
endRangeRef = 12000 # ending height of the ref grid
rangeFreq = 36 # range resolution of the ref grid
rangeTolerance = 18 # tolerance for detecting the closest neighbour

rangeRef = np.arange(beginRangeRef, endRangeRef, rangeFreq)
# tolerance for detecting closest neighbour (seconds)
timeTolerance = 1
#################################################################################################
# lets first save new binary files into nc files, for all scan patterns, since we want to have all nc files anyway. Since we are first checking if the file is already there, there is no need to change that if we later want to plot just ZEN and had CEL before.

##################################################################################################################
# now lets do the resampling
##################################################################################################################
datenow = pd.Timestamp.now() 

print(date)
if date.strftime('%y%m%d') == datenow.strftime('%y%m%d'):   
  hournow = datenow.strftime('%H')
  hour2proc = np.arange(int(hournow)+1)

else:
  hour2proc = np.arange(24)

# all variables to plot, need to do that separately for now TODO: improve that!
variables = {'Ze':{'vmax':25, 'vmin':-35,'units':'[dB]'},
              'MDV':{'vmax':0, 'vmin':-3,'units':r'[ms$^{-1}$]'},
              'ZDR':{'vmax':0, 'vmin':4,'units':r'[dB]'},
              'KDP':{'vmax':0, 'vmin':4,'units':r'[°km$^{-1}$]'},
              'sZDRmax':{'vmax':0, 'vmin':4,'units':r'[dB]'},
              'WIDTH':{'vmax':0, 'vmin':1,'units':r'[ms$^{-1}$]'},
              'sLDR':{'vmax':-20, 'vmin':-35,'units':r'[dB]'},
              }   
for var in variables:
  fig,axes = plt.subplots(nrows=4,ncols=6,figsize=(24, 12))  
  # fill up the plot according to all the hours that we already had. Therefore select the according axis: 
  for hour in hour2proc:
    if hour in np.arange(6):
      ax = axes[0,hour]
    elif hour in np.arange(6,12):
      ax = axes[1,hour-6]
    elif hour in np.arange(12,18):
      ax = axes[2,hour-12]
    else: 
      ax = axes[3,hour-18]
    timeFreq = '4S'; timeTolerance = '2S'; 
    timeRef = pd.date_range(date,date+pd.offsets.Hour(24)-pd.offsets.Second(1),freq=timeFreq)
  
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
  
    dateStr = year+month+day
  
  # defining the input file name
    filesWbandScan = sorted(glob.glob(filePathW+'{year}/{month}/{day}/FIRST_RHI__{date}_{hour}*_P09_{scantype}{fileExt}'.format(year=date.strftime('%Y'),
                                                                              month=date.strftime('%m'),
                                                                              day=date.strftime('%d'),
                                                                              date = date.strftime('%y%m%d'),
                                                                              hour = "{0:0=2d}".format(hour),
                                                                              scantype=scan2proc, # here we just use the ones specified by the bash skript
                                                                              fileExt=fileExtW)))
  
    #print(filesWbandScan)
    
    if filesWbandScan:
      f = filesWbandScan[0]
      wband_scan = xr.open_dataset(f)
              
      wband_scan = pro.mergeChirps(wband_scan)
      wband_scan = wband_scan.rename({'Time':'time'})
      
      wband_scan = pro.calcPhiDP(wband_scan)
      wband_scan = pro.calcKDP(wband_scan,timeWindow=5)
  
      
    
      filesWbandScanLV0 = sorted(glob.glob(filePathW+'{year}/{month}/{day}/FIRST_RHI__{date}_{hour}*_P09_{scantype}{fileExt}'.format(year=date.strftime('%Y'),
                                                                              month=date.strftime('%m'),
                                                                              day=date.strftime('%d'),
                                                                              date = date.strftime('%y%m%d'),
                                                                              hour = "{0:0=2d}".format(hour),
                                                                              scantype=scan2proc, # here we just use the ones specified by the bash skript
                                                                              fileExt='.LV0.nc')))
      
      #wband_scan_LV0 = xr.Dataset()
      f = filesWbandScanLV0[0]
      data = xr.open_dataset(f)
        #
      data = pro.mergeChirps(data,spec=True)
      data = data.rename({'Time':'time'})
      data['sZDR'] = 10*np.log10(data['HSpec']) - 10*np.log10(data['VSpec']) 
      data['sZDRmax'] = data['sZDR'].max(dim='Vel',keep_attrs=True)
          #for t in data.time:
          
      wband_scan = xr.merge([wband_scan, data[['sZDRmax']]])
    # converting Zg to log units
      wband_scan.time.attrs['standard_name'] = 'time'
      wband_scan.time.attrs['long_name'] = 'time'
      wband_scan.time.attrs['units']='seconds since 2001-01-01 00:00:00 UTC'
      wband_scan = xr.decode_cf(wband_scan)
      
      convert = ['Ze']#,'sLDR']#,'ZDR']
      for varconv in convert:
        wband_scan[varconv] = 10*np.log10(wband_scan[varconv])
      
      # now plot the variable in the correct space
      var2plot = (wband_scan[var]).transpose()        
      plot=ax.pcolormesh(wband_scan['range']*np.cos(np.deg2rad(wband_scan['Elv'])), wband_scan['range']*np.sin(np.deg2rad(wband_scan['Elv'])), 
                    var2plot, 
                    vmax = variables[var]['vmax'],
                    vmin = variables[var]['vmin'],
                    cmap='jet')    
      cb = plt.colorbar(plot,ax=ax)
      cb.set_label(var+variables[var]['units'])
      ax.set_ylabel('range [m]')
      ax.set_xlabel('range [m]')
      ax.set_xlim(-10000,10000)
      ax.set_ylim(0,10000)
      #ax.axis('scaled')
      ax.set_title('Hour: {hour}'.format(hour="{0:0=2d}".format(hour)))
      ax.grid()
      plt.tight_layout()
      
  print(var)
  fileName = ('_').join([dateStr,var+'_RHI_W.png'])
  filePathName = ('/').join([pathOut,fileName]) 
  plt.savefig(filePathName,dpi=200,bbox_inches='tight')   
  plt.close()

#############################################################################
# now plot Ka
#############################################################################

print(date)
if date.strftime('%y%m%d') == datenow.strftime('%y%m%d'):   
  hournow = datenow.strftime('%H')
  hour2proc = np.arange(int(hournow)+1)

else:
  hour2proc = np.arange(24)

# all variables to plot, need to do that separately for now TODO: improve that!
variables = {'Zg':{'vmax':25, 'vmin':-35,'units':'[dB]'},
                'VELg':{'vmax':0, 'vmin':-3,'units':r'[ms$^{-1}$]'},
                'RMSg':{'vmax':0, 'vmin':1,'units':r'[ms$^{-1}$]'},
                'LDRg':{'vmax':-20, 'vmin':-35,'units':r'[dB]'},
              }  
for var in variables:
  
  fig,axes = plt.subplots(nrows=4,ncols=6,figsize=(24, 12))  
  # fill up the plot according to all the hours that we already had. Therefore select the according axis: 
  for hour in hour2proc:
    if hour in np.arange(6):
      ax = axes[0,hour]
    elif hour in np.arange(6,12):
      ax = axes[1,hour-6]
    elif hour in np.arange(12,18):
      ax = axes[2,hour-12]
    else: 
      ax = axes[3,hour-18]
    
    timeFreq = '1S'; timeTolerance = '1S'; 
    timeRef = pd.date_range(date,date+pd.offsets.Hour(24)-pd.offsets.Second(1),freq=timeFreq)
  
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
  
    dateStr = year+month+day
  
  # defining the input file name
    filesKabandScan = sorted(glob.glob(filePathKa+'{year}/{month}/{day}/{date}_{hour}*{fileExt}'.format(year=date.strftime('%Y'),
                                                                              month=date.strftime('%m'),
                                                                              day=date.strftime('%d'),
                                                                              date = date.strftime('%Y%m%d'),
                                                                              hour = "{0:0=2d}".format(hour),
                                                                              fileExt=fileExtKa)))
  
    
    if filesKabandScan:
      f = filesKabandScan[0]
      KaData = xr.open_dataset(f)
      _, index_time = np.unique(KaData['time'], return_index=True)
      KaData = KaData.isel(time=index_time)
      KaData.time.attrs['units']='seconds since {0}'.format('1970-01-01 00:00:00 UTC')
      KaData = xr.decode_cf(KaData)
      
      convert = ['Zg']#,'sLDR']#,'ZDR']
      for varconv in convert:
        KaData[varconv] = 10*np.log10(KaData[varconv])
      
      # now only plot the first RHI --> select only the first one by looking for monotonically increasing elv
      diff_elv = np.diff(KaData.elv)
      diff_elv = np.insert(diff_elv,-1,0)
      
      time_increasing = KaData.time.where(diff_elv > 0)
      time_increasing = time_increasing.dropna(dim='time')
      data_increasing = KaData.sel(time=time_increasing.values)
      
      # now plot the variable in the correct space
      var2plot = (data_increasing[var]).transpose()  
      elv2plot = np.deg2rad(data_increasing['elv'])
      
      plot=ax.pcolormesh(data_increasing['range']*np.cos(elv2plot),
                          data_increasing['range']*np.sin(elv2plot), 
                          var2plot, 
                          vmax = variables[var]['vmax'],
                          vmin = variables[var]['vmin'],
                          cmap='jet')    
      cb = plt.colorbar(plot,ax=ax)
      cb.set_label(var+variables[var]['units'])
      ax.set_ylabel('range [m]')
      ax.set_xlabel('range [m]')
      ax.set_xlim(-10000,10000)
      ax.set_ylim(0,10000)
      #ax.axis('scaled')
      ax.set_title('Hour: {hour}'.format(hour="{0:0=2d}".format(hour)))
      ax.grid()
      plt.tight_layout()
      
  print(var)
  fileName = ('_').join([dateStr,var+'_RHI_Ka.png'])
  filePathName = ('/').join([pathOut,fileName]) 
  plt.savefig(filePathName,dpi=200,bbox_inches='tight')   
  plt.close()

#####################################################################
# plot difference variables
#####################################################################

print(date)
if date.strftime('%y%m%d') == datenow.strftime('%y%m%d'):   
  hournow = datenow.strftime('%H')
  hour2proc = np.arange(int(hournow)+1)

else:
  hour2proc = np.arange(24)

variables = {'Zg':{'vmax':20, 'vmin':-5, 'name':'DWR','units':'[dB]'},
              'VELg':{'vmax':0.3, 'vmin':-0.3, 'name':'DDV','units':r'[ms$^{-1}$]'},
              'RMSg':{'vmax':0.3, 'vmin':-0.3, 'name':'DSW','units':r'[ms$^{-1}$]'}
            }

  # creating the difference plots
for var in variables.keys():  
    
  fig,axes = plt.subplots(nrows=4,ncols=6,figsize=(24, 12))  
  # fill up the plot according to all the hours that we already had. Therefore select the according axis: 
  for hour in hour2proc:
    if hour in np.arange(6):
      ax = axes[0,hour]
    elif hour in np.arange(6,12):
      ax = axes[1,hour-6]
    elif hour in np.arange(12,18):
      ax = axes[2,hour-12]
    else: 
      ax = axes[3,hour-18]
  
    timeFreq = '1S'; timeTolerance = '1S'; 
    timeRef = pd.date_range(date,date+pd.offsets.Hour(24)-pd.offsets.Second(1),freq=timeFreq)
    
    beginElv = 30 ; endElv = 150 ; elvRef = np.arange(beginElv,endElv+1,3) # we need to go in steps of three here because we have 3 chrip sequences that each take 1 second (so 1° elv, so therefore we do not have a resolution of 1° but rather 3°)
    
    year = date.strftime('%Y')
    month = date.strftime('%m')
    day = date.strftime('%d')
  
    dateStr = year+month+day
  
  # defining the input file name
    filesWbandScan = sorted(glob.glob(filePathW+'{year}/{month}/{day}/FIRST_RHI__{date}_{hour}*_P09_{scantype}{fileExt}'.format(year=date.strftime('%Y'),
                                                                              month=date.strftime('%m'),
                                                                              day=date.strftime('%d'),
                                                                              date = date.strftime('%y%m%d'),
                                                                              hour = "{0:0=2d}".format(hour),
                                                                              scantype=scan2proc, # here we just use the ones specified by the bash skript
                                                                              fileExt=fileExtW)))
  
    #print(filesWbandScan)
    
    if filesWbandScan:
      f = filesWbandScan[2]
      wband_scan_chirp = xr.open_dataset(f)
      wband_scan_chirp.Time.attrs['standard_name'] = 'time'
      wband_scan_chirp.Time.attrs['long_name'] = 'time'
      wband_scan_chirp.Time.attrs['units']='seconds since 2001-01-01 00:00:00 UTC'
      wband_scan_chirp = xr.decode_cf(wband_scan_chirp)
      
      wband_scan = pro.mergeChirps(wband_scan_chirp)
      wband_scan_chirp = wband_scan_chirp.rename({'Time':'time'})
      wband_scan = wband_scan.rename({'Time':'time','Ze':'Zg','MDV':'VELg','WIDTH':'RMSg'})
      
      HourMinW = pd.to_datetime(str(wband_scan.time.values[0])).round(freq='T').strftime('%H%M') # round to closest minute
      
      # now reindex along range:
      wband_scan = wband_scan.reindex({'range':rangeRef},method='nearest',tolerance=rangeTolerance)    
      #wband_scan = wband_scan.reindex({'time':timeRef},method='nearest',tolerance=timeTolerance)
      wband_scan['time'] = wband_scan.Elv
      wband_scan = wband_scan.rename({'time':'elv'})
      wband_scan = wband_scan.drop('Elv')

      _, index_range = np.unique(wband_scan['elv'], return_index=True)
      wband_scan = wband_scan.isel(elv=index_range)
      wband_scan = wband_scan.reindex({'elv':elvRef},method='nearest',tolerance=3)  
      
      #- now the Ka-Band data
      filesKabandScan = sorted(glob.glob(filePathKa+'{year}/{month}/{day}/{date}_{hour}*{fileExt}'.format(year=date.strftime('%Y'),
                                                                              month=date.strftime('%m'),
                                                                              day=date.strftime('%d'),
                                                                              date = date.strftime('%Y%m%d'),
                                                                              hour = "{0:0=2d}".format(hour),
                                                                              fileExt=fileExtKa)))
  
      if filesKabandScan:
        print(HourMinW)
        fKa = []
        for fi in filesKabandScan: # get the same starting minute (Ka file might start at e.g. 14:59 instead of 15:00 minute, so we need to round here. Same as for W-Band)
          DateHourMinKaSec = fi.split(year+'/'+month+'/'+day+'/')[1].split('_')[0]+' '+fi.split(year+'/'+month+'/'+day+'/')[1].split('_')[1]
          HourMinKa = pd.to_datetime(DateHourMinKaSec).round(freq='T').strftime('%H%M')
          if HourMinKa == HourMinW:
            fKa = fi
        
        try:  
          KaData = xr.open_dataset(fKa)
          
          _, index_time = np.unique(KaData['time'], return_index=True)
          KaData = KaData.isel(time=index_time)
          KaData.time.attrs['units']='seconds since {0}'.format('1970-01-01 00:00:00 UTC')
          KaData = xr.decode_cf(KaData)
          KaData = KaData[[var,'elv']]
          KaData = KaData.rename({'elv':'Elv'})
        
          if var == 'Zg':
            KaData[var] = 10*np.log10(KaData[var])
          #KaData[var] = (KaData[var]/KaData[var]).fillna(1)
      
          KaData = KaData.reindex({'range':rangeRef},method='nearest',tolerance=rangeTolerance)    
          #wband_scan = wband_scan.reindex({'time':timeRef},method='nearest',tolerance=timeTolerance)
          KaData['time'] = KaData.Elv
          KaData = KaData.rename({'time':'elv'})
          KaData = KaData.drop('Elv')
          _, index_range = np.unique(KaData['elv'], return_index=True)
          KaData = KaData.isel(elv=index_range)
      
          #- now get the correct Ka-Band angle to go together with the W-Band angle (since we have e.g. 30° C1, 31° C2, 32° C3, all together saved in 30°)
          ChirpMin = [min(wband_scan_chirp.C1Range.values),min(wband_scan_chirp.C2Range.values),min(wband_scan_chirp.C3Range.values)]
          ChirpMax = [max(wband_scan_chirp.C1Range.values),max(wband_scan_chirp.C2Range.values),max(wband_scan_chirp.C3Range.values)]
          ielv = 0
          for ie, el in enumerate(elvRef):
            #print(el)
            #KaSel3 = KaData.sel(elv=el,method='nearest')
            for ielv in range(3):
              KaSel = KaData.sel(elv=(el+ielv),method='nearest')
              #print(KaSel.elv)
              if ielv == 0: # first chirp at beginning of timestep
                KaC1 = KaSel.where((KaSel.range >= ChirpMin[0]),drop=True)
                KaC1 = KaC1.where((KaC1.range <= ChirpMax[0]),drop=True)# and (KaData.range <= ChirpMax[0]))
                KaC1['elv'] = el
          
              elif ielv == 1: #second chirp at middle of time step
                KaC2 = KaSel.where((KaSel.range >= ChirpMin[1]),drop=True)
                KaC2 = KaC2.where((KaC2.range <= ChirpMax[1]),drop=True)# and (KaData.range <= ChirpMax[0]))
                KaC2['elv'] = el
            
              else: # third chirp at end of time step
                KaC3 = KaSel.where((KaSel.range >= ChirpMin[2]),drop=True)
                KaC3 = KaC3.where((KaC3.range <= ChirpMax[2]),drop=True)# and (KaData.range <= ChirpMax[0]))
                KaC3['elv'] = el
            
            
            
            if ie == 0:  
              KaAll = xr.concat([KaC1,KaC2,KaC3],dim='range')
            else:
              KaChirp = xr.concat([KaC1,KaC2,KaC3],dim='range')
              KaAll = xr.concat([KaAll,KaChirp],dim='elv')
            #print(KaAll)
          KaAll = KaAll.reindex({'elv':elvRef},method='nearest',tolerance=3)
                  
        except:
        #if fKa == '/data/obs/site/jue/joyrad35/2021/12/06/20211206_021520_tripex_pol_II_rhi.znc':
          print('no file found')
          KaData = xr.open_dataset(emptyDataPath)
          KaData = KaData.drop('elv')
          KaData = KaData.reindex({'time':wband_scan_chirp.time},method='nearest',tolerance='1s')
          KaData['time'] = wband_scan.elv.values#np.ones(len(KaData.time))*100*np.random_sample(len(KaData.time))
          KaAll = KaData.rename({'time':'elv'})
          KaAll[var] = KaAll[var]*np.nan
            
      #-- now onto the plotting (hopefully):
      diffvar = KaAll[var] - wband_scan[var]
      
      var2plot = (diffvar).transpose()        
      plot=ax.pcolormesh(diffvar['range']*np.cos(np.deg2rad(diffvar['elv'])), diffvar['range']*np.sin(np.deg2rad(diffvar['elv'])), 
                    var2plot, 
                    vmax = variables[var]['vmax'],
                    vmin = variables[var]['vmin'],
                    cmap='jet')    
      cb = plt.colorbar(plot,ax=ax)
      
      cb.set_label(variables[var]['name']+variables[var]['units'])
      ax.set_ylabel('range [m]')
      ax.set_xlabel('range [m]')
      ax.set_xlim(-10000,10000)
      ax.set_ylim(0,10000)
      #ax.axis('scaled')
      ax.set_title('Hour: {hour}'.format(hour="{0:0=2d}".format(hour)))
      ax.grid()
      plt.tight_layout()
      
  print(var)
  fileName = ('_').join([dateStr,variables[var]['name']+'_RHI.png'])
  filePathName = ('/').join([pathOut,fileName]) 
  plt.savefig(filePathName,dpi=200,bbox_inches='tight')   
  plt.close()
       
    
    
    
