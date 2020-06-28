import glob            # for file locations
import pprint          # for pretty printing
import re
from getpass import getpass
import numpy as np
import pyedflib
import pandas as pd
from scipy.signal import welch
import mat4py as m4p
import scipy.io
x=[]
y=[]
for i in range(0,4096):
	p=[]
	x.append(p)
for i in range(0,4096):
	p=[]
	y.append(p)
for k in range(1,101):
	#get patient data for all 100 patients available
	event='/home/vip/project/epilepsy/'+str(k)+'.edf'
	f=pyedflib.EdfReader(event)
	channel_names=f.getSignalLabels()
	channel_freq=f.getSampleFrequencies()
	 # get a list of the EEG channels
	selected_channels = channel_names
	# make an empty file of 0's
	sigbufs = np.zeros((f.getNSamples()[0],len(selected_channels)))
	# for each of the channels in the selected channels
	for i, channel in enumerate(selected_channels):
	    try:
	      # add the channel data into the array
	      sigbufs[:, i] = f.readSignal(channel_names.index(channel))
	    
	    except:
	      ValueError
	      # This happens if the sampling rate of that channel is 
	      # different to the others.
	      # For simplicity, in this case we just make it na.
	      sigbufs[:, i] = np.nan


	# turn to a pandas df and save a little space
	df = pd.DataFrame(sigbufs, columns = selected_channels)#.astype('float32')

	# get equally increasing numbers upto the length of the data depending
	# on the length of the data divided by the sampling frequency
	index_increase = np.linspace(0,len(df)/int(4096),len(df), endpoint=False)

	# round these to the lowest nearest decimal to get the seconds
	#seconds = np.floor(index_increase).astype('uint16')

	seconds = index_increase

	# make a column the timestamp
	df['Time'] = seconds

	# make the time stamp the index
	df = df.set_index('Time')

	# name the columns as channel
	df.columns.name = 'Channel'
	seiz_df, seiz_freq = df,int(4096)

	# lets remove channels not in EEG
	seiz_df = seiz_df[[channel for channel in seiz_df.columns if re.findall('EEG [A-Z]',channel)]]
	# ...and remove these ones too
	#seiz_df = seiz_df.drop(['EEG ROC-REF','EEG LOC-REF','EEG EKG1-REF'], 1)
	colu=seiz_df['EEG A1-REF'].to_list()
	s=len(colu)//int(4096)
	count=0;
	for l in range(0,len(colu),s):
		if(count==4096):
			break
		for o in range(0,20):
			x[count].append(colu[l+o])
		count=count+1
	print(len(y[0]))
	print("Processing ")
	print(k)	
#converted the first column to 
	# print(colu)
for k in range(1,101):
	#get patient data for all 90 patients available
	event='/home/vip/project/no_epilepsy/'+str(k)+'.edf'
	f=pyedflib.EdfReader(event)
	channel_names=f.getSignalLabels()
	channel_freq=f.getSampleFrequencies()
	 # get a list of the EEG channels
	selected_channels = channel_names
	# make an empty file of 0's
	sigbufs = np.zeros((f.getNSamples()[0],len(selected_channels)))
	# for each of the channels in the selected channels
	for i, channel in enumerate(selected_channels):
	    try:
	      # add the channel data into the array
	      sigbufs[:, i] = f.readSignal(channel_names.index(channel))
	    
	    except:
	      ValueError
	      # This happens if the sampling rate of that channel is 
	      # different to the others.
	      # For simplicity, in this case we just make it na.
	      sigbufs[:, i] = np.nan


	# turn to a pandas df and save a little space
	df = pd.DataFrame(sigbufs, columns = selected_channels)#.astype('float32')

	# get equally increasing numbers upto the length of the data depending
	# on the length of the data divided by the sampling frequency
	index_increase = np.linspace(0,len(df)/int(4096),len(df), endpoint=False)

	# round these to the lowest nearest decimal to get the seconds
	#seconds = np.floor(index_increase).astype('uint16')

	seconds = index_increase

	# make a column the timestamp
	df['Time'] = seconds

	# make the time stamp the index
	df = df.set_index('Time')

	# name the columns as channel
	df.columns.name = 'Channel'
	seiz_df, seiz_freq = df,int(4096)

	# lets remove channels not in EEG
	seiz_df = seiz_df[[channel for channel in seiz_df.columns if re.findall('EEG [A-Z]',channel)]]
	# ...and remove these ones too
	#seiz_df = seiz_df.drop(['EEG ROC-REF','EEG LOC-REF','EEG EKG1-REF'], 1)
	colu=seiz_df['EEG A1-REF'].to_list()
	s=len(colu)//int(4096)
	count=0;
	for l in range(0,len(colu),s):
		if(count==4096):
			break
		for o in range(0,20):
			y[count].append(colu[l+o])
		count=count+1
	print("Processing ")
	print(k)

data={'Epilepsy':x,'No_epilepsy':y}
m4p.savemat('final.mat',data)
