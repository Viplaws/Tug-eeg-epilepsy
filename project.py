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
# colours for printing outputs
class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

pp = pprint.PrettyPrinter()
x=[]
y=[]
for k in range(1,42):
	#get patient data for all 90 patients available
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
	print("Processing ")
	print(k)
	colu=seiz_df['EEG FP1-REF'].to_list()
	x.append(colu)
#converted the first column to 
	# print(colu)
for k in range(1,42):
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
	print("Processing ")
	print(k)
	# lets remove channels not in EEG
	seiz_df = seiz_df[[channel for channel in seiz_df.columns if re.findall('EEG [A-Z]',channel)]]
	# ...and remove these ones too
	#seiz_df = seiz_df.drop(['EEG ROC-REF','EEG LOC-REF','EEG EKG1-REF'], 1)
	colu=seiz_df['EEG FP1-REF'].to_list()
	y.append(colu)
#converted the first column to 
	# print(colu)
data={'Epilesy':x,'No_epilepsy':y}
m4p.savemat('final.mat',data)
