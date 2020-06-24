import glob            # for file locations
import pprint          # for pretty printing
import re
from getpass import getpass
import numpy as np
import pyedflib
import pandas as pd
from scipy.signal import welch

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
ls=[]
#for i in range(1,2):
	#get patient data for all 90 patients available
event='/home/vip/project/epilepsy/1.edf'
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
seiz_df = seiz_df.drop(['EEG ROC-REF','EEG LOC-REF','EEG EKG1-REF'], 1)
colu=seiz_df['EEG FP1-REF'].to_numpy()
#converted the first column to 
#print(colu)
from pywt import wavedec
level = 6

# transpose the data because its a time-series package
# get the wavelet coefficients at each level in a list
coeffs_list = wavedec(colu, wavelet='db4', level=level)

# make a list of the component names (later column rows)
nums = list(range(1,level+1))
names=[]
for num in nums:
    names.append('D' + str(num))
names.append('A' + str(nums[-1]))

# reverse the names so it counts down
names = names[::-1] 

print(names)
# make an empty dataframe
wavelets = pd.DataFrame()

for i, array in enumerate(coeffs_list):
    # turn into a dataframe and transpose
    level_df = pd.DataFrame(array)
    # name the column the appropriate level name
    level_df.columns = [names[i]]
    # if the wavelets df is empty...
    if wavelets.empty:
        # ... make the first level df the wavelets df
        wavelets = level_df
    # ..otherwise...
    else:
        # ... add the next levels df to another column
        wavelets = pd.concat([wavelets,level_df], axis=1)
regex = re.compile('D')
bad_items = [x for x in list(wavelets.columns) if not regex.match(x)]
decom_wavelets = wavelets.drop(bad_items, axis=1)

print(decom_wavelets.head())
def minus_small(data):    
  # find the smallest value for each data column (channel)...
  min_val = data.min()
  # ...and subtract it from all the data in the column and add one
  data = data.subtract(min_val).add(1)

  return data

def log_sum(data, output=False):
    absolute_sums = data.sum()
    # ...and subtract it from all the data in the column and add one
    absolute_sums_minus = minus_small(absolute_sums)
    # find the log of each elecment (datapoint)
    absolute_sums_log = absolute_sums_minus.apply(np.log)
    absolute_sums_log.index += '_LSWT'
    
    if output:
        display(absolute_sums_log)
    
    return absolute_sums_log
def ave(data, output=False):
    # get the mean
    mean_data = data.mean()
    
    mean_data.index += '_mean'
   
    return mean_data
def mean_abs(data, output=False):
    # get the mean of the absolute values
    mean_abs_data = data.abs().mean()
    
    mean_abs_data.index += '_mean_abs'
    return mean_abs_data

def coeff_std(data, output=False):
    # get the standard deviation of the coeficients
    std_data = data.std()
    
    std_data.index += '_std'
    
    if output:
        display(std_data)
    
    return std_data

example=ave(decom_wavelets,output=True)
print(example)