import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import scipy.io as sio
# descriptive statistics
import scipy as sp
import scipy.stats as stats
import pywt

db1 = pywt.Wavelet('db1')
mat_contents = sio.loadmat('final.mat')
sorted(mat_contents.keys())

Epilepsy=mat_contents['Epilepsy']
No_epilepsy=mat_contents['No_epilepsy']
temp_list = []

######################################################
#coeff = pywt.wavedec(Normal_Eyes_Open[:,1], waveletname, level)
###############################
"""Extract The Coeeficients"""
Length = 4096;    # Length of signal
Nofsignal=2000; #Number of Signal
#numrows = len(No_epilepsy)   
numrows =83 #Number of features extracted from DWT decomposition 
numcols = len(No_epilepsy[0]) 
#rms = np.sqrt(np.mean(y**2))

###############################
"""Extract The Coeeficients"""
Extracted_Features=np.ndarray(shape=(2*numcols,numrows), dtype=float, order='F')

"""NORMAL EEG SIGNAL"""  

for i in range(numcols):
    wp= pywt.WaveletPacket(No_epilepsy[:,i], db1, mode='symmetric', maxlevel=6)
    Extracted_Features[i,0]=np.mean(abs(wp['a'].data))
    Extracted_Features[i,1]=np.mean(abs(wp['aa'].data))
    Extracted_Features[i,2]=np.mean(abs(wp['aaa'].data))
    Extracted_Features[i,3]=np.mean(abs(wp['aaaa'].data))
    Extracted_Features[i,4]=np.mean(abs(wp['aaaaa'].data))
    Extracted_Features[i,5]=np.mean(abs(wp['aaaaaa'].data))
    Extracted_Features[i,6]=np.mean(abs(wp['d'].data))
    Extracted_Features[i,7]=np.mean(abs(wp['dd'].data))
    Extracted_Features[i,8]=np.mean(abs(wp['ddd'].data))
    Extracted_Features[i,9]=np.mean(abs(wp['dddd'].data))
    Extracted_Features[i,10]=np.mean(abs(wp['ddddd'].data))
    Extracted_Features[i,11]=np.mean(abs(wp['dddddd'].data))

    Extracted_Features[i,12]=np.std(wp['a'].data)
    Extracted_Features[i,13]=np.std(wp['aa'].data)
    Extracted_Features[i,14]=np.std(wp['aaa'].data)
    Extracted_Features[i,15]=np.std(wp['aaaa'].data)
    Extracted_Features[i,16]=np.std(wp['aaaaa'].data)
    Extracted_Features[i,17]=np.std(wp['aaaaaa'].data)
    Extracted_Features[i,18]=np.std(wp['d'].data)
    Extracted_Features[i,19]=np.std(wp['dd'].data)
    Extracted_Features[i,20]=np.std(wp['ddd'].data)
    Extracted_Features[i,21]=np.std(wp['dddd'].data)
    Extracted_Features[i,22]=np.std(wp['ddddd'].data)
    Extracted_Features[i,23]=np.std(wp['dddddd'].data)

    Extracted_Features[i,24]=np.median(wp['a'].data)
    Extracted_Features[i,25]=np.median(wp['aa'].data)
    Extracted_Features[i,26]=np.median(wp['aaa'].data)
    Extracted_Features[i,27]=np.median(wp['aaaa'].data)
    Extracted_Features[i,28]=np.median(wp['aaaaa'].data)
    Extracted_Features[i,29]=np.median(wp['aaaaaa'].data)
    Extracted_Features[i,30]=np.median(wp['d'].data)
    Extracted_Features[i,31]=np.median(wp['dd'].data)
    Extracted_Features[i,32]=np.median(wp['ddd'].data)
    Extracted_Features[i,33]=np.median(wp['dddd'].data)
    Extracted_Features[i,34]=np.median(wp['ddddd'].data)
    Extracted_Features[i,35]=np.median(wp['dddddd'].data)
    
    Extracted_Features[i,36]=stats.skew(wp['a'].data)
    Extracted_Features[i,37]=stats.skew(wp['aa'].data)
    Extracted_Features[i,38]=stats.skew(wp['aaa'].data)
    Extracted_Features[i,39]=stats.skew(wp['aaaa'].data)
    Extracted_Features[i,40]=stats.skew(wp['aaaaa'].data)
    Extracted_Features[i,41]=stats.skew(wp['aaaaaa'].data)
    Extracted_Features[i,42]=stats.skew(wp['d'].data)
    Extracted_Features[i,43]=stats.skew(wp['dd'].data)
    Extracted_Features[i,44]=stats.skew(wp['ddd'].data)
    Extracted_Features[i,45]=stats.skew(wp['dddd'].data)
    Extracted_Features[i,46]=stats.skew(wp['ddddd'].data)
    Extracted_Features[i,47]=stats.skew(wp['dddddd'].data)
    
    Extracted_Features[i,48]=stats.kurtosis(wp['a'].data)
    Extracted_Features[i,49]=stats.kurtosis(wp['aa'].data)
    Extracted_Features[i,50]=stats.kurtosis(wp['aaa'].data)
    Extracted_Features[i,51]=stats.kurtosis(wp['aaaa'].data)
    Extracted_Features[i,52]=stats.kurtosis(wp['aaaaa'].data)
    Extracted_Features[i,53]=stats.kurtosis(wp['aaaaaa'].data)
    Extracted_Features[i,54]=stats.kurtosis(wp['d'].data)
    Extracted_Features[i,55]=stats.kurtosis(wp['dd'].data)
    Extracted_Features[i,56]=stats.kurtosis(wp['ddd'].data)
    Extracted_Features[i,57]=stats.kurtosis(wp['dddd'].data)
    Extracted_Features[i,58]=stats.kurtosis(wp['ddddd'].data)
    Extracted_Features[i,59]=stats.kurtosis(wp['dddddd'].data)
    
    Extracted_Features[i,60]=np.sqrt(np.mean(wp['a'].data**2))   #RMS Value
    Extracted_Features[i,61]=np.sqrt(np.mean(wp['aa'].data**2))
    Extracted_Features[i,62]=np.sqrt(np.mean(wp['aaa'].data**2))
    Extracted_Features[i,63]=np.sqrt(np.mean(wp['aaaa'].data**2))
    Extracted_Features[i,64]=np.sqrt(np.mean(wp['aaaaa'].data**2))
    Extracted_Features[i,65]=np.sqrt(np.mean(wp['aaaaaa'].data**2))
    Extracted_Features[i,66]=np.sqrt(np.mean(wp['d'].data**2))
    Extracted_Features[i,67]=np.sqrt(np.mean(wp['dd'].data**2))
    Extracted_Features[i,68]=np.sqrt(np.mean(wp['ddd'].data**2))
    Extracted_Features[i,69]=np.sqrt(np.mean(wp['dddd'].data**2))
    Extracted_Features[i,70]=np.sqrt(np.mean(wp['ddddd'].data**2))
    Extracted_Features[i,71]=np.sqrt(np.mean(wp['dddddd'].data**2))
    
    Extracted_Features[i,72]=np.mean(abs(wp['a'].data))/np.mean(abs(wp['aa'].data))
    Extracted_Features[i,73]=np.mean(abs(wp['aa'].data))/np.mean(abs(wp['aaa'].data))
    Extracted_Features[i,74]=np.mean(abs(wp['aaa'].data))/np.mean(abs(wp['aaaa'].data))
    Extracted_Features[i,75]=np.mean(abs(wp['aaaa'].data))/np.mean(abs(wp['aaaaa'].data))
    Extracted_Features[i,76]=np.mean(abs(wp['aaaaa'].data))/np.mean(abs(wp['aaaaaa'].data))
    Extracted_Features[i,77]=np.mean(abs(wp['aaaaaa'].data))/np.mean(abs(wp['d'].data))
    Extracted_Features[i,78]=np.mean(abs(wp['d'].data))/np.mean(abs(wp['dd'].data))
    Extracted_Features[i,79]=np.mean(abs(wp['dd'].data))/np.mean(abs(wp['ddd'].data))
    Extracted_Features[i,80]=np.mean(abs(wp['ddd'].data))/np.mean(abs(wp['dddd'].data))
    Extracted_Features[i,81]=np.mean(abs(wp['dddd'].data))/np.mean(abs(wp['ddddd'].data))
    Extracted_Features[i,82]=np.mean(abs(wp['ddddd'].data))/np.mean(abs(wp['dddddd'].data))
    
    temp_list.append("No_epilepsy")
    
"""INTERICTAL EEG SIGNAL"""  

for i in range(numcols, 2*numcols):
    wp= pywt.WaveletPacket(Epilepsy[:,i-numcols], db1, mode='symmetric', maxlevel=6)
    Extracted_Features[i,0]=np.mean(abs(wp['a'].data))
    Extracted_Features[i,1]=np.mean(abs(wp['aa'].data))
    Extracted_Features[i,2]=np.mean(abs(wp['aaa'].data))
    Extracted_Features[i,3]=np.mean(abs(wp['aaaa'].data))
    Extracted_Features[i,4]=np.mean(abs(wp['aaaaa'].data))
    Extracted_Features[i,5]=np.mean(abs(wp['aaaaaa'].data))
    Extracted_Features[i,6]=np.mean(abs(wp['d'].data))
    Extracted_Features[i,7]=np.mean(abs(wp['dd'].data))
    Extracted_Features[i,8]=np.mean(abs(wp['ddd'].data))
    Extracted_Features[i,9]=np.mean(abs(wp['dddd'].data))
    Extracted_Features[i,10]=np.mean(abs(wp['ddddd'].data))
    Extracted_Features[i,11]=np.mean(abs(wp['dddddd'].data))

    Extracted_Features[i,12]=np.std(wp['a'].data)
    Extracted_Features[i,13]=np.std(wp['aa'].data)
    Extracted_Features[i,14]=np.std(wp['aaa'].data)
    Extracted_Features[i,15]=np.std(wp['aaaa'].data)
    Extracted_Features[i,16]=np.std(wp['aaaaa'].data)
    Extracted_Features[i,17]=np.std(wp['aaaaaa'].data)
    Extracted_Features[i,18]=np.std(wp['d'].data)
    Extracted_Features[i,19]=np.std(wp['dd'].data)
    Extracted_Features[i,20]=np.std(wp['ddd'].data)
    Extracted_Features[i,21]=np.std(wp['dddd'].data)
    Extracted_Features[i,22]=np.std(wp['ddddd'].data)
    Extracted_Features[i,23]=np.std(wp['dddddd'].data)

    Extracted_Features[i,24]=np.median(wp['a'].data)
    Extracted_Features[i,25]=np.median(wp['aa'].data)
    Extracted_Features[i,26]=np.median(wp['aaa'].data)
    Extracted_Features[i,27]=np.median(wp['aaaa'].data)
    Extracted_Features[i,28]=np.median(wp['aaaaa'].data)
    Extracted_Features[i,29]=np.median(wp['aaaaaa'].data)
    Extracted_Features[i,30]=np.median(wp['d'].data)
    Extracted_Features[i,31]=np.median(wp['dd'].data)
    Extracted_Features[i,32]=np.median(wp['ddd'].data)
    Extracted_Features[i,33]=np.median(wp['dddd'].data)
    Extracted_Features[i,34]=np.median(wp['ddddd'].data)
    Extracted_Features[i,35]=np.median(wp['dddddd'].data)
    
    Extracted_Features[i,36]=stats.skew(wp['a'].data)
    Extracted_Features[i,37]=stats.skew(wp['aa'].data)
    Extracted_Features[i,38]=stats.skew(wp['aaa'].data)
    Extracted_Features[i,39]=stats.skew(wp['aaaa'].data)
    Extracted_Features[i,40]=stats.skew(wp['aaaaa'].data)
    Extracted_Features[i,41]=stats.skew(wp['aaaaaa'].data)
    Extracted_Features[i,42]=stats.skew(wp['d'].data)
    Extracted_Features[i,43]=stats.skew(wp['dd'].data)
    Extracted_Features[i,44]=stats.skew(wp['ddd'].data)
    Extracted_Features[i,45]=stats.skew(wp['dddd'].data)
    Extracted_Features[i,46]=stats.skew(wp['ddddd'].data)
    Extracted_Features[i,47]=stats.skew(wp['dddddd'].data)
    
    Extracted_Features[i,48]=stats.kurtosis(wp['a'].data)
    Extracted_Features[i,49]=stats.kurtosis(wp['aa'].data)
    Extracted_Features[i,50]=stats.kurtosis(wp['aaa'].data)
    Extracted_Features[i,51]=stats.kurtosis(wp['aaaa'].data)
    Extracted_Features[i,52]=stats.kurtosis(wp['aaaaa'].data)
    Extracted_Features[i,53]=stats.kurtosis(wp['aaaaaa'].data)
    Extracted_Features[i,54]=stats.kurtosis(wp['d'].data)
    Extracted_Features[i,55]=stats.kurtosis(wp['dd'].data)
    Extracted_Features[i,56]=stats.kurtosis(wp['ddd'].data)
    Extracted_Features[i,57]=stats.kurtosis(wp['dddd'].data)
    Extracted_Features[i,58]=stats.kurtosis(wp['ddddd'].data)
    Extracted_Features[i,59]=stats.kurtosis(wp['dddddd'].data)
    
    Extracted_Features[i,60]=np.sqrt(np.mean(wp['a'].data**2))   #RMS Value
    Extracted_Features[i,61]=np.sqrt(np.mean(wp['aa'].data**2))
    Extracted_Features[i,62]=np.sqrt(np.mean(wp['aaa'].data**2))
    Extracted_Features[i,63]=np.sqrt(np.mean(wp['aaaa'].data**2))
    Extracted_Features[i,64]=np.sqrt(np.mean(wp['aaaaa'].data**2))
    Extracted_Features[i,65]=np.sqrt(np.mean(wp['aaaaaa'].data**2))
    Extracted_Features[i,66]=np.sqrt(np.mean(wp['d'].data**2))
    Extracted_Features[i,67]=np.sqrt(np.mean(wp['dd'].data**2))
    Extracted_Features[i,68]=np.sqrt(np.mean(wp['ddd'].data**2))
    Extracted_Features[i,69]=np.sqrt(np.mean(wp['dddd'].data**2))
    Extracted_Features[i,70]=np.sqrt(np.mean(wp['ddddd'].data**2))
    Extracted_Features[i,71]=np.sqrt(np.mean(wp['dddddd'].data**2))
    
    Extracted_Features[i,72]=np.mean(abs(wp['a'].data))/np.mean(abs(wp['aa'].data))
    Extracted_Features[i,73]=np.mean(abs(wp['aa'].data))/np.mean(abs(wp['aaa'].data))
    Extracted_Features[i,74]=np.mean(abs(wp['aaa'].data))/np.mean(abs(wp['aaaa'].data))
    Extracted_Features[i,75]=np.mean(abs(wp['aaaa'].data))/np.mean(abs(wp['aaaaa'].data))
    Extracted_Features[i,76]=np.mean(abs(wp['aaaaa'].data))/np.mean(abs(wp['aaaaaa'].data))
    Extracted_Features[i,77]=np.mean(abs(wp['aaaaaa'].data))/np.mean(abs(wp['d'].data))
    Extracted_Features[i,78]=np.mean(abs(wp['d'].data))/np.mean(abs(wp['dd'].data))
    Extracted_Features[i,79]=np.mean(abs(wp['dd'].data))/np.mean(abs(wp['ddd'].data))
    Extracted_Features[i,80]=np.mean(abs(wp['ddd'].data))/np.mean(abs(wp['dddd'].data))
    Extracted_Features[i,81]=np.mean(abs(wp['dddd'].data))/np.mean(abs(wp['ddddd'].data))
    Extracted_Features[i,82]=np.mean(abs(wp['ddddd'].data))/np.mean(abs(wp['dddddd'].data))
    
    temp_list.append("Epilepsy")

"""ICTAL EEG SIGNAL"""  
    
"""CLASSIFICATION"""
X = Extracted_Features
y = temp_list

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import metrics
from io import BytesIO #needed for plot
import seaborn as sns; sns.set()
# =============================================================================
# Define Utility Functions
# =============================================================================
def plot_model_accuracy_loss():
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
    plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
    plt.plot(history.history['loss'], 'r--', label='Loss of training data')
    plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
    plt.title('Model Accuracy and Loss')
    plt.ylabel('Accuracy and Loss')
    plt.xlabel('Training Epoch')
    plt.ylim(0)
    plt.legend()
    plt.show()

def print_confusion_matrix():
    matrix = confusion_matrix(y_test.argmax(axis=1), max_y_pred_test.argmax(axis=1))
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,cmap='coolwarm',linecolor='white',linewidths=1,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def print_performance_metrics():
    print('Accuracy:', np.round(metrics.accuracy_score(y_test, max_y_pred_test),4))
    print('Precision:', np.round(metrics.precision_score(y_test, 
                                max_y_pred_test,average='weighted'),4))
    print('Recall:', np.round(metrics.recall_score(y_test, max_y_pred_test,
                                               average='weighted'),4))
    print('F1 Score:', np.round(metrics.f1_score(y_test, max_y_pred_test,
                                               average='weighted'),4))
    print('Cohen Kappa Score:', np.round(metrics.cohen_kappa_score(y_test.argmax(axis=1), max_y_pred_test.argmax(axis=1)),4))
    print('Matthews Corrcoef:', np.round(metrics.matthews_corrcoef(y_test.argmax(axis=1), max_y_pred_test.argmax(axis=1)),4)) 
    print('\t\tClassification Report:\n', metrics.classification_report(y_test, max_y_pred_test))

def print_confusion_matrix_and_save():
    mat = confusion_matrix(y_test.argmax(axis=1), max_y_pred_test.argmax(axis=1))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    plt.savefig("Confusion.jpg")
    # Save SVG in a fake file object.
    f = BytesIO()
    plt.savefig(f, format="svg")
#%%



#%%
# =============================================================================
# KERAS Deep Lerning Model
# =============================================================================
# load data
from sklearn.model_selection import train_test_split
# split data into train and test sets
from keras.utils import np_utils
from sklearn import preprocessing
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import MaxPooling1D
lb = preprocessing.LabelBinarizer() #Binarize labels
y=lb.fit_transform(temp_list)
X = Extracted_Features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=7)
# One hot encode targets
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
InputDataDimension=83
print(X_train.shape)
print(y_train.shape)
X_train=np.reshape(X_train,(2680,1,83))
y_train=np.reshape(y_train,(2680,1,2))
X_test=np.reshape(X_test,(1320,1,83))
y_test=np.reshape(y_test,(1320,1,2))
#%%
# =============================================================================
# Build a Deep model
# =============================================================================
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=(1,83)))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=1))
model.add(Conv1D(filters=128, kernel_size=1, activation='relu'))
model.add(MaxPooling1D(pool_size=1))


model.add(Dense(83,  activation='relu'))
model.add(Dense(32,  activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
#%%
print(model.summary())
# =============================================================================
#  Compile the model
# =============================================================================
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# =============================================================================
# Train and Validate the model
# =============================================================================
history = model.fit(X_train, y_train,validation_split=0.33, epochs=50, batch_size=25,verbose=1)
#%%
# =============================================================================
# Evaluate the Model
# =============================================================================
test_loss, test_acc = model.evaluate(X_test, y_test,verbose=0)
print('\nTest accuracy:', test_acc) 
# #%%
# # =============================================================================
# # Plot the History
# # =============================================================================
# #Plot the Model Accuracy and Loss for Training and Validation dataset
plot_model_accuracy_loss()

# #%%
# #Test the Model with testing data
y_pred_test = model.predict(X_test)
print(y_pred_test.shape)
y_pred_test=np.reshape(y_pred_test,(1320,2))
print(y_pred_test)
# Round the test predictions
for i in np.nditer(y_pred_test):
  if i<0 or i>1:
    print("kat gaya")
max_y_pred_test = np.round(y_pred_test)
print(max_y_pred_test)
y_test=np.reshape(y_test,(1320,2))
#Print the Confusion Matrix
print_confusion_matrix()
#%%
#Evaluate the Model and Print Performance Metrics
print_performance_metrics()
#%%
# #Print and Save the Confusion Matrix
print_confusion_matrix_and_save()
