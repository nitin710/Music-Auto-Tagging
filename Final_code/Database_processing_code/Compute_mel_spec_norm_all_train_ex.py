import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct
from scipy.signal import spectrogram
import operator
import pickle
import time
import csv
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
import logging

################################################# LOG FILE CONFIGURATION ##################################
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='log_file_combined_dict_norm_all_examples.log',
                    filemode='w')

################################################# MEL PARAMETERS DEFINITION ##################################
class parameters:
    def __init__(self):
        # specify file path when executing 
        self.win_size = 1024
        self.hop_size = 512
        self.min_freq = 80
        self.max_freq = 4000
        self.num_mel_filts = 40
        self.n_dct = 13
param = parameters()
logging.info(param.__dict__)

##################################### FUNCTIONS FOR MEL SPECTRUM CALCULATION ##################################
# converts frequency in Hz to Mel values
# pass a numpy array to the function
def hz2mel(hzval):
    melval = 1127.01028*np.log(1+hzval/700)
    return melval
# funtion tested with example

# converts Mel values to Hz
# pass a numpy array to the function
def mel2hz(melval):
    hzval = 700*(np.exp(melval/1127.01028)-1)
    return hzval
# function tested with example

# f_fft will be the input
# rounding to the values in freq_list
def find_nearest(values,freq_list):
    q_freq_ind=[]
    for value in values.tolist():
        ind = np.argmin(np.abs(value-freq_list))
        q_freq_ind.append(ind)
    return np.asarray(q_freq_ind)

def compute_mfcc(filepath,win_size,hop_size,min_freq,max_freq,num_mel_filts,n_dct):
    melval = hz2mel(np.array([min_freq,max_freq]))
    min_mel = melval[0]
    max_mel = melval[1]
    step = (max_mel-min_mel)/(num_mel_filts-1)
    mel_freq_list = np.linspace(min_mel,max_mel,num_mel_filts)
    mel_freq_list = np.concatenate(([mel_freq_list[0]-step],mel_freq_list,[mel_freq_list[-1]+step]))
    hz_freq_list = mel2hz(mel_freq_list)
    nfft = win_size # number of ft points for the spectrogram
    # make sure librosa is imported
    x,Fs = librosa.load(filepath,sr=16000)
    f,t,Sxx = spectrogram(x,Fs,nperseg=win_size,noverlap=win_size-hop_size,nfft=nfft)
    Sxx = np.square(np.abs(Sxx))
    # the spectrogram has to be plotted flipped up-dpwn to make the lower freq show at the bottom
    fft_freq_indices = find_nearest(hz_freq_list,f)# approximate the fft freq list to the nearest value in the hz freq list got by converting mel scale
#     print(fft_freq_indices,'len=',fft_freq_indices.shape)
    filt_bank = np.zeros((1,int(nfft/2) + 1))
    for i in range(1,fft_freq_indices.shape[0]-1):# from sec ele to sec last ele
        a = fft_freq_indices[i-1]
        b = fft_freq_indices[i]
        c = fft_freq_indices[i+1]
        t1 = (1/(b-a))*np.linspace(a-a,b-a,b-a+1)
        t2 = (-1/(c-b))*np.linspace(b-c,c-c,c-b+1)
        filt = np.concatenate((t1,t2[1:]))
        filt = filt/(np.sum(filt))
        filt_zero_pad = np.zeros((1,int(nfft/2)+1))
        filt_zero_pad[0,a:c+1] = filt
        filt_bank = np.concatenate((filt_bank,filt_zero_pad),axis=0)
    filt_bank = filt_bank[1:,:]
    mel_spec = np.dot(filt_bank,Sxx)
    mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec) # for numerical stability
    mel_spec = 20*np.log10(mel_spec)
    fs_mfcc = mel_spec.shape[1]
    return mel_spec,fs_mfcc # returning the mel_spectrum


############################# CREATING COMBINED DICT BY JOINING ANNOTATIONS+CLIP_INFO ##########################

# /Users/nitin/Documents/Music Info Retrieval/project/database/magnatagatune/data_from_trey
f = open('/scratch/nn1174/MIR/data_from_trey/annotations_final.txt', 'r')

reader = csv.reader(f, delimiter='\t')
tags = next(reader)
annotation_dict = {}

while True:
    try:
        values = next(reader)
        annotation_dict[values[0]] = {}# data is a dict. values[0] is the clip id, which is the key->pointing to a dict of all tags
        for tagnames, value in zip(tags[1:], values[1:]):
            annotation_dict[values[0]][tagnames] = value
    except StopIteration:
        logging.info('end tag annotations file')
        break


ff = open('/scratch/nn1174/MIR/data_from_trey/clip_info_final.txt', 'r')

rreader = csv.reader(ff, delimiter='\t')
metadata = next(rreader)
clip_inf_dict = {}

while True:
    try:
        values = next(rreader)
        # values1 = next(reader, None)
        clip_inf_dict[values[0]] = {}
        for metdat, val in zip(metadata[1:], values[1:]):
            clip_inf_dict[values[0]][metdat] = val
    except StopIteration:
        logging.info('end clip info file')
        break


combined_dict = {}

for key in annotation_dict.keys():  # you can list as many input dicts as you want here
    combined_dict[key] = annotation_dict[key].copy()
    combined_dict[key].update(clip_inf_dict[key])

# IMPORTANT DECLARATION DEFINING VARIABLE "KEYS"
keys = list(combined_dict.keys())
logging.info('done combining the dictionaries')
logging.info(len(combined_dict.keys()))
logging.info(len(combined_dict['2'].keys()))


################################ LOADING ALL PICKLE FILES NEEDED FOR INDEXING ##################################
with open('train_ind.pickle','rb') as handle:
    train_ind = pickle.load(handle)
with open('val_ind.pickle','rb') as handle:
    val_ind = pickle.load(handle)
with open('test_ind.pickle','rb') as handle:
    test_ind = pickle.load(handle)
### loading sorted tags
with open('sorted_tags.pickle', 'rb') as handle:
    sorted_stats = pickle.load(handle)
    
################################## CALCULATING THE NORMALIZATION COEFFICIENTS ##################################
start_time = time.time()
spec_mat_train = np.zeros((len(train_ind),40,909))
datapath = '/scratch/nn1174/MIR/mp3_all'
logging.info('starting to create spec_mat_trin to generate the normalizing COEFFICIENTS')
for i,ind in enumerate(train_ind):
    if keys[ind]=='35644' or keys[ind]=='55753' or keys[ind]=='57881':
        spec_mat_train[i,:,:] = np.zeros((40,909))
    else:
        songpath = os.path.join(datapath,combined_dict[keys[ind]]['mp3_path'])
        spec,fs_spec = compute_mfcc(songpath,param.win_size,param.hop_size,param.min_freq,
                                param.max_freq,param.num_mel_filts,param.n_dct)
        spec_mat_train[i,:,:] = spec
    if i%20==0:
        logging.info(i)

###### normalizing parameters
mn = np.mean(spec_mat_train,axis=0)
stdev = np.std(spec_mat_train,axis=0)
norm_coeff = [mn,stdev]
with open('norm_coeff.pickle','wb') as handle:
    pickle.dump(norm_coeff,handle)
######
logging.info('got the mean and std')

########################## ADDING MEL SPECTRUM AND OUTPUT FIELDS IN DICTIONARY ##################################
logging.info('appending spectrum+output to validation set')
for i,ind in enumerate(val_ind):
    if keys[ind]=='35644' or keys[ind]=='55753' or keys[ind]=='57881':
        combined_dict[keys[ind]]['mel_spectrum'] = np.zeros((40,909))
        combined_dict[keys[ind]]['output'] = np.zeros((50))
    else:
        songpath = os.path.join(datapath,combined_dict[keys[ind]]['mp3_path'])
        spec,fs_spec = compute_mfcc(songpath,param.win_size,param.hop_size,param.min_freq,
                                param.max_freq,param.num_mel_filts,param.n_dct)
        spec = (spec-mn)/stdev # normalize it
        combined_dict[keys[ind]]['mel_spectrum'] = (spec,fs_spec)
        output=[]
        for j,tag in enumerate(sorted_stats):
            if j>49:
                break
            else:
                output.append(int(combined_dict[keys[ind]][tag[0]]))
        output = np.array(output)
        combined_dict[keys[ind]]['output'] = output
    if i%20==0:
        logging.info(i)
        
logging.info('appending spectrum+output to test set')
for i,ind in enumerate(test_ind):
    if keys[ind]=='35644' or keys[ind]=='55753' or keys[ind]=='57881':
        combined_dict[keys[ind]]['mel_spectrum'] = np.zeros((40,909))
        combined_dict[keys[ind]]['output'] = np.zeros((50))
    else:
        songpath = os.path.join(datapath,combined_dict[keys[ind]]['mp3_path'])
        spec,fs_spec = compute_mfcc(songpath,param.win_size,param.hop_size,param.min_freq,
                                param.max_freq,param.num_mel_filts,param.n_dct)
        spec = (spec-mn)/stdev # normalize it
        combined_dict[keys[ind]]['mel_spectrum'] = (spec,fs_spec)
        output=[]
        for j,tag in enumerate(sorted_stats):
            if j>49:
                break
            else:
                output.append(int(combined_dict[keys[ind]][tag[0]]))
        output = np.array(output)
        combined_dict[keys[ind]]['output'] = output
    if i%20 == 0:
        logging.info(i)
logging.info('appending spectrum+output to train set')
for i,ind in enumerate(train_ind):
    if keys[ind]=='35644' or keys[ind]=='55753' or keys[ind]=='57881':
        combined_dict[keys[ind]]['mel_spectrum'] = spec_mat_train[i,:,:]
        combined_dict[keys[ind]]['output'] = np.zeros((50))
    else:
        spec = spec_mat_train[i,:,:] # using already calculated spectrograms
        spec = (spec-mn)/stdev # normalize it
        combined_dict[keys[ind]]['mel_spectrum'] = (spec,909)# hard coded , but never used, so doesnt matter
        output=[]
        for j,tag in enumerate(sorted_stats):
            if j>49:
                break
            else:
                output.append(int(combined_dict[keys[ind]][tag[0]]))
        output = np.array(output)
        combined_dict[keys[ind]]['output'] = output        
    if i%20 == 0:
        logging.info(i)
        
logging.info('Done with creating the spec_matrices')
logging.info('done with generating the whole combined_dict')
with open('combined_dict_norm_all_examples.pickle', 'wb') as handle:
    pickle.dump(combined_dict, handle)

logging.info('Done with Everything')
