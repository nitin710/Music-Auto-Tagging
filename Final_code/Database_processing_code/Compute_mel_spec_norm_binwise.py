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


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='log_file_combined_dict_norm_binwise.log',
                    filemode='w')


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
#     logging.info(fft_freq_indices,'len=',fft_freq_indices.shape)
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


# /Users/nitin/Documents/Music Info Retrieval/project/database/magnatagatune/data_from_trey
f = open('/scratch/nn1174/MIR/data_from_trey/annotations_final.txt', 'r')

reader = csv.reader(f, delimiter='\t')
tags = next(reader)
annotation_dict = {}

while True:
    try:
        values = next(reader)
        # values1 = next(reader, None)
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

logging.info('done with combining all dictionaries')

with open('sorted_tags.pickle', 'rb') as handle:
    sorted_stats = pickle.load(handle)

datapath = '/scratch/nn1174/MIR/mp3_all'
start_time = time.time()
for i,key in enumerate(combined_dict):
    if key=='35644' or key=='55753' or key=='57881':
        combined_dict[key]['mel_spectrum'] = np.zeros((40,909))
        combined_dict[key]['output'] = np.zeros((50))
    else:
        songpath = os.path.join(datapath,combined_dict[key]['mp3_path'])
        spec,fs_spec = compute_mfcc(songpath,param.win_size,param.hop_size,param.min_freq,
                                param.max_freq,param.num_mel_filts,param.n_dct)

        # define normalization parameters for each filter bank in each spectrogram
        bin_mean = np.mean(spec, axis = 1).reshape(40,1)
        bin_stdev = np.std(spec, axis = 1).reshape(40,1)
        norm_melco = (spec - bin_mean)/bin_stdev        
        combined_dict[key]['mel_spectrum'] = (norm_melco,fs_spec)
        output=[]
        for j,tag in enumerate(sorted_stats):
            if j>49:
                break
            else:
                output.append(int(combined_dict[key][tag[0]]))
        output = np.array(output)
        combined_dict[key]['output'] = output
        if i%100==0:
            logging.info(i)
            done_time = time.time()
            # logging.info('Exe.time={}'.format(done_time-start_time))

done_time = time.time()
# logging.info('Exe.time={}'.format(done_time-start_time))
with open('combined_dict_norm_binwise.pickle', 'wb') as handle:
    pickle.dump(combined_dict, handle)
