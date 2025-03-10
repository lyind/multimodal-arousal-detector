# modelled after:
# ar_reader.py
# Andreas Brink-Kjaer
# Spring 2018
#
# Based on scripts by Caspar Aleksander Bang Jespersen
#
#This altered Version is an iterator over a single file and also contains code to avoid using MATLAB

import numpy as np
import tensorflow as tf
import os
import random
import ar_config
import collections
import ar_weights

import skimage_extract as skimage
import pyedflib
import scipy.io as sio
import scipy.signal as signal
import threading
import multiprocessing
import concurrent.futures
import re
from ekg_sub import loop
import pdb

Dataset = collections.namedtuple('Dataset', ['data', 'target', 'weights'])

'''the first class is leaning to the data loading from the Narcolepsie detector/sleep stage code and contains everything to read data from edf and prepsocessing
'''
class data_extraction(object):
    

    def __init__(self, path):
        self.loaded_channels = {}
        
        self.edf_pathname = path

        self.cpu_max = 4
        

    def extract(self):
        
        self.loadEDF()
        self.psg_noise_level()
        self.filtering()
        self.filter_ekg()
        result = self.shaping()
        return result

    def loadEDF(self):

        loadSignal_lock = threading.Lock()
        def load_channel(target_label, original_label, source_edf, edf_labels):
            i = edf_labels.index(original_label)
            ch_signal = None
            with loadSignal_lock:
                ch_signal = source_edf.readSignal(i)
            dimension = source_edf.getPhysicalDimension(i).lower()
            if dimension == 'mv':
                ch_signal *= 1e3
            elif dimension == 'v':
                ch_signal *= 1e6
            original_fs = int(source_edf.samplefrequency(i))
            print(target_label + ': original samplerate: ' + str(original_fs));
            if original_fs == 128:
                print(target_label + 'keep 128Hz')
                resampled_ch = ch_signal
            else:
                print(target_label + ': resampling: 128Hz');
                resampled_ch = signal.resample_poly(ch_signal, 128, original_fs, axis=0, window=('kaiser', 5.0))
            print(target_label + ': Resampling done')
            return resampled_ch


        def load_and_reference_channel(channel_state, target_label):
            label = channel_state['label']
            res_ch = load_channel(target_label, label, edf, Labels)
            ref = channel_state['ref_label']
            if ref and not channel_state['isReferenced']:
                if ref in self.loaded_channels:
                    res_ref = self.loaded_channels[ref]
                else:
                    res_ref = self.loaded_channels[ref] = load_channel(ref, ref, edf, Labels)
                
                print(target_label + ': referencing ' + label + ' with ' + ref)
                self.loaded_channels[target_label] = np.subtract(res_ch, res_ref)
            else:
                self.loaded_channels[target_label] = res_ch
        
        with pyedflib.EdfReader(str(self.edf_pathname)) as edf:
            Labels = edf.getSignalLabels()
            Channels = {'C3': {'expr': ".*C3.*", 'ref': ['A2', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'C4': {'expr': ".*C4.*", 'ref': ['A1', 'M1'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'EOG-L': {'expr': "EOG.?([1Ll]|[eE][1lL])", 'ref': ['A1', 'A2', 'M1', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'EOG-R': {'expr': "EOG.?([2Rr]|[eE][2rR])", 'ref': ['A1', 'A2', 'M1', 'M2'], 'label': None, 'isReferenced': False, 'ref_label': None },
                        'EKG': {'expr': ".*[Ee]([kK]|[Cc])[gG].*",'ref':[], 'label':None, 'isReferenced': False, 'ref_label': None}
                        }
                    
            def find_reference_label(ref_candidates):
                for ref_label in Labels:
                    for r in ref_candidates:
                        if r in ref_label:
                            return ref_label
                return None
                
            def label_is_reference(ref_candidates, label):
                for r in ref_candidates:
                    if r in label:
                        return True
                return False

        #finding channels and check if referenced
            for label in Labels:
                for ch in Channels:
                    channel_state = Channels[ch]
                    ident = re.search(channel_state['expr'], label)
                    if ident and not channel_state['isReferenced']:
                        print(ch + ': identified ' + label + ' as: ' + ch)
                        channel_state['label'] = label
                        
                        channel_state['isReferenced'] = label_is_reference(channel_state['ref'], label)
                        if channel_state['isReferenced']:
                            print(ch + ': is already referenced')
                        else:
                            print(ch + ': appears unreferenced')
                            channel_state['ref_label'] = find_reference_label(channel_state['ref'])
                            if channel_state['ref_label']:
                                print(ch + ': identified ' + channel_state['ref_label'] + ' as suitable reference')
                            else:
                                print(ch + ': No reference found')
    
                emg_ident = re.search(".*[cC][hH][iI][nN].*", label)
                if emg_ident:
                    print('EMG: found chin-EMG as: ' + label)
                    Channels['EMG'] = {'expr': "some string to prevent an ERROR", 'label':label, 'ref_label': None}
            
            if 'EMG' not in Channels:
                for label in Labels:
                    ident = re.search(".*[eE][mM][gG].*", label)
                    if ident:
                        print('EMG: found chin-EMG as: ' + label)
                        Channels['EMG'] = {'label':label, 'ref_label': None}
                        break
            
        #extracting data from edf, resampling and auto-referencing
            cpu_set = min(self.cpu_max ,max(1, (multiprocessing.cpu_count()-1)))
            
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=cpu_set) as executor:
                for ch in Channels:
                    channel_state = Channels[ch]
                    if channel_state['label']:
                        future = executor.submit(load_and_reference_channel, channel_state, ch)
                        futures.append(future)
            
            for future in futures:
                future.result()


    def psg_noise_level(self):
        if all(['C4' in self.loaded_channels, 'C3' in self.loaded_channels]):
            noiseM = sio.loadmat('./ml/noiseM.mat', squeeze_me=True)['noiseM']
            meanV = noiseM['meanV'].item()
            covM = noiseM['covM'].item()
            centrals_idx = 0
            unused_ch = self.get_loudest_channel(['C3','C4'],meanV[centrals_idx], covM[centrals_idx])
            del self.loaded_channels[unused_ch]
            
    def get_loudest_channel(self, channelTags, meanV, covM):
        noise = np.zeros(len(channelTags))
        for [idx,ch] in enumerate(channelTags):
            noise[idx] = self.channel_noise_level(ch, meanV, covM)
        return channelTags[np.argmax(noise)]

    def channel_noise_level(self, channelTag, meanV, covM):

        hjorth= self.extract_hjorth(self.loaded_channels[channelTag])
        noise_vec = np.zeros(hjorth.shape[1])
        for k in range(len(noise_vec)):
            M = hjorth[:, k][:, np.newaxis]
            x = M - meanV[:, np.newaxis]
            sigma = np.linalg.inv(covM)
            noise_vec[k] = np.sqrt(np.dot(np.dot(np.transpose(x), sigma), x))
            return np.mean(noise_vec)

    def extract_hjorth(self, x, dim=5 * 60, slide=5 * 60):

        # Length of first dimension
        dim = dim * 128

        # Overlap of segments in samples
        slide = slide * 128

        # Creates 2D array of overlapping segments
        D = skimage.view_as_windows(x, dim, dim).T

        # Extract Hjorth params for each segment
        dD = np.diff(D, 1, axis=0)
        ddD = np.diff(dD, 1, axis=0)
        mD2 = np.mean(D ** 2, axis=0)
        mdD2 = np.mean(dD ** 2, axis=0)
        mddD2 = np.mean(ddD ** 2, axis=0)

        top = np.sqrt(np.divide(mddD2, mdD2))

        mobility = np.sqrt(np.divide(mdD2, mD2))
        activity = mD2
        complexity = np.divide(top, mobility)

        hjorth = np.array([activity, complexity, mobility])
        hjorth = np.log(hjorth + np.finfo(float).eps)
        return hjorth
        #3
        ''' The filter rigth now is s potential thread to performance, since it isn´t exactly similar to the Matlab filter!
        '''
    def filtering(self):
        print('filtering signals')
        #highpass-Filters
        Fh_oe = signal.butter(5, 0.5/(128/2), btype='highpass', output='ba')
        Fh_m = signal.butter(5, 10/(128/2), btype='highpass', output='ba')
        #lowpass-Filter
        Fl = signal.butter(5, 35/(128/2), btype='lowpass', output='ba')

        for ch in self.loaded_channels:
                print('Filtering {}'.format(ch))
                if ch == 'EMG':
                    Fh = Fh_m
                else:
                    Fh = Fh_oe
                self.loaded_channels[ch] = signal.filtfilt(Fh[0], Fh[1], self.loaded_channels[ch])

    

    def filter_ekg(self):
        
        if 'EKG' in self.loaded_channels:
            ecg = self.loaded_channels['EKG']
            for entry in self.loaded_channels:
                if entry != 'EKG':
                    self.loaded_channels[entry] = loop(self.loaded_channels[entry], ecg)
            del self.loaded_channels['EKG']
     
    def shaping(self): 
        for entry in self.loaded_channels:
            if self.loaded_channels[entry].shape[0]%128 > 0:
                self.loaded_channels[entry] = np.delete(self.loaded_channels[entry], range((self.loaded_channels[entry].shape[0]//128)*128,((self.loaded_channels[entry].shape[0]//128)*128)+128,1))
            self.loaded_channels[entry] = np.reshape(self.loaded_channels[entry], (int(self.loaded_channels[entry].shape[0]/128),128))
        
        if 'C3' in self.loaded_channels:
            combine = (self.loaded_channels['C3'],self.loaded_channels['EOG-R'],self.loaded_channels['EOG-L'],self.loaded_channels['EMG'],np.zeros((self.loaded_channels['C3'].shape[0],2)))
        else:
            combine = (self.loaded_channels['C4'],self.loaded_channels['EOG-R'],self.loaded_channels['EOG-L'],self.loaded_channels['EMG'],np.zeros((self.loaded_channels['C4'].shape[0],2)))
        
        result = np.concatenate(combine, axis=1)
        return result


'''The second part is the iterator, that is calling up the edf and passes it to the main module
'''

class ArousalData:
    def __init__(self, pathname, config):

        self.pathname = pathname #location/name of edf
        self.datex = data_extraction(self.pathname)
        self.features = []
        self.logits = []
        self.weights = []
        self.config = config
        self.num_batches = 0
        self.batch_shift = 0
        self.wake_def = config.wake_def
        self.batch_size = config.batch_size
        self.iter_batch = -1
        self.iter_steps = -1
        self.batch_order = np.arange(1)

        self.load()

    def __iter__(self):
        '''Iteration function'''
        return self

    def __next__(self):
        '''This function is used to iterate the class to get next batches.'''
        # Increment counters
        # Determine stopping criteria
        if (self.iter_batch + 1) == len(self.batch_order) and self.batch_shift == 1:
            raise StopIteration()
        if (self.iter_batch + 1) == len(self.batch_order):
            load_not_ok = True
            self.num_batches = 0
            while load_not_ok:

                self.batch_shift = 1

                try:
                    self.load()                         
                except:
                    print('Error loading')

                if self.num_batches>0:
                    load_not_ok = False

                    
        self.iter_batch += 1
        self.iter_steps += 1

        # Return relevant batch

        x, y, w = self.get_batch(self.iter_batch)
        return x, y, w

    def extraction(self):
        array = self.datex.extract()
        return array

    def load_EDF(self):
        '''loading the EDF file'''

        f = self.extraction() #returns a np.array containing all the information
        excess = f.shape[0] % self.batch_size
        if excess > 0:
            f = f[:-excess]
        if self.batch_shift == 1:
            f = f[int(self.batch_size/2):-int(self.batch_size/2)]
        data = f[:,:-2]
        target = f[:,-2:]
        # Select definition of wake = {W} or wake = {W,N1}
        target[target[:,-1] == 2] = self.wake_def
        self.seq_in_file = data.shape[0]
        # Get weights
        approach = 1
        w_ar = ar_weights.train_ar_weights(target[:,0],approach)
        w_w = ar_weights.train_ar_weights(target[:,1],1)
        w = np.transpose(np.vstack((w_ar,w_w)))

        return Dataset(data=data, target=target, weights=w)

    def rewind(self):

        # Reset iter
        self.iter_batch = -1
        # Regular if not training

        self.batch_order = np.arange(self.num_batches)

    def get_batch(self, batch_num):
        '''This function chooses and selects the next batch of data.

        Args:
            batch_num: next batch number.
        '''

        # Find indices
        batch_num_ordered = self.batch_order[batch_num]
        ind = [batch_num_ordered*self.batch_size, (batch_num_ordered+1)*self.batch_size]
        #print('inds are')
        #print(ind[0])
        #print(ind[1])
        # Find batches
        x = self.features[ind[0]:ind[1], :, :]
        t = self.logits[ind[0]:ind[1],:]
        w = self.weights[ind[0]:ind[1],:]
        return x, t, w

    def load(self):
        '''Loads the next data file with load_EDF and standerdizes and reshapes input data.'''

        # Load data
        data_set = self.load_EDF()
        # Features
        self.features = data_set.data
        # Standerdization of data to (mean, std) = (0, 1)
        # Printout
        print('mean: ',np.mean(self.features[:,:128]),np.mean(self.features[:,128:256]), np.mean(self.features[:,256:384]), np.mean(self.features[:,384:]))
        print('std: ',np.std(self.features[:,:128]),np.std(self.features[:,128:256]), np.std(self.features[:,256:384]), np.std(self.features[:,384:]))
        self.features[:,:128] = (self.features[:,:128] - np.mean(self.features[:,:128]))/np.std(self.features[:,:128])
        self.features[:,128:256] = (self.features[:,128:256] - np.mean(self.features[:,128:256]))/np.std(self.features[:,128:256])
        self.features[:,256:384] = (self.features[:,256:384] - np.mean(self.features[:,256:384]))/np.std(self.features[:,256:384])
        self.features[:,384:] = (self.features[:,384:] - np.mean(self.features[:,384:]))/np.std(self.features[:,384:])
        # Labels to logits
        self.logits = np.concatenate((np.eye(2)[data_set.target[:,0].astype('int')], np.eye(2)[data_set.target[:,1].astype('int')]),axis=1)
        self.weights = data_set.weights
        # Reshape data (batches, features, 1)
        self.features = np.expand_dims(self.features,2)
        self.num_batches = self.logits.shape[0] // self.batch_size
        self.rewind()























