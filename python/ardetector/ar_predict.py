# ar_predict.py
# Andreas Brink-Kjaer
# Spring 2018
#
# Based on scripts by Caspar Aleksander Bang Jespersen
#
''' 
ar_predict iterates over all data in directory specified by flag 'pathname' using ar_reader.py and ar_config.py for data loading and 
uses a trained model specified by flags 'model' and 'ckpt' to predict arousal and wake. The model predictions are saved in 'output_dir'. 
The 'overwrite' flag can be set to 1 to overwrite predictions in output_dir.
'''

import os
import numpy as np
import tensorflow as tf
import tempfile

import ar_network
import ar_config
import single_reader as ar_reader

import matplotlib.pyplot as plt
from struct import Struct
import pyedflib
from datetime import timezone
from datetime import datetime
#import pdb

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pathname', './name.edf', 'Files to execute.')
flags.DEFINE_string('model', 'resnet', 'Which model to use')
flags.DEFINE_integer('ckpt', '350000', 'Which checkpoint to use (blank for none)')#70000
flags.DEFINE_string('output_dir','./','Directory for predictions')
flags.DEFINE_integer('overwrite',0,'Overwrite previous predictions (1 = Overwrite)')
flags.DEFINE_string('edf_file', 'name.edf', 'name of the edf-file to execute')

flags.DEFINE_float('Tar',0.225, 'Threshold for arrousal detection')
flags.DEFINE_float('Tw',0.45, 'Threshold for wake prediction')
flags.DEFINE_boolean('rm3', True, 'shall arousals shorter than 3 s get removed?')

def main(argv=None, filename = 'name'):
    '''The main function sets configurations and runs predict that computes and writes model predictions'''

    # Settings
    config = ar_config.ARConfig(num_hidden = 128, lr = 0.001, kp = 1.0, batch_size = 5*60, resnet_size = 20, model_name='resnet', is_training=False)
    
    _, _ = predict(config,FLAGS.ckpt,filename)

    edf_start = pyedflib.EdfReader(str(filename + '.edf')).getStartdatetime().replace(tzinfo=timezone.utc).timestamp()
    post_process(filename, edf_start)


def predict(config,ckpt,filename):
    '''predict uses the config and ckpt to create and load a network graph. The model reads all files in FLAGS.pathname and makes predictions.

    Args: 
        config: configurations for network and data set.
        ckpt: checkpoint number for model weights to load.
        file: Directory for data.
    '''

    # Load data and adjust batch size in CPU
    with tf.device('/cpu:0'):
        #data = ar_reader.ArousalData(FLAGS.pathname,FLAGS.edf_file, config, num_steps=config.max_steps, overwrite = FLAGS.overwrite, output_dir = FLAGS.output_dir)
        data = ar_reader.ArousalData(str(filename + '.edf'), config)

    # Creates network graph
    sess_config = tf.ConfigProto(log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Graph().as_default(), tf.Session(config=sess_config) as session:
        with tf.variable_scope('model', reuse=False) as scope:
            m = ar_network.ARModel(config)
            glob_vars = tf.global_variables()
            model_vars = [k for k in glob_vars if k.name.startswith(scope.name)]
            s = tf.train.Saver(model_vars, max_to_keep=None)

        # Loads model weights
        s.restore(session, config.checkpoint_file(ckpt))

        current_batch = 0
        for batch_input, batch_target, batch_mask in data:
            current_batch += 1
            operations = [m.softmax]
            params = {
                m.features: batch_input,
                m.targets: batch_target,
                m.mask: batch_mask,
                m.batch_size: data.batch_size,
                m.step: data.iter_steps
            }
            softmax = session.run(operations, feed_dict=params)
            # Get ar probability
            softmax = softmax[0]
            ar = np.array([x[1] for m, x in enumerate(softmax)])
            # Get W probability
            wake = np.array([x[3] for m, x in enumerate(softmax)])
            # Reshape and save
            if current_batch == 1:
                output_ar = ar
                output_w = wake
            else:
                output_ar = np.append(output_ar,ar,0)
                output_w = np.append(output_w,wake,0)
            if current_batch == data.num_batches:
                # Save output
                current_batch = 0
                #filename = data.filename
                #file = filename.split('\\')[-1]
                if data.batch_shift == 0:
                    output_file = str(filename + '._1')
                else:
                    output_file = str(filename + '._2')
                pred_file = open(output_file,'w')
                np.savetxt(pred_file,(output_ar,output_w),delimiter=',',fmt='%.2f')
                pred_file.close()

        return output_ar, output_w

def post_process(filename, edf_start):

    def ar_pp(ar, Tar=0.255, remove3=True):
        ar = np.where(ar > Tar, 1, 0)
        L = np.ndarray.tolist(np.where(ar==1)[0])
        for i in L[:-1]:
            if (i + 10) > L[L.index(i)+1] > (i + 1):
                for x in range(i, L[L.index(i)+1], 1):
                    ar[x] = 1
        if remove3:
            L = np.ndarray.tolist(np.where(ar==1)[0])
            for i in L:
                if (i-1) not in L:
                    j = i
                    while j in L:
                        j += 1
                    if (j-i) < 3:
                        for x in range(i,j,1):
                            ar[x] = 0
        return ar

    def w_pp(w, Tw = 0.45):
        w = np.where(w > Tw, 1, 0)
        L = np.ndarray.tolist(np.where(w==1)[0])
        for i in L[:-1]:
            if (i + 15) > L[L.index(i)+1] > (i + 1):
                for x in range(i, L[L.index(i)+1], 1):
                    w[x] = 1
        L = np.ndarray.tolist(np.where(w==1)[0])
        for i in L:
            if (i-1) not in L:
                j = i
                while j in L:
                    j += 1
                if (j-i) < 15:
                    for x in range(i,j,1):
                        w[x] = 0
        return w

    def write_anl(out_file, ar, starttime_seconds):
        times = np.where(ar==1)[0].tolist()
        t_start = []
        for t in times:
            if (t-1) not in times:
                t_start.append(t)
        t_end = []
        for t in times:
            if (t+1) not in times:
                t_end.append(t)
        width_non_ar = 0x00000000
        width_ar = 0x00000001 * 1000 * 1000 * 60 * 60 * 24 * 30
        color_non_ar = 0x000000FF
        color_ar = 0x00FF0000
        start_timestamp_us = (starttime_seconds + (25569 * 86400)) * 1000 * 1000
        out_file.write(b'000000CB\r\n')
        serializer = Struct('<qqIiiB')
        timestamp_us = int(start_timestamp_us)
        out_file.write(serializer.pack(timestamp_us, width_non_ar, color_non_ar, 0, 0, 0))
        for t in range(0,len(t_start),1):
            out_file.write(serializer.pack((timestamp_us + int(t_start[t] * 1000 * 1000)), width_ar, color_ar, 0, 10, 0))
            out_file.write(serializer.pack((timestamp_us + int(t_end[t] * 1000 * 1000)), width_non_ar, color_non_ar, 0, 0, 0))
    
    remove3 = FLAGS.rm3
    Tar = FLAGS.Tar
    Tw = FLAGS.Tw
    
    data_1 = str(filename + '._1')
    data_2 = str(filename + '._2')

    [ar, w] = np.split(np.loadtxt(data_1, delimiter =','), 2)
    [ar2, w2] = np.split(np.loadtxt(data_2, delimiter =','), 2)
    ar, w, ar2, w2 = ar.flatten(), w.flatten(), ar2.flatten(), w2.flatten()

    print('shapes are: ar ' + str(ar.shape) + ' w ' + str(w.shape) + ' ar2 ' + str(ar2.shape) + ' w2 ' + str(w2.shape))
    length_data = ar.shape[0]
    diff = length_data - ar2.shape[0]
    nb = length_data/diff

    ar2 = np.concatenate((np.zeros(int(diff/2)),ar2,np.zeros(int(diff/2))), axis = None)
    w2 = np.concatenate((np.zeros(int(diff/2)),w2,np.zeros(int(diff/2))), axis = None)

    ar_temp = ar
    w_temp = w

    idx2 = []
    for i in range(1,int(nb),1):
        for j in range(-int(diff/4),int(diff/4),1):
            idx2.append(i+j)

    idx2 = list(set(idx2))
    ar[idx2] = ar2[idx2]
    w[idx2] = w2[idx2]

    ar2 = ar_temp
    w2 = w_temp

    ar = ar_pp(ar, Tar, remove3)
    w = w_pp(w, Tw)

    with open(str(filename + '.arousal.anl'),"wb") as out_file:
        write_anl(out_file, ar, edf_start)

#    counter = 0
#    while counter < 4:        
#        f_name = str(filename + ['ar', 'w', 'ar2', 'w2'][counter])
#        document = open(f_name,'w')
#        np.savetxt(document, [ar, w, ar2, w2][counter], delimiter=',')
#        document.close()
#        counter += 1

    fig = plt.figure(figsize=(24,4))
    ax = fig.add_subplot(111)
    ax.plot(w, 'bs', w2, 'b-',ar,'r^', ar2, 'r-')
    plt.ylabel('probability')
    plt.xlabel('seconds')
    plt.savefig(str(filename + '.plot.png'))



if __name__ == '__main__':
    tf.app.run()
