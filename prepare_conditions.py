import copy
import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf
import pandas as pd

FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'

directory = "../VCTK-Corpus"

def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def generate_mfcc(directory, sample_rate, lc_ext_name=".csv"):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    print("files length: {}".format(len(files)))
    for filename in files:
        print filename
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20) # shape = (n_mfcc, t)
        df = pd.DataFrame(mfcc.T)
        lc_filename = copy.deepcopy(filename)
        if lc_filename.endswith('.wav'):
            lc_filename = lc_filename[:-4] + lc_ext_name
        lc_filename = lc_filename.replace("wav48", "mfcc")
        if not os.path.exists(os.path.dirname(lc_filename)):
            try:
                os.makedirs(os.path.dirname(lc_filename))
            except OSError as exc: # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        df.to_csv(lc_filename, sep=',', header=None) 

if __name__=="__main__":
    generate_mfcc("../VCTK-Corpus", 48000)