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
import ops
import json

# For training only
FILE_PATTERN = r'p([0-9]+)_([0-9]+)\.wav'
TRAINING_RANGE = 300

def get_category_cardinality(files):
    id_reg_exp = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_exp.findall(filename)[0]
        id, recording_id = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))
        yield files[file_index]


def find_files(directory, pattern="*.wav"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def align_local_condition(local_condition, length):
    # TODO
    factor = int(length / local_condition.shape[0])
    upsampled_lc = ops.upsample_fill(factor, local_condition)
    diff = length - upsampled_lc.shape[0]
    upsampled_lc = np.pad(upsampled_lc, [[diff, 0], [0, 0]],
                          'constant')
    return upsampled_lc


def load_generic_audio(directory, sample_rate, lc_maps):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
            if int(ids[0][1]) >= TRAINING_RANGE:
                continue
        # print(filename)
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        if lc_maps:
            lc_filename = os.path.realpath(os.path.join(directory, lc_maps[filename.replace(directory, "")]))
            lc = pd.read_csv(lc_filename, sep=',', header=None).values
            # TODO: upsampling to make lc same number of rows as audio
            lc = align_local_condition(lc, audio.shape[0])
        else:
            lc = None
        yield audio, filename, category_id, lc

def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return ((audio[indices[0]:indices[-1]], indices)
            if indices.size else (audio[0:0], None))


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)
        if not ids:
            return True
    return False


def not_all_have_lc(directory, files, lc_maps):
    ''' Return true iff any of the wave files isn't accompanied
        by csv file specifying local conditions.
    '''
    for file in files:
        lc_filename = os.path.realpath(os.path.join(directory, lc_maps[file.replace(directory, "")]))
        if not os.path.isfile(lc_filename):
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32,
                 lc_maps_json=None):
        self.audio_dir = os.path.abspath(audio_dir)
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.queue = tf.PaddingFIFOQueue(queue_size,
                                         ['float32'],
                                         shapes=[(None, 1)])
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'],
                                                shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])

        self.lc_maps = None
        if lc_maps_json is not None:
            try:
                with open(lc_maps_json, "r") as inputfile:
                    self.lc_maps = json.load(inputfile)
            except Exception as e:
                print(e)
                raise ValueError("Local conditioning is enabled, but json file for " 
                                 "local condition mapping is not input correctly.")
            self.lc_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
            self.lc_queue = tf.PaddingFIFOQueue(queue_size, ['float32'],
                                                shapes=[(None, None)])
            self.lc_enqueue = self.lc_queue.enqueue([self.lc_placeholder])

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(self.audio_dir)
        if not files:
            raise ValueError("No audio files found in '{}'.".format(self.audio_dir))
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names "
                             "do not conform to pattern having id.")
        # Determine the number of mutually-exclusive categories we will
        # accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1
            print("Detected --gc_cardinality={}".format(
                  self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

        # Check local conditions
        if self.lc_maps is not None:
            if not_all_have_lc(self.audio_dir, files, self.lc_maps):
                raise ValueError('''Local condition is enabled,
                    but not all wave files have local conditions.''')

    def dequeue(self, num_elements):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue_gc(self, num_elements):
        return self.gc_queue.dequeue_many(num_elements)

    def dequeue_lc(self, num_elements):
        return self.lc_queue.dequeue_many(num_elements)

    def thread_main(self, sess):
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate,
                                          self.lc_maps)
            for audio, filename, category_id, lc in iterator:
                # print filename
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio, keep_indices = trim_silence(audio[:, 0],
                                                       self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))
                    else:
                        if self.lc_maps is not None:
                            lc = lc[keep_indices[0]:keep_indices[-1], :]

                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],
                               'constant')
                
                if self.lc_maps is not None:
                    lc = np.pad(lc, [[self.receptive_field, 0], [0, 0]],
                               'constant')
                    # lc_arr = np.asarray(lc)
                    # np.savetxt(filename+"_processed.csv", lc_arr,delimiter=",")

                    assert(lc.shape[0] == audio.shape[0])

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field +
                    # sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        piece = audio[:(self.receptive_field +
                                        self.sample_size), :]
                        if self.lc_maps is not None:
                            lc_piece = lc[:(self.receptive_field + 
					    self.sample_size), :]
                        sess.run(self.enqueue,
                                 feed_dict={self.sample_placeholder: piece})
                        audio = audio[self.sample_size:, :]
                        if self.lc_maps is not None:
                            sess.run(self.lc_enqueue,
                                     feed_dict={self.lc_placeholder: lc_piece})
                            lc = lc[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.gc_enqueue, feed_dict={
                                self.id_placeholder: category_id})
                else:
                    sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: audio})
                    if self.gc_enabled:
                        sess.run(self.gc_enqueue,
                                 feed_dict={self.id_placeholder: category_id})
                    if self.lc_maps is not None:
                        sess.run(self.lc_enqueue,
                                 feed_dict={self.lc_placeholder: lc})

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

    def output_audio(self, path, wav):
        librosa.output.write_wav(path, wav, self.sample_rate)
