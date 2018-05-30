from __future__ import division
import pandas as pd

import tensorflow as tf
import numpy as np


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        padded_value = tf.pad(value, [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]])
#         if dilation > 1:
#             transformed = time_to_batch(value, dilation)
#             conv = tf.nn.conv1d(transformed, filter_, stride=1,
#                                 padding='SAME')
#             restored = batch_to_time(conv, dilation)
#         else:
#             restored = tf.nn.conv1d(padded_value, filter_, stride=1, padding='VALID')
            
        result = tf.nn.convolution(padded_value,
                                     filter_,
                                     padding="VALID",
                                     strides=[1],
                                     dilation_rate=[dilation]
                                    )
        # Remove excess elements at the end.
#         out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
#         result = tf.slice(restored,
#                           [0, 0, 0],
#                           [-1, out_width, -1])
        return result

def inject_noise(audio, quantization_channels):
    noise = tf.random_normal(tf.shape(audio),
                             mean=0.0,
                             stddev=tf.sqrt(1.0/float(quantization_channels)),
                             dtype=tf.float32
                            )
    return audio + noise

def mu_law_encode(audio, quantization_channels, noise=False):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        
        if noise:
            signal = inject_noise(signal, quantization_channels)
        
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude


# Upsample algos
def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.arange(size)
    return (1 - abs(og - center) / factor)

def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return weights

def upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """

    filter_size = get_kernel_size(factor)

    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes),
                       dtype=np.float32)

    upsample_kernel = upsample_filt(filter_size)

    for i in xrange(number_of_classes):
        weights[:, :, i, i] = upsample_kernel

    return weights


# def upsample_tconv(factor, input_time_series):
#     '''
#     input_time_series
#     axis 0: time steps, width
#     axis 1: one local condition description, in_channels
#     '''

#     number_of_classes = input_time_series.shape[1]

#     new_length = input_time_series.shape[0] * factor

#     expanded_time_series = input_time_series[np.newaxis, np.newaxis, :, :]

#     with tf.Graph().as_default():
#         with tf.Session() as sess:
#             with tf.device("/cpu:0"):

#                 upsample_filt_pl = tf.placeholder(tf.float32)
#                 logits_pl = tf.placeholder(tf.float32)

#                 upsample_filter_np = upsample_weights(factor,
#                                         number_of_classes)

#                 res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
#                         output_shape=[1, 1, new_length, number_of_classes],
#                         strides=[1, 1, factor, 1])

#                 final_result = sess.run(res,
#                                         feed_dict={upsample_filt_pl:
#                                                       upsample_filter_np,
#                                                    logits_pl:
#                                                       expanded_time_series})

#    return final_result.squeeze(axis=0).squeeze(axis=0)



def upsample_tf(factor, input_img):
    
    number_of_classes = input_img.shape[2]
    
    new_height = input_img.shape[0] * factor
    new_width = input_img.shape[1] * factor
    
    expanded_img = np.expand_dims(input_img, axis=0)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device("/cpu:0"):

                upsample_filt_pl = tf.placeholder(tf.float32)
                logits_pl = tf.placeholder(tf.float32)

                upsample_filter_np = bilinear_upsample_weights(factor,
                                        number_of_classes)

                res = tf.nn.conv2d_transpose(logits_pl, upsample_filt_pl,
                        output_shape=[1, new_height, new_width, number_of_classes],
                        strides=[1, factor, factor, 1])

                final_result = sess.run(res,
                                feed_dict={upsample_filt_pl: upsample_filter_np,
                                           logits_pl: expanded_img})
    
    return final_result.squeeze()

def upsample_fill(factor, input_time_series):
    return np.repeat(input_time_series, factor, axis=0)

def upsample_center_fill(factor, input_time_series):
    '''
    Assume input_time_series[0] is in alignment with signal at index 0
    '''
    upsampled = np.repeat(input_time_series, factor, axis=0)
    return upsampled[int(factor/2): -int(factor/2), :]
