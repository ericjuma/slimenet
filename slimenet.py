from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import os
import cv2

# Heighth and width of frames
RES = 10

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(RES, RES, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))

#opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

seq.compile(loss='binary_crossentropy', optimizer='adadelta')

def frame_data_generator(batch_size):
    frame_names = filter(lambda f: f[:3] == 'vid' and f[-4:] == '.jpg',
                         os.listdir('frames'))

    while True:
        batch_start = 0
        batch_end = batch_size
        total_num_frames = len(frame_names)
        while batch_start < total_num_frames - 1:
            limit = min(batch_end, total_num_frames - 1)
            relevant_frames = map(lambda f: cv2.imread('frames/' + f),
                                  frame_names[batch_start:limit + 1])

            x = np.stack(relevant_frames[:-1])
            y = np.stack(relevant_frames[1:])

            yield (x, y)

            batch_start += batch_size
            batch_end += batch_size


# Train the network
seq.fit_generator(frame_data_generator(batch_size=30),
                  steps_per_epoch=100,
                  epochs=1)
