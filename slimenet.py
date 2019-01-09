from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import os
import cv2
import datetime
import sys
import random

# Heighth and width of frames
RES = int(sys.argv[1])
frame_names = [f for f in os.listdir('frames') if f[:3] == 'vid']
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.
sample_size = int(sys.argv[3])

seq = Sequential()
seq.add(ConvLSTM2D(filters=RES, kernel_size=(3, 3),
                   input_shape=(None, RES, RES, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=RES, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=RES, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=RES, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))

seq.compile(loss='binary_crossentropy', optimizer='adadelta')

seed = None
seed_num = random.randint(0, len(frame_names) - 10)

def frame_data_generator(batch_size, sample_size, batch_start=0):

    buffer = sample_size + 1
    while True:
        batch_start = batch_start + buffer
        batch_end = batch_size + batch_start
        total_num_frames = len(frame_names)

        # each time through this loop a batch is yielded until end of data
        while batch_start < total_num_frames:
            limit = min(batch_end, total_num_frames)
            relevant_frames = []
            for frame in frame_names[batch_start - buffer:limit]:
                relevant_frames.append(cv2.imread('frames/' + frame)[:, :, :1] / 255.0)
            relevant_frames = [f for f in relevant_frames if f.shape == (RES, RES, 1)]
            x = []
            for i in range(batch_size):
                x.append(np.stack(
                    relevant_frames[i:i + sample_size]
                ))
            x = np.stack(x)

            y = []
            for i in range(1, batch_size + 1):
                y.append(np.stack(
                    relevant_frames[i:i + sample_size]
                ))
            y = np.stack(y)
            assert(np.array_equal(x[1],y[0]))

            yield (x, y)

            batch_start += batch_size
            batch_end += batch_size

# Train the network
seq.fit_generator(frame_data_generator(batch_size=5, sample_size=sample_size),
                  steps_per_epoch=200,
                  epochs=int(sys.argv[2]))

_, seed = next(frame_data_generator(batch_size=5,
                                    sample_size=sample_size,
                                    batch_start=random.randint(0,
                                    len(frame_names) - 20)))
# for _ in range(random.randint(0, len(frame_names) - 10)):
#     _, seed = next(frame_data_generator(batch_size=10, sample_size=7))


for f in range(12):
    cv2.imwrite("generated/gen_%s__%d___res_%s___epochs_%s___ss_%s.jpg" % (datetime.datetime.now(), f, sys.argv[1], sys.argv[2], sys.argv[3]), (seed[0, 0, :, :, :] * 255))
    seed = seq.predict(seed)
