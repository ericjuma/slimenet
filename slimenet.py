from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import os
import cv2

# Heighth and width of frames
RES = 240
frame_names = [f for f in os.listdir('frames') if f[:3] == 'vid']
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()
seq.add(ConvLSTM2D(filters=10, kernel_size=(3, 3),
                   input_shape=(None, RES, RES, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=10, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=10, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=10, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))

seq.compile(loss='binary_crossentropy', optimizer='adadelta')

def frame_data_generator(batch_size, sample_size):

    buffer = sample_size + 1
    while True:
        batch_start = buffer
        batch_end = batch_size + buffer
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
seq.fit_generator(frame_data_generator(batch_size=100, sample_size=7),
                  steps_per_epoch=20,
                  epochs=1)

_, seed = next(frame_data_generator(batch_size=10, sample_size=7))


for f in range(100):
    cv2.imwrite("generated/generated_frame%d.jpg" % (f), (seed[0, 0, :, :, :] * 255))
    seed = seq.predict(seed)
