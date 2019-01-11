from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
import os
import cv2
from datetime import datetime
import sys
import random

# Heighth and width of frames
RES = int(sys.argv[1])
sample_size = int(sys.argv[3])
batch_size = 3
frame_names = [f for f in os.listdir('frames') if f[:3] == 'vid']
total_num_frames = len(frame_names)
# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

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
                   padding='same', return_sequences=False))
seq.add(BatchNormalization())

seq.add(Conv2D(filters=1, kernel_size=(3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))

seq.compile(loss='binary_crossentropy', optimizer='adadelta')

seed = None
seed_num = random.randint(0, len(frame_names) - 10)

def frame_data_generator(batch_size, sample_size, start_on_frame=0):

    is_beginning_of_new_epoch = True
    while True:
        if is_beginning_of_new_epoch:
            batch_start = start_on_frame + sample_size + 1
            batch_end = batch_size + batch_start

        # each time through this loop a batch is yielded until end of data
        limit = min(batch_end, total_num_frames)
        this_batch_size = limit - batch_start
        relevant_frames = []
        for frame in frame_names[batch_start - (sample_size + 1):limit]:
            relevant_frames.append(cv2.imread('frames/' + frame)[:, :, :1] / 255.0)
        #relevant_frames = [f for f in relevant_frames if f.shape == (RES, RES, 1)]
        x = np.ndarray((0, sample_size, RES, RES, 1))
        y = np.ndarray((0, RES, RES, 1))

        for i in range(this_batch_size):
            x = np.concatenate((x, np.stack(relevant_frames[i:i + sample_size])[np.newaxis, :]))
            y = np.concatenate((y, relevant_frames[i + sample_size][np.newaxis, :]))

        assert(np.array_equal(x[1, -1],y[0]))
        yield (x, y)

        batch_start += batch_size
        batch_end += batch_size
        is_beginning_of_new_epoch = batch_start >= total_num_frames


# Train the network
seq.fit_generator(frame_data_generator(batch_size=batch_size, sample_size=sample_size),
                  steps_per_epoch=int(total_num_frames / batch_size),
                  epochs=int(sys.argv[2]))

seed = next(frame_data_generator(batch_size=batch_size,
                                    sample_size=sample_size,
                                    start_on_frame=random.randint(0,
                                    len(frame_names) - 20)))[0][:1,:,:,:,:]
# seed = next(frame_data_generator(batch_size=batch_size,
#                                     sample_size=sample_size,
#                                     start_on_frame=2613)
for f in range(10):
    filename = "generated/%s_ _frame_%d_ _res_%s_ _epochs_%s_ _ss_%s.jpg" % (datetime.now().strftime('%Y.%m.%d %H.%M'), f, sys.argv[1], sys.argv[2], sys.argv[3])
    cv2.imwrite(filename, (seed[0, 0, :, :, :] * 255))
    new_frame = seq.predict(seed)
    seed = np.concatenate((seed[:, 1:, :, :, :],
                           new_frame[np.newaxis, :]),
                          axis=1)
    # print ("old seed chopped shape", seed[:1, 1:, :, :, :].shape)
    # print ("new prediction shape", seq.predict(seed[:1, 1:, :, :, :]).shape)
