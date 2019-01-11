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
# Num of epochs
EPOCHS = int(sys.argv[2])
# Sample size - number of frames given to the network at a time
SS = int(sys.argv[3])
# Number of frames to generate
GEN = int(sys.argv[4])
# Number of serie of frames given to the network at a time
BATCH_SIZE = 3
# Paths of frame images relative to vids/
FRAME_NAMES = [f for f in os.listdir('frames') if f[:3] == 'vid']
TOTAL_NUM_FRAMES = len(FRAME_NAMES)

# Network takes in videos of shape (n_frames, width, height, channels)
# and returns a prediction of the next frame in each video

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


def frame_data_generator(start_on_frame=0):

    is_beginning_of_new_epoch = True
    while True:
        if is_beginning_of_new_epoch:
            batch_start = start_on_frame + SS + 1
            batch_end = BATCH_SIZE + batch_start

        # each time through this loop a batch is yielded until end of data
        limit = min(batch_end, TOTAL_NUM_FRAMES)
        this_batch_size = limit - batch_start
        relevant_frames = []
        for frame in FRAME_NAMES[batch_start - (SS + 1):limit]:
            relevant_frames.append(
                cv2.imread('frames/' + frame)[:, :, :1] / 255.0
            )

        x = np.ndarray((0, SS, RES, RES, 1))
        y = np.ndarray((0, RES, RES, 1))

        for i in range(this_batch_size):
            x = np.concatenate((
                x,
                np.stack(relevant_frames[i:i + SS])[np.newaxis, :]
            ))
            y = np.concatenate((
                y,
                relevant_frames[i + SS][np.newaxis, :]
            ))

        assert(np.array_equal(x[1, -1], y[0]))
        yield (x, y)

        batch_start += BATCH_SIZE
        batch_end += BATCH_SIZE
        is_beginning_of_new_epoch = batch_start >= TOTAL_NUM_FRAMES


# Train the network
seq.fit_generator(
        frame_data_generator(),
        steps_per_epoch=int(TOTAL_NUM_FRAMES / BATCH_SIZE),
        epochs=EPOCHS
)

# Generate and save new frames

seed = next(frame_data_generator(
    start_on_frame=random.randint(0, TOTAL_NUM_FRAMES - 20)))[0][:1, :]

for f in range(GEN):
    filename = "generated/%s_frame%d_res%s_epochs%s_ss%s.jpg" % \
        (datetime.now().strftime('%Y.%m.%d %H.%M'), f, RES, EPOCHS, SS)
    cv2.imwrite(filename, (seed[0, -1, :, :, :] * 255))
    new_frame = seq.predict(seed)
    seed = np.concatenate((seed[:, 1:, :, :, :],
                           new_frame[np.newaxis, :]),
                          axis=1)
