"""
takes all the vids and splits them into evenly sized square frames
saved as jpegs in the frames folder
"""

import os
import cv2
import sys

# resolution that will be the height and width of each frame
RES = int(sys.argv[1])

# goes through each vid in vids
for vid in os.listdir('vids')[1:]:
    if vid[-4:] != '.mp4':
        continue
    # load video
    vidcap = cv2.VideoCapture('vids/' + vid)
    # load first frame
    success, image = vidcap.read()
    # get aspect ratio of video
    aspect_ratio = image.shape[1] / image.shape[0]
    # set new width and height that the video will be scaled to fit in
    if aspect_ratio > 1:
        new_frame_width, new_frame_height = int(aspect_ratio * RES), RES
    else:
        new_frame_width, new_frame_height = RES, int(aspect_ratio * RES)

    # goes thru each frame in the vid, scales and saves it
    count = 0
    while success:
        # proportionally scale image so one dimension is desired resolution
        # and the other dimension is larger or equal to that resolution
        image = cv2.resize(image, (new_frame_width, new_frame_height))
        # compute the amount to crop off from each side and crop it
        height_crop, width_crop = [int((n - RES) / 2) for n in image.shape[:2]]
        image = image[height_crop:image.shape[0] - height_crop,
                      width_crop:image.shape[1] - width_crop]
        # shave off any extra pixel on either side from rounding errors
        image = image[0:RES, 0:RES]
        # write frame to jpg
        cv2.imwrite("frames/vid_%s_frame%d.jpg" % (vid, count), image)
        # read next frame
        success, image = vidcap.read()
        count += 1

    print("%d frames written from %s" % (count, vid))
