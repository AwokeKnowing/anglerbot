#!/usr/bin/env python3
#
#  USB Camera - Simple
#
#  Copyright (C) 2021-22 JetsonHacks (info@jetsonhacks.com)
#
#  MIT License
#!/usr/bin/env python

'''
Multithreaded video processing sample.
Usage:
   video_threaded.py {<video device number>|<video file name>}

   Shows how python threading capabilities can be used
   to organize parallel captured frame processing pipeline
   for smoother playback.

Keyboard shortcuts:

   ESC - exit
   space - switch between multi and single threaded processing
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from multiprocessing.pool import ThreadPool
from collections import deque

from common import clock, draw_str, StatValue
import video


class DummyTask:
    def __init__(self, data):
        self.data = data
    def ready(self):
        return True
    def get(self):
        return self.data

def main():
    import sys

    try:
        fn = sys.argv[1]
    except:
        fn = 0
    cap = video.create_capture(fn)


    def process_frame(frame, t0):
        # some intensive computation...
        frame = cv.medianBlur(frame, 19)
        frame = cv.medianBlur(frame, 19)
        return frame, t0

    threadn = cv.getNumberOfCPUs()
    pool = ThreadPool(processes = threadn)
    pending = deque()

    threaded_mode = True

    latency = StatValue()
    frame_interval = StatValue()
    last_frame_time = clock()
    while True:
        while len(pending) > 0 and pending[0].ready():
            res, t0 = pending.popleft().get()
            latency.update(clock() - t0)
            draw_str(res, (20, 20), "threaded      :  " + str(threaded_mode))
            draw_str(res, (20, 40), "latency        :  %.1f ms" % (latency.value*1000))
            draw_str(res, (20, 60), "frame interval :  %.1f ms" % (frame_interval.value*1000))
            cv.imshow('threaded video', res)
        if len(pending) < threadn:
            _ret, frame = cap.read()
            t = clock()
            frame_interval.update(t - last_frame_time)
            last_frame_time = t
            if threaded_mode:
                task = pool.apply_async(process_frame, (frame.copy(), t))
            else:
                task = DummyTask(process_frame(frame, t))
            pending.append(task)
        ch = cv.waitKey(1)
        if ch == ord(' '):
            threaded_mode = not threaded_mode
        if ch == 27:
            break

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
window_title = "USB Camera"


def show_camera():
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video12"
    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    print("start")

    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    print("started")
    """ 
    # How to set video capture properties using V4L2:
    # Full list of Video Capture Properties for OpenCV: https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    #Select Pixel Format:
    # video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
    # Two common formats, MJPG and H264
    # video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # Default libopencv on the Jetson is not linked against libx264, so H.264 is not available
    # video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
    # Select frame size, FPS:
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    """
    #video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    #video_capture.set(cv2.CAP_PROP_FPS, 30)
    print("setted")
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(
                window_title, cv2.WINDOW_AUTOSIZE )
            
            # Window
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    continue
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


