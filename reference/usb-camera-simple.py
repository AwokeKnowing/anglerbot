#!/usr/bin/env python3
#
#  USB Camera - Simple
#
#  Copyright (C) 2021-22 JetsonHacks (info@jetsonhacks.com)
#
#  MIT License
#

import sys
import time
import cv2

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
    
    video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    video_capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
    video_capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("setted")
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(
                window_title, cv2.WINDOW_AUTOSIZE )
            
            # Window
            while True:
                try:
                    ret_val, frame = video_capture.read()
                    if not ret_val:
                        continue
                except cv2.error as e:
    
                    # inspect error object
                    print(e)
                    for k in dir(e):
                        if k[0:2] != "__":
                            print("e.%s = %s" % (k, getattr(e, k)))

                        # handle error: empty frame
                        if e.err == "!_src.empty()":
                            continue # break the while loop

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


if __name__ == "__main__":

    show_camera()
