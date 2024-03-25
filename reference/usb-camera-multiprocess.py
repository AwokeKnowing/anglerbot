#!/usr/bin/env python3
import numpy as np
import cv2
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import SharedMemory

def produce_frames(q, camera_id='/dev/video12'):
    #get the first frame to calculate size of buffer
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    success, frame = cap.read()
    shm = SharedMemory(create=True, size=frame.nbytes)
    framebuffer = np.ndarray(frame.shape, frame.dtype, buffer=shm.buf) #could also maybe use array.array instead of numpy, but I'm familiar with numpy
    framebuffer[:] = frame #in case you need to send the first frame to the main process
    q.put(shm) #send the buffer back to main
    q.put(frame.shape) #send the array details
    q.put(frame.dtype)
    try:
        while True:
            cap.read(framebuffer)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close() #call this in all processes where the shm exists
        shm.unlink() #call from only one process

def consume_frames(q):
    shm = q.get() #get the shared buffer
    shape = q.get()
    dtype = q.get()
    framebuffer = np.ndarray(shape, dtype, buffer=shm.buf) #reconstruct the array
    try:
        while True:
            cv2.imshow("window title", framebuffer)
            cv2.waitKey(100)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()

if __name__ == "__main__":
    q = Queue()
    producer = Process(target=produce_frames, args=(q,))
    producer.start()
    consume_frames(q)

#import sys
#import time
#import cv2

#window_title = "USB Camera"


def show_camera():
    # ASSIGN CAMERA ADDRESS HERE
    camera_id = "/dev/video12"
    # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
    # For webcams, we use V4L2
    print("start")

    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
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
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 3000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("setted")
    if cap.isOpened():
        try:
            window_handle = cv2.namedWindow(
                window_title, cv2.WINDOW_AUTOSIZE )
            
            # Window
            while True:
                ret_val, frame = cap.read()
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
            cap.release()
            cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


#if __name__ == "__main__":

#    show_camera()
