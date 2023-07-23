#!/usr/bin/env python3
#
#  USB Camera - Simple
#
#  Copyright (C) 2021-22 JetsonHacks (info@jetsonhacks.com)
#
#  MIT License
#

import sys

import cv2

window_title = "USB Camera"

class UpperEyeCamera:

    def __init__(self):
        # ASSIGN CAMERA ADDRESS HERE
        self.camera_id = "/dev/video12"
        # Full list of Video Capture APIs (video backends): https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
        # For webcams, we use V4L2
        self.video_capture = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
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
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.video_capture.set(cv2.CAP_PROP_FPS, 30)

    def stop(self):
        self.video_capture.release()


    def frame(self):
        if self.video_capture.isOpened():
            try:
                ret_val, frame = self.video_capture.read()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return frame
            except:
                 print("error getting frame from uppereye")
        else:
            print("Unable to open camera")

        return None


if __name__ == "__main__":

   cam = UpperEyeCamera() 
   cam.frame()
