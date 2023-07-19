import numpy as np
import cv2
from .topdown import RealsenseTopdownCamera

def polar():
    center=(618,608)

    import sys
    try:
        fn = sys.argv[1]
    except:
        fn = 'polar.png'

    img = cv2.imread(fn)
    if img is None:
        print('Failed to load image file:', fn)
        sys.exit(1)
    
    #img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    imgo = img.copy()
    imgo = cv2.rotate(imgo, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.circle(imgo, center, 3, (255, 0, 0), -1)
    imgo = cv2.rotate(imgo, cv2.ROTATE_90_CLOCKWISE)

    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img2 = cv2.logPolar(img, center, 618*.3, cv2.WARP_FILL_OUTLIERS)
    img3 = cv2.linearPolar(img, center, 618*4, cv2.WARP_FILL_OUTLIERS)

    img2 = cv2.rotate(img2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img3 = cv2.rotate(img3, cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow('before', imgo)
    cv2.imshow('logpolar', img2)
    cv2.imshow('linearpolar', img3)

    cv2.waitKey(0)




def start(config, whiteFiber, brainSleeping):
    print("starting vision")

    axon = whiteFiber.axon(
        get_topics = [

        ],
        put_topics = [
            "/vision/images/topdown"
        ]
    )

   
    rs_forward_sn=config['vision.realsense_forward_serial']
    # Create a context object for the Intel RealSense camera

    topdown=RealsenseTopdownCamera(config['vision.realsense_topdown_serial'])

    print("vision ready")
    while not brainSleeping.isSet():
        topdown_depth,topdown_mask,topdown_overlay,topdown_color=topdown.frame()
        
        rcx,rcy= topdown.robot_center
        roi_fwd1_tl = (rcx-130, rcy-125)
        roi_fwd1_br = (rcx+130, rcy-75)
        roi_fwd2_tl = (rcx-130, rcy-175)
        roi_fwd2_br = (rcx+130, rcy-125)
        # draw visual indicators

        # obstacle sensors
        roi_fwd_img = topdown_mask[roi_fwd1_tl[1]:roi_fwd1_br[1], roi_fwd1_tl[0]:roi_fwd1_br[0]]
        
        # Calculate the total number of pixels in the image
        total_pixels = roi_fwd_img.size
        zero_pixels = np.count_nonzero(roi_fwd_img == 0)
        percent_zero = (zero_pixels / total_pixels) * 100
        
        # Define the threshold for considering the ROI as "more than 99% white"
        threshold = 0.1

        # Check if the ROI is more than 99% white
        if percent_zero > threshold:
            roi_fwd_color = (0, 0, 255)  # Red
        else:
            roi_fwd_color = (0, 255, 0)  # Green
            
        # Draw the rectangle on the image
        cv2.rectangle(topdown_overlay, roi_fwd1_tl, roi_fwd1_br, roi_fwd_color, 1)



        # obstacle sensors
        roi_fwd_img = topdown_mask[roi_fwd2_tl[1]:roi_fwd2_br[1], roi_fwd2_tl[0]:roi_fwd2_br[0]]
        
        # Calculate the total number of pixels in the image
        total_pixels = roi_fwd_img.size
        zero_pixels = np.count_nonzero(roi_fwd_img == 0)
        percent_zero = (zero_pixels / total_pixels) * 100
        
        # Define the threshold for considering the ROI as "more than 99% white"
        threshold = 0.1

        # Check if the ROI is more than 99% white
        if percent_zero > threshold:
            roi_fwd_color = (0, 0, 255)  # Red
        else:
            roi_fwd_color = (0, 255, 0)  # Green
            
        # Draw the rectangle on the image
        cv2.rectangle(topdown_overlay, roi_fwd2_tl, roi_fwd2_br, roi_fwd_color, 1)




        
        

        #output1 = cv2.add(roi_with_mask, overlay)
        
        
        

        # Display the ROI with the mask
        cv2.imshow('ROI with Mask', topdown_overlay)
        #cv2.imshow('ROI Mask', mask)
        cv2.imshow('color', topdown_color)

        # Break the loop if 'q' is pressed
        cv2.waitKey(1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Stop the pipeline and close the camera
    topdown.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("stopped vision")
