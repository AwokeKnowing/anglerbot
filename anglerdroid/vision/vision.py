import pyrealsense2 as rs
import numpy as np
import cv2


def start(config, whiteFiber, brainSleeping):
    print("starting vision")

    axon = whiteFiber.axon(
        get_topics = [

        ],
        put_topics = [
            "/vision/images/topdown"
        ]
    )
    rs_devices=['815412070676','815412070180']
    rs_topdown_sn=rs_devices[0]
    rs_forward_sn=rs_devices[1]
    # Create a context object for the Intel RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(rs_topdown_sn)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)

    # Start the pipeline
    pipeline.start(config)

    # Define the region of interest (ROI) for plane estimation
    depth_cam_to_floor_mm = 920
    min_obstacle_mm = 73
    roi_topdown_cam_x = 0
    roi_topdown_cam_y = 32
    roi_topdown_cam_w = 480
    roi_topdown_cam_h = 816


    # visual center of robot (middle of wheel axis)
    rcx=240
    rcy=396
    

    # Create polygon points with shifted x coordinates
    rdy1=72
    rdy2=122
    rdw=112
    rdw2=30
    rdw3=40
    robot_polygon_points = np.array([
        [rcx-rdw, rcy-rdy1], [rcx-rdw, rcy+rdy2], 
        [rcx-rdw2,rcy+rdy2], 
        [rcx-rdw3,roi_topdown_cam_h-1], [rcx+rdw3,roi_topdown_cam_h-1], 
        [rcx+rdw2, rcy+rdy2], [rcx+rdw, rcy+rdy2], 
        [rcx+rdw,rcy-rdy1]]) + [0, 0]
    

    roi_fwd1_tl = (rcx-130, rcy-125)
    roi_fwd1_br = (rcx+130, rcy-75)
    roi_fwd2_tl = (rcx-130, rcy-175)
    roi_fwd2_br = (rcx+130, rcy-125)


    # Function to calculate the floor mask
    def floor_mask(img, floor_mm, obst_mm):
        mask = np.where(img - floor_mm < obst_mm, 255, 0).astype(np.uint8)
        return mask

    print("vision ready")
    while not brainSleeping.isSet():
        # Create an empty mask with the size of ROI for each pass
        mask_accumulated = np.zeros((roi_topdown_cam_h, roi_topdown_cam_w), dtype=np.uint8)

        # Collect 3 frames and accumulate the mask
        for _ in range(3):
            # Wait for the next frame from the camera
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            # Convert the depth frame to a NumPy array
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)

            # Extract the ROI from the depth image
            roi = depth_image[roi_topdown_cam_y:roi_topdown_cam_y + roi_topdown_cam_h, roi_topdown_cam_x:roi_topdown_cam_x + roi_topdown_cam_w]

            # Calculate the floor mask for the current frame
            mask = floor_mask(roi, depth_cam_to_floor_mm, min_obstacle_mm)

            # Accumulate the mask
            mask_accumulated = cv2.bitwise_or(mask_accumulated, mask)

        # Apply morphological close operation
        kernel = np.ones((5, 5), np.uint8)
        mask_accumulated = cv2.morphologyEx(mask_accumulated, cv2.MORPH_CLOSE, kernel)
        

        # Apply color mapping to the ROI
        roi_colorized = cv2.applyColorMap(cv2.convertScaleAbs(roi, alpha=0.03), cv2.COLORMAP_JET)

        # Composite the accumulated mask on the ROI
        mask_rgb = cv2.cvtColor(mask_accumulated, cv2.COLOR_GRAY2BGR)
        mask_rgb = cv2.addWeighted(mask_rgb, 0.8, np.zeros_like(mask_rgb), 0.2, 0)
        roi_with_mask = cv2.add(roi_colorized, mask_rgb)

        # Apply morphological close operation
        kernel = np.ones((5, 5), np.uint8)
        roi_with_mask = cv2.morphologyEx(roi_with_mask, cv2.MORPH_CLOSE, kernel)

        # Draw visual indicators on the roi_with_mask with 30% opacity
        overlay = roi_with_mask.copy()
        overlayblend = np.zeros_like(roi_with_mask)
        
        # draw visual indicators

        # obstacle sensors
        roi_fwd_img = mask_accumulated[roi_fwd1_tl[1]:roi_fwd1_br[1], roi_fwd1_tl[0]:roi_fwd1_br[0]]
        
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
        cv2.rectangle(overlay, roi_fwd1_tl, roi_fwd1_br, roi_fwd_color, 1)



        # obstacle sensors
        roi_fwd_img = mask_accumulated[roi_fwd2_tl[1]:roi_fwd2_br[1], roi_fwd2_tl[0]:roi_fwd2_br[0]]
        
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
        cv2.rectangle(overlay, roi_fwd2_tl, roi_fwd2_br, roi_fwd_color, 1)




        
        # Add a circle to the image
        cv2.fillPoly(overlayblend, [robot_polygon_points], (255, 255, 0))
        overlay = cv2.addWeighted(overlay, .6, overlayblend, 0.4, 0)

        cv2.circle(overlay, (rcx, rcy), 3, (0, 255, 0), -1)

        #output1 = cv2.add(roi_with_mask, overlay)
        
        
        

        # Display the ROI with the mask
        cv2.imshow('ROI with Mask', overlay)
        #cv2.imshow('ROI Mask', mask_accumulated)

        # Break the loop if 'q' is pressed
        cv2.waitKey(1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Stop the pipeline and close the camera
    pipeline.stop()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
    print("stopped vision")
