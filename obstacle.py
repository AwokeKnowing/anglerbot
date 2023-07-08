import pyrealsense2 as rs
import numpy as np
import cv2

# Create a context object for the Intel RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)

# Start the pipeline
pipeline.start(config)

# Define the region of interest (ROI) for plane estimation
depth_cam_to_floor_mm = 920
min_obstacle_mm = 73
roi_x = 0
roi_y = 32
roi_width = 480
roi_height = 816

# Function to calculate the floor mask
def floor_mask(img, floor_mm, obst_mm):
    mask = np.where(img - floor_mm < obst_mm, 255, 0).astype(np.uint8)
    return mask

while True:
    # Create an empty mask with the size of ROI for each pass
    mask_accumulated = np.zeros((roi_height, roi_width), dtype=np.uint8)

    # Collect 3 frames and accumulate the mask
    for _ in range(3):
        # Wait for the next frame from the camera
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        
        # Convert the depth frame to a NumPy array
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_90_CLOCKWISE)

        # Extract the ROI from the depth image
        roi = depth_image[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

        # Calculate the floor mask for the current frame
        mask = floor_mask(roi, depth_cam_to_floor_mm, min_obstacle_mm)

        # Accumulate the mask
        mask_accumulated = cv2.bitwise_or(mask_accumulated, mask)

    # Apply color mapping to the ROI
    roi_colorized = cv2.applyColorMap(cv2.convertScaleAbs(roi, alpha=0.03), cv2.COLORMAP_JET)

    # Composite the accumulated mask on the ROI
    mask_rgb = cv2.cvtColor(mask_accumulated, cv2.COLOR_GRAY2BGR)
    mask_rgb = cv2.addWeighted(mask_rgb, 0.9, np.zeros_like(mask_rgb), 0.5, 0)
    roi_with_mask = cv2.add(roi_colorized, mask_rgb)

    # Apply morphological close operation
    kernel = np.ones((5, 5), np.uint8)
    roi_with_mask = cv2.morphologyEx(roi_with_mask, cv2.MORPH_CLOSE, kernel)

    
    # visual center of robot (middle of wheel axis)
    rcx=240
    rcy=396
    

    # Create polygon points with shifted x coordinates
    rdy1=72
    rdy2=122
    rdw=112
    rdw2=30
    rdw3=40
    polygon_points = np.array([
        [rcx-rdw, rcy-rdy1], [rcx-rdw, rcy+rdy2], 
        [rcx-rdw2,rcy+rdy2], 
        [rcx-rdw3,roi_height-1], [rcx+rdw3,roi_height-1], 
        [rcx+rdw2, rcy+rdy2], [rcx+rdw, rcy+rdy2], 
        [rcx+rdw,rcy-rdy1]]) + [0, 0]

    # Draw and fill the polygon on the roi_with_mask with 20% opacity
    overlay = roi_with_mask.copy()
    # Add a circle to the image
    

    cv2.fillPoly(overlay, [polygon_points], (255, 255, 0))
    roi_with_mask = cv2.addWeighted(roi_with_mask, 0.8, overlay, 0.2, 0)
    
    # if calibrate
    if True:
        cv2.circle(roi_with_mask, (rcx, rcy), 3, (0, 255, 0), -1)

    # Display the ROI with the mask
    cv2.imshow('ROI with Mask', roi_with_mask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the pipeline and close the camera
pipeline.stop()

# Close all OpenCV windows
cv2.destroyAllWindows()
