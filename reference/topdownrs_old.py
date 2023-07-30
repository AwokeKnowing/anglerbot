import pyrealsense2 as rs
import numpy as np
import cv2

import time


import numpy as np



class TopdownRealsenseCamera:
    

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
    

    


    def __init__(self,serial):
        self.serial =  serial

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 90)
        config.enable_stream(rs.stream.color,640,480, rs.format.bgr8, 30)
        pipeline.start(config)

        #public props
        self.robot_center=(self.rcx,self.rcy)
        self.pipeline = pipeline
        self.config = config

    def stop(self):
        self.pipeline.stop()

    def frame(self,depth_accumulate=3):
        # Create an empty mask with the size of ROI for each pass
        mask_accumulated = np.zeros((self.roi_topdown_cam_h, self.roi_topdown_cam_w), dtype=np.uint8)

        # Collect 3 frames and accumulate the mask
        for _ in range(depth_accumulate):
            # Wait for the next frame from the camera
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            
            # Convert the depth frame to a NumPy array
            depth_image1 = np.asanyarray(depth_frame.get_data())
            depth_image = cv2.rotate(depth_image1, cv2.ROTATE_90_CLOCKWISE)

            # Extract the ROI from the depth image
            topdownimg = depth_image[self.roi_topdown_cam_y:self.roi_topdown_cam_y + self.roi_topdown_cam_h, 
                                     self.roi_topdown_cam_x:self.roi_topdown_cam_x + self.roi_topdown_cam_w]

            # Calculate the floor mask for the current frame
            # Function to calculate the floor mask
            mask = np.where(topdownimg - self.depth_cam_to_floor_mm < self.min_obstacle_mm, 255, 0).astype(np.uint8)
            # Accumulate the mask
            mask_accumulated = cv2.bitwise_or(mask_accumulated, mask)

        color_image = frames.get_color_frame()
        color_image= np.asanyarray(color_image.get_data())
        color_image = cv2.rotate(color_image, cv2.ROTATE_90_CLOCKWISE)
        # Apply morphological close operation
        kernel = np.ones((5, 5), np.uint8)
        mask_accumulated = cv2.morphologyEx(mask_accumulated, cv2.MORPH_CLOSE, kernel)
        

        # Apply color mapping to the ROI
        roi_colorized = cv2.applyColorMap(cv2.convertScaleAbs(topdownimg, alpha=0.3), cv2.COLORMAP_JET)

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

        # Add a circle to the image
        cv2.fillPoly(overlayblend, [self.robot_polygon_points], (255, 255, 0))
        overlay = cv2.addWeighted(overlay, .9, overlayblend, 0.1, 0)

        cv2.circle(overlay, (self.rcx, self.rcy), 3, (0, 255, 0), -1)




        intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()

        # Camera calibration parameters (from RealSense intrinsics)
        fx, fy = intrinsics.fx, intrinsics.fy  # Focal lengths
        cx, cy = intrinsics.ppx, intrinsics.ppy  # Principal points

        # Create the camera intrinsic matrix
        camera_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])

        # Assuming the camera is facing down at the floor
        vertical_direction = np.array([0, -1]) 

        points = depth_frame.as_points()
        pc = rs.pointcloud()
        points2 = pc.calculate(depth_frame)
        #verts = points2.get_vertices()
        verts = np.asanyarray(points2.get_vertices()).view(np.float32).reshape(-1, 3)

        result = plane_from_points(verts)
        if result is not None:
            centroid, normal = result
            #print("Centroid:", centroid)
            print("Normal:", normal)
        else:
            print("Not enough points to determine a plane.")
        #print(verts[480*100+200])
        #print(depth_image1.astype('float32')[1][1])
        #ret, rvec, tvec = cv2.solvePnPRansac(verts, depth_image1.astype('float32'), camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE)

        # Print the results
        ##print(ret)
        #print(rvec)
        #print(tvec)

        # points3d = 
        # normal_estimator = cv2.rgbd.RgbdNormals_create(16, 16, cv2.CV_32F, camera_matrix, 3)
        # normals = normal_estimator.apply( points3d )
        # print("normals", normals)
        # time.sleep(1)

        # # Assuming the camera is facing down at the floor
        # vertical_direction = np.array([0, 0, 1])  # Assuming upward direction from the floor

        # # Calculate the tilt angle (angle between the plane normal and the vertical direction)
        # angle_tilt_radians = np.arccos(np.dot(normals, vertical_direction) /
        #                             (np.linalg.norm(normals) * np.linalg.norm(vertical_direction)))
        # angle_tilt_degrees = np.degrees(angle_tilt_radians)

        # print("Tilt angle:", angle_tilt_degrees)
                        
        # Load the depth image (replace 'depth_image.png' with your file path)
        #depth_image = cv2.imread('depth_image.png', cv2.IMREAD_GRAYSCALE)

        
        #print("adjust",adjusted_gradient)
        time.sleep(.1)
        points=0

        return topdownimg,mask_accumulated,overlay,color_image, points


if __name__ == '__main__':
    topdown = TopdownRealsenseCamera("815412070676")
    depth,mask,overlay,color=topdown.frame()