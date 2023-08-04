# pyrealsense2 is required.
# Please see instructions in https://github.com/IntelRealSense/librealsense/tree/master/wrappers/python
import pyrealsense2 as rs
import open3d as o3d
from open3d import *
import numpy as np
from enum import IntEnum

from datetime import datetime
from open3d.visualization import *
from open3d.geometry import *
from open3d.camera import *

import cv2

import anglerdroid as a7

import matplotlib.pyplot as plt

def mb_box(sx,sy,sz,t,c):
    #create a box and return mesh and bounding box
    rb1 = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
    rb1.paint_uniform_color(c)
    rb1.translate(t)
    return rb1.get_axis_aligned_bounding_box(),rb1


boxbot_last_mesh=None
boxbot_last_bboxes=None
def extract_boxbot(pcd,recalculate=False):
    global boxbot_last_mesh,boxbot_last_bboxes
    if recalculate or boxbot_last_mesh is None:
        fl=-1.0
        rb1sx=.44
        rb1sy=.36
        rb1sz=.55
        rb1bb, rb1 = mb_box(rb1sx,rb1sy,rb1sz,
                            [(-rb1sx/2-.07),-rb1sy/2,fl],
                            [0.9, 0.1, 0.1])

        rb2sx=.33
        rb2sy=.43
        rb2sz=.55
        rb2bb, rb2 = mb_box(rb2sx,rb2sy,rb2sz,
                            [(-rb2sx/2-.04),-rb2sy/2,fl],
                            [0.9, 0.5, 0.1])
        
        rb3sx=.22
        rb3sy=.52
        rb3sz=.22
        rb3bb, rb3 = mb_box(rb3sx,rb3sy,rb3sz,
                            [(-rb3sx/2+.00),-rb3sy/2,fl],
                            [0.1, 0.1, 0.1])

        rb4sx=.16
        rb4sy=.22
        rb4sz=1.0
        rb4bb, rb4 = mb_box(rb4sx,rb4sy,rb4sz,[(-rb4sx/2-.22),-rb4sy/2,fl],[0.4, 0.4, 0.6])

        boxbot_last_mesh = rb1+rb2+rb3+rb4
        boxbot_last_bboxes = (rb1bb,rb2bb,rb3bb,rb4bb)
    else:
        rb1bb,rb2bb,rb3bb,rb4bb = boxbot_last_bboxes

    rbp1 = rb1bb.get_point_indices_within_bounding_box(pcd.points)
    rbp2 =(rb2bb.get_point_indices_within_bounding_box(pcd.points))
    rbp3 =(rb3bb.get_point_indices_within_bounding_box(pcd.points))
    rbp4 =(rb4bb.get_point_indices_within_bounding_box(pcd.points))

    boxbot_points = list(set(rbp1+rbp2+rbp3+rbp4))
    
    boxbot_pcd=pcd.select_by_index(boxbot_points)
    not_boxbot_pcd=pcd.select_by_index(boxbot_points, invert=True) #select outside points
    
    

    return not_boxbot_pcd,boxbot_pcd, boxbot_last_mesh

def bbox_from_xxyyzz(x1,x2,y1,y2,z1,z2,color=(1, 0, 0)):
    vec= o3d.utility.Vector3dVector(np.asarray([
        [x1, y1, z1],
        [x2, y1, z1],
        [x1, y2, z1],
        [x1, y1, z2],
        [x2, y2, z2],
        [x1, y2, z2],
        [x2, y1, z2],
        [x2, y2, z1]
    ]))

    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(vec)
    bbox.color = color
    return bbox


def calc_unrotate_floor(pcd,ransac_dist=.003,ransac_n=3,ransac_iter=100):
    plane, points = pcd.segment_plane(distance_threshold=ransac_dist,
                                             ransac_n=ransac_n,
                                             num_iterations=ransac_iter)
    #[a, b, c, d] = plane
    #print(f"Plane equation: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")
    
    in_pcd=topdown_pcd.select_by_index(points)
    
    obb=in_pcd.get_oriented_bounding_box()
    obb.color=(0, 1, 0)
    R=obb.R
    #print(R) #this can be backward because obb direction is random

    sy = np.sqrt(R[0][0] * R[0][0] +  R[1][0] * R[1][0])
    x = np.arctan2(R[2][1] , R[2][2])
    y = np.arctan2(-R[2][0], sy)
    z = 0 #we don't want to rotate on z #np.arctan2(R[1][0], R[0][0])
    
    #at least when detecting floor plane, avoids flip when R is oriented backward
    if(x>np.pi/2):
        x-=np.pi
    if(x<-np.pi/2):
        x+=np.pi
    
    R = o3d.geometry.get_rotation_matrix_from_zyx(np.array([z,y,x]))
    inv_R = np.linalg.inv(R)
    
    Rcenter=obb.get_center()
    
    #in_pcd.paint_uniform_color([1.0, 0, 0])
    #in_pcd=inpcd.rotate(inv_R,center=Rcenter)

    return inv_R, Rcenter,in_pcd

def topview(vis):
    rq=523.598775 # sigh...
    vis.get_view_control().rotate(0,rq,0,0)
    vis.get_view_control().rotate(-rq,0,0,0)
    vis.get_view_control().rotate(0,-rq,0,0)

    vis.get_view_control().set_zoom(.48)

    #ortho
    vis.get_view_control().change_field_of_view(step=-90)





if __name__ == "__main__":
    with_color=False   #setting false really slows down on small voxel size
    with_forward=True
    with_clusters=False
    with_shadow_ground=True

    show_voxels=False 
    show_pointcloud=True
    show_axis=False
    show_topdown_roi_box=True
    show_boxbot=True
    show_floor_basis_points=False

    crop_floor=True

    voxel_detail = 2
    voxel_size=.1/float(2.**max(0,min(4,voxel_detail))) #clamp detail 0-5

    topdowncam = a7.RealsenseCamera("815412070676",with_color)

    if with_forward:
        forwardcam = a7.RealsenseCamera("815412070180",with_color)
    
    vis = Visualizer()
    
    #need to make a function to set view and capture frame and call it 1 fps
    vis.create_window(width=240,height=424)
    vis.get_render_option().point_size=28/voxel_detail

    floor_depth_size=(240,424)
    floor_acc= np.zeros_like(floor_depth_size)
    
    pcd = PointCloud()
    floor_basis_points_pcd = PointCloud()

    #topdown roi volume points (for obstacles, ie excluding floor)
    
    tdx1 = -0.91874238
    tdx2 = +0.88713181
    tdy1 = -0.50779997
    tdy2 = +0.54382918
    tdz1 = -0.91435421
    tdz2 = -0.22804920
    floor_z = tdz1

    morph_kernel = np.ones((5, 5), np.uint8)
    last_forward_transform = np.identity(4)

    #preallocate image

    # Streaming loop
    frame_count = 0
    try:
        while True:

            dt0=datetime.now()
            tempsource,rgbd,intrinsic = topdowncam.frame()
            topdown_pcd=tempsource.voxel_down_sample(voxel_size=voxel_size)
            
            if frame_count % 30 == 0:
                floor_inv_R,floor_Rcenter,floor_basis_points=calc_unrotate_floor(topdown_pcd)
            
            topdown_pcd = topdown_pcd.rotate(floor_inv_R, center=floor_Rcenter)
            #topdown_pcd.estimate_normals()
            
            if with_forward:
                forward_pcd,rgbd2,intrinsic = forwardcam.frame()
                forward_pcd = forward_pcd.voxel_down_sample(voxel_size=voxel_size)

                p=np.pi/2
                pp=np.pi
                p3=np.pi/6
                p10=np.pi/18
                deg=np.pi/180
                forward_pcd = forward_pcd.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.array([-p+15.5*deg+p10,0,0])))
                forward_pcd = forward_pcd.rotate(o3d.geometry.get_rotation_matrix_from_xyz(np.array([0,0,p])))
                if with_color:
                    forward_pcd = forward_pcd.translate([1.57,.20,.64])
                else:
                    forward_pcd = forward_pcd.translate([.88,.40,.32])

                #reg_p2p = o3d.pipelines.registration.registration_icp(
                #    forward_pcd, topdown_pcd, .02, np.identity(4),
                #    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                #    #o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
                #    )
                #forward_pcd.paint_uniform_color((0,1,0))
                #topdown_pcd.paint_uniform_color((1,0,0))
                #print("reg",reg_p2p)
                #if reg_p2p.fitness > .04 and len(reg_p2p.correspondence_set)>150:
                #    #forward_pcd.transform(reg_p2p.transformation)
                #    last_forward_transform = reg_p2p.transformation
                #else:
                #    print("bad fit")
                #    #forward_pcd.transform(last_forward_transform)
         
            
            
                            
            
            # because topdown 0,0 is the 'origin' of robot, we set 
            # the roi in worldspace rather than pcd space
            #above_ground_bbox = topdown_pcd.get_axis_aligned_bounding_box()
            above_ground_bbox = bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,floor_z,tdz2,(1, 0, 0))
            below_ground_bbox = bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,floor_z,floor_z-.5,(0, 1, 0))
            
            # get rid of the floor
            if with_shadow_ground:
                ground_pcd=topdown_pcd.crop(below_ground_bbox)

            if crop_floor:
                topdown_pcd=topdown_pcd.crop(above_ground_bbox)

            topdown_pcd, boxbot_pcd, boxbot_mesh = extract_boxbot(topdown_pcd)

            if with_clusters:
                labels = np.array(topdown_pcd.cluster_dbscan(eps=0.1, min_points=int(2.**voxel_detail)))

                try:
                    max_label = labels.max()    
                except ValueError:
                    max_label=0
                #print(f"point cloud has {max_label + 1} clusters")
                colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
                #colors[labels < 1] = 0
                #temp.colors = o3d.utility.Vector3dVector(colors[:, :3])


            if with_forward:
                topdown_pcd.points.extend(forward_pcd.points)
                topdown_pcd.colors.extend(forward_pcd.colors)            
                topdown_pcd.normals.extend(forward_pcd.normals)

            if show_pointcloud:
                #topdown_pcd.points[:,:,2]=0
                pcd.points = topdown_pcd.points
                pcd.colors = topdown_pcd.colors
                pcd.normals = topdown_pcd.normals

            if show_floor_basis_points:
                floor_basis_points_pcd.points = floor_basis_points.points
                floor_basis_points_pcd.colors = floor_basis_points.colors
                floor_basis_points_pcd.normals = floor_basis_points.normals
            
            if show_voxels:
                vxgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size=voxel_size)

            
            
        # draw 
            if frame_count ==0:
                
                if show_floor_basis_points:
                    vis.add_geometry(floor_basis_points_pcd)

                if show_topdown_roi_box:
                    vis.add_geometry(above_ground_bbox) #everything

                if show_boxbot:
                    vis.add_geometry(boxbot_mesh) #robot

                if show_axis:
                    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                    vis.add_geometry(mesh_frame)
                    mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[1, 0, 0])
                    vis.add_geometry(mesh_frame1)

                if show_voxels:
                    vxgrid_old = vxgrid
                    vis.add_geometry(vxgrid_old)
                
                if show_pointcloud:
                    vis.add_geometry(pcd)
                
                topview(vis)

            
            
            if show_topdown_roi_box:
                # in case we are dynamically adjusting
                vis.update_geometry(above_ground_bbox)
            
            if show_floor_basis_points:
                vis.update_geometry(floor_basis_points_pcd)
            
            if show_voxels:
                vis.remove_geometry(vxgrid_old,False)
                vis.add_geometry(vxgrid,False)
                vxgrid_old = vxgrid
                
            if show_pointcloud:
                vis.update_geometry(pcd)

            vis.poll_events()
            
#2d render
            
            vis.update_renderer()
            floor_depth = np.array(vis.capture_depth_float_buffer()).astype(np.uint8)
            floor_depth = cv2.resize(floor_depth, floor_depth_size, interpolation= cv2.INTER_LINEAR)
            floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, morph_kernel)
            ret, floor_depth = cv2.threshold(floor_depth, 0, 255, cv2.THRESH_BINARY)
            #floor_depth = np.where(floor_depth>0, 255, 0).astype(np.uint8).copy()

            #floor_depth_save = floor_depth
            #cv2.imshow('floor depth save', floor_depth_save)

            if with_shadow_ground:
                #recalculate every 4 frames
                if frame_count % 4 == 0:
                    vis.add_geometry(ground_pcd,False)

                    shadow_ground_depth = np.array(vis.capture_depth_float_buffer(True)).astype(np.uint8)
                    vis.remove_geometry(ground_pcd,False)
                    shadow_ground_depth = cv2.resize(shadow_ground_depth, floor_depth_size, interpolation=cv2.INTER_LINEAR)
                    shadow_ground_depth = cv2.morphologyEx(shadow_ground_depth, cv2.MORPH_CLOSE, morph_kernel)
                    ret, shadow_ground_depth = cv2.threshold(shadow_ground_depth, 0, 255, cv2.THRESH_BINARY_INV)
                    floor_acc = shadow_ground_depth
                    #floor_acc = np.where(shadow_ground_depth>0, 0, 255).astype(np.uint8).copy()
                    
                    #cover the tail shadow
                    cv2.rectangle(floor_acc, (100,276), (240-100,423), (0,0,0), -1)
                #print(floor_acc,floor_depth)
                floor_depth=cv2.bitwise_or(floor_acc,floor_depth)
                floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, morph_kernel)
            
            cv2.imshow('floor depth', floor_depth)
            #cv2.imwrite('botmask.png',floor_depth)
            cv2.waitKey(1)

            #cv2.imshow('floor depth', floor_depth)
            #cv2.imshow('floor depth acc', floor_acc)

            process_time = datetime.now() - dt0
            frame_count += 1

            if frame_count % 20 == 0:
                print("FPS: "+str(1/process_time.total_seconds()))
            

    finally:
        cv2.destroyAllWindows()
        topdowncam.stop()
        vis.destroy_window()