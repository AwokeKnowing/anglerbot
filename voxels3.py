import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

from datetime import datetime
from functools import lru_cache

import anglerdroid as a7


class Util3d:
    @staticmethod
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
    
    
    @staticmethod
    def mb_box(sx,sy,sz,t,c):
        c=[1,1,1]
        #create a box and return mesh and bounding box
        rb1 = o3d.geometry.TriangleMesh.create_box(width=sx, height=sy, depth=sz)
        rb1.paint_uniform_color(c)
        rb1.translate(t)
        return rb1.get_axis_aligned_bounding_box(),rb1
    

    @staticmethod
    def calc_unrotate_floor(pcd,ransac_dist=.015,ransac_n=3,ransac_iter=100):
        
        plane, points = pcd.segment_plane(distance_threshold=ransac_dist,
                                                ransac_n=ransac_n,
                                                num_iterations=ransac_iter)
        #[a, b, c, d] = plane
        #print(f"Plane equation: {a:.5f}x + {b:.5f}y + {c:.5f}z + {d:.5f} = 0")
        
        in_pcd=pcd.select_by_index(points)
        
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
    
    
    @staticmethod
    def topview(vis):
        rq=523.598775 # sigh...
        vis.get_view_control().rotate(0,rq,0,0)
        vis.get_view_control().rotate(-rq,0,0,0)
        vis.get_view_control().rotate(0,-rq,0,0)

        vis.get_view_control().set_zoom(.48)

        #ortho
        vis.get_view_control().change_field_of_view(step=-90)


class BotExtractor:
    @staticmethod
    @lru_cache
    def getRobotBoundingMeshAndBoundingBoxes():
            fl = -1.0
            rb1sx = .44
            rb1sy = .36
            rb1sz = .55
            rb1bb, rb1 = Util3d.mb_box(rb1sx,rb1sy,rb1sz,
                                [(-rb1sx/2-.07),-rb1sy/2,fl],
                                [0.9, 0.1, 0.1])

            rb2sx = .36
            rb2sy = .43
            rb2sz = .55
            rb2bb, rb2 = Util3d.mb_box(rb2sx,rb2sy,rb2sz,
                                [(-rb2sx/2-.03),-rb2sy/2,fl],
                                [0.9, 0.5, 0.1])
            
            rb3sx = .22
            rb3sy = .46
            rb3sz = .22
            rb3bb, rb3 = Util3d.mb_box(rb3sx,rb3sy,rb3sz,
                                [(-rb3sx/2+.00),-rb3sy/2,fl],
                                [0.1, 0.1, 0.1])

            rb4sx = .16
            rb4sy = .22
            rb4sz = 1.0
            rb4bb, rb4 = Util3d.mb_box(rb4sx,rb4sy,rb4sz,[(-rb4sx/2-.21),-rb4sy/2,fl],[0.4, 0.4, 0.6])

            boxbot_mesh = rb1+rb2+rb3+rb4
            boxbot_bboxes = (rb1bb,rb2bb,rb3bb,rb4bb)

            return boxbot_mesh,boxbot_bboxes


    @staticmethod
    def extract(pcd,recalculate=False):
        
        if recalculate:
            BotExtractor.getRobotBoundingMeshAndBoundingBoxes.cache_clear()

        boxbot_mesh, boxbot_bboxes = BotExtractor.getRobotBoundingMeshAndBoundingBoxes()
        rb1bb,rb2bb,rb3bb,rb4bb = boxbot_bboxes

        rbp1 = rb1bb.get_point_indices_within_bounding_box(pcd.points)
        rbp2 = rb2bb.get_point_indices_within_bounding_box(pcd.points)
        rbp3 = rb3bb.get_point_indices_within_bounding_box(pcd.points)
        rbp4 = rb4bb.get_point_indices_within_bounding_box(pcd.points)

        boxbot_points = list(set(rbp1+rbp2+rbp3+rbp4))
        
        boxbot_pcd = pcd.select_by_index(boxbot_points)
        not_boxbot_pcd = pcd.select_by_index(boxbot_points, invert=True) #select outside points
        
        return not_boxbot_pcd, boxbot_pcd, boxbot_mesh
    

class AnglerDroidCameras:

    def __init__(self, *, rsTopdownSerial, rsForwardSerial=None):
        self.with_color = False
        self.with_forward = True
        self.with_forward_alignment = False
        self.with_clusters = False
        self.with_shadow_ground = False
        self.with_voxels = False

        self.show_pointcloud = True
        self.show_axis = False 
        self.show_topdown_roi_box = False 
        self.show_boxbot = False
        self.show_floor_basis_points = False 

        self.crop_floor = True

        self.voxel_detail = 2
        self.voxel_size = .1/float(2.**max(0,min(4, self.voxel_detail))) #clamp detail 0-5

        #topdown camera is rotated 180 so x will be forward
        self.topdown2world = np.identity(4)
        self.topdown2world[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.deg2rad(0),
            np.deg2rad(0),
            np.deg2rad(180)])
        
        self.forward2topdown = np.identity(4)
        self.forward2topdown[:3,:3] = o3d.geometry.get_rotation_matrix_from_xyz(
            [np.deg2rad(26-90), #camera is pointed ~26 degrees down from straight ahead
            np.deg2rad(0),
            np.deg2rad(270)])
                      
        self.last_forward_transform = np.identity(4)
        
        self.forwardcam = None

        self.topdown_size = (224,224)
        self.topdown_size2d = (224,224)

        self.boxbot_mesh = None
        self.boxbot_bboxes = None
        
        self.topdowncam = a7.RealsenseCamera(rsTopdownSerial,self.with_color,self.topdown2world)
        if self.with_forward:
            self.forwardcam = a7.RealsenseCamera(rsForwardSerial,self.with_color,self.forward2topdown)
        
        self.vis_topdown = o3d.visualization.Visualizer()
        
        #need to make a function to set view and capture frame and call it 1 fps
        self.vis_topdown.create_window(width=self.topdown_size[0],height=self.topdown_size[1])
        self.vis_topdown.get_render_option().point_size=4/self.voxel_detail

        #preallocate image
        self.floor_acc = np.zeros_like(self.topdown_size2d)
        
        self.pcd = o3d.geometry.PointCloud()
        self.floor_basis_points_pcd = o3d.geometry.PointCloud()


    def get3dViewBounds(self, with_forward, min_obstacle_height=.005):
        #topdown roi volume points (for obstacles, ie excluding floor)
        min_obstacle_height = .005 #obstacles smaller than this will be ignored
        
        tdx1 = -0.92
        tdx2 = +0.92
        tdy1 = -0.54
        tdy2 = +0.54
        tdz1 = -0.92
        tdz2 = -0.0
        floor_z = tdz1 + min_obstacle_height 

        if with_forward:
            tdx1 = -0.92
            tdx2 = +3.56
            tdy1 = -2.24
            tdy2 = +2.24

        return [tdx1,tdx2,tdy1,tdy2,floor_z,tdz2]


    def getTopwdownPointCloud(self):
        tempsource,rgbd,intrinsic = self.topdowncam.frame(3,4)
        topdown_pcd = tempsource
        topdown_pcd = tempsource.voxel_down_sample(voxel_size=self.voxel_size)
        
        if frame_count % 30 == 0:
            #must be done on first frame to get initial rotation 
            floor_inv_R,floor_Rcenter,floor_basis_points = Util3d.calc_unrotate_floor(topdown_pcd)

            if self.show_floor_basis_points:
                self.floor_basis_points_pcd.points = floor_basis_points.points
                self.floor_basis_points_pcd.colors = floor_basis_points.colors
                self.floor_basis_points_pcd.normals = floor_basis_points.normals
        
        topdown_pcd = topdown_pcd.rotate(floor_inv_R, center=floor_Rcenter)

        return topdown_pcd, rgbd, intrinsic
    

    def getForwardPointCloud(self, topdown_pcd, align=True):
        forward_pcd,rgbd,intrinsic = self.forwardcam.frame(3,4)
        forward_pcd = forward_pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        if self.with_color:
            forward_pcd = forward_pcd.translate([1.57,.20,.64])
        else:
            #forward_pcd = forward_pcd.translate([.88,.40,.32])
            forward_pcd = forward_pcd.translate([-.105,.035,-.525])

        if self.with_forward_alignment and self.frame_count % 30 ==15:
            estimator = o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2,max_nn=30)
            topdown_pcd.estimate_normals(estimator)
            
            if align:
                reg_p2p = o3d.pipelines.registration.registration_icp(
                    forward_pcd, topdown_pcd, .01, np.identity(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)
                    )
                #forward_pcd.paint_uniform_color((0,1,0))
                #topdown_pcd.paint_uniform_color((1,0,0))
                print("reg",reg_p2p)
                if  reg_p2p.fitness > .15 and len(reg_p2p.correspondence_set)>250:
                    last_forward_transform = reg_p2p.transformation
                else:
                    print("bad fit")
                    #forward_pcd.transform(last_forward_transform)

        forward_pcd.transform(last_forward_transform)

        return forward_pcd,rgbd,intrinsic
    
    def segmentClusters(self,pcd, applyColor=True):
        labels = np.array(pcd.cluster_dbscan(eps=0.1, min_points=int(2.**voxel_detail)))

        try:
            max_label = labels.max()    
        except ValueError:
            max_label = 0
        #print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        #colors[labels < 1] = 0
        if applyColor:
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        return labels

    
    def get3dFrame(self):
        topdown_pcd, topdown_rgbd, topdown_intrinsic = self.getTopwdownPointCloud()
        
        if self.with_forward:
            forward_pcd,forward_rgbd, forward_intrinsic = self.getForwardPointCloud()

        #cut the bot self cloud out before merging
        topdown_pcd, boxbot_pcd, boxbot_mesh = BotExtractor.extract(topdown_pcd)

        if self.with_forward:
            topdown_pcd.points.extend(forward_pcd.points)
            topdown_pcd.colors.extend(forward_pcd.colors)            
            topdown_pcd.normals.extend(forward_pcd.normals)

        # because topdown 0,0 is the 'origin' of robot, we set 
        # the roi in worldspace rather than pcd space
        #above_ground_bbox = topdown_pcd.get_axis_aligned_bounding_box()
        above_ground_bbox = Util3d.bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,floor_z,tdz2,(1, 0, 0))
        below_ground_bbox = Util3d.bbox_from_xxyyzz(tdx1,tdx2,tdy1,tdy2,floor_z,floor_z-.5,(0, 1, 0))
        
        # get rid of the floor
        if self.with_shadow_ground:
            ground_pcd=topdown_pcd.crop(below_ground_bbox)

        if self.crop_floor:
            topdown_pcd=topdown_pcd.crop(above_ground_bbox)

        if self.show_pointcloud:
            self.pcd.points = topdown_pcd.points
            self.pcd.colors = topdown_pcd.colors
            self.pcd.normals = topdown_pcd.normals

        if self.with_voxels:
            vxgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pcd,voxel_size=self.voxel_size)

        if self.with_clusters:
            clusters = self.segmentClusters(self.pcd)

        return self.pcd

        


if __name__ == "__main__":
    frame_count = 0
    a7vis = AnglerDroidCameras(rsTopdownSerial="815412070676", rsForwardSerial="815412070180")
    try:
        while True:

            dt0=datetime.now()
            a7vis.get3dFrame()

            
            
        # draw 
            if frame_count == 0:

                vis.add_geometry(above_ground_bbox) #everything

                if show_floor_basis_points:
                    vis.add_geometry(floor_basis_points_pcd)

                if show_topdown_roi_box:
                    c=-.48
                    above_ground_bbox_topdown = Util3d.bbox_from_xxyyzz(tdx1-c,tdx2+c,tdy1-c,tdy2+c,floor_z,tdz2,(1, 0, 0))
                    vis.add_geometry(above_ground_bbox_topdown) #everything

                if show_boxbot:
                    vis.add_geometry(boxbot_mesh) #robot

                if show_axis:
                    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
                    vis.add_geometry(mesh_frame)
                    mesh_frame1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[1, 0, 0])
                    vis.add_geometry(mesh_frame1)

                if with_voxels:
                    vxgrid_old = vxgrid
                    vis.add_geometry(vxgrid_old)
                
                if show_pointcloud:
                    vis.add_geometry(pcd)
                
                Util3d.topview(vis)

            
            
            if show_topdown_roi_box:
                # in case we are dynamically adjusting
                vis.update_geometry(above_ground_bbox)
            
            if show_floor_basis_points:
                vis.update_geometry(floor_basis_points_pcd)
            
            if with_voxels:
                vis.remove_geometry(vxgrid_old,False)
                vis.add_geometry(vxgrid,False)
                vxgrid_old = vxgrid
                
            if show_pointcloud:
                vis.update_geometry(pcd)

            vis.poll_events()
            
#2d render
            
            vis.update_renderer()
            
                
            floor_depth = np.array(vis.capture_depth_float_buffer()).astype(np.uint8)
            
            #floor_depth = cv2.resize(floor_depth, floor_depth_size, interpolation= cv2.INTER_LINEAR)
            
            #floor_depth = cv2.cvtColor(floor_depth, cv2.COLOR_BGR2GRAY)
            ret, floor_depth = cv2.threshold(floor_depth, 0, 255, cv2.THRESH_BINARY)
            #floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, morph_kernel)
            #floor_depth = np.where(floor_depth>0, 255, 0).astype(np.uint8).copy()

            #floor_depth_save = floor_depth
            #cv2.imshow('floor depth save', floor_depth_save)

            if with_shadow_ground:
                #recalculate every 4 frames
                if frame_count % 4 == 0:
                    vis.add_geometry(ground_pcd,False)

                    shadow_ground_depth = np.array(vis.capture_depth_float_buffer(True)).astype(np.uint8)
                    vis.remove_geometry(ground_pcd,False)
                    #shadow_ground_depth = cv2.resize(shadow_ground_depth, floor_depth_size, interpolation=cv2.INTER_LINEAR)
                    shadow_ground_depth = cv2.morphologyEx(shadow_ground_depth, cv2.MORPH_CLOSE, morph_kernel)
                    ret, shadow_ground_depth = cv2.threshold(shadow_ground_depth, 0, 255, cv2.THRESH_BINARY_INV)
                    floor_acc = shadow_ground_depth
                    #floor_acc = np.where(shadow_ground_depth>0, 0, 255).astype(np.uint8).copy()
                    
                    #cover the tail shadow
                    cv2.rectangle(floor_acc, (100,276), (240-100,423), (0,0,0), 1)
                #print(floor_acc,floor_depth)
                floor_depth=cv2.bitwise_or(floor_acc,floor_depth)
                #floor_depth = cv2.morphologyEx(floor_depth, cv2.MORPH_CLOSE, morph_kernel)
            
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