import numpy as np
import cv2


class DynamicPathfinder:
    def __init__(self, axle_x, axle_y, cm_between_wheels=10, px_per_cm=1):
        self.axle_x = axle_x
        self.axle_y = axle_y
        self.cm_between_wheels = cm_between_wheels
        self.px_per_cm = px_per_cm

        self.current_left_vel = 0
        self.current_right_vel = 0


    def get_distance_factor(self, desired_length, radius):
        circumference = 2 * np.pi * radius
        distance_factor = desired_length / circumference
        return distance_factor
    

    def calculate_radius(self, distance_between_wheels, velocity_left_wheel, velocity_right_wheel):
        # Calculate the difference in wheel velocities
        velocity_difference = velocity_right_wheel - velocity_left_wheel

        # Calculate the turning radius based on the velocity ratio
        if velocity_difference == 0:
            radius = 2000000000 #go straight
        else:
            radius = distance_between_wheels / (2 * velocity_difference)
        return radius
    

    def select_path(self, image, vels, distance_between_wheels,dist_factor):
        #distance = 6  # in cm

        image_sum=np.sum(image)
        n=0
        for vel in vels:
            img=image.copy()

            
            v1 = vel[0]/100 * dist_factor
            v2 = vel[1]/100 * dist_factor
            distance = v1+v2/2
            r = self.calculate_radius(distance_between_wheels, v1,v2)
            
            distance_factor = self.get_distance_factor(distance, r)
            theta = np.linspace(0, 2 * np.pi * distance_factor, 8)
            x = r * np.cos(theta) - r  # Offset the x-values by -r
            y = r * np.sin(theta)
            
            # Convert to pixel coordinates (scaling factor of 20 for visualization)
            pixel_x = (x * 20 + 124).astype(int)
            pixel_y = (-y * 20 + 209).astype(int)

            r=68
            # Plot points on the image
            for px, py in zip(pixel_x, pixel_y):
                cv2.circle(img, (px,py), 68, (0, 0,0), -1)

            if np.sum(img) != image_sum:
                for px, py in zip(pixel_x, pixel_y):
                    cv2.circle(img, (px,py), 3, (0, 0,255), -1)
                cv2.circle(img, (pixel_x[-1],pixel_y[-1]), 68, (0, 0,255), 1)
                print("hit")
            else:
                for px, py in zip(pixel_x, pixel_y):
                    cv2.circle(img, (px,py), 3, (0, 255,0), -1)
                cv2.circle(img, (pixel_x[-1],pixel_y[-1]), 68, (0, 255,0), 1)
                print("clear")

            cv2.imshow("Turning Radii "+str(0), img)
            n+=1

            cv2.waitKey(100)
        cv2.destroyAllWindows()

        #TODO SCORE BY DISTANCE (AND FACTOR) TO GOAL
        return (0,0)



    def draw_bot2d(self, image, axle_x, axle_y, color=(255,255,255), show_circle=False):
        rx = axle_x
        ry = axle_y

        cv2.rectangle(image, (rx-64, ry-29),(rx+64,ry+26),  color, -1)
        cv2.rectangle(image, (rx-53, ry-32),(rx+53,ry+50),  color, -1)
        cv2.rectangle(image, (rx-45, ry-39),(rx+45,ry+73),  color, -1)

        if show_circle:
            cv2.circle(image, (rx,ry),68,  (255, 255, 255), 1)


    def get_possible_wheel_vels(self, current_v1,current_v2,steps=5,step_by=10,spin_penalty=.7,max=100,min=-100):

        vels=[]
        
        i = -steps*step_by
        while i <= steps*step_by:
            j = -steps*step_by
            while j <= steps*step_by:
                vh1,vh2=current_v1+i,current_v2+j
                if min <= vh1 <= max and min <= vh2 <= max:
                    vels.append((vh1,vh2))
                j += step_by
            i += step_by

        vels=sorted(vels, key=lambda v: -(v[0]+v[1]-abs(v[0]-v[1])*spin_penalty))

        for v in vels:
            print(v,-(v[0]+v[1]-abs(v[0]-v[1])*spin_penalty))

        return vels
    
    
    def next_wheel_vels(self, image=None, from_vels=None, steps=2, axle_x=None, axle_y=None):
        if image is not None:  self.image = image
        if axle_x is not None: self.axle_x = axle_x
        if axle_y is not None: self.axle_y = axle_y
        if from_vels is not None:  self.current_vels = from_vels

        self.draw_bot2d(image, self.axle_x, self.axle_y,(0,0,0))

        vels = self.get_possible_wheel_vels(self.current_vels[0], self.current_vels[1], steps)
        
        distance_lookahead_factor = 4.5

        left_vel, right_vel = self.select_path(image, vels, self.cm_between_wheels, distance_lookahead_factor)

        return (left_vel, right_vel)


# Create an OpenCV image of size 424x240
image = cv2.imread('floorexample.png')
botmask = cv2.imread('botmask.png')

rx = 124
ry = 209
wheel_space = 10  # in cm
px_per_cm = 1

pathfinder = DynamicPathfinder(rx, ry, wheel_space, px_per_cm)
pathfinder.draw_bot2d(image, rx, ry, (255,255,255))

cv2.imshow("mask",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

pathfinder.next_wheel_vels(image, (30, 30)) 

for i in range(20):
    vels =  pathfinder.next_wheel_vels(image, (30, 30+i))
