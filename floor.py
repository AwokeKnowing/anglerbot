import numpy as np
import cv2

def get_distance_factor(desired_length, radius):
    circumference = 2 * np.pi * radius
    distance_factor = desired_length / circumference
    return distance_factor

def calculate_radius(distance_between_wheels, velocity_left_wheel, velocity_right_wheel):
    # Calculate the difference in wheel velocities
    velocity_difference = velocity_right_wheel - velocity_left_wheel

    # Calculate the turning radius based on the velocity ratio
    if velocity_difference == 0:
        radius = 2000000000 #go straight
    else:
        radius = distance_between_wheels / (2 * velocity_difference)
    return radius

def draw_paths(image, vels, distance_between_wheels,dist_factor):
    #distance = 6  # in cm

    image_sum=np.sum(image)
    n=0
    for vel in vels:
        img=image.copy()

        
        v1 = vel[0]/100 * dist_factor
        v2 = vel[1]/100 * dist_factor
        distance = v1+v2/2
        r = calculate_radius(distance_between_wheels, v1,v2)
        
        distance_factor = get_distance_factor(distance, r)
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
                cv2.circle(img, (px,py), 68, (0, 0,255), -1)
            print("hit")
        else:
            for px, py in zip(pixel_x, pixel_y):
                cv2.circle(img, (px,py), 68, (0, 255,0), -1)
            print("clear")

        cv2.imshow("Turning Radii "+str(0), img)
        n+=1

        cv2.waitKey(1000)
    cv2.destroyAllWindows()

# Create an OpenCV image of size 424x240
image = cv2.imread('floorexample.png')
botmask = cv2.imread('botmask.png')
rx=124
ry=209
cv2.rectangle(botmask, (50, 160),(200,294),  (255, 255, 255), 1)
cv2.circle(botmask, (rx,ry),68,  (255, 255, 255), 1)
#cv2.imcrop(image, (x1, y1, x2, y2))
cv2.imshow("mask",botmask)
cv2.waitKey(0)
cv2.destroyAllWindows()

def get_possible_wheel_vels(current_v1,current_v2,steps=5,step_by=10,spin_penalty=.7,max=100,min=-100):

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

    
    

vels = get_possible_wheel_vels(20,20,1)


#exit()
# Calculate the turning radius for given distance between wheels and wheel velocities
distance_between_wheels = 10  # in cm


# Calculate radii for each right wheel velocity using a loop
radii = []


#print("Calculated Turning Radii:", radii, "cm")

# Plot points on the OpenCV image with the calculated turning radii
cv2.rectangle(image, (50, 160),(200,294),  (0, 0, 0), -1)
draw_paths(image, vels,distance_between_wheels,4.5)
