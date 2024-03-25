import numpy as np
import matplotlib.pyplot as plt

def get_distance_factor(desired_length, radius):
    circumference = 2 * np.pi * radius
    distance_factor = desired_length / circumference
    return distance_factor

def calculate_radius(distance_between_wheels, velocity_left_wheel, velocity_right_wheel):
    # Calculate the difference in wheel velocities
    velocity_difference = velocity_right_wheel - velocity_left_wheel

    # Calculate the turning radius based on the velocity ratio
    if velocity_difference == 0:
        radius = np.inf
    else:
        radius = distance_between_wheels / (2 * velocity_difference)
    return radius

def draw_segment(radius):
    desired_segment_length = 3  # in cm

    if isinstance(radius, (int, float)):
        radius = [radius]

    for r in radius:
        distance_factor = get_distance_factor(desired_segment_length, r)
        theta = np.linspace(0, 2 * np.pi * distance_factor, 100)
        x = r * np.cos(theta) - r  # Offset the x-values by -r
        y = r * np.sin(theta)
        plt.plot(x, y, label=f"Turning Radius: {r:.2f} cm")

    plt.axis('equal')
    plt.legend()
    plt.show()

# Calculate the turning radius for given distance between wheels and wheel velocities
distance_between_wheels = 30  # in cm
velocity_left_wheel = 20-.0001       # in cm/s
velocity_right_wheels = np.linspace(20, 40, 10)     # Generate an array of velocities between 1 and 2

# Calculate radii for each right wheel velocity using a loop
radii = []
for velocity_right_wheel in velocity_right_wheels:
    radius = calculate_radius(distance_between_wheels, velocity_right_wheel,velocity_left_wheel)
    radii.append(radius)

print("Calculated Turning Radii:", radii, "cm")

# Plot the line segments with the calculated turning radii
draw_segment(radii)