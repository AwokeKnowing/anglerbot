import numpy as np
import matplotlib.pyplot as plt

def calculate_grid_cell_means(points, num_grid_cells=8, num_bins=10):
    # 1. Grid Setup
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    cell_width = (x_max - x_min) / num_grid_cells
    cell_height = (y_max - y_min) / num_grid_cells

    # 2. Point Assignment to Grid Cells
    grid_assignments = np.floor((points[:, :2] - np.array([x_min, y_min])) /
                                np.array([cell_width, cell_height])).astype(int)

    # 3. Histograms per Cell
    grid_cell_means = np.zeros((num_grid_cells, num_grid_cells))

    for cell_x in range(num_grid_cells):
        for cell_y in range(num_grid_cells):
            points_in_cell = points[np.where((grid_assignments[:, 0] == cell_x) &
                                             (grid_assignments[:, 1] == cell_y))]
            if points_in_cell.size > 0:
                z_values = points_in_cell[:, 2]
                hist, bin_edges = np.histogram(z_values, bins=num_bins)
                highest_bin_idx = np.argmax(hist)
                mean_of_highest_bin = np.mean(z_values[(bin_edges[highest_bin_idx] <= z_values) &
                                                       (z_values < bin_edges[highest_bin_idx + 1])])
                grid_cell_means[cell_x, cell_y] = mean_of_highest_bin

    return grid_cell_means


import random
# Generate sample points
num_points = 1000
points = np.random.rand(num_points, 3)  # Points with coordinates in the range [0, 1)
for i in range(1000):

    # Generate random x and y values
    x = points[i][0]
    y = points[i][1]

    # Calculate z with the gradient and randomness
    z = x * 10 + y * 5 + random.random() * 3  # Adjust the '3' for randomness
    points[i][2]=z

# Calculate and visualize
grid_cell_means = calculate_grid_cell_means(points)
print(grid_cell_means)



# Plotting
fig, ax = plt.subplots()

# Scatter plot of points
ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], s=15, cmap='viridis')

# Markers for cell means
x_coords, y_coords = np.meshgrid(np.arange(grid_cell_means.shape[1]),
                                 np.arange(grid_cell_means.shape[0]))

x_coords = x_coords.astype(float)
y_coords = y_coords.astype(float)
x_coords += 0.5  # Center markers within cells
y_coords += 0.5

ax.scatter(x_coords, y_coords, c=grid_cell_means, s=50, marker='x', cmap='plasma')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Points and Grid Cell Representative Z-Values')
plt.show()
