from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import time

def calculate_grid_cell_means(points, num_cells_x=16,num_cells_y=16, decimate=1,num_bins=10):
    # 1. Grid Setup
    points=points[::decimate]
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    cell_width = (x_max - x_min) / num_cells_x
    cell_height = (y_max - y_min) / num_cells_y

    # 2. Point Assignment to Grid Cells
    grid_assignments = np.floor((points[:, :2] - np.array([x_min, y_min])) /
                                np.array([cell_width, cell_height])).astype(int)

    # 3. Histograms per Cell
    grid_cell_means = np.zeros((num_cells_y, num_cells_x))

    for cell_x in range(num_cells_x):
        for cell_y in range(num_cells_y):
            #print("grid",time.time()-tm,flush=True)
            points_in_cell = points[np.where((grid_assignments[:, 0] == cell_x) &
                                             (grid_assignments[:, 1] == cell_y))]
            if points_in_cell.size > 0:
                z_values = points_in_cell[:, 2]
                mean_of_highest_bin=np.mean(z_values)
                if False:
                    
                    #print("grid hb",time.time()-tm,flush=True)
                    hist, bin_edges = np.histogram(z_values, bins=num_bins)
                    #print("grid ha",time.time()-tm,flush=True)
                    highest_bin_idx = np.argmax(hist)
                    mean_of_highest_bin = np.mean(z_values[(bin_edges[highest_bin_idx] <= z_values) &
                                                        (z_values < bin_edges[highest_bin_idx + 1])])
                grid_cell_means[cell_y, cell_x] = mean_of_highest_bin

    return grid_cell_means


def calculate_grid_cell_means2(points, num_cells_x=16, num_cells_y=16, decimate=1, num_bins=10):
    # 1. Grid Setup
    points = points[::decimate]
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    cell_width = (x_max - x_min) / num_cells_x
    cell_height = (y_max - y_min) / num_cells_y

    # 2. Pre-calculate Global Histogram
    hist, bin_edges = np.histogramdd(points[:, :3], bins=num_bins, range=((x_min, x_max), (y_min, y_max), (points[:,2].min(), points[:,2].max())))

    # 3. Grid Assignments (Vectorized)
    grid_assignments = np.floor((points[:, :2] - np.array([x_min, y_min])) /
                                np.array([cell_width, cell_height])).astype(int)

    # 4. Unique Cells
    unique_cells = np.unique(grid_assignments, axis=0)  

    # 5. Iteration and Calculation
    grid_cell_means = np.zeros((num_cells_y, num_cells_x))
    for cell_x, cell_y in unique_cells:  # Only iterate over necessary cells
        points_in_cell = points[np.where((grid_assignments[:, 0] == cell_x) &
                                         (grid_assignments[:, 1] == cell_y))]
        if points_in_cell.size > 0:
            # Map z_values to bin number 
           bin_idx = np.digitize(points_in_cell[:, 2], bin_edges[2]) - 1 

           # Most frequent bin with points in it
           highest_bin = np.argmax(np.bincount(bin_idx))  

           # Calculate mean
           mean_of_highest_bin = np.mean(points_in_cell[:, 2][ bin_idx == highest_bin])
           grid_cell_means[cell_y, cell_x] = mean_of_highest_bin

    return grid_cell_means

def fit_plane(grid_cell_means):

    # Markers for cell means
    x_coords, y_coords = np.meshgrid(np.arange(grid_cell_means.shape[1]),
                                    np.arange(grid_cell_means.shape[0]))

    x_coords = x_coords.astype(float)
    y_coords = y_coords.astype(float)
    x_coords += 0.5  # Center markers within cells
    y_coords += 0.5

    sy,sx=grid_cell_means.shape
    sX1 = x_coords * (x/sx) #scale back out eg 16x16 to 256x256
    sX2 = y_coords * (y/sy)
    sY = grid_cell_means
    
    #Regression
    X = np.hstack(  ( np.reshape(sX1, (sx*sy, 1)) , np.reshape(sX2, (sx*sy, 1)) )  )
    X = np.hstack(   ( np.ones((sx*sy, 1)) , X ))
    YY = np.reshape(sY, (sx*sy, 1))

    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)
    #plane = np.reshape(np.dot(X, theta), (sy, sx))
    
    return theta#, plane


def subtract_plane(points,theta):
    Xorig = np.hstack(  ( np.reshape(points[:,0], (len(points), 1)) , np.reshape(points[:,1], (len(points), 1)) )  )
    Xorig = np.hstack(  ( np.ones((len(points), 1)) , Xorig ))
    planeorig = np.dot(Xorig, theta)
    #Subtraction
    Y_sub = points[:,2][:,None] - planeorig

    return Y_sub


def get_points_grid(x=128,y=128,F=3,tilt_a=.005,tilt_b=.002):
    X1, X2 = np.mgrid[:x, :y]
         
    i = np.minimum(X1, x-X1-1)
    j = np.minimum(X2, y-X2-1)
    H = np.exp(-.5*(np.power(i, 2)  +  np.power(j, 2)   )/(F*F))
    Y = np.real(  np.fft.ifft2   (H  *  np.fft.fft2(  np.random.randn(x, y))))
    Y = Y + (tilt_a*X1 + tilt_b*X2); #adding the plane
    Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y)) #data scaling
    return np.column_stack((X1.flatten(), X2.flatten(), Y.flatten()))

x=108
y=60

pointsa=[]
for i in range(1000):
    points = get_points_grid(x,y)
    pointsa.append(points)



for i in range(1000):
    tm= time.time()

    grid_cell_means = calculate_grid_cell_means(pointsa[i],4,3,1)
    #print(time.time()-tm,flush=True)

    theta = fit_plane(grid_cell_means)
    #print(time.time()-tm,flush=True)

    Y_sub = subtract_plane(pointsa[i],theta)
    fps=1.0/(time.time()-tm)
    print(f"total {fps:.2f}" )






fig = plt.figure()
jet = plt.get_cmap('jet')

ax = fig.add_subplot(3,1,1, projection='3d')
ax.scatter(points[:,0],points[:,1],points[:,2], c=points[:,2], s=1,  cmap='plasma')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Points and Grid Cell Representative Z-Values')

#ax = fig.add_subplot(3,1,2, projection='3d')
#ax.plot_surface(sX1,sX2,plane)
#ax.plot_surface(sX1,sX2,sY, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)

ax = fig.add_subplot(3,1,3, projection='3d')
ax.scatter(points[:,0],points[:,1],Y_sub,       c=Y_sub,       s=1,  cmap='plasma')
ax.set_xlabel('X')
ax.set_ylabel('Y')

plt.show()
