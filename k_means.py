import numpy as np

# Step 1: Set k = 3
k = 3

# Step 2: Hard-code the dataset
data_points = np.array([[0.22, 0.33], [0.45, 0.76], [0.73, 0.39], [0.25, 0.35], [0.51, 0.69], [0.69, 0.42],
                       [0.41, 0.49], [0.15, 0.29], [0.81, 0.32], [0.50, 0.88], [0.23, 0.31], [0.77, 0.30],
                       [0.56, 0.75], [0.11, 0.38], [0.81, 0.33], [0.59, 0.77], [0.10, 0.89], [0.55, 0.09],
                       [0.75, 0.35], [0.44, 0.55]])

# Step 3: Read in the initial cluster centers from user input
initial_centers = []
for i in range(k):
    x = float(input("Enter x-coordinate for cluster center {}: ".format(i + 1)))
    y = float(input("Enter y-coordinate for cluster center {}: ".format(i + 1)))
    initial_centers.append([x, y])

initial_centers = np.array(initial_centers)

# Step 4: Perform one execution of the k-means algorithm
new_cluster_centers = []
new_cluster_assignments = np.argmin(np.sum((data_points[:,np.newaxis] - initial_centers)**2, axis=-1), axis=1)
for i in range(k):
    cluster_points = data_points[new_cluster_assignments==i]
    new_cluster_centers.append(np.mean(cluster_points, axis=0))

new_cluster_centers = np.array(new_cluster_centers)

# Step 5: Compute the sum-of-squares error with respect to the initial cluster centers
initial_error = np.sum(np.min(np.sum((data_points[:, np.newaxis] - initial_centers) ** 2, axis=-1), axis=1))

# Step 6: Compute the sum-of-squares error with respect to the new cluster centers

new_error = np.sum(np.min(np.sum((data_points[:,np.newaxis] - new_cluster_centers)**2, axis=-1), axis=1))
# Step 7: Output the sum-of-squares errors
initial_error = round(initial_error, 4)
new_error = round(new_error, 4)
print("Initial Error:", initial_error)
print("New Error:", new_error)
