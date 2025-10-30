import numpy as np

def distance_pure_python(data):
    """
    Calculate pairwise Euclidean distances using basic Python loops.
    
    Parameters:
        data: numpy array of shape (n, d) containing n points in d dimensions
        
    Returns:
        numpy array of shape (n, n) containing distance matrix
    """
    rows, cols = data.shape
    distance_matrix = np.empty((rows, rows))
    
    # Iterate through each pair of points
    for point_i in range(rows):
        for point_j in range(rows):
            # Calculate squared differences for each dimension
            sum_of_squares = 0.0
            for dim in range(cols):
                diff = data[point_i, dim] - data[point_j, dim]
                sum_of_squares += diff * diff
            
            # Store the Euclidean distance
            distance_matrix[point_i, point_j] = sum_of_squares ** 0.5
    
    return distance_matrix

def distance_numpy(data):
    """
    Vectorized distance matrix computation using NumPy operations.
    
    Parameters:
        data: numpy array of shape (n, d) representing point coordinates
        
    Returns:
        Distance matrix as numpy array of shape (n, n)
    """
    # Method: expand dimensions and use broadcasting
    points_expanded = np.expand_dims(data, axis=1)  # Shape: (n, 1, d)
    points_transposed = np.expand_dims(data, axis=0)  # Shape: (1, n, d)
    
    # Calculate element-wise differences
    differences = points_expanded - points_transposed  # Shape: (n, n, d)
    
    # Square the differences and sum along the last axis
    squared_diffs = np.square(differences)
    sum_squared = np.sum(squared_diffs, axis=-1)
    
    # Take square root to get final distances
    distance_matrix = np.sqrt(sum_squared)
    
    return distance_matrix

# DO NOT MODIFY ANYTHING BELOW THIS POINT IN YOUR SUBMITTED CODE
def main():
    rng = np.random.default_rng()
    n, d = 500, 500

    data = rng.random((n, d))

    dist_mat_pure_python = distance_pure_python(data)
    dist_mat_numpy = distance_numpy(data)

    print(f"Are the two results the same?: {np.max(np.abs(dist_mat_pure_python - dist_mat_numpy)) < 1e-5}")

if __name__ == "__main__":
    main()