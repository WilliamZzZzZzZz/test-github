import numpy as np

def distance_pure_python(data):
    """
    Compute distance matrix using nested for loops.
    
    Args:
        data: n x d numpy array of Cartesian coordinates
    
    Returns:
        n x n distance matrix where D[i,j] is Euclidean distance between point i and j
    """
    n, d = data.shape
    dist_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Compute Euclidean distance between point i and j
            dist_mat[i, j] = np.sqrt(np.sum((data[i] - data[j]) ** 2))
    
    return dist_mat


def distance_numpy(data):
    """
    Compute distance matrix using only numpy functions without loops.
    
    Args:
        data: n x d numpy array of Cartesian coordinates
    
    Returns:
        n x n distance matrix where D[i,j] is Euclidean distance between point i and j
    """
    # Broadcasting approach: compute all pairwise differences at once
    # data[:, None, :] has shape (n, 1, d)
    # data[None, :, :] has shape (1, n, d)
    # Their difference has shape (n, n, d)
    diff = data[:, None, :] - data[None, :, :]
    
    # Compute squared distances and take square root
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=2))
    
    return dist_mat

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