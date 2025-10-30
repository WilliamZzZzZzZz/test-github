from mpi4py import MPI
import numpy as np

def parallel_histogram(data, low, high, n_bins):
    """
    Compute histogram using MPI processes.
    
    Args:
        data: an array of numbers (only on rank 0)
        low: the lower range of the bins
        high: the upper range of the bins
        n_bins: number of bins used
    
    Returns:
        An array that contains counts of elements in each bin (only on rank 0)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    
    if rank == 0:
        # Split data into chunks for each process
        chunk_size = len(data) // size
        remainder = len(data) % size
        chunks = []
        start = 0
        for i in range(size):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(data[start:end])
            start = end
    else:
        chunks = None
    
    # Scatter chunks to all processes
    local_data = comm.scatter(chunks, root=0)
    
    # Compute local histogram
    local_hist, _ = np.histogram(local_data, bins=n_bins, range=(low, high))
    
    # Reduce results to rank 0
    global_hist = comm.reduce(local_hist, op=MPI.SUM, root=0)
    
    # Only rank 0 returns the result, others return None
    if rank == 0:
        return global_hist
    else:
        return None

# DO NOT MODIFY ANYTHING BELOW THIS POINT IN YOUR SUBMITTED CODE
def main():
    """
    Compute histogram using MPI
    Run with: mpiexec -n <num_processes> python histogram_mpi.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        test_data = np.random.rand(100000)
        low = 0.0
        high = 1.0
        n_bins = 50

        hist = parallel_histogram(test_data, low, high, n_bins)
        print(f"Histogram: {hist}")
    else:
        # Non-root processes also need to call parallel_histogram
        # but they pass None as data
        parallel_histogram(None, 0.0, 1.0, 50)

if __name__ == "__main__":
    main()