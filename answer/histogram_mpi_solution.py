from mpi4py import MPI
import numpy as np


def distribute_data_evenly(data_array, num_processes):
    """
    Divide data array into approximately equal chunks for MPI processes.
    
    Args:
        data_array: Input data to be distributed
        num_processes: Number of MPI processes
        
    Returns:
        List of data chunks
    """
    total_elements = len(data_array)
    base_chunk_size = total_elements // num_processes
    extra_elements = total_elements % num_processes
    
    data_chunks = []
    current_index = 0
    
    for process_id in range(num_processes):
        # Some processes get an extra element if total doesn't divide evenly
        current_chunk_size = base_chunk_size + (1 if process_id < extra_elements else 0)
        
        chunk = data_array[current_index:current_index + current_chunk_size]
        data_chunks.append(chunk)
        current_index += current_chunk_size
    
    return data_chunks


def compute_local_histogram_bins(local_data, bin_min, bin_max, num_bins):
    """
    Calculate histogram for local data chunk using manual binning.
    
    Args:
        local_data: Data chunk assigned to this process
        bin_min: Lower bound of histogram range
        bin_max: Upper bound of histogram range
        num_bins: Number of histogram bins
        
    Returns:
        Array of bin counts
    """
    bin_counts = np.zeros(num_bins, dtype=int)
    bin_width = (bin_max - bin_min) / num_bins
    
    for data_point in local_data:
        if bin_min <= data_point < bin_max:
            # Calculate which bin this data point belongs to
            bin_index = int((data_point - bin_min) / bin_width)
            
            # Handle edge case where data_point equals bin_max
            if bin_index >= num_bins:
                bin_index = num_bins - 1
                
            bin_counts[bin_index] += 1
    
    return bin_counts


def parallel_histogram(data, low, high, n_bins):
    """
    MPI-based parallel histogram computation with custom data distribution.
    
    Args:
        data: Input data array (only valid on root process)
        low: Minimum value for histogram range
        high: Maximum value for histogram range
        n_bins: Number of histogram bins
        
    Returns:
        Complete histogram (only on root process), None elsewhere
    """
    # Get MPI communicator and process information
    mpi_comm = MPI.COMM_WORLD
    current_rank = mpi_comm.Get_rank()
    total_processes = mpi_comm.Get_size()
    
    # Root process prepares and distributes data
    if current_rank == 0:
        # Custom data distribution
        distributed_chunks = distribute_data_evenly(data, total_processes)
    else:
        distributed_chunks = None
    
    # Each process receives its data chunk
    my_data_chunk = mpi_comm.scatter(distributed_chunks, root=0)
    
    # Compute local histogram using custom binning
    local_histogram = compute_local_histogram_bins(my_data_chunk, low, high, n_bins)
    
    # Collect all local histograms at root process
    collected_histograms = mpi_comm.gather(local_histogram, root=0)
    
    # Root process combines all local histograms
    if current_rank == 0:
        final_histogram = np.zeros(n_bins, dtype=int)
        for hist in collected_histograms:
            final_histogram += hist
        return final_histogram
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