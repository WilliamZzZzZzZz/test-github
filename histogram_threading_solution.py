import threading
import time
import numpy as np

def worker(thread_id, data, low, high, n_bins, n_threads, results, lock):
    """
    Worker function for each thread.
    Each thread processes a chunk of data and updates the shared results array.
    """
    # Calculate the chunk of data this thread should process
    chunk_size = len(data) // n_threads
    start_idx = thread_id * chunk_size
    
    # Last thread handles any remaining data
    if thread_id == n_threads - 1:
        end_idx = len(data)
    else:
        end_idx = start_idx + chunk_size
    
    # Process this thread's chunk of data
    chunk = data[start_idx:end_idx]
    
    # Calculate bin width
    bin_width = (high - low) / n_bins
    
    # Local histogram for this thread
    local_hist = [0] * n_bins
    
    # Count elements in each bin for this chunk
    for value in chunk:
        if low <= value < high:
            bin_idx = int((value - low) / bin_width)
            # Handle edge case where value equals high
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            local_hist[bin_idx] += 1
    
    # Update shared results with thread safety
    with lock:
        for i in range(n_bins):
            results[i] += local_hist[i]

def parallel_histogram(data, low, high, n_bins, n_threads):
    """
    Compute histogram using multiple threads.
    
    Args:
        data: an array of numbers from which the histogram is built
        low: the lower range of the bins
        high: the upper range of the bins
        n_bins: number of bins used
        n_threads: number of threads to use
    
    Returns:
        An array that contains counts of elements in each bin
    """
    results = [0] * n_bins
    lock = threading.Lock()
    threads = []

    for i in range(n_threads):
        t = threading.Thread(target=worker, args=(i, data, low, high, n_bins, n_threads, results, lock))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results

def serial_histogram(data, low, high, n_bins):
    """
    Serial implementation for comparison.
    """
    results = [0] * n_bins
    bin_width = (high - low) / n_bins
    
    for value in data:
        if low <= value < high:
            bin_idx = int((value - low) / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            results[bin_idx] += 1
    
    return results

# DO NOT MODIFY ANYTHING BELOW THIS POINT IN YOUR SUBMITTED CODE
if __name__ == "__main__":
    test_data = np.random.rand(100000)
    n_threads = 8
    low = 0.0
    high = 1.0
    n_bins = 50

    # Test parallel implementation
    start_time = time.time()
    hist_parallel = parallel_histogram(test_data, low, high, n_bins, n_threads)
    parallel_time = time.time() - start_time

    # Test serial implementation
    start_time = time.time()
    hist_serial = serial_histogram(test_data, low, high, n_bins)
    serial_time = time.time() - start_time

    print(f"Parallel Histogram: {hist_parallel}")
    print(f"Serial Histogram: {hist_serial}")
    print(f"Results match: {hist_parallel == hist_serial}")
    print(f"Parallel time: {parallel_time:.4f}s")
    print(f"Serial time: {serial_time:.4f}s")
    print(f"Speedup: {serial_time / parallel_time:.2f}x")