import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor


class HistogramCalculator:
    """A class-based approach to parallel histogram computation."""
    
    def __init__(self, data, low, high, n_bins, n_threads):
        self.data = data
        self.low = low
        self.high = high
        self.n_bins = n_bins
        self.n_threads = n_threads
        self.bin_width = (high - low) / n_bins
        self.global_counts = [0] * n_bins
        self.mutex = threading.RLock()  # Using RLock instead of Lock
        
    def _process_data_chunk(self, thread_index):
        """Process a specific chunk of data assigned to this thread."""
        # Calculate which portion of data this thread handles
        data_size = len(self.data)
        chunk_size = data_size // self.n_threads
        
        start_position = thread_index * chunk_size
        if thread_index == self.n_threads - 1:
            # Last thread takes all remaining data
            end_position = data_size
        else:
            end_position = start_position + chunk_size
            
        # Extract the chunk for this thread
        data_chunk = self.data[start_position:end_position]
        
        # Create thread-local histogram
        thread_local_histogram = [0] * self.n_bins
        
        # Process each value in the chunk
        for val in data_chunk:
            if self.low <= val < self.high:
                # Determine which bin this value belongs to
                bin_index = int((val - self.low) / self.bin_width)
                # Handle boundary case
                if bin_index >= self.n_bins:
                    bin_index = self.n_bins - 1
                thread_local_histogram[bin_index] += 1
        
        # Safely merge with global histogram
        self._merge_results(thread_local_histogram)
    
    def _merge_results(self, local_histogram):
        """Thread-safe merging of local results into global histogram."""
        with self.mutex:
            for idx in range(self.n_bins):
                self.global_counts[idx] += local_histogram[idx]
    
    def compute(self):
        """Execute the parallel histogram computation."""
        # Use ThreadPoolExecutor for a different threading approach
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            # Submit tasks for each thread
            futures = [executor.submit(self._process_data_chunk, i) 
                      for i in range(self.n_threads)]
            
            # Wait for all threads to complete
            for future in futures:
                future.result()
        
        return self.global_counts


def parallel_histogram(data, low, high, n_bins, n_threads):
    """
    Multi-threaded histogram computation with object-oriented design.
    
    Arguments:
        data: input array of numerical values
        low: minimum value for histogram range
        high: maximum value for histogram range  
        n_bins: number of histogram bins
        n_threads: number of worker threads
    
    Returns:
        List containing count of elements in each bin
    """
    calculator = HistogramCalculator(data, low, high, n_bins, n_threads)
    return calculator.compute()


def serial_histogram(data, low, high, n_bins):
    """
    Single-threaded histogram implementation for performance comparison.
    """
    histogram_counts = [0] * n_bins
    bin_size = (high - low) / n_bins
    
    for value in data:
        if low <= value < high:
            bin_idx = int((value - low) / bin_size)
            # Edge case handling
            bin_idx = min(bin_idx, n_bins - 1)
            histogram_counts[bin_idx] += 1
    
    return histogram_counts


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