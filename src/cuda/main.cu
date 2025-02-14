#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/gather.h>

//#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include "utils_gpu.cuh"
#include "btree_gpu.cuh"
#include "octree_gpu.cuh"

#define TIMER_START(start) cudaEventRecord(start);
#define TIMER_STOP(msg, start, stop) {                   \
    cudaEventRecord(stop);                               \
    cudaEventSynchronize(stop);                          \
    float ms;                                            \
    cudaEventElapsedTime(&ms, start, stop);              \
    std::cout << msg << ": " << ms << "ms" << std::endl; \
}

constexpr int NUM_POINTS = 2 << 17;

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    thrust::host_vector<float> h_x(NUM_POINTS);
    thrust::host_vector<float> h_y(NUM_POINTS);
    thrust::host_vector<float> h_z(NUM_POINTS);

    thrust::default_random_engine rng(100);
    thrust::uniform_real_distribution<float> dist;

    thrust::generate(h_x.begin(), h_x.end(), [&] { return dist(rng); });
    thrust::generate(h_y.begin(), h_y.end(), [&] { return dist(rng); });
    thrust::generate(h_z.begin(), h_z.end(), [&] { return dist(rng); });

    // Allocate device memory to store points coordinates
    thrust::device_vector<float> d_x(h_x);
    thrust::device_vector<float> d_y(h_y);
    thrust::device_vector<float> d_z(h_z);
    // Initializes points SoA and copy to device
    Points *h_points = new Points(
        thrust::raw_pointer_cast(&d_x[0]),
        thrust::raw_pointer_cast(&d_y[0]),
        thrust::raw_pointer_cast(&d_z[0]));
    Points *d_points = alloc_device_soa(h_points, sizeof(Points));

    // Allocate device memory to store morton codes
    thrust::device_vector<uint32_t> d_codes(NUM_POINTS);
    uint32_t *d_codes_ptr = thrust::raw_pointer_cast(&d_codes[0]);

    // Kernel launch to compute morton codes of points
    morton_encode<<<NUM_POINTS / THREADS_PER_BLOCK,
                    THREADS_PER_BLOCK>>>(d_points, d_codes_ptr);

    LessOp custom_op;
    // Determine sort tmp storage size
    void *d_sort_tmp = nullptr;
    size_t sort_tmp_size;
    cub::DeviceMergeSort::SortKeys(d_sort_tmp,
                                   sort_tmp_size,
                                   d_codes_ptr,
                                   NUM_POINTS,
                                   custom_op);
    // Allocate sort tmp storage
    cudaMalloc(&d_sort_tmp, sort_tmp_size);
    // Sorting
    TIMER_START(start)
    // Faster than DeviceRadixSort::SortKeys and thrust::sort
    cub::DeviceMergeSort::SortKeys(d_sort_tmp,
                                   sort_tmp_size,
                                   d_codes_ptr,
                                   NUM_POINTS,
                                   custom_op);
    TIMER_STOP("sort-codes", start, stop)
    cudaFree(d_sort_tmp);

    // Allocating Btree for NUM_POINTS number of leaves,
    // the actual number of leaves will be smaller
    Btree h_btree(NUM_POINTS);

    // Obtaining unique codes and counting occurrences
    // using run-length encoding
    thrust::device_vector<uint32_t> d_unique_codes(NUM_POINTS);
    thrust::device_vector<int> d_codes_occurrences(NUM_POINTS);
    uint32_t *d_unique_codes_ptr =
        thrust::raw_pointer_cast(&d_unique_codes[0]);
    int *d_codes_occurrences_ptr =
        thrust::raw_pointer_cast(&d_codes_occurrences[0]);

    // Btree device copy will store the actual number of leaves
    int *d_num_unique_codes = h_btree.get_dev_num_leaves_ptr();

    void *d_runlength_tmp = nullptr;
    size_t runlength_tmp_size;
    cub::DeviceRunLengthEncode::Encode(d_runlength_tmp,
                                       runlength_tmp_size,
                                       d_codes_ptr,
                                       d_unique_codes_ptr,
                                       d_codes_occurrences_ptr,
                                       d_num_unique_codes,
                                       NUM_POINTS);
    cudaMalloc(&d_runlength_tmp, runlength_tmp_size);

    TIMER_START(start)
    cub::DeviceRunLengthEncode::Encode(d_runlength_tmp,
                                       runlength_tmp_size,
                                       d_codes_ptr,
                                       d_unique_codes_ptr,
                                       d_codes_occurrences_ptr,
                                       d_num_unique_codes,
                                       NUM_POINTS);
    TIMER_STOP("run-length", start, stop)
    cudaFree(d_runlength_tmp);

    int h_num_unique_codes;
    cudaMemcpy(&h_num_unique_codes,
               d_num_unique_codes,
               sizeof(int),
               cudaMemcpyDeviceToHost);

    std::cout << "num_unique_codes=" << h_num_unique_codes << std::endl;

    // TODO: compute scan of d_codes_occurrences, x, y and z

    // TODO: is it correct?
    // Octree h_octree(ceil(log2(num_unique_points) / 3) + 1)
    Octree h_octree(7);

    TIMER_START(start)
    h_btree.build(d_unique_codes_ptr);
    TIMER_STOP("btree-build", start, stop)

    // WARNING: Perhaps sort octree instead?
    // Octree nodes are ~1/3 of the btree nodes,
    // sorting would be faster

    TIMER_START(start)
    h_btree.sort_to_bfs_order();
    TIMER_STOP("btree-sort", start, stop)

    TIMER_START(start)
    h_btree.compute_octree_map();
    TIMER_STOP("btree-scan", start, stop)

    TIMER_START(start)
    h_octree.build(h_btree);
    TIMER_STOP("octree-build", start, stop)

    //h_btree.print();
    //h_octree.print();

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    free(h_points);
    cudaFree(d_points);

    return 0;
}

