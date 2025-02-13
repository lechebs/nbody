#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/gather.h>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_merge_sort.cuh>

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
    // Kernel launch to compute morton codes of points
    morton_encode<<<NUM_POINTS / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        d_points, thrust::raw_pointer_cast(&d_codes[0]));

    LessOp custom_op;
    // Determine sort tmp storage size
    void *d_sort_tmp = nullptr;
    size_t sort_tmp_size;
    cub::DeviceMergeSort::SortKeys(d_sort_tmp,
                                   sort_tmp_size,
                                   thrust::raw_pointer_cast(&d_codes[0]),
                                   // thrust::raw_pointer_cast(&d_codes_sorted[0]),
                                   NUM_POINTS,
                                   custom_op);
    // Allocate sort tmp storage
    cudaMalloc(&d_sort_tmp, sort_tmp_size);
    // Sorting
    TIMER_START(start)
    // Faster than DeviceRadixSort::SortKeys and thrust::sort
    cub::DeviceMergeSort::SortKeys(d_sort_tmp,
                                   sort_tmp_size,
                                   thrust::raw_pointer_cast(&d_codes[0]),
                                   // thrust::raw_pointer_cast(&d_codes_sorted[0]),
                                   NUM_POINTS,
                                   custom_op);
    TIMER_STOP("sort-codes", start, stop)

    /*
    TIMER_START(start)
    thrust::sort(d_codes_sorted.begin(), d_codes_sorted.end());
    TIMER_STOP(start, stop)
    */

    auto unique_end = thrust::unique(d_codes.begin(), d_codes.end());
    int num_unique_points = unique_end - d_codes.begin();

    // std::cout << "num_unique_points=" << num_unique_points << std::endl;

    Btree h_btree(num_unique_points);
    // TODO: is it correct?
    // Octree h_octree(ceil(log2(num_unique_points) / 3) + 1)
    Octree h_octree(7);

    TIMER_START(start)
    h_btree.build(thrust::raw_pointer_cast(&d_codes[0]));
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

