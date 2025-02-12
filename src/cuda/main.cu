#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/gather.h>

#include "utils_gpu.cuh"
#include "btree_gpu.cuh"
#include "octree_gpu.cuh"

constexpr int NUM_POINTS = 2 << 17;

void print_bits(uint32_t u)
{
    for (int i = 0; i < 32; ++i) {
        printf("%d", (u >> (31 - i)) & 0x01);
    }
}

int main()
{
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

    // Sorting and removing duplicates
    thrust::sort(d_codes.begin(), d_codes.end());
    auto unique_end = thrust::unique(d_codes.begin(), d_codes.end());
    int num_unique_points = unique_end - d_codes.begin();

    std::cout << "num_unique_points=" << num_unique_points << std::endl;

    thrust::host_vector<uint32_t> h_unique_codes(d_codes);
    for (int i = 0; i < 32 - 1; ++i) {
        printf("%4d: %12u ", i, h_unique_codes[i]);
        print_bits(h_unique_codes[i]);
        printf("\n");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Btree h_btree(num_unique_points);
    // TODO: is it correct?
    // Octree h_octree(ceil(log2(num_unique_points) / 3) + 1)
    Octree h_octree(7);

    cudaEventRecord(start);
    h_btree.build(thrust::raw_pointer_cast(&d_codes[0]));
    // WARNING: Perhaps sort octree instead?
    // Octree nodes are ~1/3 of the btree nodes,
    // sorting would be faster
    h_btree.sort_to_bfs_order();
    h_btree.compute_octree_map();
    h_octree.build(h_btree);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // h_btree.print();

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << ms << "ms" << std::endl;

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    free(h_points);

    cudaFree(d_points);

    return 0;
}

