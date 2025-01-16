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

constexpr int NUM_POINTS = 2 << 4;

void print_bits(uint32_t u)
{
    for (int i = 0; i < 32; ++i) {
        printf("%d", (u >> (31 - i)) & 0x01);
    }
}

__global__ void gather_child_pointers(int *left,
                                      int *right,
                                      int *indices,
                                      int num_leaves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (left[idx] < num_leaves - 1) {
        // Inefficient scatter read
        left[idx] = indices[left[idx]];
    }

    if (right[idx] < num_leaves - 1) {
        right[idx] = indices[right[idx]];
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
    // thrust::fill(h_z.begin(), h_z.end(), 0.0f);

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

    thrust::host_vector<uint32_t> h_unique_codes(d_codes);
    for (int i = 0; i < 32; ++i) {
        printf("%4d: %12u ", i, h_unique_codes[i]);
        print_bits(h_unique_codes[i]);
        printf("\n");
    }

    Btree h_btree(num_unique_points);
    h_btree.build(thrust::raw_pointer_cast(&d_codes[0]));
    h_btree.sort_to_bfs_order();

    // cudaDeviceSynchronize();

    // Correct pointers to child nodes
    // WARNING: pointers to leaf nodes should remaing intact
    /*
    thrust::device_vector<int> d_range(h_indices);
    thrust::device_vector<int> d_scattered_indices(num_unique_points - 1);
    thrust::scatter(d_range.begin(),
                    d_range.end(),
                    d_indices.begin(),
                    d_scattered_indices.begin());
    // Can we use a gather if?
    gather_child_pointers<<<
         num_unique_points / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            h_btree.get_left_ptr()
            h_btree.get_right_ptr()
            thrust::raw_pointer_cast(&d_right[0]),
            thrust::raw_pointer_cast(&d_scattered_indices[0]),
            num_unique_points);

    thrust::device_vector<int> d_bin2oct(num_unique_points - 1);
    // Exclusive scan to obtain binary node to octree node mapping
    thrust::exclusive_scan(d_edge_delta.begin(),
                           d_edge_delta.begin() + num_unique_points - 1,
                           d_bin2oct.begin());

    thrust::host_vector<int> h_left(d_left);
    thrust::host_vector<int> h_right(d_right);
    thrust::host_vector<int> h_edge_delta(d_edge_delta);
    thrust::host_vector<int> h_depth(d_depth);
    thrust::host_vector<int> h_bin2oct(d_bin2oct);

    std::cout << "Unique pts: " << num_unique_points << std::endl;

    for (int i = 0; i < 32 - 1; ++i) {
        printf("%4d: %4d %4d - d: %d %d - depth: %d\n",
               i, h_left[i], h_right[i], h_edge_delta[i],
               h_bin2oct[i], h_depth[i]);
    }
    */

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    free(h_points);

    cudaFree(d_points);

    return 0;
}

