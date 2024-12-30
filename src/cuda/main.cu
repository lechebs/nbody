#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "radix_tree_gpu.cuh"

constexpr int NUM_POINTS = 2 << 10;
constexpr int THREADS_PER_BLOCK = 128;

// Allocates device memory to store SoA
// and copies member data from host
template<class T>
T *alloc_device_soa(T *data, std::size_t size)
{
    void *device_data;

    cudaMalloc(&device_data, size);
    cudaMemcpy(device_data, data, size, cudaMemcpyHostToDevice);

    return (T *) device_data;
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
    thrust::fill(h_z.begin(), h_z.end(), 0.0f);

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

    // Allocate device memory to store the radix tree nodes
    thrust::device_vector<int> d_left(NUM_POINTS - 1);
    thrust::device_vector<int> d_right(NUM_POINTS - 1);
    // Initializes nodes SoA and copy to device
    Nodes *h_nodes = new Nodes(
        NUM_POINTS,
        thrust::raw_pointer_cast(&d_left[0]),
        thrust::raw_pointer_cast(&d_right[0]));
    Nodes *d_nodes = alloc_device_soa(h_nodes, sizeof(Nodes));

    // Sorting and removing duplicates
    thrust::sort(d_codes.begin(), d_codes.end());
    auto unique_end = thrust::unique(d_codes.begin(), d_codes.end());

    int num_unique_points = unique_end - d_codes.begin();

    build_radix_tree
    <<<num_unique_points / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(&d_codes[0]),
        d_nodes,
        num_unique_points);

    cudaDeviceSynchronize();

    std::cout << cudaGetErrorString(cudaGetLastError()) << std::endl;

    free(h_points);
    free(h_nodes);

    cudaFree(d_points);
    cudaFree(d_nodes);

    return 0;
}

