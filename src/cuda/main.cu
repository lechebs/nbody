#include <iostream>

#include "cuda/utils_gpu.cuh"
#include "cuda/points_gpu.cuh"
#include "cuda/btree_gpu.cuh"
#include "cuda/octree_gpu.cuh"

#define TIMER_START(start) cudaEventRecord(start);
#define TIMER_STOP(msg, start, stop) {                   \
    cudaEventRecord(stop);                               \
    cudaEventSynchronize(stop);                          \
    std::cout << cudaGetErrorString(cudaGetLastError())  \
              << std::endl;                              \
    float ms;                                            \
    cudaEventElapsedTime(&ms, start, stop);              \
    std::cout << msg << ": " << ms << "ms" << std::endl; \
}

constexpr int NUM_POINTS = 2 << 18;
constexpr int MAX_CODES_PER_LEAF = 16;

void print_bits(uint32_t u)
{
    for (int i = 0; i < 32; ++i) {
        printf("%d", (u >> (31 - i)) & 0x01);
    }
}

int main()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Points<float> points(NUM_POINTS);
    points.sample_uniform();

    Btree btree(NUM_POINTS);

    // If the octree is completely unbalanced, the number of internal
    // nodes should be O((NUM_POINTS-1) / 3), while if it is completely
    // balanced it should be O(geometric_sum(8, log2(NUM_POINTS) / 3))
    // The latter expression bounds the former from above

    // Note that the specified formulas assume that each leaf contains
    // one point only, the number of internal nodes will actually be much
    // smaller

    int num_octree_nodes = min(
        2 * NUM_POINTS,
        geometric_sum(8, ceil(log2(NUM_POINTS) / 3.0) + 1));
    Octree<float> octree(num_octree_nodes);

    TIMER_START(start)
    points.compute_morton_codes();
    TIMER_STOP("morton", start, stop);

    TIMER_START(start)
    points.sort_by_codes();
    TIMER_STOP("sort-codes", start, stop)

    TIMER_START(start)
    points.compute_unique_codes(btree.get_d_num_leaves_ptr());
    TIMER_STOP("run-length", start, stop)

    int num_unique_codes = btree.get_num_leaves();
    std::cout << "num_unique_codes=" << num_unique_codes << std::endl;

    TIMER_START(start)
    points.scan_attributes();
    TIMER_STOP("points-scan", start, stop)

    TIMER_START(start)
    btree.generate_leaves(points.get_d_unique_codes_ptr(),
                          MAX_CODES_PER_LEAF);
    TIMER_STOP("btree-leaves", start, stop)

    /*
    thrust::host_vector<uint32_t> h_unique_codes(d_unique_codes);
    for (int i = 0; i < 32; ++i) {
        printf("%4d: %12u ", i, h_unique_codes[i]);
        print_bits(h_unique_codes[i]);
        printf("\n");
    }
    */

    TIMER_START(start);
    int num_leaves = btree.get_num_leaves();
    TIMER_STOP("memcpy-num-leaves", start, stop);
    std::cout << "num_leaves=" << num_leaves << std::endl;

    // Setting max values to actual ones to speedup kernel launches
    btree.set_max_num_leaves(num_leaves);
    octree.set_max_num_nodes(btree.get_max_num_nodes());

    TIMER_START(start)
    btree.build(points.get_d_unique_codes_ptr());
    TIMER_STOP("btree-build", start, stop)

    // WARNING: Perhaps sort octree instead?
    // Octree nodes are ~1/3 of the btree nodes,
    // sorting would be faster

    TIMER_START(start)
    btree.sort_to_bfs_order();
    TIMER_STOP("btree-sort", start, stop)

    TIMER_START(start)
    btree.compute_octree_map();
    TIMER_STOP("btree-scan", start, stop)

    TIMER_START(start)
    octree.build(btree);
    TIMER_STOP("octree-build", start, stop)

    TIMER_START(start)
    octree.compute_nodes_barycenter(points,
                                    btree.get_d_leaf_first_code_idx_ptr());
    TIMER_STOP("octree-barycenters", start, stop)

    // h_btree.print();
    octree.print();

    return 0;
}

