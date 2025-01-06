#ifndef RADIX_TREE_GPU_CUH
#define RADIX_TREE_GPU_CUH

typedef unsigned int uint32_t;

constexpr int MAX_LCP = 32;

// SoA to store the points coordinates, allows coalescing
class Points
{
public:
    Points(float *x, float *y, float *z) : _x(x), _y(y), _z(z) {}

    __device__ __forceinline__ float get_x(int idx) { return _x[idx]; }
    __device__ __forceinline__ float get_y(int idx) { return _y[idx]; }
    __device__ __forceinline__ float get_z(int idx) { return _z[idx]; }

private:
    float *_x;
    float *_y;
    float *_z;
};

// SoA to store the nodes of the radix tree
class Nodes
{
public:
    Nodes(int num_leaves, int *left, int *right, int *edge_delta) :
        _num_leaves(num_leaves),
        _left(left),
        _right(right),
        _edge_delta(edge_delta) {}

    __device__ __forceinline__ bool is_leaf(int node) {
        // Pointers to leaf nodes are offset by the
        // number of internal nodes
        return node >= _num_leaves - 1;
    }

    __device__ __forceinline__ int get_left(int idx) { return _left[idx]; }
    __device__ __forceinline__ int get_right(int idx) { return _right[idx]; }

    __device__ __forceinline__ void set_left(int idx,
                                             int left,
                                             int lcp,
                                             bool is_leaf)
    {
        _set_child(_left, idx, left, lcp, is_leaf);
    }

    __device__ __forceinline__ void set_right(int idx,
                                              int right,
                                              int lcp,
                                              bool is_leaf)
    {
        _set_child(_right, idx, right, lcp, is_leaf);
    }

private:
    // Computes how many octree nodes each edge
    // of the radix tree generates
    __device__ __forceinline__ void _compute_edge_delta(int node,
                                                        int parent_lcp,
                                                        bool is_leaf)
    {
        int node_lcp = is_leaf ? MAX_LCP : parent_lcp + 1;
        _edge_delta[node] = node_lcp / 3 - parent_lcp / 3;
    }

    // Setter for either the left or right child of an internal node
    __device__ __forceinline__ void _set_child(int *dest,
                                               int parent,
                                               int child,
                                               int lcp,
                                               bool is_leaf)
    {
        child += (_num_leaves - 1) * is_leaf;
        dest[parent] = child;
        _compute_edge_delta(child, lcp, is_leaf);
    }

    // Number of internal nodes is _num_leaves - 1
    int _num_leaves;
    // Arrays to store pointers (indices) to left and right children
    // of the internal nodes
    int *_left;
    int *_right;
    // Array to store the number of octree nodes that correspond
    // to the ascending edge of each radix tree node (both internal
    // and leaves)
    int *_edge_delta;
};

__global__ void morton_encode(Points *points, uint32_t *codes);
__global__ void build_radix_tree(uint32_t *codes,
                                 Nodes *internal,
                                 int num_leaves);

#endif
