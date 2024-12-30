#ifndef RADIX_TREE_GPU_CUH
#define RADIX_TREE_GPU_CUH

typedef unsigned int uint32_t;

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
    Nodes(int num_leaves, int *left, int *right) :
        _num_leaves(num_leaves),
        _left(left),
        _right(right) {}

    __device__ __forceinline__ bool is_leaf(int node) {
        // Pointers to leaf nodes are offset by the
        // number of internal nodes
        return node >= _num_leaves - 1;
    }

    __device__ __forceinline__ int get_left(int idx) { return _left[idx]; }
    __device__ __forceinline__ int get_right(int idx) { return _right[idx]; }

    __device__ __forceinline__ void set_left(int idx, int left, bool is_leaf)
    {
        _left[idx] = left + (_num_leaves - 1) * is_leaf;
    }

    __device__ __forceinline__ void set_right(int idx, int right, bool is_leaf)
    {
        _right[idx] = right + (_num_leaves - 1) * is_leaf;
    }

private:
    // Number of internal nodes is num_leaves - 1
    int _num_leaves;
    // Arrays to store pointers (indices) to left and right children
    int *_left;
    int *_right;
};

__global__ void morton_encode(Points *points, uint32_t *codes);
__global__ void build_radix_tree(uint32_t *codes,
                                 Nodes *internal,
                                 int num_leaves);

#endif
