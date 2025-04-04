#ifndef SOA_OCTREE_NODES_CUH
#define SOA_OCTREE_NODES_CUH

class SoAOctreeNodes
{
    template<typename T> friend class Octree;

public:
    // Device getters
    __device__ __forceinline__ int first_child(int idx) const
    {
        return _first_child[idx];
    }

    __device__ __forceinline__ int num_children(int idx) const
    {
        return _num_children[idx];
    }

    __device__ __forceinline__ int leaves_begin(int idx) const
    {
        return _leaves_begin[idx];
    }

    __device__ __forceinline__ int leaves_end(int idx) const
    {
        return _leaves_end[idx];
    }

    __device__ __forceinline__ float size(int idx) const
    {
        return _size[idx];
    }

    // Device setters
    __device__ __forceinline__ int &first_child(int idx)
    {
        return _first_child[idx];
    }

    __device__ __forceinline__ int &num_children(int idx)
    {
        return _num_children[idx];
    }

    __device__ __forceinline__ int &leaves_begin(int idx)
    {
        return _leaves_begin[idx];
    }

    __device__ __forceinline__ int &leaves_end(int idx)
    {
        return _leaves_end[idx];
    }

    __device__ __forceinline__ float &size(int idx)
    {
        return _size[idx];
    }

    void alloc(int num_nodes)
    {
        cudaMalloc(&_first_child, num_nodes * sizeof(int));
        cudaMalloc(&_num_children, num_nodes * sizeof(int));
        cudaMalloc(&_leaves_begin, num_nodes * sizeof(int));
        cudaMalloc(&_leaves_end, num_nodes * sizeof(int));
        cudaMalloc(&_size, num_nodes * sizeof(float));
    }

    void free()
    {
        cudaFree(_first_child);
        cudaFree(_num_children);
        cudaFree(_leaves_begin);
        cudaFree(_leaves_end);
        cudaFree(_size);
    }

private:
    // Array to store the index of the first child of each node
    int *_first_child;
    // Array to store the number of children of each node
    int *_num_children;
    // Arrays to store the range of leaves covered by each node
    int *_leaves_begin;
    int *_leaves_end;
    // Array to store the cube side length covered by each node
    float *_size;
};

#endif
