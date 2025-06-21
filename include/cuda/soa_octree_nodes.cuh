#ifndef SOA_OCTREE_NODES_CUH
#define SOA_OCTREE_NODES_CUH

class SoAOctreeNodes
{
    template<typename T> friend class Octree;

public:
    // Device getters
    __device__ __forceinline__ int first_child(int idx) const
    {
        return first_child_[idx];
    }

    __device__ __forceinline__ int num_children(int idx) const
    {
        return num_children_[idx];
    }

    __device__ __forceinline__ int leaves_begin(int idx) const
    {
        return leaves_begin_[idx];
    }

    __device__ __forceinline__ int leaves_end(int idx) const
    {
        return leaves_end_[idx];
    }

    __device__ __forceinline__ int depth(int idx) const
    {
        return depth_[idx];
    }

    // Device setters
    __device__ __forceinline__ int &first_child(int idx)
    {
        return first_child_[idx];
    }

    __device__ __forceinline__ int &num_children(int idx)
    {
        return num_children_[idx];
    }

    __device__ __forceinline__ int &leaves_begin(int idx)
    {
        return leaves_begin_[idx];
    }

    __device__ __forceinline__ int &leaves_end(int idx)
    {
        return leaves_end_[idx];
    }

    __device__ __forceinline__ int &depth(int idx)
    {
        return depth_[idx];
    }

    void alloc(int num_nodes)
    {
        cudaMalloc(&first_child_, num_nodes * sizeof(int));
        cudaMalloc(&num_children_, num_nodes * sizeof(int));
        cudaMalloc(&leaves_begin_, num_nodes * sizeof(int));
        cudaMalloc(&leaves_end_, num_nodes * sizeof(int));
        cudaMalloc(&depth_, num_nodes * sizeof(int));
    }

    void free()
    {
        cudaFree(first_child_);
        cudaFree(num_children_);
        cudaFree(leaves_begin_);
        cudaFree(leaves_end_);
        cudaFree(depth_);
    }

private:
    // Array to store the index of the first child of each node
    int *first_child_;
    // Array to store the number of children of each node
    int *num_children_;
    // Arrays to store the range of leaves covered by each node
    int *leaves_begin_;
    int *leaves_end_;
    // Array to store the depth of each node
    int *depth_;
};

#endif
