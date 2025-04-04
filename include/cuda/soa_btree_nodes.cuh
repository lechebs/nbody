#ifndef SOA_BTREE_NODES_CUH
#define SOA_BTREE_NODES_CUH

class SoABtreeNodes
{
    friend class Btree;

public:
    // Device getters
    __device__ __forceinline__ int parent(int idx) const
    {
        return _parent[idx];
    }

    __device__ __forceinline__ int depth(int idx) const
    {
        return _depth[idx];
    }

    __device__ __forceinline__ int *left()
    {
        return _left;
    }
    __device__ __forceinline__ const int *left() const
    {
        return _left;
    }
    __device__ __forceinline__ int left(int idx) const
    {
        return _left[idx];
    }

    __device__ __forceinline__ int *right()
    {
        return _right;
    }
    __device__ __forceinline__ const int *right() const
    {
        return _right;
    }
    __device__ __forceinline__ int right(int idx) const
    {
        return _right[idx];
    }

    __device__ __forceinline__ int lcp(int idx) const
    {
        return _lcp[idx];
    }

    __device__ __forceinline__ const int *edge_delta() const
    {
        return _edge_delta;
    }
    __device__ __forceinline__ int edge_delta(int idx) const
    {
        return _edge_delta[idx];
    }

    __device__ __forceinline__ int leaves_begin(int idx) const
    {
        return _leaves_begin[idx];
    }

    __device__ __forceinline__ int leaves_end(int idx) const
    {
        return _leaves_end[idx];
    }

    // Device setters
    __device__ __forceinline__ int &parent(int idx)
    {
        return _parent[idx];
    }

    __device__ __forceinline__ int &depth(int idx)
    {
        return _depth[idx];
    }

    __device__ __forceinline__ int &left(int idx)
    {
        return _left[idx];
    }

    __device__ __forceinline__ int &right(int idx)
    {
        return _right[idx];
    }

    __device__ __forceinline__ int &lcp(int idx)
    {
        return _lcp[idx];
    }

    __device__ __forceinline__ int &edge_delta(int idx)
    {
        return _edge_delta[idx];
    }

    __device__ __forceinline__ int &leaves_begin(int idx)
    {
        return _leaves_begin[idx];
    }

    __device__ __forceinline__ int &leaves_end(int idx)
    {
        return _leaves_end[idx];
    }

    void alloc(int num_nodes)
    {
        cudaMalloc(&_parent, num_nodes * sizeof(int));
        cudaMalloc(&_depth, num_nodes * sizeof(int));
        cudaMalloc(&_left, num_nodes * sizeof(int));
        cudaMalloc(&_right, num_nodes * sizeof(int));
        cudaMalloc(&_lcp, num_nodes * sizeof(int));
        cudaMalloc(&_edge_delta, num_nodes * sizeof(int));
        cudaMalloc(&_leaves_begin, num_nodes * sizeof(int));
        cudaMalloc(&_leaves_end, num_nodes * sizeof(int));
    }

    void free()
    {
        cudaFree(_parent);
        cudaFree(_depth);
        cudaFree(_left);
        cudaFree(_right);
        cudaFree(_lcp);
        cudaFree(_edge_delta);
        cudaFree(_leaves_begin);
        cudaFree(_leaves_end);
    }

private:
    int *_parent;
    int *_depth;
    int *_left;
    int *_right;
    int *_lcp;
    int *_edge_delta;
    int *_leaves_begin;
    int *_leaves_end;
};

#endif
