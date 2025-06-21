#ifndef SOA_BTREE_NODES_CUH
#define SOA_BTREE_NODES_CUH

class SoABtreeNodes
{
    friend class Btree;

public:
    // Device getters
    __device__ __forceinline__ int parent(int idx) const
    {
        return parent_[idx];
    }

    __device__ __forceinline__ int depth(int idx) const
    {
        return depth_[idx];
    }

    __device__ __forceinline__ int *left()
    {
        return left_;
    }
    __device__ __forceinline__ const int *left() const
    {
        return left_;
    }
    __device__ __forceinline__ int left(int idx) const
    {
        return left_[idx];
    }

    __device__ __forceinline__ int *right()
    {
        return right_;
    }
    __device__ __forceinline__ const int *right() const
    {
        return right_;
    }
    __device__ __forceinline__ int right(int idx) const
    {
        return right_[idx];
    }

    __device__ __forceinline__ int lcp(int idx) const
    {
        return lcp_[idx];
    }

    __device__ __forceinline__ const int *edge_delta() const
    {
        return edge_delta_;
    }
    __device__ __forceinline__ int edge_delta(int idx) const
    {
        return edge_delta_[idx];
    }

    __device__ __forceinline__ int leaves_begin(int idx) const
    {
        return leaves_begin_[idx];
    }

    __device__ __forceinline__ int leaves_end(int idx) const
    {
        return leaves_end_[idx];
    }

    // Device setters
    __device__ __forceinline__ int &parent(int idx)
    {
        return parent_[idx];
    }

    __device__ __forceinline__ int &depth(int idx)
    {
        return depth_[idx];
    }

    __device__ __forceinline__ int &left(int idx)
    {
        return left_[idx];
    }

    __device__ __forceinline__ int &right(int idx)
    {
        return right_[idx];
    }

    __device__ __forceinline__ int &lcp(int idx)
    {
        return lcp_[idx];
    }

    __device__ __forceinline__ int &edge_delta(int idx)
    {
        return edge_delta_[idx];
    }

    __device__ __forceinline__ int &leaves_begin(int idx)
    {
        return leaves_begin_[idx];
    }

    __device__ __forceinline__ int &leaves_end(int idx)
    {
        return leaves_end_[idx];
    }

    void alloc(int num_nodes)
    {
        cudaMalloc(&parent_, num_nodes * sizeof(int));
        cudaMalloc(&depth_, num_nodes * sizeof(int));
        cudaMalloc(&left_, num_nodes * sizeof(int));
        cudaMalloc(&right_, num_nodes * sizeof(int));
        cudaMalloc(&lcp_, num_nodes * sizeof(int));
        cudaMalloc(&edge_delta_, num_nodes * sizeof(int));
        cudaMalloc(&leaves_begin_, num_nodes * sizeof(int));
        cudaMalloc(&leaves_end_, num_nodes * sizeof(int));
    }

    void free()
    {
        cudaFree(parent_);
        cudaFree(depth_);
        cudaFree(left_);
        cudaFree(right_);
        cudaFree(lcp_);
        cudaFree(edge_delta_);
        cudaFree(leaves_begin_);
        cudaFree(leaves_end_);
    }

private:
    int *parent_;
    int *depth_;
    int *left_;
    int *right_;
    int *lcp_;
    int *edge_delta_;
    int *leaves_begin_;
    int *leaves_end_;
};

#endif
