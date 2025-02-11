#ifndef OCTREE_GPU_CUH
#ifndef OCTREE_GPU_CUH

// SoA to store the octree
class Octree
{
public:
    // max_depth ~= btree_height / 3
    Octree(int max_depth);

    void build();

    ~Octree();

    __device__ __forceinline__
    bool is_leaf(int node)
    {
        return node >= _max_num_internal;
    }

    __device__ __forceinline__
    void add_child(int parent, int child, bool is_leaf)
    {
        _children[(parent << 3) + _num_children[parent]++] =
            // Pointers to leaf nodes are offset by the
            // maximum number of internal nodes
            child + is_leaf * _max_num_internal;
    }

private:
    int _max_depth;
    int _max_num_internal;

    // Array to store pointers (indices) to the children of each node,
    // each groups of 8 siblings is contiguous in memory
    int *_children;
    // Array to store the number of children of each internal node
    int *_num_children;

    // TODO: hold a pointer to the corresponding Btree object
    // so that it doesn't get redundantly copied to gpu memory
}

#endif

