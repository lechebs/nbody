#ifndef UTILS_GPU_CUH
#define UTILS_GPU_CUH

typedef unsigned int uint32_t;

constexpr int THREADS_PER_BLOCK = 32;
// TODO: compute it such that a SM is fully utilized
constexpr int MAX_THREADS_PER_BLOCK = 512;

template<typename T> class SoAVec3
{
public:
    __device__ __forceinline__ T x(int idx) const { return _x[idx]; }
    __device__ __forceinline__ T y(int idx) const { return _y[idx]; }
    __device__ __forceinline__ T z(int idx) const { return _z[idx]; }

    __device__ __forceinline__ T &x(int idx) { return _x[idx]; }
    __device__ __forceinline__ T &y(int idx) { return _y[idx]; }
    __device__ __forceinline__ T &z(int idx) { return _z[idx]; }

    T *&x() { return _x; }
    T *&y() { return _y; }
    T *&z() { return _z; }

    void alloc(int n)
    {
        cudaMalloc(&_x, n * sizeof(T));
        cudaMalloc(&_y, n * sizeof(T));
        cudaMalloc(&_z, n * sizeof(T));
    }

    void free()
    {
        cudaFree(_x);
        cudaFree(_y);
        cudaFree(_z);
    }

private:
    T *_x;
    T *_y;
    T *_z;
};

struct LessOp
{
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs)
  {
    return lhs < rhs;
  }
};

// TODO: move to utils_gpu.cu
inline int log2(int n)
{
    int it = 0;
    int val = 1;
    while (val < n) {
        val *= 2;
        it++;
    }
    return it;
}

// TODO: move to utils_gpu.cu
inline int geometric_sum(int base, int n)
{
    int pow = 1;
    for (int i = 0; i < n + 1; i++) {
        pow *= base;
    }
    return (1 - pow) / (1 - base);
}

template<typename T> void swap_ptr(T **ptr1, T **ptr2)
{
    T *tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

#endif

