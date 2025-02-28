#ifndef UTILS_GPU_CUH
#define UTILS_GPU_CUH

constexpr int THREADS_PER_BLOCK = 32;
// TODO: compute it such that a SM is fully utilized
constexpr int MAX_THREADS_PER_BLOCK = 512;

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

// SoA to store the points coordinates, allows coalescing
class Points
{
public:
    Points(float *x, float *y, float *z) : _x(x), _y(y), _z(z) {}

    __device__ __forceinline__ float get_x(int idx) const { return _x[idx]; }
    __device__ __forceinline__ float get_y(int idx) const { return _y[idx]; }
    __device__ __forceinline__ float get_z(int idx) const { return _z[idx]; }

private:
    float *_x;
    float *_y;
    float *_z;
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

inline void swap_ptr(int **ptr1, int **ptr2)
{
    int *tmp = *ptr1;
    *ptr1 = *ptr2;
    *ptr2 = tmp;
}

#endif

