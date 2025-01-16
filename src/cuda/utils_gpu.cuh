#ifndef UTILS_GPU_CUH
#define UTILS_GPU_CUH

constexpr int THREADS_PER_BLOCK = 32;

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

    __device__ __forceinline__ float get_x(int idx) { return _x[idx]; }
    __device__ __forceinline__ float get_y(int idx) { return _y[idx]; }
    __device__ __forceinline__ float get_z(int idx) { return _z[idx]; }

private:
    float *_x;
    float *_y;
    float *_z;
};

#endif

