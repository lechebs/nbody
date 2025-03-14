#ifndef SOA_VEC3_CUH
#define SOA_VEC3_CUH

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

#endif
