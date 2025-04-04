#ifndef SOA_VEC3_CUH
#define SOA_VEC3_CUH

#include <cstdlib>

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

    void zeros(int n)
    {
        cudaMemset(_x, 0, n * sizeof(T));
        cudaMemset(_y, 0, n * sizeof(T));
        cudaMemset(_z, 0, n * sizeof(T));
    }

    void rand(int n)
    {
        T *buff = (T *) std::malloc(n * sizeof(T));
        T *dst[3] = { _x, _y, _z };

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < n; ++j) {
                buff[j] = ((T) std::rand() - RAND_MAX / 2) / RAND_MAX / 2 * 10;
            }
            cudaMemcpy(dst[i], buff, n * sizeof(T), cudaMemcpyHostToDevice);
        }

        std::free(buff);
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
