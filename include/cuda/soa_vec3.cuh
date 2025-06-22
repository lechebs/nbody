#ifndef SOA_VEC3_CUH
#define SOA_VEC3_CUH

#include <cstdlib>
#include <cmath>

#include <thrust/execution_policy.h>
#include <thrust/gather.h>

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

    void copy(const SoAVec3<T> &src, int n)
    {
        cudaMemcpy(_x, src._x, n * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_y, src._y, n * sizeof(T), cudaMemcpyDeviceToDevice);
        cudaMemcpy(_z, src._z, n * sizeof(T), cudaMemcpyDeviceToDevice);
    }

    void swap(SoAVec3<T> &other)
    {
        T *tmp;

        tmp = _x;
        _x = other._x;
        other._x = tmp;

        tmp = _y;
        _y = other._y;
        other._y = tmp;

        tmp = _z;
        _z = other._z;
        other._z = tmp;
    }

    void gather(const SoAVec3<T> &src, const int *map, int n)
    {
        thrust::gather(thrust::device, map, map + n, src._x, _x);
        thrust::gather(thrust::device, map, map + n, src._y, _y);
        thrust::gather(thrust::device, map, map + n, src._z, _z);
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
