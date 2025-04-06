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
                buff[j] = ((T) std::rand() - RAND_MAX / 2) / RAND_MAX / 2;
            }
            cudaMemcpy(dst[i], buff, n * sizeof(T), cudaMemcpyHostToDevice);
        }

        std::free(buff);
    }

    void tangent(const SoAVec3<T> _pos, int n)
    {
        T *x_pos = (T *) std::malloc(n * sizeof(T));
        T *y_pos = (T *) std::malloc(n * sizeof(T));
        T *z_pos = (T *) std::malloc(n * sizeof(T));

        cudaMemcpy(x_pos, _pos._x, n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(y_pos, _pos._y, n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(z_pos, _pos._z, n * sizeof(T), cudaMemcpyDeviceToHost);

        T *x_dst = (T *) std::malloc(n * sizeof(T));
        T *y_dst = (T *) std::malloc(n * sizeof(T));
        T *z_dst = (T *) std::malloc(n * sizeof(T));

        for (int i = 0; i < n; ++i) {
            T x = x_pos[i] - 0.5;
            T y = y_pos[i] - 0.5;
            x_dst[i] = -y * 500;
            y_dst[i] = x * 500;
            z_dst[i] = 0;//(std::rand() - RAND_MAX / 2 ) / RAND_MAX / 2 * 500;
        }

        cudaMemcpy(_x, x_dst, n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(_y, y_dst, n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(_z, z_dst, n * sizeof(T), cudaMemcpyHostToDevice);

        std::free(x_pos);
        std::free(y_pos);
        std::free(z_pos);
        std::free(x_dst);
        std::free(y_dst);
        std::free(z_dst);
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
