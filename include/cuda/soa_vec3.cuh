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

    void hubble(const SoAVec3<T> _pos, int n, float h0)
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
            T z = z_pos[i] - 0.5;
            T r = std::sqrt(x * x + y * y + z * z);
            x_dst[i] = x * h0 * r;
            y_dst[i] = y * h0 * r;
            z_dst[i] = z * h0 * r;
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

    void plummer(int n, T a)
    {
        std::srand(100);

        T *x = (T *) std::malloc(n * sizeof(T));
        T *y = (T *) std::malloc(n * sizeof(T));
        T *z = (T *) std::malloc(n * sizeof(T));

        for (int i = 0; i < n; ++i) {
            T u = (T) std::rand() / RAND_MAX;
            T r = a * std::pow((std::pow(u, -2.0 / 3) - 1), -0.5);

            z[i] = (T) std::rand() / RAND_MAX * 2 - 1.0;

            T r_xy = std::sqrt(1 - z[i] * z[i]);
            T theta = (T) std::rand() / RAND_MAX * 2 * M_PI;

            x[i] = r_xy * std::cos(theta) * r * 0.5 + 0.5;
            y[i] = r_xy * std::sin(theta) * r * 0.5 + 0.5;
            z[i] = r * z[i] * 0.5 + 0.5;

            x[i] = std::max<T>(0.0, std::min<T>(1.0, x[i]));
            y[i] = std::max<T>(0.0, std::min<T>(1.0, y[i]));
            z[i] = std::max<T>(0.0, std::min<T>(1.0, z[i]));
        }

        cudaMemcpy(_x, x, n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(_y, y, n * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(_z, z, n * sizeof(T), cudaMemcpyHostToDevice);

        std::free(x);
        std::free(y);
        std::free(z);
    }

    void plummer_vel(const SoAVec3<T> pos, int n, T a)
    {
        T *x_pos = (T *) std::malloc(n * sizeof(T));
        T *y_pos = (T *) std::malloc(n * sizeof(T));
        T *z_pos = (T *) std::malloc(n * sizeof(T));

        cudaMemcpy(x_pos, pos._x, n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(y_pos, pos._y, n * sizeof(T), cudaMemcpyDeviceToHost);
        cudaMemcpy(z_pos, pos._z, n * sizeof(T), cudaMemcpyDeviceToHost);

        T *x_dst = (T *) std::malloc(n * sizeof(T));
        T *y_dst = (T *) std::malloc(n * sizeof(T));
        T *z_dst = (T *) std::malloc(n * sizeof(T));

        for (int i = 0; i < n; ++i) {
            T x = x_pos[i] - 0.5;
            T y = y_pos[i] - 0.5;
            T z = z_pos[i] - 0.5;
            T r = std::sqrt(x * x + y * y + z * z);

            /*
            T phi = -n / std::sqrt(r * r + a * a);
            T v_esc = std::sqrt(-2 * phi);

            T v;
            while (true) {
                v = (T) std::rand() / RAND_MAX * v_esc;
                T vv = v / v_esc;
                T g = vv * vv * std::pow((1 - vv * vv), 3.5);
                if ((T) std::rand() / RAND_MAX < g) {
                    break;
                }
            }
            */

            // T v_z = (T) std::rand() / RAND_MAX * 2 - 1.0;
            T theta = std::sqrt(n / r);//(T) std::rand() / RAND_MAX * 2 * M_PI;
            T vx = -theta * y / r;
            T vy = theta * x / r;

            x_dst[i] = vx * 0.2 + std::rand() / RAND_MAX * 2 - 1.0;
            y_dst[i] = vy * 0.2 + std::rand() / RAND_MAX * 2 - 1.0;
            z_dst[i] = 0;
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
