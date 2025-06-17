#ifndef POINTS_CUH
#define POINTS_CUH

#include <iostream>

#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/fill.h>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_run_length_encode.cuh>

#include "cuda/utils.cuh"
#include "cuda/soa_vec3.cuh"

__device__ __forceinline__ morton_t _expand_bits(morton_t u)
{
    u &= 0x00000000001fffffu;
    u = (u * 0x0000000100000001u) & 0x1F00000000FFFFu;
    u = (u * 0x0000000000010001u) & 0x1F0000FF0000FFu;
    u = (u * 0x0000000000000101u) & 0x100F00F00F00F00Fu;
    u = (u * 0x0000000000000011u) & 0x10C30C30C30C30C3u;
    u = (u * 0x0000000000000005u) & 0x1249249249249249u;
    return u;
}

// Computes 63-bit morton code by interleaving the bits
// of the normalized coordinates
template<typename T> __global__ void _morton_encode(const SoAVec3<T> pos,
                                                    morton_t *codes,
                                                    float domain_size,
                                                    int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    // Map coordinates to [0, 2^(21)-1]
    morton_t x = (morton_t) (pos.x(idx) / domain_size * 2097151.0f);
    morton_t y = (morton_t) (pos.y(idx) / domain_size * 2097151.0f);
    morton_t z = (morton_t) (pos.z(idx) / domain_size * 2097151.0f);

    x = _expand_bits(x);
    y = _expand_bits(y);
    z = _expand_bits(z);

    // Left shift x by 2 bits, y by 1 bit, then bitwise or
    codes[idx] = x * 4 + y * 2 + z;
}

template<typename T> __global__
void compute_weighted_pos(const SoAVec3<T> pos,
                          const T *mass,
                          SoAVec3<T> weighted_pos,
                          int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    T m = mass[idx];

    weighted_pos.x(idx) = pos.x(idx) * m;
    weighted_pos.y(idx) = pos.y(idx) * m;
    weighted_pos.z(idx) = pos.z(idx) * m;
}

template<typename T> class Points
{
public:
    Points(int num_points, float domain_size) :
        _num_points(num_points),
        _domain_size(domain_size),
        _gl_buffers(false)
    {
        _pos.alloc(num_points);
        _init(num_points);
    }

    Points(int num_points, float domain_size, T *x, T *y, T *z) :
        _num_points(num_points),
        _domain_size(domain_size),
        _gl_buffers(true)
    {
        _pos.x() = x;
        _pos.y() = y;
        _pos.z() = z;
        _init(num_points);
    }

    morton_t *get_d_unique_codes_ptr()
    {
        return _unique_codes;
    }

    int *get_d_codes_first_point_idx_ptr()
    {
        return _codes_first_point_idx;
    }

    SoAVec3<T> &get_d_pos()
    {
        return _pos;
    }

    const SoAVec3<T> &get_d_pos() const
    {
        return _pos;
    }

    const SoAVec3<T> &get_d_weighted_pos() const
    {
        return _weighted_pos;
    }

    const SoAVec3<T> &get_d_scan_weighted_pos() const
    {
        return _scan_weighted_pos;
    }

    T *get_d_mass()
    {
        return mass_;
    }

    const T *get_d_mass() const
    {
        return mass_;
    }

    const T *get_d_scan_mass() const
    {
        return scan_mass_;
    }

    const int *get_d_sort_indices_ptr() const
    {
        return _range;
    }

    void compute_morton_codes()
    {
        _morton_encode<<<_num_points / MAX_THREADS_PER_BLOCK +
                         (_num_points % MAX_THREADS_PER_BLOCK > 0),
                         MAX_THREADS_PER_BLOCK>>>(_pos,
                                                  _codes,
                                                  _domain_size,
                                                  _num_points);
    }

    void sort_by_codes()
    {
        thrust::sequence(thrust::device, _range, _range + _num_points);

        // Sort range once, then gather arrays to be sorted
        cub::DeviceMergeSort::SortPairs(_tmp_sort,
                                        _tmp_sort_size,
                                        _codes,
                                        _range,
                                        _num_points,
                                        less_op);

        _tmp_pos.gather(_pos, _range, _num_points);

        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       mass_,
                       tmp_mass_);

        if (_gl_buffers) {
            // Copying back to original buffer since it's where
            // OpenGL expects to read the particles data
            cudaMemcpy(_pos.x(),
                       _tmp_pos.x(),
                       _num_points * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(_pos.y(),
                       _tmp_pos.y(),
                       _num_points * sizeof(T),
                       cudaMemcpyDeviceToDevice);
            cudaMemcpy(_pos.z(),
                       _tmp_pos.z(),
                       _num_points * sizeof(T),
                       cudaMemcpyDeviceToDevice);
        } else {
            _pos.swap(_tmp_pos);
        }

        swap_ptr(&mass_, &tmp_mass_);
    }

    void compute_unique_codes(int *d_num_unique_codes)
    {
        // Obtaining unique codes and counting occurrences
        // using run-length encoding
        cub::DeviceRunLengthEncode::Encode(_tmp_runlength,
                                           _tmp_runlength_size,
                                           _codes,
                                           _unique_codes,
                                           _codes_occurrences,
                                           d_num_unique_codes,
                                           _num_points);
    }

    void scan_attributes()
    {
        // For each unique code, obtains the index of the 
        // first point that got mapped to it; useful to recover
        // all of the points encoded to the same code
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _codes_occurrences,
                                      _codes_first_point_idx,
                                      _num_points + 1);

        /*
        // Scan mass array
        cub::DeviceScan::ExclusiveSum(_tmp_scan_mass,
                                      _tmp_scan_mass_size,
                                      mass_,
                                      scan_mass_,
                                      _num_points);

        compute_weighted_pos<<<_num_points / MAX_THREADS_PER_BLOCK +
                               (_num_points % MAX_THREADS_PER_BLOCK > 0),
                               MAX_THREADS_PER_BLOCK>>>(_pos,
                                                        mass_,
                                                        _weighted_pos,
                                                        _num_points);

        // Scan coordinates to later compute the barycenter
        // of the octree nodes
        cub::DeviceScan::ExclusiveSum(_tmp_scan_mass,
                                      _tmp_scan_mass_size,
                                      _weighted_pos.x(),
                                      _scan_weighted_pos.x(),
                                      _num_points);
        cub::DeviceScan::ExclusiveSum(_tmp_scan_mass,
                                      _tmp_scan_mass_size,
                                      _weighted_pos.y(),
                                      _scan_weighted_pos.y(),
                                      _num_points);
        cub::DeviceScan::ExclusiveSum(_tmp_scan_mass,
                                      _tmp_scan_mass_size,
                                      _weighted_pos.z(),
                                      _scan_weighted_pos.z(),
                                      _num_points);
        */
    }

    ~Points()
    {
        if (!_gl_buffers) {
            _pos.free();
        }
        _tmp_pos.free();
        _weighted_pos.free();
        _scan_weighted_pos.free();

        cudaFree(mass_);
        cudaFree(tmp_mass_);
        cudaFree(scan_mass_);

        cudaFree(_codes);
        cudaFree(_unique_codes);
        cudaFree(_codes_occurrences);
        cudaFree(_codes_first_point_idx);

        cudaFree(_range);

        cudaFree(_tmp_sort);
        cudaFree(_tmp_runlength);
        cudaFree(_tmp_scan);
        cudaFree(_tmp_scan_mass);
        cudaFree(_tmp_scan_weighted_pos);
    }

private:
    void _init(int num_points)
    {
        _tmp_pos.alloc(num_points);
        _weighted_pos.alloc(num_points);
        _scan_weighted_pos.alloc(num_points);

        cudaMalloc(&mass_, num_points * sizeof(T));
        cudaMalloc(&tmp_mass_, num_points * sizeof(T));
        thrust::fill(thrust::device, mass_, mass_ + num_points, (T) 1.0);
        cudaMalloc(&scan_mass_, num_points * sizeof(T));

        cudaMalloc(&_codes, num_points * sizeof(morton_t));
        cudaMalloc(&_unique_codes, num_points * sizeof(morton_t));
        cudaMalloc(&_codes_occurrences, (num_points + 1) * sizeof(int));
        cudaMalloc(&_codes_first_point_idx, (num_points + 1) * sizeof(int));

        cudaMalloc(&_range, num_points * sizeof(int));

        // Determining and allocating storage for cub operations
        _tmp_sort = nullptr;
        cub::DeviceMergeSort::SortPairs(_tmp_sort,
                                        _tmp_sort_size,
                                        _codes,
                                        _range,
                                        num_points,
                                        less_op);
        cudaMalloc(&_tmp_sort, _tmp_sort_size);

        int *tmp = nullptr;
        _tmp_runlength = nullptr;
        cub::DeviceRunLengthEncode::Encode(_tmp_runlength,
                                           _tmp_runlength_size,
                                           _codes,
                                           _unique_codes,
                                           _codes_occurrences,
                                           tmp,
                                           num_points);
        cudaMalloc(&_tmp_runlength, _tmp_runlength_size);

        _tmp_scan = nullptr;
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _codes_occurrences,
                                      _codes_first_point_idx,
                                      num_points);
        cudaMalloc(&_tmp_scan, _tmp_scan_size);

        _tmp_scan_mass = nullptr;
        cub::DeviceScan::ExclusiveSum(_tmp_scan_mass,
                                      _tmp_scan_mass_size,
                                      mass_,
                                      scan_mass_,
                                      num_points);
        cudaMalloc(&_tmp_scan_mass, _tmp_scan_mass_size);
        cudaMalloc(&_tmp_scan_weighted_pos, _tmp_scan_mass_size);
    }

    // Whether _pos buffers are OpenGL mapped resources
    const bool _gl_buffers;

    int _num_points;
    float _domain_size;

    SoAVec3<T> _pos;
    SoAVec3<T> _tmp_pos;
    SoAVec3<T> _weighted_pos;
    SoAVec3<T> _scan_weighted_pos;

    T *mass_;
    T *tmp_mass_;
    T *scan_mass_;

    morton_t *_codes;
    morton_t *_unique_codes;
    int *_codes_occurrences;
    int *_codes_first_point_idx;

    int *_range;

    LessOp less_op;
    int *_tmp_sort;
    size_t _tmp_sort_size;

    int *_tmp_runlength;
    size_t _tmp_runlength_size;

    int *_tmp_scan;
    size_t _tmp_scan_size;

    T *_tmp_scan_mass;
    T *_tmp_scan_weighted_pos;
    size_t _tmp_scan_mass_size;
};

#endif
