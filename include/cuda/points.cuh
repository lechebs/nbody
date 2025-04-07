#ifndef POINTS_CUH
#define POINTS_CUH

#include "cuda/utils.cuh"
#include "cuda/soa_vec3.cuh"

#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_run_length_encode.cuh>

__device__ __forceinline__ uint32_t _expand_bits(uint32_t u)
{
    u = (u * 0x00010001u) & 0xFF0000FFu;
    u = (u * 0x00000101u) & 0x0F00F00Fu;
    u = (u * 0x00000011u) & 0xC30C30C3u;
    u = (u * 0x00000005u) & 0x49249249u;

    return u;
}

// Computes 30-bit morton code by interleaving the bits
// of the coordinates, supposing that they are normalized
// in the range [0.0, 1.0]
template<typename T> __global__ void _morton_encode(const SoAVec3<T> pos,
                                                    uint32_t *codes,
                                                    int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) {
        return;
    }

    // Scale coordinates to [0, 2^10)
    uint32_t x = (uint32_t) (pos.x(idx) * 1023.0f);
    uint32_t y = (uint32_t) (pos.y(idx) * 1023.0f);
    uint32_t z = (uint32_t) (pos.z(idx) * 1023.0f);

    x = _expand_bits(x);
    y = _expand_bits(y);
    z = _expand_bits(z);

    // Left shift x by 2 bits, y by 1 bit, then bitwise or
    codes[idx] = x * 4 + y * 2 + z;
}

template<typename T> class Points
{
public:
    Points(int num_points) :
        _num_points(num_points),
        _rng(_SEED),
        _gl_buffers(false)
    {
        _pos.alloc(num_points);
        _init(num_points);
    }

    Points(int num_points, T *x, T *y, T *z) :
        _num_points(num_points),
        _rng(_SEED),
        _gl_buffers(true)
    {
        _pos.x() = x;
        _pos.y() = y;
        _pos.z() = z;
        _init(num_points);
    }

    uint32_t *get_d_unique_codes_ptr()
    {
        return _unique_codes;
    }

    int *get_d_codes_first_point_idx_ptr()
    {
        return _codes_first_point_idx;
    }

    const SoAVec3<T> &get_d_pos() const
    {
        return _pos;
    }

    const SoAVec3<T> &get_d_scan_pos() const
    {
        return _scan_pos;
    }

    void sample_uniform()
    {
        thrust::host_vector<T> x(_num_points);
        thrust::host_vector<T> y(_num_points);
        thrust::host_vector<T> z(_num_points);

        thrust::uniform_real_distribution<T> dist;
        // thrust::normal_distribution<T> dist(0.5, 0.1);
        auto dist_gen = [&] { return max(0.0, min(1.0, dist(_rng))); };

        thrust::generate(x.begin(), x.end(), dist_gen);
        thrust::generate(y.begin(), y.end(), dist_gen);
        thrust::generate(z.begin(), z.end(), dist_gen);

        cudaMemcpy(_pos.x(),
                   thrust::raw_pointer_cast(&x[0]),
                   _num_points * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(_pos.y(),
                   thrust::raw_pointer_cast(&y[0]),
                   _num_points * sizeof(T),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(_pos.z(),
                   thrust::raw_pointer_cast(&z[0]),
                   _num_points * sizeof(T),
                   cudaMemcpyHostToDevice);
    }

    void sample_plummer()
    {
        _pos.plummer(_num_points, 0.2);
    }

    void compute_morton_codes()
    {
        _morton_encode<<<_num_points / MAX_THREADS_PER_BLOCK +
                         (_num_points % MAX_THREADS_PER_BLOCK > 0),
                         MAX_THREADS_PER_BLOCK>>>(_pos, _codes, _num_points);
    }

    void sort_by_codes(SoAVec3<T> &vel, SoAVec3<T> &acc)
    {
        thrust::sequence(thrust::device, _range, _range + _num_points);

        // Sort range once, then gather arrays to be sorted
        cub::DeviceMergeSort::SortPairs(_tmp_sort,
                                        _tmp_sort_size,
                                        _codes,
                                        _range,
                                        _num_points,
                                        less_op);

        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       _pos.x(),
                       _tmp_pos.x());
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       _pos.y(),
                       _tmp_pos.y());
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       _pos.z(),
                       _tmp_pos.z());

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

            swap_ptr(&_pos.x(), &_tmp_pos.x());
            swap_ptr(&_pos.y(), &_tmp_pos.y());
            swap_ptr(&_pos.z(), &_tmp_pos.z());
        }

        // TODO: refactor
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       vel.x(),
                       _tmp_vel.x());
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       vel.y(),
                       _tmp_vel.y());
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       vel.z(),
                       _tmp_vel.z());

        swap_ptr(&vel.x(), &_tmp_vel.x());
        swap_ptr(&vel.y(), &_tmp_vel.y());
        swap_ptr(&vel.z(), &_tmp_vel.z());

        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       acc.x(),
                       _tmp_acc.x());
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       acc.y(),
                       _tmp_acc.y());
        thrust::gather(thrust::device,
                       _range,
                       _range + _num_points,
                       acc.z(),
                       _tmp_acc.z());

        swap_ptr(&acc.x(), &_tmp_acc.x());
        swap_ptr(&acc.y(), &_tmp_acc.y());
        swap_ptr(&acc.z(), &_tmp_acc.z());
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

        // Scan coordinates to later compute the barycenter
        // of the octree nodes
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _pos.x(),
                                      _scan_pos.x(),
                                      _num_points);
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _pos.y(),
                                      _scan_pos.y(),
                                      _num_points);
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _pos.z(),
                                      _scan_pos.z(),
                                      _num_points);
    }

    ~Points()
    {
        if (!_gl_buffers) {
            _pos.free();
        }
        _tmp_pos.free();
        _scan_pos.free();
        _tmp_vel.free();
        _tmp_acc.free();

        cudaFree(_codes);
        cudaFree(_unique_codes);
        cudaFree(_codes_occurrences);
        cudaFree(_codes_first_point_idx);

        cudaFree(_range);

        cudaFree(_tmp_sort);
        cudaFree(_tmp_runlength);
        cudaFree(_tmp_scan);
    }

private:
    void _init(int num_points)
    {
        _tmp_pos.alloc(num_points);
        _scan_pos.alloc(num_points);
        _tmp_vel.alloc(num_points);
        _tmp_acc.alloc(num_points);

        cudaMalloc(&_codes, num_points * sizeof(uint32_t));
        cudaMalloc(&_unique_codes, num_points * sizeof(uint32_t));
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
    }

    const static int _SEED = 100;

    thrust::default_random_engine _rng;

    // Whether _pos buffers are OpenGL mapped resources
    const bool _gl_buffers;

    int _num_points;

    SoAVec3<T> _pos;
    SoAVec3<T> _tmp_pos;
    SoAVec3<T> _scan_pos;
    SoAVec3<T> _tmp_vel;
    SoAVec3<T> _tmp_acc;

    uint32_t *_codes;
    uint32_t *_unique_codes;
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
};

#endif
