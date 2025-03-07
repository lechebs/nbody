#include "utils_gpu.cuh"

#include <cub/device/device_merge_sort.cuh>

typedef unsigned int uint32_t;

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
template<typename T> __global__ void morton_encode(const T *x_,
                                                   const T *y_,
                                                   const T *z_,
                                                   uint32_t *codes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Scale coordinates to [0, 2^10)
    uint32_t x = (uint32_t) (x_[idx] * 1023.0f);
    uint32_t y = (uint32_t) (y_[idx] * 1023.0f);
    uint32_t z = (uint32_t) (z_[idx] * 1023.0f);

    x = _expand_bits(x);
    y = _expand_bits(y);
    z = _expand_bits(z);

    // Left shift x by 2 bits, y by 1 bit, then bitwise or
    codes[idx] = x * 4 + y * 2 + z;
}

template<typename T>
class Points
{
public:
    Points(int num_points) : _num_points(num_points), _rng(_SEED)
    {
        cudaMalloc(&_num_unique_codes, sizeof(int));

        cudaMalloc(&_pos.x, num_points * sizeof(T));
        cudaMalloc(&_pos.y, num_points * sizeof(T));
        cudaMalloc(&_pos.z, num_points * sizeof(T));

        cudaMalloc(&_tmp_pos.x, num_points * sizeof(T));
        cudaMalloc(&_tmp_pos.y, num_points * sizeof(T));
        cudaMalloc(&_tmp_pos.z, num_points * sizeof(T));

        cudaMalloc(&_scan_pos.x, num_points * sizeof(T));
        cudaMalloc(&_scan_pos.y, num_points * sizeof(T));
        cudaMalloc(&_scan_pos.z, num_points * sizeof(T));

        cudaMalloc(&_codes, num_points * sizeof(uint32_t));
        cudaMalloc(&_unique_codes, num_points * sizeof(uint32_t));
        cudaMalloc(&_codes_occurrences, num_points * sizeof(int));
        cudaMalloc(&_unique_codes_first_point, num_points * sizeof(int));

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

        _tmp_runlength = nullptr;
        cub::DeviceRunLengthEncode::Encode(_tmp_runlength,
                                           _tmp_runlength_size,
                                           _codes,
                                           _unique_codes,
                                           _codes_occurrences,
                                           _num_unique_codes,
                                           num_points);
        cudaMalloc(&_tmp_runlength, _tmp_runlength_size);

        _tmp_scan = nullptr;
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _codes_occurrences,
                                      _unique_codes_first_point,
                                      num_points);
        cudaMalloc(&_tmp_scan, _tmp_scan_size);
    }

    int get_num_unique_codes()
    {
        int num_unique_codes;
        // TODO: cudaMemcpyAsync?
        cudaMemcpy(&num_unique_codes, sizeof(int), cudaMemcpyDeviceToHost);
        return num_unique_codes;
    }

    int *get_d_num_unique_codes_ptr()
    {
        return _num_unique_codes;
    }

    int *get_d_unique_codes_first_point_ptr()
    {
        return _unique_codes_first_point;
    }

    void sample_uniform()
    {
        thrust::uniform_real_distribution<T> dist;
        auto dist_gen = [&] { return dist(_rng); };

        thrust::generate(_pos.x, _pos.x + _num_points, dist_gen);
        thrust::generate(_pos.y, _pos.y + _num_points, dist_gen);
        thrust::generate(_pos.z, _pos.z + _num_points, dist_gen);
    }

    void compute_morton_codes()
    {
        morton_encode<<<_num_points / MAX_THREADS_PER_BLOCK +
                        (_num_points % MAX_THREADS_PER_BLOCK > 0),
                        MAX_THREADS_PER_BLOCK>>>(_pos.x,
                                                 _pos.y,
                                                 _pos.z,
                                                 _codes);
    }

    void sort_by_codes()
    {
        thrust::sequence(thrust::device, _range, _range + _num_points);

        // Sort range once, then gather arrays to be sorted
        cub::DeviceMergeSort::SortPairs(_tmp_sort,
                                        _tmp_sort_size,
                                        _codes,
                                        _range,
                                        num_points,
                                        less_op);

        thrust::gather(
            thrust::device, _range, _range + _num_points, _pos.x, _tmp_pos.x);
        thrust::gather(
            thrust::device, _range, _range + _num_points, _pos.y, _tmp_pos.y);
        thrust::gather(
            thrust::device, _range, _range + _num_points, _pos.z, _tmp_pos.z);

        swap_ptr(_pos.x, _tmp_pos.x);
        swap_ptr(_pos.y, _tmp_pos.y);
        swap_ptr(_pos.z, _tmp_pos.z);
    }

    void compute_unique_codes()
    {
        // Obtaining unique codes and counting occurrences
        // using run-length encoding
        cub::DeviceRunLengthEncode::Encode(_tmp_runlength,
                                           _tmp_runlength_size,
                                           _codes,
                                           _unique_codes,
                                           _codes_occurrences,
                                           _num_unique_codes,
                                           _num_points);

        // For each unique code, obtains the index of the 
        // first point that got mapped to it; useful to recover
        // all of the points encoded to a single code
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _codes_occurrences,
                                      _unique_codes_first_point,
                                      _num_points);
    }

    void scan_attributes()
    {
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _pos.x,
                                      _scan_pos.x,
                                      _num_points);
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _pos.y,
                                      _scan_pos.y,
                                      _num_points);
        cub::DeviceScan::ExclusiveSum(_tmp_scan,
                                      _tmp_scan_size,
                                      _pos.z,
                                      _scan_pos.z,
                                      _num_points);
    }

    ~Points()
    {
        cudaFree(_num_unique_codes);

        cudaFree(_pos.x);
        cudaFree(_pos.y);
        cudaFree(_pos.z);

        cudaFree(_tmp_pos.x);
        cudaFree(_tmp_pos.y);
        cudaFree(_tmp_pos.z);

        cudaFree(_scan_pos.x);
        cudaFree(_scan_pos.y);
        cudaFree(_scan_pos.z);

        cudaFree(_codes);
        cudaFree(_unique_codes);
        cudaFree(_codes_occurrences);
        cudaFree(_unique_codes_first_point);

        cudaFree(_range);

        cudaFree(_tmp_sort);
        cudaFree(_tmp_runlength);
        cudaFree(_tmp_scan);
    }

private:
    struct Position {
        T *x;
        T *y;
        T *z;
    };

    const static int _SEED = 100;
    const thrust::default_random_engine _rng;

    int _num_points;
    int *_num_unique_codes;

    struct Position _pos;
    struct Position _tmp_pos;
    struct Position _scan_pos;

    uint32_t *_codes;
    uint32_t *_unique_codes;
    int *_codes_occurrences;
    int *_unique_codes_first_point;

    int *_range;

    LessOp less_op;
    int *_tmp_sort;
    size_t _tmp_sort_size;

    int *_tmp_runlength;
    size_t _tmp_runlength_size;

    int *_tmp_scan;
    size_t _tmp_scan_size;
};
