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

template<typename T> class Points
{
public:
    Points(int num_points, float domain_size);
    Points(int num_points, float domain_size, T *x, T *y, T *z);

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

    void compute_morton_codes();

    void sort_by_codes();

    void compute_unique_codes(int *d_num_unique_codes);

    void compute_codes_first_point_idx();

    ~Points();

private:
    void init_(int num_points);

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
