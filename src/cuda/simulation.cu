#include "simulation.hpp"

#include <string>
#include <vector>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda/points.cuh"
#include "cuda/btree.cuh"
#include "cuda/octree.cuh"
#include "cuda/barnes_hut.cuh"
#include "cuda/validator.cuh"
#include "cuda/initial_conditions.cuh"

template<typename T>
template<typename U>
struct Simulation<T>::Impl
{
    Impl(const Simulation<T>::Params &params) :
        points(params.num_points, params.domain_size),
        btree(params.num_points),
        octree(params.num_points, params.domain_size),
        bh(points.get_d_pos(), params.num_points, params.theta, params.dt),
        validator(points.get_d_pos(),
                  bh.get_d_vel(),
                  bh.get_d_acc(),
                  points.get_d_mass(),
                  points.get_d_sort_indices_ptr(),
                  params.num_points,
                  params.dt,
                  params.num_steps_validator) {}

    Impl(const Simulation<T>::Params &params, void *mapped_ptrs[7]) :
        points(params.num_points,
               params.domain_size,
               static_cast<U *>(mapped_ptrs[0]),
               static_cast<U *>(mapped_ptrs[1]),
               static_cast<U *>(mapped_ptrs[2])),
        btree(params.num_points),
        octree(params.num_points,
               params.domain_size,
               static_cast<U *>(mapped_ptrs[3]),
               static_cast<U *>(mapped_ptrs[4]),
               static_cast<U *>(mapped_ptrs[5]),
               static_cast<U *>(mapped_ptrs[6])),
        bh(points.get_d_pos(), params.num_points, params.theta, params.dt),
        validator(points.get_d_pos(),
                  bh.get_d_vel(),
                  bh.get_d_acc(),
                  points.get_d_mass(),
                  points.get_d_sort_indices_ptr(),
                  params.num_points,
                  params.dt,
                  params.num_steps_validator)
    {
        PhysicsCommon<U>::set_params(params.gravity,
                                     params.softening_factor *
                                        params.softening_factor,
                                     params.velocity_dampening,
                                     params.domain_size);
    }

    void update_points()
    {
        points.compute_morton_codes();
        points.sort_by_codes();
        bh.sort_bodies(points.get_d_sort_indices_ptr());
        points.compute_unique_codes(btree.get_d_num_leaves_ptr());
        points.compute_codes_first_point_idx();
    }

    void update_octree(int max_num_codes_per_leaf)
    {
        btree.reset_max_num_leaves();
        btree.generate_leaves(points.get_d_unique_codes_ptr(),
                              max_num_codes_per_leaf);
        _num_leaves = btree.get_num_leaves();
        btree.set_max_num_leaves(_num_leaves);

        btree.build(points.get_d_unique_codes_ptr());
        btree.sort_to_bfs_order();
        btree.compute_octree_map();

        octree.set_max_num_nodes(btree.get_max_num_nodes());
        octree.build(btree);
        octree.compute_nodes_points_range(
            btree.get_d_leaf_first_code_idx_ptr(),
            points.get_d_codes_first_point_idx_ptr());
        octree.compute_nodes_barycenter(points);
    }

    void update_bodies_pos()
    {
        bh.solve_pos(octree,
                     points.get_d_codes_first_point_idx_ptr(),
                     btree.get_d_leaf_first_code_idx_ptr(),
                     _num_leaves);
    }

    void update_bodies_vel()
    {
        bh.solve_vel(octree,
                     points.get_d_mass(),
                     points.get_d_codes_first_point_idx_ptr(),
                     btree.get_d_leaf_first_code_idx_ptr(),
                     _num_leaves);
    }

    Points<T> points;
    Btree btree;
    Octree<T> octree;
    BarnesHut<T> bh;
    Validator<T> validator;

private:
    int _num_leaves;
};

template<typename T>
Simulation<T>::Simulation(Simulation<T>::Params &params) :
    params_(params)
{
    _impl = std::make_unique<Simulation::Impl<T>>(params);
}

template<typename T>
Simulation<T>::Simulation(Simulation<T>::Params &params, GLuint buffers[7]) :
    params_(params)
{
    void *mapped_ptrs[7];

    for (int i = 0; i < 7; ++i) {
        cudaGraphicsResource_t res;
        // Mapping OpenGL buffer for access by CUDA
        cudaGraphicsGLRegisterBuffer(&res,
                                     buffers[i],
                                     cudaGraphicsRegisterFlagsNone);
        cudaGraphicsMapResources(1, &res);
        // Obtaining device pointer
        size_t size;
        cudaGraphicsResourceGetMappedPointer(&mapped_ptrs[i], &size, res);
    }

    _impl = std::make_unique<Simulation::Impl<T>>(params,
                                                  mapped_ptrs);
}

template<typename T>
void Simulation<T>::spawn_points()
{
    InitialConditions<T>::set_seed(42);

    InitialConditions<T>::sample_disk(10.0,
                                      1.0,
                                      0.03,
                                      0.30,
                                      -1.5,
                                      0.05,
                                      0.5,
                                      0.5,
                                      0.5,
                                      params_.domain_size,
                                      { 0.0, 0.0, 0.0 },
                                      _impl->points.get_d_pos(),
                                      _impl->bh.get_d_vel(),
                                      _impl->points.get_d_mass(),
                                      params_.num_points,
                                      0);

    if (params_.num_steps_validator > 0) {
        _impl->validator.copy_initial_conditions();
    }
    _impl->update_points();
    _impl->update_octree(params_.max_num_codes_per_leaf);
}

template<typename T>
void Simulation<T>::update()
{
    if (params_.num_steps_validator > 0) {
        _impl->validator.update_all_pairs();
    }

    // Solve for position
    _impl->update_bodies_pos();
    // Update octree
    _impl->update_points();
    _impl->update_octree(params_.max_num_codes_per_leaf);
    // Solve for velocity
    _impl->update_bodies_vel();

    if (render_all_pairs_) {
        _impl->points.get_d_pos().copy(_impl->validator.get_d_pos_ap(),
                                       params_.num_points);
    }
}

template<typename T>
int Simulation<T>::get_num_octree_nodes()
{
    return _impl->octree.get_num_nodes();
}

template<typename T>
void Simulation<T>::set_render_all_pairs(bool value)
{
    render_all_pairs_ = value;
}

template<typename T>
void Simulation<T>::write_validation_history(const std::string &csv_file_path)
{
    _impl->validator.dump_history_to_csv(csv_file_path);
}

template<typename T>
Simulation<T>::~Simulation() {}

template class Simulation<float>;
template class Simulation<double>;
