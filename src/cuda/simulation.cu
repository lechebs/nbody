#include "Simulation.hpp"

#include <string>

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
    Impl(int num_points, float domain_size, float theta, float dt) :
        points(num_points, domain_size),
        btree(num_points),
        octree(num_points, domain_size),
        bh(points.get_d_pos(), num_points, theta, dt),
        validator(points.get_d_pos(),
                  bh.get_d_vel(),
                  bh.get_d_acc(),
                  points.get_d_sort_indices_ptr(),
                  num_points,
                  dt,
                  5000) {}

    Impl(int num_points,
         float domain_size,
         float theta,
         float dt,
         void *mapped_ptrs[7]) :
        points(num_points,
               domain_size,
               static_cast<U *>(mapped_ptrs[0]),
               static_cast<U *>(mapped_ptrs[1]),
               static_cast<U *>(mapped_ptrs[2])),
        btree(num_points),
        octree(num_points,
               domain_size,
               static_cast<U *>(mapped_ptrs[3]),
               static_cast<U *>(mapped_ptrs[4]),
               static_cast<U *>(mapped_ptrs[5]),
               static_cast<U *>(mapped_ptrs[6])),
        bh(points.get_d_pos(), num_points, theta, dt),
        validator(points.get_d_pos(),
                  bh.get_d_vel(),
                  bh.get_d_acc(),
                  points.get_d_sort_indices_ptr(),
                  num_points,
                  dt,
                  5000) {}

    void updatePoints()
    {
        points.compute_morton_codes();
        points.sort_by_codes();
        bh.sort_bodies(points.get_d_sort_indices_ptr());
        points.compute_unique_codes(btree.get_d_num_leaves_ptr());
        points.scan_attributes();
    }

    void updateOctree(int max_num_codes_per_leaf)
    {
        //std::cout << "num_unique_codes="
        //          << btree.get_num_leaves() << std::endl;
        btree.reset_max_num_leaves();
        btree.generate_leaves(points.get_d_unique_codes_ptr(),
                              max_num_codes_per_leaf);
        _num_leaves = btree.get_num_leaves();
        btree.set_max_num_leaves(_num_leaves);
        //std::cout << "num_leaves=" << _num_leaves << std::endl;

        btree.build(points.get_d_unique_codes_ptr());
        btree.sort_to_bfs_order();
        btree.compute_octree_map();

        //btree.print();

        octree.set_max_num_nodes(btree.get_max_num_nodes());
        octree.build(btree);
        octree.compute_nodes_points_range(
            btree.get_d_leaf_first_code_idx_ptr(),
            points.get_d_codes_first_point_idx_ptr());
        octree.compute_nodes_barycenter(points);

        //octree.print();
    }

    void updateBodiesPos()
    {
        bh.solve_pos(octree,
                     points.get_d_codes_first_point_idx_ptr(),
                     btree.get_d_leaf_first_code_idx_ptr(),
                     _num_leaves);
    }

    void updateBodiesVel()
    {
        bh.solve_vel(octree,
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
    _params(params)
{
    _impl = std::make_unique<Simulation::Impl<T>>(params.num_points,
                                                  params.domain_size,
                                                  params.theta,
                                                  params.dt);
}

template<typename T>
Simulation<T>::Simulation(Simulation<T>::Params &params, GLuint buffers[7]) :
    _params(params)
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

    _impl = std::make_unique<Simulation::Impl<T>>(params.num_points,
                                                  params.domain_size,
                                                  params.theta,
                                                  params.dt,
                                                  mapped_ptrs);
}

template<typename T>
void Simulation<T>::spawnBodies()
{
    //_impl->spawnBodies();

    InitialConditions<T>::set_seed(42);

    /*
    InitialConditions<T>::sample_uniform(_params.domain_size,
                                         _impl->points.get_d_pos(),
                                         _params.num_points);
    */
    InitialConditions<T>::sample_sphere(_params.domain_size * 0.2,
                                        _params.domain_size / 3,
                                        _params.domain_size / 3,
                                        _params.domain_size / 2,
                                        _params.domain_size,
                                        _impl->points.get_d_pos(),
                                        _params.num_points / 3,
                                        0);

    InitialConditions<T>::sample_sphere(_params.domain_size * 0.2,
                                        _params.domain_size / 3 * 2,
                                        _params.domain_size / 2,
                                        _params.domain_size / 2,
                                        _params.domain_size,
                                        _impl->points.get_d_pos(),
                                        _params.num_points / 3,
                                        _params.num_points / 3);

    InitialConditions<T>::sample_sphere(_params.domain_size * 0.2,
                                        _params.domain_size / 3 * 2,
                                        _params.domain_size / 3 * 2,
                                        _params.domain_size / 2,
                                        _params.domain_size,
                                        _impl->points.get_d_pos(),
                                        _params.num_points / 3,
                                        _params.num_points / 3 * 2);

    //_impl->validator.copy_initial_conditions();
    _impl->updatePoints();
    _impl->updateOctree(_params.max_num_codes_per_leaf);
}

template<typename T>
void Simulation<T>::update()
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    //_impl->validator.update_all_pairs();

    // Solve for position
    _impl->updateBodiesPos();
    // Update octree
    _impl->updatePoints();
    _impl->updateOctree(_params.max_num_codes_per_leaf);
    // Solve for velocity
    _impl->updateBodiesVel();


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "elapsed=" << ms << std::endl;
}

template<typename T>
int Simulation<T>::get_num_octree_nodes()
{
    return _impl->octree.get_num_nodes();
}

template<typename T>
void Simulation<T>::writeHistory(const std::string &csv_file_path)
{
    _impl->validator.dump_history_to_csv(csv_file_path);
}

template<typename T>
Simulation<T>::~Simulation() {}

template class Simulation<float>;
template class Simulation<double>;
