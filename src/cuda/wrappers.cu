#include "CUDAWrappers.hpp"

#include <string>

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda/points.cuh"
#include "cuda/btree.cuh"
#include "cuda/octree.cuh"
#include "cuda/barnes_hut.cuh"
#include "cuda/validator.cuh"

namespace CUDAWrappers
{
    struct Simulation::Impl
    {
        Impl(int num_points, float theta, float dt) :
            points(num_points),
            btree(num_points),
            octree(num_points),
            bh(points.get_d_pos(), num_points, theta, dt),
            validator(points.get_d_pos(),
                      bh.get_d_vel(),
                      bh.get_d_acc(),
                      points.get_d_sort_indices_ptr(),
                      num_points,
                      dt,
                      5000) {}

        Impl(int num_points, float theta, float dt, void *mapped_ptrs[5]) :
            points(num_points,
                   static_cast<float *>(mapped_ptrs[0]),
                   static_cast<float *>(mapped_ptrs[1]),
                   static_cast<float *>(mapped_ptrs[2])),
            btree(num_points),
            octree(num_points,
                   static_cast<int *>(mapped_ptrs[3]),
                   static_cast<int *>(mapped_ptrs[4])),
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
            points.sort_by_codes(bh.get_d_vel(),
                                 bh.get_d_vel_half(),
                                 bh.get_d_acc());
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

        Points<float> points;
        Btree btree;
        Octree<float> octree;
        BarnesHut<float> bh;
        Validator<float> validator;

    private:
        int _num_leaves;
    };

    Simulation::Simulation(Simulation::Params &params) :
        _params(params)
    {
        _impl = std::make_unique<Simulation::Impl>(params.num_points,
                                                   params.theta,
                                                   params.dt);
    }

    Simulation::Simulation(Simulation::Params &params, GLuint buffers[5]) :
        _params(params)
    {
        void *mapped_ptrs[5];

        for (int i = 0; i < 5; ++i) {
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

        _impl = std::make_unique<Simulation::Impl>(params.num_points,
                                                   params.theta,
                                                   params.dt,
                                                   mapped_ptrs);
    }

    void Simulation::samplePoints()
    {
        //_impl->points.sample_uniform();
        //_impl->points.sample_plummer();
        _impl->points.sample_sphere();
        //_impl->points.sample_disk();

        // TODO: Sample velocities here as well

        _impl->validator.copy_initial_conditions();

        _impl->updatePoints();
        _impl->updateOctree(_params.max_num_codes_per_leaf);
    }

    void Simulation::update()
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

    void Simulation::writeHistory(const std::string &csv_file_path)
    {
        _impl->validator.dump_history_to_csv(csv_file_path);
    }

    Simulation::~Simulation() {}
};
