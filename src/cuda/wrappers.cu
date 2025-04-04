#include "CUDAWrappers.hpp"

#include <GL/glew.h>
#include <cuda_gl_interop.h>

#include "cuda/points.cuh"
#include "cuda/btree.cuh"
#include "cuda/octree.cuh"
#include "cuda/barnes_hut.cuh"

namespace CUDAWrappers
{
    struct Simulation::Impl
    {
        Impl(int num_points) :
            points(num_points),
            btree(num_points),
            octree(num_points),
            bh(points.get_d_pos(), num_points) {}

        Impl(int num_points, void *mapped_ptrs[5]) :
            points(num_points,
                   static_cast<float *>(mapped_ptrs[0]),
                   static_cast<float *>(mapped_ptrs[1]),
                   static_cast<float *>(mapped_ptrs[2])),
            btree(num_points),
            octree(num_points,
                   static_cast<int *>(mapped_ptrs[3]),
                   static_cast<int *>(mapped_ptrs[4])),
            bh(points.get_d_pos(), num_points) {}

        void updatePoints()
        {
            points.compute_morton_codes();
            points.sort_by_codes(bh.get_d_vel());
            points.compute_unique_codes(btree.get_d_num_leaves_ptr());
            points.scan_attributes();
        }

        int updateOctree(int max_num_codes_per_leaf)
        {
            //std::cout << "num_unique_codes="
            //          << btree.get_num_leaves() << std::endl;
            btree.generate_leaves(points.get_d_unique_codes_ptr(),
                                  max_num_codes_per_leaf);
            int num_leaves = btree.get_num_leaves();
            // btree.set_max_num_leaves(num_leaves);
            // std::cout << "num_leaves=" << num_leaves << std::endl;

            btree.build(points.get_d_unique_codes_ptr());
            btree.sort_to_bfs_order();
            btree.compute_octree_map();

            // btree.print();

            // octree.set_max_num_nodes(btree.get_max_num_nodes());
            octree.build(btree);
            octree.compute_nodes_points_range(
                btree.get_d_leaf_first_code_idx_ptr(),
                points.get_d_codes_first_point_idx_ptr());
            octree.compute_nodes_barycenter(points);

            // octree.print();

            return num_leaves;
        }

        void updateBodies(int num_leaves)
        {
            bh.compute_forces(octree,
                              points.get_d_codes_first_point_idx_ptr(),
                              btree.get_d_leaf_first_code_idx_ptr(),
                              num_leaves);
            bh.update_bodies();
        }

        Points<float> points;
        Btree btree;
        Octree<float> octree;
        BarnesHut<float> bh;
    };

    Simulation::Simulation(Simulation::Params &params) :
        _params(params)
    {
        _impl = std::make_unique<Simulation::Impl>(params.num_points);
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
                                                   mapped_ptrs);
    }

    void Simulation::samplePoints()
    {
        _impl->points.sample_uniform();
    }

    void Simulation::update()
    {
        _impl->updatePoints();
        int num_leaves = _impl->updateOctree(_params.max_num_codes_per_leaf);
        _impl->updateBodies(num_leaves);
    }

    Simulation::~Simulation() {}
};
