#include "cuda/validator.cuh"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include <cub/device/device_reduce.cuh>

#include "cuda/soa_vec3.cuh"
#include "cuda/utils.cuh"
#include "cuda/physics_common.cuh"

namespace
{
    template<typename T>
    __device__ __inline__ T compute_kinetic_energy(int body_idx,
                                                   const SoAVec3<T> vel)
    {
        T vx = vel.x(body_idx);
        T vy = vel.y(body_idx);
        T vz = vel.z(body_idx);

        return 0.5f * (vx * vx + vy * vy + vz * vz);
    }

    template<typename T>
    __device__ __inline__ T compute_pairwise_potential_energy(T dist_sq, T gravity)
    {
        return -0.5f * gravity * __frsqrt_rn(dist_sq);
    }

    template<typename T>
    __global__ void compute_acc_error(const SoAVec3<T> acc,
                                      const SoAVec3<T> acc_ap,
                                      T *acc_err,
                                      int num_bodies)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_bodies) {
            return;
        }

        T ax = acc.x(idx);
        T ay = acc.y(idx);
        T az = acc.z(idx);

        T ax_ap = acc_ap.x(idx);
        T ay_ap = acc_ap.y(idx);
        T az_ap = acc_ap.z(idx);

        T dist_sq = PhysicsCommon<T>::
                    compute_dist_sq(ax, ay, az, ax_ap, ay_ap, az_ap);

        acc_err[idx] = __fsqrt_rn(dist_sq) * __frsqrt_rn(ax_ap * ax_ap +
                                                         ay_ap * ay_ap +
                                                         az_ap * az_ap);
    }

    template<typename T> __global__ void
    compute_conserved_quantities(const SoAVec3<T> pos,
                                 const SoAVec3<T> vel,
                                 const SoAVec3<T> pos_ap,
                                 const SoAVec3<T> vel_ap,
                                 T *energy,
                                 T *energy_ap,
                                 int curr_step,
                                 int num_bodies)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_bodies) {
            return;
        }

        T body_energy = compute_kinetic_energy(idx, vel);
        T body_energy_ap = compute_kinetic_energy(idx, vel_ap);

        T p1x = pos.x(idx);
        T p1y = pos.y(idx);
        T p1z = pos.z(idx);

        T p1x_ap = pos_ap.x(idx);
        T p1y_ap = pos_ap.y(idx);
        T p1z_ap = pos_ap.z(idx);

        T fx = 0.0f;
        T fy = 0.0f;
        T fz = 0.0f;

        T gravity = PhysicsCommon<T>::get_gravity();

    #pragma unroll 32
        for (int i = 0; i < num_bodies; ++i) {
            if (idx != i) {

                T p2x_ap = pos_ap.x(i);
                T p2y_ap = pos_ap.y(i);
                T p2z_ap = pos_ap.z(i);

                T dist_sq_ap = PhysicsCommon<T>::
                               compute_dist_sq(p1x_ap, p1y_ap, p1z_ap,
                                               p2x_ap, p2y_ap, p2z_ap);
                body_energy_ap +=
                    compute_pairwise_potential_energy(dist_sq_ap, gravity);

                T p2x = pos.x(i);
                T p2y = pos.y(i);
                T p2z = pos.z(i);
                T dist_sq = PhysicsCommon<T>::
                            compute_dist_sq(p1x, p1y, p1z, p2x, p2y, p2z);
                body_energy += compute_pairwise_potential_energy(dist_sq,
                                                                 gravity);
            }
        }

        energy[idx] = body_energy;
        energy_ap[idx] = body_energy_ap;
    }

    template<typename T>
    __global__ void compute_ap_forces(const SoAVec3<T> pos_ap,
                                      SoAVec3<T> acc_ap,
                                      int num_bodies)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_bodies) {
            return;
        }

        T p1x = pos_ap.x(idx);
        T p1y = pos_ap.y(idx);
        T p1z = pos_ap.z(idx);

        T fx = 0.0f;
        T fy = 0.0f;
        T fz = 0.0f;

        T gravity = PhysicsCommon<T>::get_gravity();
        T softening_factor = PhysicsCommon<T>::get_softening_factor();

    #pragma unroll 32
        for (int i = 0; i < num_bodies; ++i) {

            T p2x = pos_ap.x(i);
            T p2y = pos_ap.y(i);
            T p2z = pos_ap.z(i);

            PhysicsCommon<T>::
            accumulate_pairwise_force(p1x, p1y, p1z,
                                      p2x, p2y, p2z,
                                      (T) 1.0f, (T) 1.0f,
                                      fx, fy, fz,
                                      gravity,
                                      softening_factor);
        }

        acc_ap.x(idx) = fx;
        acc_ap.y(idx) = fy;
        acc_ap.z(idx) = fz;
    }
}

template<typename T>
Validator<T>::Validator(const SoAVec3<T> &pos,
                        const SoAVec3<T> &vel,
                        const SoAVec3<T> &acc,
                        const int *d_sort_indices,
                        int num_bodies,
                        float dt,
                        int max_timesteps) :
    pos_(pos),
    vel_(vel),
    acc_(acc),
    sort_indices_(d_sort_indices),
    num_bodies_(num_bodies),
    dt_(dt),
    max_timesteps_(max_timesteps),
    curr_step_(0)
{
    pos_ap_.alloc(num_bodies);
    vel_ap_.alloc(num_bodies);
    vel_half_ap_.alloc(num_bodies);
    acc_ap_.alloc(num_bodies);
    tmp_pos_ap_.alloc(num_bodies);
    tmp_vel_ap_.alloc(num_bodies);
    tmp_acc_ap_.alloc(num_bodies);

    cudaMalloc(&energy_, num_bodies * sizeof(T));
    cudaMalloc(&energy_ap_, num_bodies * sizeof(T));
    cudaMalloc(&acc_err_, num_bodies * sizeof(T));

    tmp_reduce_ = nullptr;
    cub::DeviceReduce::Sum(tmp_reduce_,
                           tmp_reduce_size_,
                           energy_,
                           sys_energy_,
                           num_bodies);
    cudaMalloc(&tmp_reduce_, tmp_reduce_size_);

    cudaMalloc(&sys_energy_, max_timesteps * sizeof(T));
    cudaMalloc(&sys_energy_ap_, max_timesteps * sizeof(T));
    cudaMalloc(&sys_momentum_mag_, max_timesteps * sizeof(T));
    cudaMalloc(&sys_momentum_mag_ap_, max_timesteps * sizeof(T));
    cudaMalloc(&avg_acc_err_, max_timesteps * sizeof(T));
}

template<typename T>
void Validator<T>::copy_initial_conditions()
{
    pos_ap_.copy(pos_, num_bodies_);
    vel_ap_.copy(vel_, num_bodies_);
    acc_ap_.copy(acc_, num_bodies_);
}

template<typename T>
void Validator<T>::update_all_pairs()
{
    int num_blocks = (num_bodies_ - 1) / MAX_THREADS_PER_BLOCK + 1;

    tmp_pos_ap_.gather(pos_ap_, sort_indices_, num_bodies_);
    tmp_vel_ap_.gather(vel_ap_, sort_indices_, num_bodies_);
    tmp_acc_ap_.gather(acc_ap_, sort_indices_, num_bodies_);

    pos_ap_.swap(tmp_pos_ap_);
    vel_ap_.swap(tmp_vel_ap_);
    acc_ap_.swap(tmp_acc_ap_);

    compute_acc_error<<<num_blocks, MAX_THREADS_PER_BLOCK>>>(acc_,
                                                             acc_ap_,
                                                             acc_err_,
                                                             num_bodies_);
    cub::DeviceReduce::Sum(tmp_reduce_,
                           tmp_reduce_size_,
                           acc_err_,
                           avg_acc_err_ + curr_step_,
                           num_bodies_);

    compute_conserved_quantities<<<num_blocks,
                                   MAX_THREADS_PER_BLOCK>>>(pos_,
                                                            vel_,
                                                            pos_ap_,
                                                            vel_ap_,
                                                            energy_,
                                                            energy_ap_,
                                                            curr_step_,
                                                            num_bodies_);
    // Computing total energies
    cub::DeviceReduce::Sum(tmp_reduce_,
                           tmp_reduce_size_,
                           energy_,
                           sys_energy_ + curr_step_,
                           num_bodies_);
    cub::DeviceReduce::Sum(tmp_reduce_,
                           tmp_reduce_size_,
                           energy_ap_,
                           sys_energy_ap_ + curr_step_,
                           num_bodies_);

    // Solving for all-pairs pos
    leapfrog_integrate_pos<<<num_blocks,
                             MAX_THREADS_PER_BLOCK>>>(pos_ap_,
                                                      vel_ap_,
                                                      vel_half_ap_,
                                                      acc_ap_,
                                                      dt_,
                                                      num_bodies_);
    // Computing all-pairs forces
    compute_ap_forces<<<num_blocks,
                        MAX_THREADS_PER_BLOCK>>>(pos_ap_,
                                                 acc_ap_,
                                                 num_bodies_);
    // Solving for all-pairs vel
    leapfrog_integrate_vel<<<num_blocks,
                             MAX_THREADS_PER_BLOCK>>>(vel_ap_,
                                                      vel_half_ap_,
                                                      acc_ap_,
                                                      dt_,
                                                      num_bodies_);

    curr_step_ = (curr_step_ + 1) % max_timesteps_;
}

template<typename T>
void Validator<T>::dump_history_to_csv(const std::string &file_path)
{
    std::ofstream out(file_path);

    std::vector<T> energy;
    std::vector<T> energy_ap;
    std::vector<T> acc_err;
    energy.reserve(curr_step_ + 1);
    energy_ap.reserve(curr_step_ + 1);
    acc_err.reserve(curr_step_ + 1);

    cudaMemcpy(energy.data(),
               sys_energy_,
               (curr_step_ + 1) * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(energy_ap.data(),
               sys_energy_ap_,
               (curr_step_ + 1) * sizeof(T),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(acc_err.data(),
               avg_acc_err_,
               (curr_step_ + 1) * sizeof(T),
               cudaMemcpyDeviceToHost);

    out << "sys_energy,sys_energy_ap,avg_acc_err" << std::endl;
    for (int i = 0; i < curr_step_; ++i) {
        out << energy[i] << ","
            << energy_ap[i] << ","
            << acc_err[i] / num_bodies_ << std::endl;
    }
}

template<typename T>
Validator<T>::~Validator()
{
    pos_ap_.free();
    vel_ap_.free();
    vel_half_ap_.free();
    acc_ap_.free();
    tmp_pos_ap_.free();
    tmp_vel_ap_.free();
    tmp_acc_ap_.free();

    cudaFree(energy_);
    cudaFree(energy_ap_);
    cudaFree(acc_err_);
    cudaFree(tmp_reduce_);
    cudaFree(sys_energy_),
    cudaFree(sys_energy_ap_),
    cudaFree(sys_momentum_mag_);
    cudaFree(sys_momentum_mag_ap_);
    cudaFree(avg_acc_err_);
}

template class Validator<float>;
template class Validator<double>;
