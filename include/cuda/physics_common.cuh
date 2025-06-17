#ifndef PHYSICS_COMMON_CUH
#define PHYSICS_COMMON_CUH

#include <type_traits>

#include "cuda/soa_vec3.cuh"

/*
#define GRAVITY 1.0f
#define SOFTENING_FACTOR 1e-3f
#define VELOCITY_DAMPENING 0.0f
//#define VELOCITY_DAMPENING (1.0f - 0.8236f)
//#define VELOCITY_DAMPENING (1.0f - 0.8136f) for 2mln
#define DIST_SCALE 1.0f
*/

template<typename T>
class PhysicsCommon
{
public:
    static void set_params(T gravity,
                           T softening_factor,
                           T velocity_dampening,
                           T domain_size);

    static T get_gravity_h()
    {
        return gravity_;
    }

    static __device__ __forceinline__ T get_gravity();
    static __device__ __forceinline__ T get_softening_factor();
    static __device__ __forceinline__ T get_velocity_dampening();
    static __device__ __forceinline__ T get_domain_size();

    static __device__ __forceinline__
    T compute_dist_sq(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z)
    {
        T dx = p1x - p2x;
        T dy = p1y - p2y;
        T dz = p1z - p2z;

        return dx * dx + dy * dy + dz * dz;
    }

    static __device__ __inline__
    void accumulate_pairwise_force(
        T p1x, T p1y, T p1z,
        T p2x, T p2y, T p2z,
        T mass,
        T &dst_x, T &dst_y, T &dst_z,
        T gravity,
        T softening_factor)
    {
        T dist_x = p2x - p1x;
        T dist_y = p2y - p1y;
        T dist_z = p2z - p1z;

        T dist_sq = dist_x * dist_x + dist_y * dist_y + dist_z * dist_z;

        T inv_den = dist_sq + softening_factor * softening_factor;

        if constexpr (std::is_same_v<T, float>) {
            // Use fast inverse sqrt intrinsic
            inv_den = __frsqrt_rn(inv_den * inv_den * inv_den);
        } else {
            // Invert sqrt intrinsic
            inv_den = 1 / __dsqrt_rn(inv_den * inv_den * inv_den);
        }

        dst_x += mass * gravity * dist_x * inv_den;
        dst_y += mass * gravity * dist_y * inv_den;
        dst_z += mass * gravity * dist_z * inv_den;
    }

    static __device__
    void impose_boundary_conditions(T &x, T &y, T &z,
                                    T &vx, T &vy, T &vz);

private:
    static void *get_params_addr();

    static T gravity_;
};

template<typename T> __global__
void leapfrog_integrate_pos(SoAVec3<T> pos,
                            const SoAVec3<T> vel,
                            SoAVec3<T> vel_half,
                            const SoAVec3<T> acc,
                            float dt,
                            int num_bodies);

template<typename T> __global__
void leapfrog_integrate_vel(SoAVec3<T> vel,
                            const SoAVec3<T> vel_half,
                            const SoAVec3<T> acc,
                            float dt,
                            int num_bodies);

template class PhysicsCommon<float>;
template class PhysicsCommon<double>;

#endif
