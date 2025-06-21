#include "cuda/physics_common.cuh"

#include <type_traits>

template<typename T> struct Params_
{
    T gravity;
    T softening_factor_sq;
    T velocity_dampening;
    T domain_size;
};

__constant__ Params_<float> F_PARAMS_;
__constant__ Params_<double> D_PARAMS_;

template<typename T>
T PhysicsCommon<T>::gravity_ = 1.0f;

template<typename T>
void *PhysicsCommon<T>::get_params_addr()
{
    void *ptr;

    if constexpr (std::is_same_v<T, float>) {
        cudaGetSymbolAddress(&ptr, F_PARAMS_);
    } else {
        cudaGetSymbolAddress(&ptr, D_PARAMS_);
    }

    return ptr;
}

template<typename T>
void PhysicsCommon<T>::set_params(T gravity,
                                  T softening_factor_sq,
                                  T velocity_dampening,
                                  T domain_size)
{
    Params_<T> params;
    params.gravity = gravity;
    params.softening_factor_sq = softening_factor_sq;
    params.velocity_dampening = velocity_dampening;
    params.domain_size = domain_size;

    cudaMemcpy(get_params_addr(),
               &params,
               sizeof(struct Params_<T>),
               cudaMemcpyHostToDevice);

    gravity_ = gravity;
}

#define DEV_GETTER(var)                          \
template<typename T> __device__ __forceinline__  \
T PhysicsCommon<T>::get_##var()                  \
{                                                \
    if constexpr (std::is_same_v<T, float>) {    \
        return F_PARAMS_.var;                    \
    } else {                                     \
        return D_PARAMS_.var;                    \
    }                                            \
}                                                \

DEV_GETTER(gravity)
DEV_GETTER(softening_factor_sq)
DEV_GETTER(velocity_dampening)
DEV_GETTER(domain_size)

template<typename T> __device__
void PhysicsCommon<T>::impose_boundary_conditions(T &x, T &y, T &z,
                                                  T &vx, T &vy, T &vz)
{
    // Reflective boundary conditions

    if (x < 0.0f || x > get_domain_size()) {
        vx *= -1.0f;
        x = max(0.0f, min(get_domain_size(), x));
    }

    if (y < 0.0f || y > get_domain_size()) {
        vy *= -1.0f;
        y = max(0.0f, min(get_domain_size(), y));
    }

    if (z < 0.0f || z > get_domain_size()) {
        vz *= -1.0f;
        z = max(0.0f, min(get_domain_size(), z));
    }
}

template<typename T> __global__
void leapfrog_integrate_pos(SoAVec3<T> pos,
                            const SoAVec3<T> vel,
                            SoAVec3<T> vel_half,
                            const SoAVec3<T> acc,
                            float dt,
                            int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    T x = pos.x(idx);
    T y = pos.y(idx);
    T z = pos.z(idx);

    T vx = vel.x(idx) + (T) 0.5f * acc.x(idx) * dt;
    T vy = vel.y(idx) + (T) 0.5f * acc.y(idx) * dt;
    T vz = vel.z(idx) + (T) 0.5f * acc.z(idx) * dt;

    x += vx * dt;
    y += vy * dt;
    z += vz * dt;

    PhysicsCommon<T>::
    impose_boundary_conditions(x, y, z, vx, vy, vz);

    pos.x(idx) = x;
    pos.y(idx) = y;
    pos.z(idx) = z;

    vel_half.x(idx) = vx;
    vel_half.y(idx) = vy;
    vel_half.z(idx) = vz;
}

template<typename T> __global__
void leapfrog_integrate_vel(SoAVec3<T> vel,
                            const SoAVec3<T> vel_half,
                            const SoAVec3<T> acc,
                            float dt,
                            int num_bodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_bodies) {
        return;
    }

    T vx = vel_half.x(idx) + (T) 0.5f * acc.x(idx) * dt;
    T vy = vel_half.y(idx) + (T) 0.5f * acc.y(idx) * dt;
    T vz = vel_half.z(idx) + (T) 0.5f * acc.z(idx) * dt;

    vel.x(idx) = vx * ((T) 1.0f - PhysicsCommon<T>::get_velocity_dampening());
    vel.y(idx) = vy * ((T) 1.0f - PhysicsCommon<T>::get_velocity_dampening());
    vel.z(idx) = vz * ((T) 1.0f - PhysicsCommon<T>::get_velocity_dampening());
}

// Explicit instantiations

template __global__
void leapfrog_integrate_pos(SoAVec3<float> pos,
                            const SoAVec3<float> vel,
                            SoAVec3<float> vel_half,
                            const SoAVec3<float> acc,
                            float dt,
                            int num_bodies);
template __global__
void leapfrog_integrate_pos(SoAVec3<double> pos,
                            const SoAVec3<double> vel,
                            SoAVec3<double> vel_half,
                            const SoAVec3<double> acc,
                            float dt,
                            int num_bodies);

template __global__
void leapfrog_integrate_vel(SoAVec3<float> vel,
                            const SoAVec3<float> vel_half,
                            const SoAVec3<float> acc,
                            float dt,
                            int num_bodies);
template __global__
void leapfrog_integrate_vel(SoAVec3<double> vel,
                            const SoAVec3<double> vel_half,
                            const SoAVec3<double> acc,
                            float dt,
                            int num_bodies);
