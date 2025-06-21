#ifndef CAMERA_H
#define CAMERA_H

#include "vector.hpp"
#include "matrix.hpp"

using vec3 = Vector<float, 3>;
using mat4 = Matrix<float, 4, 4>;

class Camera
{
public:
    Camera(float fovy, /* Vertical field of view */
           float aspect_ratio, /* Window width/height */
           float near, /* z coordinate of near clipping plane */
           float far   /* z coordinate of far clipping plane */);

    const vec3 &get_position() const;
    // Getters for camera transformation matrices
    const mat4 &get_world_to_camera() const;
    const mat4 &get_perspective_projection() const;

    void set_position(const vec3 &position);
    void set_spherical_position(const vec3 &position);
    // Sets camera frame of reference axes
    void set_frame_of_reference(const vec3 &u, /* x axis on view plane */
                             const vec3 &v, /* y axis on view plane */
                             const vec3 &n  /* normal to the view plane */);
    void set_orbit_mode(bool flag);
    void set_orbit_mode_center(const vec3 &center);
    // Points the camera to the given point
    void look_at(const vec3 &point,
                const vec3 &view_up = { 0.0, 1.0, 0.0 });

    // Moves the camera around the look at point
    // using spherical coordinates
    void orbit(const vec3 &delta);
    // Moves the camera using cartesian coordinates
    void move(const vec3 &delta);
    // Updates camera position and recomputes
    // world to camera transformation matrix
    void update(float dt);

private:
    // Computes the transformation matrix
    // to move from world coordinates to
    // camera coordinates
    void compute_world_to_camera();
    // Computes the transformation matrix
    // to project points in camera coordinates
    // to the view plane
    void compute_perspective_projection(float fovy,
                                        float aspect_ratio,
                                        float near,
                                        float far);
    // Converts current spherical coordinates position
    // to cartesian coordinates
    void spherical_to_cartesian();

    // Cartesian coordinates
    vec3 _position;
    // Spherical coordinates used to orbit camera
    vec3 _spherical_position;

    // Vertical field of view
    float _fovy;
     // Window width / height
    float _aspect_ratio;
    // Camera frame of reference axes
    vec3 _axis_u, _axis_v, _axis_n;
    // Transformation matrices
    mat4 _worldToCamera;
    mat4 _perspectiveProjection;

    bool _orbit_mode;
    vec3 _orbit_mode_center;
    vec3 _orbit_mode_view_up;

    vec3 _target_position;
    vec3 _target_spherical_position;

    const float _MOVE_SPEED = 10.0f;
    const float _ORBIT_SPEED = 10.0f;
};

#endif
