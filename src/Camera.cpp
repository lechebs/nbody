#include "Camera.hpp"

#include <cmath>

#include "Vector.hpp"
#include "Matrix.hpp"

Camera::Camera(float fovy,
               float aspect_ratio,
               float near,
               float far) :
    _fovy(fovy),
    _aspect_ratio(aspect_ratio),
    _axis_u({ 1.0, 0.0, 0.0 }),
    _axis_v({ 0.0, 1.0, 0.0 }),
    _axis_n({ 0.0, 0.0, -1.0 })
{
    _computeWorldToCamera();
    _computePerspectiveProjection(fovy, aspect_ratio, near, far);
}

const vec3 &Camera::getPosition() const
{
    return _position;
}

const mat4 &Camera::getWorldToCamera() const
{
    return _worldToCamera;
}

const mat4 &Camera::getPerspectiveProjection() const
{
    return _perspectiveProjection;
}

void Camera::setPosition(const vec3 &position)
{
    _position = position;
    _target_position = position;
}

void Camera::setSphericalPosition(const vec3 &position)
{
    _spherical_position = position;
    _target_spherical_position = position;
    _sphericalToCartesian();
}

void Camera::setFrameOfReference(const vec3 &u, const vec3 &v, const vec3 &n)
{
    _axis_u = u;
    _axis_v = v;
    _axis_n = n;
}

void Camera::setOrbitMode(bool flag)
{
    _orbit_mode = flag;
}

void Camera::setOrbitModeCenter(const vec3 &center)
{
    _orbit_mode_center = center;
    _orbit_mode_view_up = { 0.0f, 1.0f, 0.0f };
}

void Camera::lookAt(const vec3 &point, const vec3 &view_up)
{
    _axis_n = _position - point;
    _axis_u = vec3::cross(view_up, _axis_n);
    _axis_v = vec3::cross(_axis_n, _axis_u);

    _axis_n.normalize();
    _axis_u.normalize();
    _axis_v.normalize();
}

void Camera::move(const vec3 &delta)
{
    // Projecting delta back to xy plane
    vec3 camera_delta({
        delta[0] * _aspect_ratio * _position[2] * std::tan(_fovy / 2),
        delta[1] * _position[2] * std::tan(_fovy / 2),
        delta[2]
    });

    _target_position += camera_delta;
}

void Camera::orbit(const vec3 &delta)
{
    _target_spherical_position -= delta;

    // Clamp radius
    if (_target_spherical_position[0] < 0.0001) {
        _target_spherical_position[0] = 0.0001;
    }

    // Clamp theta
    float theta = _target_spherical_position[1];
    if (theta <= -M_PI / 2) {
        theta = -M_PI / 2 + 0.0001;
    } else if (theta >= M_PI / 2) {
        theta = M_PI / 2 - 0.0001;
    }
    _target_spherical_position[1] = theta;
}

void Camera::update(float dt)
{
    if (_orbit_mode) {
        _spherical_position += dt * _ORBIT_SPEED *
            (_target_spherical_position - _spherical_position);
        _sphericalToCartesian();
        lookAt(_orbit_mode_center, _orbit_mode_view_up);
    } else {
        // Linearly interpolating
        _position += dt * _MOVE_SPEED *
            (_target_position - _position);
    }

    _computeWorldToCamera();
}

void Camera::_computeWorldToCamera()
{
    mat4 rotation = mat4::identity();
    mat4 translation = mat4::identity();

    // TODO: write a more elegant way
    for (int i = 0; i < 3; ++i) {
        rotation(0, i) = _axis_u[i];
        rotation(1, i) = _axis_v[i];
        rotation(2, i) = _axis_n[i];
        translation(i, 3) = -_position[i];
    }

    _worldToCamera = rotation * translation;
}

void Camera::_computePerspectiveProjection(float fovy,
                                           float aspect_ratio,
                                           float near,
                                           float far)
{
    mat4 &mat = _perspectiveProjection;

    mat(0, 0) = 1 / (aspect_ratio * std::tan(fovy / 2));
    mat(1, 1) = 1 / std::tan(fovy / 2);
    mat(2, 2) = - (near + far) / (far - near);
    mat(2, 3) = 2 * near * far / (far - near);
    mat(3, 2) = - 1.0;
}

void Camera::_sphericalToCartesian()
{ // TODO: clamp values
    float theta = _spherical_position[1];
    float phi = _spherical_position[2];

    _position[0] = std::cos(theta) * std::cos(phi);
    _position[1] = std::sin(theta);
    _position[2] = std::cos(theta) * std::sin(phi);
    // Multiply by radius
    _position *= _spherical_position[0];
}
