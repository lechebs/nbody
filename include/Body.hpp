#ifndef BODY_HPP_
#define BODY_HPP_
#include <Vector.hpp>

typedef double massT;

template <typename T, int N>
class Body
{
private:
    massT mass;
    Vector<T, N> pos;
    Vector<T, N> vel;
    Vector<T, N> acc;

public:
    Body(const massT &mass_init, const std::initializer_list<T> &pos_init, const std::initializer_list<T> &vel_init) : mass(mass_init),
                                                                                                                      pos(pos_init),
                                                                                                                      vel(vel_init) {}
    Body(const massT &mass_init, const Vector<T, N> &pos_init, const Vector<T, N> &vel_init) : mass(mass_init),
                                                                                            pos(pos_init),
                                                                                            vel(vel_init) {}

    const massT getMass() const
    {
        return mass;
    }

    const Vector<T, N> &getPosition() const
    {
        return pos;
    }

    const Vector<T, N> &getVelocity() const
    {
        return vel;
    }

    const Vector<T, N> &getAcceleration() const
    {
        return acc;
    }

    void updatePosition(const Vector<T, N> &deltaPos)
    {
        pos += deltaPos;
    }

    void setPosition(const Vector<T, N> &newPos)
    {
        pos = newPos;
    }

    void updateVelocity(const Vector<T, N> &deltaVel)
    {
        vel += deltaVel;
    }

    void setVelocity(const Vector<T, N> &newVel)
    {
        vel = newVel;
    }

    void updateAcceleration(const Vector<T, N> &deltaAcc)
    {
        acc += deltaAcc;
    }

    void setAcceleration(const Vector<T, N> &newAcc)
    {
        acc = newAcc;
    }

    // Stream operators
    inline friend std::ostream &operator<<(std::ostream &os, const Body &body)
    {
        os << body.getMass() << " "
           << body.getPosition() << " "
           << body.getVelocity() << " "
           << body.getAcceleration();

        return os;
    }
};

// Definition of some useful types
typedef Body<double, 2> Body2d;
typedef Body<double, 3> Body3d;

#endif