#include <Body.hpp>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    Body2d b1(1.0,{1.0,2.0},{2.0,-3.0});
    Body2d b2(4.0,{3.0,-1.0},{6.0,4.0});

    cout << b1 << b2 << endl;

    Vector2d delta({10.0,20.0});
    b1.updatePosition(2*delta);
    b2.updateVelocity(3*delta);

    cout << b1 << b2 << endl;
    return 0;
}