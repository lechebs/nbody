#include <Vector.hpp>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    Vector3d vec1({1, 2, 3});
    Vector3d vec2({1, 2, 3});
    Vector3d vec3(vec1);
    Vector3d vec4 = 2*vec3;

    cout << vec1 << endl
         << vec2 << endl
         << vec3 << endl
         << vec4 << endl;

    cout << (vec1 == vec2) << endl;
    cout << (vec1 == vec3) << endl;
    vec1 = 2 * vec1;
    cout << vec1 << endl;
    vec1 += vec2;
    cout << vec1 << endl;
    vec1 -= 2 * vec2;
    cout << vec1 << endl;
    cout << vec1.norm() << endl;
    return 0;
}