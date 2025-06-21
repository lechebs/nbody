#include "renderer.hpp"

int main()
{
    Renderer renderer(1600, 900);

    if (renderer.init()) {
        renderer.run();
    }

    return 0;
}
