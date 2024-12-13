#include "Renderer.hpp"

int main()
{
    Renderer renderer(1280, 720);

    if (renderer.init()) {
        renderer.run();
    }

    return 0;
}
