#include "Renderer.hpp"

int main()
{
    Renderer renderer(900, 900);

    if (renderer.init()) {
        renderer.run();
    }

    return 0;
}
