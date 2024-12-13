#include "IO.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <iomanip>

void testReadBodies()
{
    std::ofstream tempInput("testInput.txt");
    // Number of bodies
    tempInput << "2" << std::endl;
    // First body
    tempInput << "1.0" << std::endl;      // mass
    tempInput << "1.0 2.0" << std::endl;  // position
    tempInput << "2.0 -3.0" << std::endl; // velocity
    // Second body
    tempInput << "4.0" << std::endl;      // mass
    tempInput << "3.0 -1.0" << std::endl; // position
    tempInput << "6.0 4.0" << std::endl;  // velocity
    tempInput.close();

    // Read bodies using IO class
    std::vector<Body2d> bodies = IO<double, 2>::readBodiesFromFile("testInput.txt");

    // Detailed checks
    assert(bodies.size() == 2);

    // First body checks
    assert(bodies[0].getMass() == 1.0);
    assert(bodies[0].getPosition()[0] == 1.0);
    assert(bodies[0].getPosition()[1] == 2.0);
    assert(bodies[0].getVelocity()[0] == 2.0);
    assert(bodies[0].getVelocity()[1] == -3.0);

    // Second body checks
    assert(bodies[1].getMass() == 4.0);
    assert(bodies[1].getPosition()[0] == 3.0);
    assert(bodies[1].getPosition()[1] == -1.0);
    assert(bodies[1].getVelocity()[0] == 6.0);
    assert(bodies[1].getVelocity()[1] == 4.0);

    std::cout << "Test read bodies works." << std::endl;
    // std::remove("testInput.txt");
}

void testWriteBodies()
{
    std::vector<Body2d> bodies = {
        Body2d(1.0, {1.0, 2.0}, {2.0, -3.0}),
        Body2d(4.0, {3.0, -1.0}, {6.0, 4.0})};

    IO<double, 2>::writeBodiesToFile("testOutput.txt", bodies);

    // More robust file reading
    std::ifstream tempOutput("testOutput.txt");
    std::string line;
    std::vector<std::string> lines;

    while (std::getline(tempOutput, line))
    {
        if (!line.empty())
        {
            lines.push_back(line);
        }
    }

    // Check number of lines
    assert(lines.size() == 7);

    // Direct string comparison
    assert(lines[0] == "2");
    assert(lines[1] == "1");
    assert(lines[2] == "1 2");
    assert(lines[3] == "2 -3");
    assert(lines[4] == "4");
    assert(lines[5] == "3 -1");
    assert(lines[6] == "6 4");

    std::cout << "Test write bodies works." << std::endl;

    std::remove("testOutput.txt");
}

int main()
{
    testReadBodies();
    testWriteBodies();

    std::cout << "IO test works." << std::endl;
    return 0;
}