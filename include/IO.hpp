#ifndef IO_HPP
#define IO_HPP

#include "Body.hpp"
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

template <typename T, int dim>
class IO
{
public:
    // Read body data from a file
    static std::vector<Body<T, dim>> readBodiesFromFile(const std::string &filename)
    {
        std::ifstream file(filename);

        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return {};
        }

        std::vector<Body<T, dim>> bodies;
        std::string line;

        int numBodies;
        {
            std::getline(file, line);
            std::istringstream iss(line);
            iss >> numBodies;
        }
        for (int i = 0; i < numBodies; i++)
        {

            // Read mass
            massT mass;
            {
                std::getline(file, line);
                std::istringstream iss(line);
                // Read mass
                iss >> mass;
            }

            // Read position
            std::vector<T> pos;
            {
                std::getline(file, line);
                std::istringstream iss(line);

                for (int i = 0; i < dim; i++)
                {
                    T value;
                    iss >> value;
                    pos.emplace_back(value);
                }
            }

            // Read velocity
            std::vector<T> vel;
            {
                std::getline(file, line);
                std::istringstream iss(line);
                for (int i = 0; i < dim; i++)
                {
                    T value;
                    iss >> value;
                    vel.emplace_back(value);
                }
            }

            bodies.emplace_back(Body<T, dim>(mass, Vector<T, dim>(pos), Vector<T, dim>(vel)));
        }

        file.close();
        return bodies;
    }

    // Write body data to a file
    static void writeBodiesToFile(const std::string &filename, const std::vector<Body2d> &bodies)
    {
        std::ofstream file(filename);

        if (!file.is_open())
        {
            std::cerr << "Error: Cannot open file " << filename << std::endl;
            return;
        }

        file << bodies.size() << std::endl;

        for (const auto &body : bodies)
        {
            file << body.getMass() << std::endl
                 << body.getPosition() << std::endl
                 << body.getVelocity() << std::endl;
        }

        file.close();
    }
};

#endif