#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <string>
#include <iostream>

#include <GL/glew.h>

#include "Matrix.hpp"

class ShaderProgram
{
public:
    void create();

    // Loads a shader from file, compiles it and
    // attaches it to the program
    bool loadShader(const std::string &source_path,
                    GLenum shader_type);
    // Links the shaders attached to the program
    bool link();
    // Enables the program
    void enable();
    // Loads a single integer inside a int GLSL uniform
    bool loadUniformInt(const std::string &name, int value);
    // Loads a matrix inside a mat4 GLSL uniform
    bool loadUniformMat4(const std::string &name,
                         const Matrix<float, 4, 4> &value);

    ~ShaderProgram();

private:
    GLuint _program_id;

    static const unsigned int _MAX_LOG_LENGTH = 1024;
};

#endif
