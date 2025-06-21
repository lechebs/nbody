#ifndef SHADERPROGRAM_H
#define SHADERPROGRAM_H

#include <string>
#include <iostream>

#include <GL/glew.h>

#include "matrix.hpp"

class ShaderProgram
{
public:
    void create();

    // Loads a shader from file, compiles it and
    // attaches it to the program
    bool load_shader(const std::string &source_path,
                    GLenum shader_type,
                    const std::string &FTYPE_ = "float");
    // Links the shaders attached to the program
    bool link();
    // Enables the program
    void enable();
    // Loads a single integer inside a int GLSL uniform
    bool load_uniform_int(const std::string &name, int value);
    // Loads a single float inside a float GLSL uniform
    bool load_uniform_float(const std::string &name, float value);
    // Loads a matrix inside a mat4 GLSL uniform
    bool load_uniform_mat4(const std::string &name,
                         const Matrix<float, 4, 4> &value);

    ~ShaderProgram();

private:
    GLuint program_id_;

    static const unsigned int _MAX_LOG_LENGTH = 1024;
};

#endif
