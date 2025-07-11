#include "shader_program.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <GL/glew.h>

#include "matrix.hpp"

void ShaderProgram::create()
{
    // Requires an active OpenGL context
    program_id_ = glCreateProgram();
}

bool ShaderProgram::load_shader(const std::string &source_path,
                                GLenum shader_type,
                                const std::string &FTYPE_)
{
    const GLuint shader = glCreateShader(shader_type);

    // TODO: handle io error
    std::ifstream source_file(source_path);
    // Get file length
    source_file.seekg(0, source_file.end);
    int length = source_file.tellg();
    source_file.seekg(0, source_file.beg);

    std::string buffer;
    buffer.resize(length);
    source_file.read(buffer.data(), length);
    buffer[length] = 0;

    // Inserting FTYPE__ #define
    buffer.insert(buffer.find_first_of('\n'),
                  "\n#define FTYPE_ " + FTYPE_);

    // Loading shader source code
    char *ptr = buffer.data();
    glShaderSource(shader, 1, &ptr, nullptr);

    glCompileShader(shader);
    // Checking for compilation status.
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        buffer.clear();
        buffer.reserve(_MAX_LOG_LENGTH);
        // Getting log string
        glGetShaderInfoLog(shader, _MAX_LOG_LENGTH, nullptr, buffer.data());
        std::cout << "Error while compiling shader " << source_path <<
            ": " << buffer.data();
    } else {
        glAttachShader(program_id_, shader);
    }

    glDeleteShader(shader);

    return success;
}

bool ShaderProgram::link()
{
    glLinkProgram(program_id_);

    // Checking for link status.
    int success;
    glGetProgramiv(program_id_, GL_LINK_STATUS, &success);
    if (!success) {
        std::string buffer;
        buffer.reserve(_MAX_LOG_LENGTH);
        buffer.clear();

        glGetProgramInfoLog(program_id_,
                            _MAX_LOG_LENGTH,
                            nullptr,
                            buffer.data());
        std::cout << "Error while linking program: " << buffer.data();
    }

    return success;
}

void ShaderProgram::enable()
{
    glUseProgram(program_id_);
}

bool ShaderProgram::load_uniform_int(const std::string &name, int value)
{
    GLuint location = glGetUniformLocation(program_id_, name.c_str());

    if (location != -1u) {
        enable();
        glUniform1i(location, value);
    }

    return location > 0;
}

bool ShaderProgram::load_uniform_float(const std::string &name, float value)
{
    GLuint location = glGetUniformLocation(program_id_, name.c_str());

    if (location != -1u) {
        enable();
        glUniform1f(location, value);
    }

    return location > 0;
}


bool ShaderProgram::load_uniform_mat4(const std::string &name,
                                    const Matrix<float, 4, 4> &value)
{
    GLuint location = glGetUniformLocation(program_id_, name.c_str());

    if (location != -1u) {
        enable();
        // GL_TRUE transposes the matrix since GLSL works
        // with colum major ordering
        glUniformMatrix4fv(location, 1, GL_TRUE, value.data());
    }

    return location > 0;
}

ShaderProgram::~ShaderProgram()
{
    glDeleteProgram(program_id_);
}
