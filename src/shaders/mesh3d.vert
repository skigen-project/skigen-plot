#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
    vec4 lightParams;
};

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_position;

void main()
{
    v_normal = normal;
    v_position = position;
    gl_Position = mvp * vec4(position, 1.0);
}
