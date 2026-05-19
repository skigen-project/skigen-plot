#version 440

layout(location = 0) in vec2 position;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;
};

void main()
{
    gl_PointSize = params.x;
    gl_Position = mvp * vec4(position, 0.0, 1.0);
}
