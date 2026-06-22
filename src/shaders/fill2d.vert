#version 440

layout(location = 0) in vec2 position;
layout(location = 1) in vec4 vcolor;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;  // unused for per-vertex fill, kept for SRB layout compatibility
};

void main()
{
    fragColor = vcolor;
    gl_Position = mvp * vec4(position, 0.0, 1.0);
}
