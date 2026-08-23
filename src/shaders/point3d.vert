#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 corner;
layout(location = 0) out vec2 markerCoord;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;
};

void main()
{
    gl_Position = mvp * vec4(position, 1.0);
    vec2 viewport = max(params.yz, vec2(1.0));
    gl_Position.xy += corner * params.x * 2.0 / viewport * gl_Position.w;
    markerCoord = corner;
}
