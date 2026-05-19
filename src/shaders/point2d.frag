#version 440

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;
};

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float radius = length(coord);
    if (radius > 0.5)
        discard;

    float edgeAlpha = 1.0 - smoothstep(0.40, 0.50, radius);
    float centerLift = 1.0 + 0.08 * (1.0 - smoothstep(0.00, 0.45, radius));
    fragColor = vec4(color.rgb * centerLift, color.a * edgeAlpha);
}
