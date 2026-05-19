#version 440

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;  // x = pointSize, y = hollow (1.0 = ring, 0.0 = disc)
};

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float radius = length(coord);
    if (radius > 0.5)
        discard;

    float outerAlpha = 1.0 - smoothstep(0.42, 0.50, radius);

    if (params.y > 0.5) {
        // Ring: keep only the annular band between 0.30 and 0.50
        float innerAlpha = smoothstep(0.26, 0.34, radius);
        float alpha = innerAlpha * outerAlpha;
        if (alpha < 0.01)
            discard;
        fragColor = vec4(color.rgb, color.a * alpha);
    } else {
        float centerLift = 1.0 + 0.08 * (1.0 - smoothstep(0.00, 0.45, radius));
        fragColor = vec4(color.rgb * centerLift, color.a * outerAlpha);
    }
}
