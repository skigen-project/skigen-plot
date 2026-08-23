#version 440

layout(location = 0) out vec4 fragColor;
layout(location = 0) in vec2 markerCoord;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;  // x = pointSize, y = hollow, z = MarkerShape
};

float markerDistance(vec2 point, float radius, float shape)
{
    if (shape < 0.5)
        return length(point) - radius;
    if (shape < 1.5)
        return max(abs(point.x), abs(point.y)) - radius;
    if (shape < 2.5) {
        vec2 scaled = point / radius;
        float triangle = max(-scaled.y - 1.0,
                             max(scaled.y - 1.0,
                                 abs(scaled.x) - 0.5 * (scaled.y + 1.0)));
        return triangle * radius;
    }

    vec2 p = point;
    if (shape >= 3.5)
        p = vec2(point.x + point.y, point.y - point.x) * 0.70710678;
    float vertical = max(abs(p.x) - radius * 0.22, abs(p.y) - radius);
    float horizontal = max(abs(p.x) - radius, abs(p.y) - radius * 0.22);
    return min(vertical, horizontal);
}

void main()
{
    vec2 coord = markerCoord;
    float distance = markerDistance(coord, 0.5, params.z);
    float outerAlpha = 1.0 - smoothstep(-0.04, 0.0, distance);
    if (outerAlpha < 0.01)
        discard;

    if (params.y > 0.5 && params.z < 2.5) {
        float innerDistance = markerDistance(coord, 0.30, params.z);
        float innerAlpha = 1.0 - smoothstep(-0.04, 0.0, innerDistance);
        float alpha = outerAlpha * (1.0 - innerAlpha);
        if (alpha < 0.01)
            discard;
        fragColor = vec4(color.rgb, color.a * alpha);
    } else {
        fragColor = vec4(color.rgb, color.a * outerAlpha);
    }
}
