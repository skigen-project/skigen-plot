#version 440

layout(location = 0) in vec2 position;
layout(location = 1) in vec2 corner;
layout(location = 0) out vec2 markerCoord;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;
    vec4 axisParams; // x/y: log10 scale, z/w: viewport size
};

void main()
{
    if ((axisParams.x > 0.5 && position.x <= 0.0)
        || (axisParams.y > 0.5 && position.y <= 0.0)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }
    vec2 transformed = position;
    if (axisParams.x > 0.5)
        transformed.x = log2(position.x) / log2(10.0);
    if (axisParams.y > 0.5)
        transformed.y = log2(position.y) / log2(10.0);
    gl_Position = mvp * vec4(transformed, 0.0, 1.0);
    vec2 viewport = max(axisParams.zw, vec2(1.0));
    gl_Position.xy += corner * params.x * 2.0 / viewport * gl_Position.w;
    markerCoord = corner;
}
