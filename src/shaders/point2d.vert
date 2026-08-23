#version 440

layout(location = 0) in vec2 position;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 params;
    vec4 axisParams; // x/y: log10 scale
};

void main()
{
    gl_PointSize = params.x;
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
}
