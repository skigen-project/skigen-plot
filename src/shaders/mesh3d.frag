#version 440

layout(location = 0) in vec3 v_normal;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
    vec4 lightParams;
};

void main()
{
    vec3 N = normalize(v_normal);
    vec3 L = normalize(lightDir.xyz);
    float diff = max(dot(N, L), 0.0);
    float intensity = lightParams.x + lightParams.y * diff;
    fragColor = vec4(color.rgb * intensity, color.a);
}
