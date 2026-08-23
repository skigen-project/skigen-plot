#version 440

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
    vec4 lightParams;
};

void main()
{
    vec3 faceNormal = normalize(cross(dFdx(v_position), dFdy(v_position)));
    if (dot(faceNormal, normalize(v_normal)) < 0.0)
        faceNormal = -faceNormal;

    vec3 N = faceNormal;
    vec3 L = normalize(lightDir.xyz);
    vec3 V = normalize(L);
    vec3 fillL = normalize(vec3(-0.70, -0.25, 0.70));
    vec3 topL = normalize(vec3(-0.20, 0.95, 0.45));

    float key = max(dot(N, L), 0.0);
    float fill = max(dot(N, fillL), 0.0);
    float top = max(dot(N, topL), 0.0);

    vec3 H = normalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), 28.0);
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 albedo = mix(vec3(luminance), color.rgb, 0.68);

    float diffuse = lightParams.x
        + lightParams.y * key
        + 0.14 * fill
        + 0.08 * top;
    diffuse = clamp(diffuse, 0.34, 0.92);
    vec3 litColor = albedo * diffuse;
    litColor += lightParams.z * specular * vec3(0.92);

    fragColor = vec4(clamp(litColor, 0.0, 1.0), color.a);
}
