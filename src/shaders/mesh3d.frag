#version 440

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_position;

layout(location = 0) out vec4 fragColor;

layout(std140, binding = 0) uniform buf {
    mat4 mvp;
    vec4 color;
    vec4 lightDir;
    vec4 lightParams;
    vec4 surfaceRange;
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

    float key = smoothstep(-0.20, 0.90, dot(N, L));
    float fill = smoothstep(-0.35, 0.85, dot(N, fillL));
    float top = smoothstep(-0.15, 0.95, dot(N, topL));

    vec3 H = normalize(L + V);
    float specular = pow(max(dot(N, H), 0.0), 22.0);
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    vec3 albedo = mix(vec3(luminance), color.rgb, 0.78);
    float height = clamp((v_position.y - surfaceRange.x)
        / max(surfaceRange.y - surfaceRange.x, 1e-6), 0.0, 1.0);
    height = smoothstep(0.0, 1.0, height);
    vec3 lowColor = mix(albedo, vec3(0.035, 0.085, 0.13), 0.52);
    vec3 highColor = mix(albedo, vec3(0.76, 0.90, 0.86), 0.46);
    vec3 surfaceColor = height < 0.5
        ? mix(lowColor, albedo, smoothstep(0.0, 0.5, height))
        : mix(albedo, highColor, smoothstep(0.5, 1.0, height));
    vec3 coolAmbient = surfaceColor * vec3(0.78, 0.92, 1.04);
    vec3 warmKey = surfaceColor * vec3(1.08, 0.99, 0.88);

    vec3 litColor = lightParams.x * coolAmbient;
    litColor += lightParams.y * key * warmKey;
    litColor += lightParams.w * fill * surfaceColor;
    litColor += 0.10 * top * mix(surfaceColor, vec3(luminance), 0.28);
    litColor += lightParams.z * specular * vec3(0.92, 0.96, 0.94);
    litColor = pow(max(litColor, vec3(0.0)), vec3(0.92));

    fragColor = vec4(clamp(litColor, 0.0, 1.0), color.a);
}
