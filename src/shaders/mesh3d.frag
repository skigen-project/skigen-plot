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
    vec3 H2 = normalize(topL + V);
    float broadSpec = pow(max(dot(N, H), 0.0), 14.0);
    float crispSpec = pow(max(dot(N, H2), 0.0), 70.0);
    float fresnel = pow(1.0 - clamp(dot(N, V), 0.0, 1.0), 2.4);

    float topFacing = max(N.y, 0.0);
    float rightFacing = max(N.x, 0.0);
    float frontFacing = max(N.z, 0.0);
    float cameraGloss = pow(max(dot(N, V), 0.0), 3.2);
    vec3 brandBlue = vec3(0.05, 0.50, 0.74);
    vec3 deepCyan = vec3(0.02, 0.26, 0.31);
    vec3 albedo = mix(color.rgb, brandBlue, 0.38);
    albedo = mix(albedo, vec3(0.12, 0.78, 0.94), 0.22 * topFacing);
    albedo += vec3(0.02, 0.10, 0.16) * rightFacing;
    albedo += vec3(0.06, 0.16, 0.18) * topFacing;
    albedo -= vec3(0.02, 0.03, 0.03) * frontFacing;

    float diffuse = lightParams.x + lightParams.y * key + 0.22 * fill + 0.22 * top + 0.30 * topFacing;
    vec3 litColor = albedo * diffuse;
    litColor += (0.30 * broadSpec + lightParams.z * crispSpec + 0.16 * cameraGloss)
        * vec3(0.78, 0.96, 1.00);
    litColor += lightParams.w * fresnel * vec3(0.04, 0.44, 0.62);
    litColor += deepCyan * 0.08 * (1.0 - key);

    fragColor = vec4(clamp(litColor, 0.0, 1.0), color.a);
}
