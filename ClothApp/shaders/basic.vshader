#version 430

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
uniform vec3 uLight;

layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoord;

out vec3 vNormal;
out vec3 vLightDir;
out vec3 vViewPosition;
out vec2 vTexCoord;

void main(){
    mat3 normalMatrix = transpose(inverse(mat3(uModelViewMatrix)));
    vec4 viewPosition = uModelViewMatrix * vec4(aPosition, 1.0);
    vNormal = normalize(normalMatrix * aNormal);
    vLightDir = normalize(mat3(uModelViewMatrix) * (-uLight));
    vViewPosition = viewPosition.xyz;
    vTexCoord = aTexCoord;
    gl_Position = uProjectionMatrix * viewPosition;
}