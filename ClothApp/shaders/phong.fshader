#version 430

in vec3 vNormal;
in vec3 vLightDir;
in vec3 vViewPosition;
out vec4 fragColor;

uniform vec3 uAlbedo;
uniform vec3 uAmbient;
uniform float uSpecularStrength;
uniform float uShininess;
void main(){
    vec3 normal = normalize(vNormal);
	vec3 albedo = uAlbedo;
    if(!gl_FrontFacing) {
		normal = -normal;
		albedo = 1.0 - albedo;
	}

    vec3 toLight = normalize(vLightDir);
    vec3 toView = normalize(-vViewPosition);
    float diffuse = max(0, dot(toLight, normal));
    vec3 halfVector = normalize(toLight + toView);
    float specular = 0.0;
    if (diffuse > 0.0) {
        specular = uSpecularStrength * pow(max(0.0, dot(normal, halfVector)), uShininess);
    }

    vec3 color = diffuse * albedo + uAmbient * albedo + specular * vec3(1.0);
    fragColor = vec4(color, 1.0);
}