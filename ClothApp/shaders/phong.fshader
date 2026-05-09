#version 430

in vec3 vNormal;
in vec3 vLightDir;
in vec3 vViewPosition;
in vec2 vTexCoord;
out vec4 fragColor;

uniform vec3 uAlbedo;
uniform vec3 uAmbient;
uniform float uSpecularStrength;
uniform float uShininess;
uniform bool uUseFlagPattern;

float starMask(vec2 uv, vec2 center, float radiusOuter, float radiusInner) {
    vec2 delta = uv - center;
    float angle = atan(delta.y, delta.x);
    float distanceValue = length(delta);
    float sector = floor((angle + 3.14159265) * 5.0 / 3.14159265);
    float localAngle = angle + sector * 3.14159265 / 5.0;
    float targetRadius = mod(sector, 2.0) < 0.5 ? radiusOuter : radiusInner;
    return step(distanceValue, targetRadius * max(0.35, abs(cos(localAngle))));
}

vec3 americanFlagColor(vec2 uv) {
    const vec3 red = vec3(0.698, 0.132, 0.203);
    const vec3 white = vec3(0.96, 0.96, 0.94);
    const vec3 blue = vec3(0.235, 0.233, 0.431);

    vec2 clampedUv = clamp(uv, 0.0, 1.0);
    float stripeIndex = floor((1.0 - clampedUv.y) * 13.0);
    vec3 color = mod(stripeIndex, 2.0) < 0.5 ? red : white;

    bool inUnion = clampedUv.x <= 0.4 && clampedUv.y >= (1.0 - 7.0 / 13.0);
    if (inUnion) {
        color = blue;
        vec2 unionUv = vec2(clampedUv.x / 0.4, (clampedUv.y - (1.0 - 7.0 / 13.0)) / (7.0 / 13.0));
        float stars = 0.0;
        for (int row = 0; row < 9; ++row) {
            float starsInRow = (row % 2 == 0) ? 6.0 : 5.0;
            float xOffset = (row % 2 == 0) ? 1.0 / 12.0 : 1.0 / 6.0;
            for (int col = 0; col < int(starsInRow); ++col) {
                vec2 center = vec2(xOffset + float(col) / starsInRow, 1.0 - (float(row) + 0.5) / 9.0);
                stars = max(stars, starMask(unionUv, center, 0.045, 0.018));
            }
        }
        color = mix(color, white, stars);
    }

    return color;
}

void main(){
    vec3 normal = normalize(vNormal);
	vec3 albedo = uAlbedo;
    vec2 texCoord = vec2(clamp(vTexCoord.x, 0.0, 1.0), 1.0 - clamp(vTexCoord.y, 0.0, 1.0));

    if(!gl_FrontFacing) {
        normal = -normal;
        if (!uUseFlagPattern) {
            albedo = 1.0 - albedo;
        }
    }

	if (uUseFlagPattern) {
		albedo = americanFlagColor(texCoord);
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