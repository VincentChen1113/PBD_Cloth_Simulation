#version 430

out vec4 fragColor;

uniform vec4 uShadowColor;

void main() {
	fragColor = uShadowColor;
}