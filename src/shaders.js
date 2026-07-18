export const G_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;
layout(location=1) in float aDepth;
layout(location=2) in vec2 aGradient;
uniform float uYaw,uPitch,uDepthScale,uSign,uZoom,uAspect,uSourceCrop,uSourceAspect;
out vec2 vSourceUV;
out vec3 vNormal;
out vec3 vWorldPos;
vec3 rotate3(vec3 p){
  float cy=cos(uYaw), sy=sin(uYaw), cp=cos(uPitch), sp=sin(uPitch);
  vec3 q=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);
  return vec3(q.x,q.y*cp-q.z*sp,q.y*sp+q.z*cp);
}
void main(){
  vec3 p=rotate3(vec3(aXY,aDepth*uDepthScale*uSign));
  vec3 n=normalize(vec3(-aGradient.x*uDepthScale*uSign,-aGradient.y*uDepthScale*uSign,1.0));
  n=normalize(rotate3(n));
  float camDist=3.2;
  float cz=camDist-p.z;
  float f=2.41421356;
  float nearP=.1, farP=30.0;
  float zEye=-cz;
  float A=(farP+nearP)/(nearP-farP);
  float B=(2.0*farP*nearP)/(nearP-farP);
  gl_Position=vec4(p.x*f*uZoom/uAspect,p.y*f*uZoom,A*zEye+B,cz);
  vSourceUV=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),0.5-aXY.y/(2.0*uSourceAspect));
  vNormal=n; vWorldPos=p;
}`;

export const G_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 vSourceUV;
in vec3 vNormal;
in vec3 vWorldPos;
uniform sampler2D uSourceTex;
layout(location=0) out vec4 outAlbedo;
layout(location=1) out vec4 outNormal;
layout(location=2) out vec4 outPosition;
void main(){
  outAlbedo=vec4(texture(uSourceTex,clamp(vSourceUV,0.0,1.0)).rgb,1.0);
  outNormal=vec4(normalize(vNormal)*0.5+0.5,1.0);
  outPosition=vec4(vWorldPos,1.0);
}`;

export const SHADOW_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;
layout(location=1) in float aDepth;
uniform float uYaw,uPitch,uDepthScale,uSign;
uniform mat4 uLightVP;
vec3 rotate3(vec3 p){
  float cy=cos(uYaw), sy=sin(uYaw), cp=cos(uPitch), sp=sin(uPitch);
  vec3 q=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);
  return vec3(q.x,q.y*cp-q.z*sp,q.y*sp+q.z*cp);
}
void main(){
  vec3 p=rotate3(vec3(aXY,aDepth*uDepthScale*uSign));
  gl_Position=uLightVP*vec4(p,1.0);
}`;

export const SHADOW_FRAGMENT_SHADER = `#version 300 es
precision highp float;
void main(){}`;

export const LIGHT_VERTEX_SHADER = `#version 300 es
precision highp float;
const vec2 POS[6]=vec2[6](vec2(-1,-1),vec2(1,-1),vec2(-1,1),vec2(-1,1),vec2(1,-1),vec2(1,1));
out vec2 vUV;
void main(){
  vec2 p=POS[gl_VertexID];
  vUV=p*0.5+0.5;
  gl_Position=vec4(p,0.0,1.0);
}`;

export const LIGHT_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uAlbedoTex;
uniform sampler2D uNormalTex;
uniform sampler2D uPositionTex;
uniform sampler2D uShadowTex;
uniform vec2 uShadowTexel;
uniform mat4 uLightVP;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform float uAmbient,uDiffuse,uFill,uShadowStrength,uSelfShade,uSpecular,uShininess,uBiasBase,uBiasSlope,uPcfRadius,uShadowHardness;
out vec4 outColor;
float shadowFactor(vec3 worldPos, vec3 N, vec3 L){
  vec4 sp=uLightVP*vec4(worldPos,1.0);
  vec3 proj=sp.xyz/max(sp.w, 1e-6);
  proj=proj*0.5+0.5;
  if(proj.x<=0.0||proj.x>=1.0||proj.y<=0.0||proj.y>=1.0||proj.z<=0.0||proj.z>=1.0) return 0.0;
  float bias=max(uBiasBase, uBiasSlope*(1.0-max(dot(N,L),0.0)));
  float shadow=0.0;
  for(int yy=-1;yy<=1;yy++) for(int xx=-1;xx<=1;xx++){
    vec2 off=vec2(float(xx),float(yy))*uShadowTexel*uPcfRadius;
    float depth=texture(uShadowTex, proj.xy+off).r;
    shadow += (proj.z-bias>depth) ? 1.0 : 0.0;
  }
  float s=shadow/9.0;
  float width=mix(0.45,0.02,clamp(uShadowHardness,0.0,1.0));
  return smoothstep(0.5-width,0.5+width,s);
}
void main(){
  vec4 a=texture(uAlbedoTex,vUV);
  vec4 nTex=texture(uNormalTex,vUV);
  vec4 pTex=texture(uPositionTex,vUV);
  if(a.a<0.5 || pTex.a<0.5){ outColor=vec4(0.045,0.05,0.06,1.0); return; }
  vec3 albedo=a.rgb;
  vec3 N=normalize(nTex.xyz*2.0-1.0);
  vec3 worldPos=pTex.xyz;
  vec3 L=normalize(uLightDir);
  vec3 V=normalize(uCameraPos-worldPos);
  float ndl=max(dot(N,L),0.0);
  float shadow=shadowFactor(worldPos,N,L);
  float selfShade=0.0;
  if(uSelfShade>0.0){
    float curvature=clamp((length(dFdx(N))+length(dFdy(N)))*2.0,0.0,1.0);
    float grazing=1.0-max(dot(N,V),0.0);
    selfShade=uSelfShade*curvature*(0.35+0.65*grazing);
  }
  vec3 H=normalize(L+V);
  float spec=pow(max(dot(N,H),0.0), max(1.0,uShininess));
  float direct=(1.0-shadow*uShadowStrength)*uDiffuse*ndl;
  float illumination=uAmbient + uFill + direct - selfShade;
  vec3 color=albedo*illumination + vec3(uSpecular*spec*(1.0-shadow*uShadowStrength));
  outColor=vec4(clamp(color,0.0,1.0),1.0);
}`;
