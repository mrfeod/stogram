export const G_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;
layout(location=1) in float aDepth;
layout(location=2) in vec2 aGradient;
uniform float uYaw,uPitch,uDepthScale,uSign,uZoom,uAspect,uSourceCrop,uSourceAspect,uPatternTime;
uniform sampler2D uDepthTex;
uniform float uUseDepthTex;
out vec2 vSourceUV;
out vec3 vNormal;
out vec3 vWorldPos;
vec3 rotate3(vec3 p){
  float cy=cos(uYaw), sy=sin(uYaw), cp=cos(uPitch), sp=sin(uPitch);
  vec3 q=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);
  return vec3(q.x,q.y*cp-q.z*sp,q.y*sp+q.z*cp);
}
vec2 cellHash(vec2 p){
  return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}
float livingVoronoi(vec2 uv){
  vec2 size=vec2(textureSize(uDepthTex,0));
  vec2 p=uv*size/32.0, cell=floor(p), local=fract(p);
  float nearest=10.0;
  for(int yy=-1;yy<=1;yy++) for(int xx=-1;xx<=1;xx++){
    vec2 neighbour=vec2(float(xx),float(yy));
    vec2 seed=cellHash(cell+neighbour);
    vec2 motion=sin(uPatternTime*0.7+seed*6.2831853)*0.28;
    vec2 point=neighbour+0.5+motion;
    nearest=min(nearest,length(point-local));
  }
  return 1.0-smoothstep(0.08,0.72,nearest);
}
float depthWithVoidPattern(vec2 uv,float sourceDepth){
  vec2 size=vec2(textureSize(uDepthTex,0)), pixel=uv*size;
  float left=uSourceCrop*size.x;
  bool border=pixel.x<=left+1.0||pixel.x>=size.x-1.0||
      pixel.y<=1.0||pixel.y>=size.y-1.0;
  if(border) return 0.0;
  float pattern=(5.0+5.0*livingVoronoi(uv))/255.0;
  if(sourceDepth>=0.995) return sourceDepth+pattern;
  if(sourceDepth>0.00001) return sourceDepth;
  return pattern;
}
float signedMeshDepth(vec2 uv,float sourceDepth){
  float meshDepth=depthWithVoidPattern(uv,sourceDepth);
  if(meshDepth<=0.00001) return 0.0;
  float base=uSign<0.0?1.0-sourceDepth:sourceDepth;
  if(sourceDepth<=0.00001) return base+meshDepth;
  if(sourceDepth>=0.995) return base+(meshDepth-sourceDepth);
  return base;
}
void main(){
  vec2 depthUV=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),0.5-aXY.y/(2.0*uSourceAspect));
  vec2 texel=1.0/vec2(textureSize(uDepthTex,0));
  vec2 uvL=depthUV-vec2(texel.x,0.0), uvR=depthUV+vec2(texel.x,0.0);
  vec2 uvU=depthUV-vec2(0.0,texel.y), uvD=depthUV+vec2(0.0,texel.y);
  float dL=signedMeshDepth(uvL,texture(uDepthTex,uvL).r);
  float dR=signedMeshDepth(uvR,texture(uDepthTex,uvR).r);
  float dU=signedMeshDepth(uvU,texture(uDepthTex,uvU).r);
  float dD=signedMeshDepth(uvD,texture(uDepthTex,uvD).r);
  float sampledDepth=texture(uDepthTex,depthUV).r;
  sampledDepth=signedMeshDepth(depthUV,sampledDepth);
  float cpuDepth=uSign<0.0?1.0-aDepth:aDepth;
  float depth=mix(cpuDepth,sampledDepth,uUseDepthTex);
  vec2 sampledGradient=vec2(dR-dL,dD-dU)*float(textureSize(uDepthTex,0).x)*0.23;
  vec2 gradient=mix(aGradient*uSign,sampledGradient,uUseDepthTex);
  vec3 p=rotate3(vec3(aXY,depth*uDepthScale));
  vec3 n=normalize(vec3(-gradient.x*uDepthScale,-gradient.y*uDepthScale,1.0));
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
uniform sampler2D uDepthTex;
uniform float uUseDepthTex,uSourceCrop,uSourceAspect,uPatternTime;
vec3 rotate3(vec3 p){
  float cy=cos(uYaw), sy=sin(uYaw), cp=cos(uPitch), sp=sin(uPitch);
  vec3 q=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);
  return vec3(q.x,q.y*cp-q.z*sp,q.y*sp+q.z*cp);
}
vec2 cellHash(vec2 p){
  return fract(sin(vec2(dot(p,vec2(127.1,311.7)),dot(p,vec2(269.5,183.3))))*43758.5453);
}
float livingVoronoi(vec2 uv){
  vec2 size=vec2(textureSize(uDepthTex,0));
  vec2 p=uv*size/32.0, cell=floor(p), local=fract(p);
  float nearest=10.0;
  for(int yy=-1;yy<=1;yy++) for(int xx=-1;xx<=1;xx++){
    vec2 neighbour=vec2(float(xx),float(yy));
    vec2 seed=cellHash(cell+neighbour);
    vec2 motion=sin(uPatternTime*0.7+seed*6.2831853)*0.28;
    nearest=min(nearest,length(neighbour+0.5+motion-local));
  }
  return 1.0-smoothstep(0.08,0.72,nearest);
}
float depthWithVoidPattern(vec2 uv,float sourceDepth){
  vec2 size=vec2(textureSize(uDepthTex,0)), pixel=uv*size;
  float left=uSourceCrop*size.x;
  bool border=pixel.x<=left+1.0||pixel.x>=size.x-1.0||
      pixel.y<=1.0||pixel.y>=size.y-1.0;
  if(border) return 0.0;
  float pattern=(5.0+5.0*livingVoronoi(uv))/255.0;
  if(sourceDepth>=0.995) return sourceDepth+pattern;
  if(sourceDepth>0.00001) return sourceDepth;
  return pattern;
}
float signedMeshDepth(vec2 uv,float sourceDepth){
  float meshDepth=depthWithVoidPattern(uv,sourceDepth);
  if(meshDepth<=0.00001) return 0.0;
  float base=uSign<0.0?1.0-sourceDepth:sourceDepth;
  if(sourceDepth<=0.00001) return base+meshDepth;
  if(sourceDepth>=0.995) return base+(meshDepth-sourceDepth);
  return base;
}
void main(){
  vec2 uv=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),0.5-aXY.y/(2.0*uSourceAspect));
  float sampledDepth=texture(uDepthTex,uv).r;
  sampledDepth=signedMeshDepth(uv,sampledDepth);
  float cpuDepth=uSign<0.0?1.0-aDepth:aDepth;
  float depth=mix(cpuDepth,sampledDepth,uUseDepthTex);
  vec3 p=rotate3(vec3(aXY,depth*uDepthScale));
  gl_Position=uLightVP*vec4(p,1.0);
}`;

export const SHADOW_FRAGMENT_SHADER = `#version 300 es
precision highp float;
void main(){}`;

export const LIGHT_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aPosition;
out vec2 vUV;
void main(){
  vec2 p=aPosition;
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
  for(int yy=-2;yy<=2;yy++) for(int xx=-2;xx<=2;xx++){
    vec2 off=vec2(float(xx),float(yy))*uShadowTexel*uPcfRadius;
    float depth=texture(uShadowTex, proj.xy+off).r;
    shadow += (proj.z-bias>depth) ? 1.0 : 0.0;
  }
  float s=shadow/25.0;
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
