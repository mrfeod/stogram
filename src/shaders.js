export const G_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;
layout(location=1) in float aDepth;
layout(location=2) in vec2 aGradient;
uniform float uYaw,uPitch,uDepthScale,uSign,uZoom,uAspect,uSourceCrop,uSourceAspect,uPatternTime,uBackPass;
uniform float uCameraDistance,uFov;
uniform sampler2D uDepthTex;
uniform float uUseDepthTex;
out vec2 vSourceUV;
out vec3 vNormal;
out vec3 vWorldPos;
flat out float vSurfaceKind;
out float vWallT;
out vec2 vSideNormal;
out vec3 vLocalNormal;
out float vLocalZ;
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
  float blackWeight=1.0-smoothstep(0.0,0.035,sourceDepth);
  float whiteWeight=smoothstep(0.965,1.0,sourceDepth);
  float patternWeight=max(blackWeight,whiteWeight);
  if(patternWeight<=0.0) return sourceDepth;
  float pattern=(5.0+5.0*livingVoronoi(uv))/255.0;
  return sourceDepth+pattern*patternWeight;
}
float signedMeshDepth(vec2 uv,float sourceDepth){
  float meshDepth=depthWithVoidPattern(uv,sourceDepth);
  float base=uSign<0.0?1.0-sourceDepth:sourceDepth;
  return base+(meshDepth-sourceDepth);
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
  bool auxiliary=uUseDepthTex>0.5&&aDepth<-0.5;
  bool side=auxiliary&&aDepth<-1.5;
  bool forcedBase=auxiliary&&(aDepth>-1.5||aDepth<-2.5);
  if(forcedBase) depth=0.0;
  bool layerBack=uUseDepthTex<0.5&&uBackPass>0.5;
  float localZ=depth*(uUseDepthTex<0.5?max(uDepthScale,0.01):uDepthScale);
  if(forcedBase) localZ=-0.0025;
  if(layerBack) localZ-=0.00025;
  vec3 p=rotate3(vec3(aXY,localZ));
  vec3 n;
  if(side) n=normalize(vec3(aGradient,0.0));
  else if(auxiliary) n=vec3(0.0,0.0,-1.0);
  else if(layerBack) n=vec3(0.0,0.0,-1.0);
  else n=normalize(vec3(-gradient.x*uDepthScale,-gradient.y*uDepthScale,1.0));
  vLocalNormal=n;
  vLocalZ=localZ;
  n=normalize(rotate3(n));
  float camDist=uCameraDistance;
  float cz=camDist-p.z;
  float f=1.0/tan(radians(uFov)*0.5);
  float nearP=.1, farP=30.0;
  float zEye=-cz;
  float A=(farP+nearP)/(nearP-farP);
  float B=(2.0*farP*nearP)/(nearP-farP);
  gl_Position=vec4(p.x*f*uZoom/uAspect,p.y*f*uZoom,A*zEye+B,cz);
  vSourceUV=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),0.5-aXY.y/(2.0*uSourceAspect));
  vSurfaceKind=side?1.0:((auxiliary||layerBack)?2.0:0.0);
  vWallT=side&&aDepth<-2.5?1.0:0.0;
  vSideNormal=side?aGradient:vec2(0.0);
  vNormal=n; vWorldPos=p;
}`;

export const G_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 vSourceUV;
in vec3 vNormal;
in vec3 vWorldPos;
flat in float vSurfaceKind;
in float vWallT;
in vec2 vSideNormal;
in vec3 vLocalNormal;
in float vLocalZ;
uniform sampler2D uSourceTex;
uniform float uSourceAspect;
layout(location=0) out vec4 outAlbedo;
layout(location=1) out vec4 outNormal;
layout(location=2) out vec4 outPosition;
float mirrorRepeat(float value){
  float wrapped=mod(value,2.0);
  return 1.0-abs(wrapped-1.0);
}
vec3 sampleSource(vec2 uv){
  vec2 texel=1.0/vec2(textureSize(uSourceTex,0));
  vec3 center=texture(uSourceTex,uv).rgb;
  vec3 neighbours=(
      texture(uSourceTex,uv+vec2(texel.x,0.0)).rgb+
      texture(uSourceTex,uv-vec2(texel.x,0.0)).rgb+
      texture(uSourceTex,uv+vec2(0.0,texel.y)).rgb+
      texture(uSourceTex,uv-vec2(0.0,texel.y)).rgb)*0.25;
  return clamp(center+(center-neighbours)*0.38,0.0,1.0);
}
void main(){
  vec2 sourceUV=clamp(vSourceUV,0.0,1.0);
  vec3 albedo;
  if(vSurfaceKind>0.5&&vSurfaceKind<1.5){
    float depthX=mirrorRepeat(vLocalZ*0.5);
    float depthY=mirrorRepeat(
        vLocalZ/(2.0*max(uSourceAspect,1e-5)));
    vec3 projectedX=sampleSource(vec2(depthX,sourceUV.y));
    vec3 projectedY=sampleSource(vec2(sourceUV.x,depthY));
    vec2 wallWeights=abs(vSideNormal);
    wallWeights/=max(wallWeights.x+wallWeights.y,1e-5);
    vec3 projected=projectedX*wallWeights.x+projectedY*wallWeights.y;
    vec3 edgeColor=sampleSource(sourceUV);
    albedo=mix(edgeColor,projected,smoothstep(0.0,0.18,vWallT));
  }else if(vSurfaceKind<0.5){
    vec3 localNormal=normalize(vLocalNormal);
    vec3 weights=pow(abs(localNormal),vec3(4.0));
    weights/=max(weights.x+weights.y+weights.z,1e-5);
    float depthX=mirrorRepeat(vLocalZ*0.5);
    float depthY=mirrorRepeat(
        vLocalZ/(2.0*max(uSourceAspect,1e-5)));
    vec2 projectionX=vec2(depthX,sourceUV.y);
    vec2 projectionY=vec2(sourceUV.x,depthY);
    vec3 colorX=sampleSource(projectionX);
    vec3 colorY=sampleSource(projectionY);
    vec3 colorZ=sampleSource(sourceUV);
    albedo=colorX*weights.x+colorY*weights.y+colorZ*weights.z;
  }else{
    albedo=sampleSource(sourceUV);
  }
  outAlbedo=vec4(albedo,1.0);
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
uniform float uUseDepthTex,uSourceCrop,uSourceAspect,uPatternTime,uBackPass;
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
  float blackWeight=1.0-smoothstep(0.0,0.035,sourceDepth);
  float whiteWeight=smoothstep(0.965,1.0,sourceDepth);
  float patternWeight=max(blackWeight,whiteWeight);
  if(patternWeight<=0.0) return sourceDepth;
  float pattern=(5.0+5.0*livingVoronoi(uv))/255.0;
  return sourceDepth+pattern*patternWeight;
}
float signedMeshDepth(vec2 uv,float sourceDepth){
  float meshDepth=depthWithVoidPattern(uv,sourceDepth);
  float base=uSign<0.0?1.0-sourceDepth:sourceDepth;
  return base+(meshDepth-sourceDepth);
}
void main(){
  vec2 uv=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),0.5-aXY.y/(2.0*uSourceAspect));
  float sampledDepth=texture(uDepthTex,uv).r;
  sampledDepth=signedMeshDepth(uv,sampledDepth);
  float cpuDepth=uSign<0.0?1.0-aDepth:aDepth;
  float depth=mix(cpuDepth,sampledDepth,uUseDepthTex);
  bool auxiliary=uUseDepthTex>0.5&&aDepth<-0.5;
  bool forcedBase=auxiliary&&(aDepth>-1.5||aDepth<-2.5);
  if(forcedBase) depth=0.0;
  bool layerBack=uUseDepthTex<0.5&&uBackPass>0.5;
  float localZ=depth*(uUseDepthTex<0.5?max(uDepthScale,0.01):uDepthScale);
  if(forcedBase) localZ=-0.0025;
  if(layerBack) localZ-=0.00025;
  vec3 p=rotate3(vec3(aXY,localZ));
  gl_Position=uLightVP*vec4(p,1.0);
}`;

export const SHADOW_FRAGMENT_SHADER = `#version 300 es
precision highp float;
void main(){}`;

export const LAYER_G_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;
uniform float uYaw,uPitch,uDepthScale,uSign,uZoom,uAspect,uSourceCrop,uSourceAspect,uBackPass;
uniform float uCameraDistance,uFov,uOrthographic;
uniform float uLayerLevels[16];
out vec2 vSourceUV;
out vec2 vMaskUV;
out vec3 vNormal;
out vec3 vWorldPos;
flat out int vLayerIndex;
vec3 rotate3(vec3 p){
  float cy=cos(uYaw),sy=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);
  vec3 q=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);
  return vec3(q.x,q.y*cp-q.z*sp,q.y*sp+q.z*cp);
}
void main(){
  int layer=gl_InstanceID;
  float depth=uSign<0.0?1.0-uLayerLevels[layer]:uLayerLevels[layer];
  float z=depth*max(uDepthScale,0.01)-uBackPass*0.00025;
  vec3 p=rotate3(vec3(aXY,z));
  vec3 n=normalize(rotate3(vec3(0.0,0.0,uBackPass>0.5?-1.0:1.0)));
  if(uOrthographic>0.5){
    // Match the apparent scale of the default 45 degree / 3.2 perspective
    // camera, while keeping every layer equally large regardless of depth.
    float orthoScale=0.75444174*uZoom;
    gl_Position=vec4(p.x*orthoScale/uAspect,p.y*orthoScale,-p.z*0.5,1.0);
    vSourceUV=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),
        0.5-aXY.y/(2.0*uSourceAspect));
    vMaskUV=vec2(aXY.x*0.5+0.5,0.5-aXY.y/(2.0*uSourceAspect));
    vNormal=n;vWorldPos=p;vLayerIndex=layer;
    return;
  }
  float cz=uCameraDistance-p.z,f=1.0/tan(radians(uFov)*0.5);
  float nearP=.1,farP=30.0,zEye=-cz;
  float A=(farP+nearP)/(nearP-farP),B=(2.0*farP*nearP)/(nearP-farP);
  gl_Position=vec4(p.x*f*uZoom/uAspect,p.y*f*uZoom,A*zEye+B,cz);
  vSourceUV=vec2(mix(uSourceCrop,1.0,aXY.x*0.5+0.5),
      0.5-aXY.y/(2.0*uSourceAspect));
  vMaskUV=vec2(aXY.x*0.5+0.5,0.5-aXY.y/(2.0*uSourceAspect));
  vNormal=n;vWorldPos=p;vLayerIndex=layer;
}`;

export const LAYER_G_FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler2DArray;
in vec2 vSourceUV;
in vec2 vMaskUV;
in vec3 vNormal;
in vec3 vWorldPos;
flat in int vLayerIndex;
uniform sampler2D uSourceTex;
uniform sampler2DArray uLayerMasks;
layout(location=0) out vec4 outAlbedo;
layout(location=1) out vec4 outNormal;
layout(location=2) out vec4 outPosition;
void main(){
  float mask=texture(uLayerMasks,vec3(clamp(vMaskUV,0.0,1.0),float(vLayerIndex))).r;
  float edgeWidth=max(fwidth(mask)*1.35,1.0/255.0);
  float coverage=smoothstep(0.5-edgeWidth,0.5+edgeWidth,mask);
  // Screen-door dithering forms a visible 4x4 grid once a layer is minified.
  // The mask texture already has filtered mip levels, so a stable threshold
  // gives a cleaner distant silhouette without a repeating screen pattern.
  if(coverage<0.5) discard;
  vec2 uv=clamp(vSourceUV,0.0,1.0),texel=1.0/vec2(textureSize(uSourceTex,0));
  vec3 center=texture(uSourceTex,uv).rgb;
  vec3 neighbours=(texture(uSourceTex,uv+vec2(texel.x,0)).rgb+
      texture(uSourceTex,uv-vec2(texel.x,0)).rgb+
      texture(uSourceTex,uv+vec2(0,texel.y)).rgb+
      texture(uSourceTex,uv-vec2(0,texel.y)).rgb)*0.25;
  vec3 albedo=clamp(center+(center-neighbours)*0.38,0.0,1.0);
  outAlbedo=vec4(albedo,1.0);
  vec3 faceNormal=gl_FrontFacing?vNormal:-vNormal;
  outNormal=vec4(normalize(faceNormal)*0.5+0.5,1.0);
  outPosition=vec4(vWorldPos,float(vLayerIndex)+1.0);
}`;

export const LAYER_SHADOW_VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location=0) in vec2 aXY;
uniform float uYaw,uPitch,uDepthScale,uSign,uSourceAspect,uBackPass;
uniform float uLayerLevels[16];
uniform mat4 uLightVP;
out vec2 vMaskUV;
flat out int vLayerIndex;
vec3 rotate3(vec3 p){
  float cy=cos(uYaw),sy=sin(uYaw),cp=cos(uPitch),sp=sin(uPitch);
  vec3 q=vec3(p.x*cy+p.z*sy,p.y,-p.x*sy+p.z*cy);
  return vec3(q.x,q.y*cp-q.z*sp,q.y*sp+q.z*cp);
}
void main(){
  int layer=gl_InstanceID;
  float depth=uSign<0.0?1.0-uLayerLevels[layer]:uLayerLevels[layer];
  float z=depth*max(uDepthScale,0.01)-uBackPass*0.00025;
  gl_Position=uLightVP*vec4(rotate3(vec3(aXY,z)),1.0);
  vMaskUV=vec2(aXY.x*0.5+0.5,0.5-aXY.y/(2.0*uSourceAspect));
  vLayerIndex=layer;
}`;

export const LAYER_SHADOW_FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler2DArray;
in vec2 vMaskUV;
flat in int vLayerIndex;
uniform sampler2DArray uLayerMasks;
void main(){
  if(texture(uLayerMasks,vec3(clamp(vMaskUV,0.0,1.0),float(vLayerIndex))).r<0.5)
    discard;
}`;

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
    vec2 sampleUV=proj.xy+off;
    vec2 edge=uShadowTexel*0.5;
    if(any(lessThan(sampleUV,edge))||any(greaterThan(sampleUV,vec2(1.0)-edge))) continue;
    float depth=texture(uShadowTex,sampleUV).r;
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
  // Layer cards deliberately do not cast shadows. Besides matching their
  // photographic look, this branch avoids 25 shadow-map taps per pixel.
  float shadow=uShadowStrength>0.001?shadowFactor(worldPos,N,L):0.0;
  float selfShade=0.0;
  if(uSelfShade>0.0){
    float curvature=clamp((length(dFdx(N))+length(dFdy(N)))*2.0,0.0,1.0);
    float grazing=1.0-max(dot(N,V),0.0);
    selfShade=uSelfShade*curvature*(0.35+0.65*grazing);
  }
  vec3 H=normalize(L+V);
  float spec=pow(max(dot(N,H),0.0), max(1.0,uShininess));
  float shadowAmount=clamp(shadow*uShadowStrength,0.0,1.0);
  float direct=(1.0-shadowAmount)*uDiffuse*ndl;
  float illumination=uAmbient + uFill + direct - selfShade;
  vec3 color=albedo*illumination + vec3(uSpecular*spec*(1.0-shadowAmount));
  outColor=vec4(clamp(color,0.0,1.0),1.0);
}`;

export const DOF_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uColorTex;
uniform sampler2D uPositionTex;
uniform sampler2D uBackdropTex;
uniform vec2 uTexel;
uniform vec2 uDirection;
uniform float uMaxBlur;
uniform int uFinalPass;
uniform int uFocusLayers;
uniform float uCurveRate,uCurvePower,uSigmaScale,uSigmaMin;
uniform float uDepthFalloff,uMixStart,uMixEnd;
uniform float uLayerDepths[16];
out vec4 outColor;
float blurRadiusForLayer(int layerIndex){
  // Depths are sorted from front to back. Measure the defocus strictly along
  // the local layer stack, not along camera/world Z, so rotation cannot alter
  // the blur assigned to a plane.
  int focusIndex=clamp(uFocusLayers-1,0,15);
  float delta=max(0.0,uLayerDepths[focusIndex]-uLayerDepths[layerIndex]);
  float curve=1.0-exp(-delta*uCurveRate);
  return uMaxBlur*pow(max(curve,0.0),uCurvePower);
}
float radiusAt(vec4 position){
  if(position.a<0.5) return 0.0;
  int layerIndex=clamp(int(position.a-1.0+0.5),0,15);
  if(layerIndex<uFocusLayers) return 0.0;
  return blurRadiusForLayer(layerIndex);
}
void main(){
  vec4 centerPosition=texture(uPositionTex,vUV);
  vec4 centerSample=texture(uColorTex,vUV);
  vec3 backdrop=texture(uBackdropTex,vUV).rgb;
  bool centerGeometry=centerPosition.a>=0.5;
  int layerIndex=centerGeometry?
      clamp(int(centerPosition.a-1.0+0.5),0,15):0;
  float centerCoverage=uFinalPass==0?(centerGeometry?1.0:0.0):centerSample.a;
  if(centerGeometry&&centerPosition.a<=float(uFocusLayers)){
    outColor=uFinalPass==0?vec4(centerSample.rgb,1.0):
        vec4(centerSample.rgb+backdrop*(1.0-centerSample.a),1.0);
    return;
  }
  float radius=radiusAt(centerPosition);
  // Outside a card there is no center depth from which to derive a radius.
  // Gather the largest nearby circle of confusion so a blurred silhouette can
  // extend into empty space instead of being clipped by the original mask.
  if(!centerGeometry){
    for(int i=-5;i<=5;i++){
      vec2 probeUV=clamp(vUV+uDirection*uTexel*uMaxBlur*float(i)/5.0,
          vec2(0.0),vec2(1.0));
      radius=max(radius,radiusAt(texture(uPositionTex,probeUV)));
    }
  }
  if(radius<uMixStart){
    if(uFinalPass==0)
      outColor=vec4(centerSample.rgb*centerCoverage,centerCoverage);
    else
      outColor=vec4(centerSample.rgb+backdrop*(1.0-centerSample.a),1.0);
    return;
  }
  vec2 axis=uDirection*uTexel*max(radius/5.0,1.0);
  float centerDepth=centerGeometry?uLayerDepths[layerIndex]:0.0;
  float sigma=max(radius*uSigmaScale,uSigmaMin);
  vec3 sum=vec3(0.0);
  float alphaSum=0.0;
  float kernelWeightSum=0.0;
  // A sparse 3x3 kernel leaves large unsampled gaps on high-frequency
  // stereogram textures, which shows up as tiled ghost copies instead of blur.
  // Blur along each axis with dense Gaussian taps so the footprint stays smooth
  // as the radius changes across layers.
  for(int i=-5;i<=5;i++){
    float stepIndex=float(i);
    float pixelOffset=abs(stepIndex)*max(radius/5.0,1.0);
    float spatialWeight=exp(-0.5*pixelOffset*pixelOffset/(sigma*sigma));
    vec2 sampleUV=clamp(vUV+axis*stepIndex,vec2(0.0),vec2(1.0));
    vec4 samplePosition=texture(uPositionTex,sampleUV);
    vec4 colorSample=texture(uColorTex,sampleUV);
    float coverage=uFinalPass==0?(samplePosition.a>=0.5?1.0:0.0):colorSample.a;
    float accepted=1.0;
    float depthWeight=1.0;
    if(centerGeometry&&samplePosition.a>=0.5){
      int sampleLayer=clamp(int(samplePosition.a-1.0+0.5),0,15);
      float sampleDepth=uLayerDepths[sampleLayer];
      float inFront=max(0.0,sampleDepth-centerDepth);
      if(inFront>1e-4){
        coverage=0.0;
        accepted=0.0;
      }
      float behind=max(0.0,centerDepth-sampleDepth);
      depthWeight=exp(-behind*uDepthFalloff);
    }
    float weight=spatialWeight*depthWeight;
    vec3 premultiplied=uFinalPass==0?
        colorSample.rgb*coverage:colorSample.rgb*accepted;
    sum+=premultiplied*weight;
    alphaSum+=coverage*weight;
    kernelWeightSum+=weight;
  }
  vec3 blurred=sum/max(kernelWeightSum,1e-5);
  float blurredCoverage=alphaSum/max(kernelWeightSum,1e-5);
  float amount=smoothstep(uMixStart,max(uMixStart+0.001,uMixEnd),radius);
  vec3 centerPremultiplied=uFinalPass==0?
      centerSample.rgb*centerCoverage:centerSample.rgb;
  vec3 premultiplied=mix(centerPremultiplied,blurred,amount);
  float coverage=mix(centerCoverage,blurredCoverage,amount);
  if(uFinalPass==0)
    outColor=vec4(premultiplied,coverage);
  else{
    // Reconstruct the edge color before compositing. Using premultiplied RGB
    // directly makes dark fringes; adding the original foreground underneath
    // makes the same edge look emissive. A slightly tightened coverage ramp
    // keeps the contour soft without either halo.
    vec3 edgeColor=coverage>1e-4?premultiplied/coverage:backdrop;
    float edgeCoverage=smoothstep(0.12,0.88,coverage);
    outColor=vec4(mix(backdrop,edgeColor,edgeCoverage),1.0);
  }
}`;
