import {G_FRAGMENT_SHADER, G_VERTEX_SHADER, LIGHT_FRAGMENT_SHADER, LIGHT_VERTEX_SHADER, SHADOW_FRAGMENT_SHADER, SHADOW_VERTEX_SHADER} from './shaders.js';

'use strict';
const $ = s => document.querySelector(s);
const cv = $('#view'), depthCv = $('#depthView'),
      depthCtx = depthCv.getContext('2d'), fi = $('#file'),
      textureFi = $('#textureFile'), drop = $('#drop'), thumb = $('#thumb'),
      progressOverlay = $('#progressOverlay'),
      progressText = $('#progressText'), progressBar = $('#progressBar');
const depthR = $('#depth'), depthN = $('#depthNum');
const layersR = $('#layers'), layersN = $('#layersNum');
const periodR = $('#period'), periodN = $('#periodNum');
const autoBtn = $('#auto');
const MESH_STEP_PX = 2;
const MESH_SMOOTH_AMOUNT = 0.6;
const gl = cv.getContext(
    'webgl2',
    {alpha: false, antialias: true, powerPreference: 'high-performance'});
if (!gl) {
  alert('Нужен WebGL2');
  throw new Error('WebGL2 unavailable');
}

let img = null, loadedDepthMap = false, textureLoaded = false;
let mapW = 0, mapH = 0, cropX = 0;
let rgbaMap = null, rawDepth = null, processedDepth = null,
    cleanDepthMap = null, depthPreview = null, depthPreviewW = 0,
    depthPreviewH = 0;
let yaw = -.22, pitch = -.16, zoom = 1, drag = false, lastX = 0, lastY = 0,
    pinch = 0;
let autoRotate = false, autoPauseUntil = 0, lastFrame = performance.now(),
    dirty = true, autoTime = 0, autoBaseYaw = -.22, autoBasePitch = -.16;
let meshIndexCount = 0, processTimer = 0, depthWorker = null,
    analysisJob = 0;
let meshSourceBounds = {minX: -1, maxX: 1, minY: -1, maxY: 1, minD: 0, maxD: 0};
function setProgress(value, _label) {
  const v = Math.max(0, Math.min(1, Number(value) || 0));
  progressOverlay.hidden = false;
  progressBar.style.width = (v * 100).toFixed(1) + '%';
  progressText.textContent = (v * 100).toFixed(0) + '%';
}
function hideProgress() {
  progressOverlay.hidden = true;
  progressBar.style.width = '0%';
}
function nextFrame() {
  return new Promise(resolve => requestAnimationFrame(() => resolve()));
}
function createDepthWorker() {
  return new Worker(new URL('./depth-worker.js', import.meta.url));
}
function currentShape() {
  return +document.querySelector('input[name="shape"]:checked').value;
}
function currentViewMode() {
  return document.querySelector('input[name="viewmode"]:checked').value;
}
function clampSliderValue(el, v) {
  let n = Number(v);
  if (!Number.isFinite(n)) n = Number(el.value) || 0;
  const min = el.min !== '' ? Number(el.min) : -Infinity;
  const max = el.max !== '' ? Number(el.max) : Infinity;
  const step = (el.step && el.step !== 'any') ? Number(el.step) : 0;
  n = Math.max(min, Math.min(max, n));
  if (step > 0) {
    const base = Number.isFinite(min) ? min : 0;
    n = Math.round((n - base) / step) * step + base;
    const d = (String(step).split('.')[1] || '').length;
    n = Number(n.toFixed(d));
  }
  return n;
}
function formatValue(el, v) {
  const step = (el.step && el.step !== 'any') ? Number(el.step) : 0;
  const d = step > 0 ? (String(step).split('.')[1] || '').length : 0;
  return d > 0 ? Number(v).toFixed(d) : String(Math.round(v));
}
function commitPair(r, n, fireChange = true) {
  const v = clampSliderValue(r, n.value);
  r.value = String(v);
  n.value = formatValue(r, v);
  r.dispatchEvent(new Event('input', {bubbles: true}));
  if (fireChange) r.dispatchEvent(new Event('change', {bubbles: true}));
}
function syncPair(r, n) {
  const fromRange = () => {
    const v = clampSliderValue(r, r.value);
    r.value = String(v);
    n.value = formatValue(r, v);
  };
  r.addEventListener('input', fromRange);
  r.addEventListener('change', fromRange);
  fromRange();
  n.addEventListener('change', () => commitPair(r, n, true));
  n.addEventListener('blur', () => commitPair(r, n, true));
  n.addEventListener('keydown', e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      commitPair(r, n, true);
      n.blur();
    }
  });
}
function setPair(r, n, v) {
  const x = clampSliderValue(r, v);
  r.value = String(x);
  n.value = formatValue(r, x);
}
function axisValues(start, end, step) {
  const a = [];
  for (let v = start; v < end; v += step) a.push(v);
  if (!a.length || a[a.length - 1] !== end - 1) a.push(end - 1);
  return a;
}
function sched() {
  dirty = true;
}
function pauseAuto(ms = 1800) {
  autoPauseUntil = performance.now() + ms;
}
function resetCamera() {
  yaw = -.22;
  pitch = -.16;
  zoom = 1;
  autoBaseYaw = yaw;
  autoBasePitch = pitch;
  autoTime = 0;
  pauseAuto(700);
  sched();
}
function updateAutoButton() {
  autoBtn.textContent = '▶';
  autoBtn.classList.toggle('active', !!autoRotate);
}
function disableAutoRotate() {
  if (autoRotate) {
    autoRotate = false;
    updateAutoButton();
  }
}
function lightParams() {
  return {
    x: -0.5,
    y: 0.4,
    z: 0.5,
    ambient: 0.7,
    diffuse: 0.9,
    fill: 0.0,
    shadowStrength: 1.4,
    selfShade: 0.5,
    specular: 0.1,
    shininess: 5.0,
    biasBase: 0.02,
    biasSlope: 0.0,
    pcfRadius: 1.0,
    shadowHardness: 0.7
  };
}
function lightDirCameraSpace() {
  const p = lightParams();
  const len = Math.hypot(p.x, p.y, p.z) || 1;
  return {x: p.x / len, y: p.y / len, z: p.z / len};
}

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
    throw new Error(gl.getShaderInfoLog(s));
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p));
  return p;
}
const gProg = program(G_VERTEX_SHADER, G_FRAGMENT_SHADER);

const shadowProg = program(SHADOW_VERTEX_SHADER, SHADOW_FRAGMENT_SHADER);

const lightProg = program(LIGHT_VERTEX_SHADER, LIGHT_FRAGMENT_SHADER);

const vao = gl.createVertexArray(), vbo = gl.createBuffer(),
      ebo = gl.createBuffer();
gl.bindVertexArray(vao);
gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
const stride = 5 * 4;
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, stride, 0);
gl.enableVertexAttribArray(1);
gl.vertexAttribPointer(1, 1, gl.FLOAT, false, stride, 2 * 4);
gl.enableVertexAttribArray(2);
gl.vertexAttribPointer(2, 2, gl.FLOAT, false, stride, 3 * 4);
gl.bindVertexArray(null);
const quadVao = gl.createVertexArray();

const GU = {
  yaw: gl.getUniformLocation(gProg, 'uYaw'),
  pitch: gl.getUniformLocation(gProg, 'uPitch'),
  depth: gl.getUniformLocation(gProg, 'uDepthScale'),
  sign: gl.getUniformLocation(gProg, 'uSign'),
  zoom: gl.getUniformLocation(gProg, 'uZoom'),
  aspect: gl.getUniformLocation(gProg, 'uAspect'),
  sourceTex: gl.getUniformLocation(gProg, 'uSourceTex'),
  sourceCrop: gl.getUniformLocation(gProg, 'uSourceCrop'),
  sourceAspect: gl.getUniformLocation(gProg, 'uSourceAspect')
};
const SU = {
  yaw: gl.getUniformLocation(shadowProg, 'uYaw'),
  pitch: gl.getUniformLocation(shadowProg, 'uPitch'),
  depth: gl.getUniformLocation(shadowProg, 'uDepthScale'),
  sign: gl.getUniformLocation(shadowProg, 'uSign'),
  lightVP: gl.getUniformLocation(shadowProg, 'uLightVP')
};
const LU = {
  albedo: gl.getUniformLocation(lightProg, 'uAlbedoTex'),
  normal: gl.getUniformLocation(lightProg, 'uNormalTex'),
  position: gl.getUniformLocation(lightProg, 'uPositionTex'),
  shadowTex: gl.getUniformLocation(lightProg, 'uShadowTex'),
  shadowTexel: gl.getUniformLocation(lightProg, 'uShadowTexel'),
  lightVP: gl.getUniformLocation(lightProg, 'uLightVP'),
  lightDir: gl.getUniformLocation(lightProg, 'uLightDir'),
  cameraPos: gl.getUniformLocation(lightProg, 'uCameraPos'),
  ambient: gl.getUniformLocation(lightProg, 'uAmbient'),
  diffuse: gl.getUniformLocation(lightProg, 'uDiffuse'),
  fill: gl.getUniformLocation(lightProg, 'uFill'),
  shadowStrength: gl.getUniformLocation(lightProg, 'uShadowStrength'),
  selfShade: gl.getUniformLocation(lightProg, 'uSelfShade'),
  specular: gl.getUniformLocation(lightProg, 'uSpecular'),
  shininess: gl.getUniformLocation(lightProg, 'uShininess'),
  biasBase: gl.getUniformLocation(lightProg, 'uBiasBase'),
  biasSlope: gl.getUniformLocation(lightProg, 'uBiasSlope'),
  pcfRadius: gl.getUniformLocation(lightProg, 'uPcfRadius'),
  shadowHardness: gl.getUniformLocation(lightProg, 'uShadowHardness')
};

gl.getExtension('EXT_color_buffer_float');
gl.enable(gl.DEPTH_TEST);
gl.depthFunc(gl.LEQUAL);
gl.disable(gl.CULL_FACE);
gl.clearColor(.045, .05, .06, 1);
const sourceTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, sourceTex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
    new Uint8Array([255, 255, 255, 255]));
let shadowSize = 2048;
const shadowTex = gl.createTexture();
const shadowFbo = gl.createFramebuffer();
function initShadowMap() {
  gl.bindTexture(gl.TEXTURE_2D, shadowTex);
  gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.DEPTH_COMPONENT24, shadowSize, shadowSize, 0,
      gl.DEPTH_COMPONENT, gl.UNSIGNED_INT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFbo);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, shadowTex, 0);
  gl.drawBuffers([gl.NONE]);
  gl.readBuffer(gl.NONE);
  const shadowFboStatus = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (shadowFboStatus !== gl.FRAMEBUFFER_COMPLETE) {
    console.error('Shadow framebuffer incomplete:', shadowFboStatus);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
initShadowMap();

const gFbo = gl.createFramebuffer();
const gAlbedoTex = gl.createTexture();
const gNormalTex = gl.createTexture();
const gPositionTex = gl.createTexture();
const gDepthRb = gl.createRenderbuffer();
function initRenderTex(tex, internalFormat, format, type) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(
      gl.TEXTURE_2D, 0, internalFormat, cv.width, cv.height, 0, format, type,
      null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}
function initGBuffer() {
  initRenderTex(gAlbedoTex, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
  initRenderTex(gNormalTex, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
  initRenderTex(gPositionTex, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
  gl.bindRenderbuffer(gl.RENDERBUFFER, gDepthRb);
  gl.renderbufferStorage(
      gl.RENDERBUFFER, gl.DEPTH_COMPONENT24, cv.width, cv.height);
  gl.bindFramebuffer(gl.FRAMEBUFFER, gFbo);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, gAlbedoTex, 0);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, gNormalTex, 0);
  gl.framebufferTexture2D(
      gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, gPositionTex, 0);
  gl.framebufferRenderbuffer(
      gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, gDepthRb);
  gl.drawBuffers(
      [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2]);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

function mat4Identity() {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}
function mat4Mul(a, b) {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      out[c * 4 + r] = a[0 * 4 + r] * b[c * 4 + 0] +
          a[1 * 4 + r] * b[c * 4 + 1] + a[2 * 4 + r] * b[c * 4 + 2] +
          a[3 * 4 + r] * b[c * 4 + 3];
  return out;
}
function mat4Ortho(l, r, b, t, n, f) {
  const m = mat4Identity();
  m[0] = 2 / (r - l);
  m[5] = 2 / (t - b);
  m[10] = -2 / (f - n);
  m[12] = -(r + l) / (r - l);
  m[13] = -(t + b) / (t - b);
  m[14] = -(f + n) / (f - n);
  return m;
}
function vec3Norm(x, y, z) {
  const len = Math.hypot(x, y, z) || 1;
  return {x: x / len, y: y / len, z: z / len};
}
function vec3Cross(a, b) {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x
  };
}
function mat4LookAt(eye, center, up) {
  const f = vec3Norm(center.x - eye.x, center.y - eye.y, center.z - eye.z);
  let s = vec3Cross(f, up);
  s = vec3Norm(s.x, s.y, s.z);
  const u = vec3Cross(s, f);
  const m = mat4Identity();
  m[0] = s.x;
  m[4] = s.y;
  m[8] = s.z;
  m[1] = u.x;
  m[5] = u.y;
  m[9] = u.z;
  m[2] = -f.x;
  m[6] = -f.y;
  m[10] = -f.z;
  m[12] = -(s.x * eye.x + s.y * eye.y + s.z * eye.z);
  m[13] = -(u.x * eye.x + u.y * eye.y + u.z * eye.z);
  m[14] = (f.x * eye.x + f.y * eye.y + f.z * eye.z);
  return m;
}

function transformPoint(m, p) {
  return {
    x: m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12],
    y: m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13],
    z: m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14]
  };
}
function rotatePoint(p, yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch),
        sp = Math.sin(pitch);
  const qx = p.x * cy + p.z * sy, qy = p.y, qz = -p.x * sy + p.z * cy;
  return {x: qx, y: qy * cp - qz * sp, z: qy * sp + qz * cp};
}
function updateMeshBoundsFromVerts(verts) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity,
      minD = Infinity, maxD = -Infinity;
  for (let i = 0; i < verts.length; i += 5) {
    const x = verts[i], y = verts[i + 1], d = verts[i + 2];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
    if (d < minD) minD = d;
    if (d > maxD) maxD = d;
  }
  if (Number.isFinite(minX))
    meshSourceBounds = {minX, maxX, minY, maxY, minD, maxD};
}
function computeLightVP() {
  const d = lightDirCameraSpace();
  const L = vec3Norm(d.x, d.y, d.z);
  const scale = (+depthR.value || 0) / 100;
  const sign = currentShape();
  const b = meshSourceBounds;
  const corners = [];
  for (const x of [b.minX, b.maxX])
    for (const y of [b.minY, b.maxY])
      for (const d0 of [b.minD, b.maxD]) {
        corners.push(rotatePoint({x, y, z: d0 * scale * sign}, yaw, pitch));
      }
  let cminX = Infinity, cmaxX = -Infinity, cminY = Infinity, cmaxY = -Infinity,
      cminZ = Infinity, cmaxZ = -Infinity;
  for (const p of corners) {
    if (p.x < cminX) cminX = p.x;
    if (p.x > cmaxX) cmaxX = p.x;
    if (p.y < cminY) cminY = p.y;
    if (p.y > cmaxY) cmaxY = p.y;
    if (p.z < cminZ) cminZ = p.z;
    if (p.z > cmaxZ) cmaxZ = p.z;
  }
  const center = {
    x: (cminX + cmaxX) * 0.5,
    y: (cminY + cmaxY) * 0.5,
    z: (cminZ + cmaxZ) * 0.5
  };
  let radius = 0.1;
  for (const p of corners) {
    const dx = p.x - center.x, dy = p.y - center.y, dz = p.z - center.z;
    radius = Math.max(radius, Math.hypot(dx, dy, dz));
  }
  const eye = {
    x: center.x + L.x * (radius * 2.2),
    y: center.y + L.y * (radius * 2.2),
    z: center.z + L.z * (radius * 2.2)
  };
  const up = Math.abs(L.y) > 0.92 ? {x: 0, y: 0, z: 1} : {x: 0, y: 1, z: 0};
  const view = mat4LookAt(eye, center, up);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity,
      minZ = Infinity, maxZ = -Infinity;
  for (const p of corners) {
    const q = transformPoint(view, p);
    if (q.x < minX) minX = q.x;
    if (q.x > maxX) maxX = q.x;
    if (q.y < minY) minY = q.y;
    if (q.y > maxY) maxY = q.y;
    if (q.z < minZ) minZ = q.z;
    if (q.z > maxZ) maxZ = q.z;
  }
  const pad = Math.max(0.08, radius * 0.12);
  minX -= pad;
  maxX += pad;
  minY -= pad;
  maxY += pad;
  const nearP = Math.max(0.05, -(maxZ + pad));
  const farP = Math.max(nearP + 0.1, -(minZ - pad));
  const proj = mat4Ortho(minX, maxX, minY, maxY, nearP, farP);
  return mat4Mul(proj, view);
}
function renderShadowMap(lightVP) {
  gl.bindFramebuffer(gl.FRAMEBUFFER, shadowFbo);
  gl.viewport(0, 0, shadowSize, shadowSize);
  gl.clear(gl.DEPTH_BUFFER_BIT);
  gl.colorMask(false, false, false, false);
  gl.enable(gl.POLYGON_OFFSET_FILL);
  gl.polygonOffset(1.5, 4.0);
  gl.useProgram(shadowProg);
  gl.bindVertexArray(vao);
  gl.uniform1f(SU.yaw, yaw);
  gl.uniform1f(SU.pitch, pitch);
  gl.uniform1f(SU.depth, (+depthR.value || 0) / 100);
  gl.uniform1f(SU.sign, currentShape());
  gl.uniformMatrix4fv(SU.lightVP, false, lightVP);
  gl.drawElements(gl.TRIANGLES, meshIndexCount, gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);
  gl.disable(gl.POLYGON_OFFSET_FILL);
  gl.colorMask(true, true, true, true);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, cv.width, cv.height);
}
function renderGBuffer() {
  gl.bindFramebuffer(gl.FRAMEBUFFER, gFbo);
  gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  gl.useProgram(gProg);
  gl.bindVertexArray(vao);
  gl.uniform1f(GU.yaw, yaw);
  gl.uniform1f(GU.pitch, pitch);
  gl.uniform1f(GU.depth, (+depthR.value || 0) / 100);
  gl.uniform1f(GU.sign, currentShape());
  gl.uniform1f(GU.zoom, zoom);
  gl.uniform1f(GU.aspect, cv.width / Math.max(1, cv.height));
  gl.uniform1f(GU.sourceCrop, cropX / Math.max(1, mapW));
  gl.uniform1f(GU.sourceAspect, mapH / Math.max(1, mapW - cropX));
  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_2D, sourceTex);
  gl.uniform1i(GU.sourceTex, 4);
  gl.drawElements(gl.TRIANGLES, meshIndexCount, gl.UNSIGNED_INT, 0);
  gl.bindVertexArray(null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2),
        r = cv.getBoundingClientRect();
  cv.width = Math.max(1, Math.round(r.width * dpr));
  cv.height = Math.max(1, Math.round(r.height * dpr));
  depthCv.width = cv.width;
  depthCv.height = cv.height;
  initGBuffer();
  gl.viewport(0, 0, cv.width, cv.height);
  sched();
}
addEventListener('resize', resize);
resize();
function frame(now) {
  const dt = Math.min(50, now - lastFrame);
  lastFrame = now;
  if (processedDepth && autoRotate && !drag && now > autoPauseUntil) {
    autoTime += dt * 0.001;
    yaw = autoBaseYaw + Math.sin(autoTime * 0.72) * 0.55;
    pitch = autoBasePitch + Math.sin(autoTime * 0.41) * 0.14;
    dirty = true;
  }
  if (dirty) {
    dirty = false;
    render();
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function updateLayerUi() {
  const mode = currentViewMode();
  layersR.parentElement.classList.toggle('disabled', mode !== 'layers');
  const depthMode = mode === 'depthmap';
  cv.style.display = depthMode ? 'none' : 'block';
  depthCv.style.display = depthMode ? 'block' : 'none';
  depthR.parentElement.classList.toggle('disabled', depthMode);
}
updateLayerUi();

function getImagePixels(image) {
  const w = image.naturalWidth;
  const h = image.naturalHeight;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const x = c.getContext('2d', {willReadFrequently: true});
  x.drawImage(image, 0, 0, w, h);
  return {d: x.getImageData(0, 0, w, h).data, w, h};
}
function blockMedian(src, w, h, x, y, r) {
  const vals = [];
  const x0 = Math.max(cropX, x - r), x1 = Math.min(w - 1, x + r),
        y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r);
  for (let yy = y0; yy <= y1; yy++)
    for (let xx = x0; xx <= x1; xx++) vals.push(src[yy * w + xx]);
  vals.sort((a, b) => a - b);
  return vals[(vals.length / 2) | 0];
}

function updateSourceModeUi() {
  periodR.parentElement.classList.toggle('disabled', loadedDepthMap);
}

function uploadTextureImage(image) {
  gl.bindTexture(gl.TEXTURE_2D, sourceTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  textureLoaded = true;
  sched();
}

function activateImage(image, url, revokeUrl = false) {
  img = image;
  loadedDepthMap = false;
  updateSourceModeUi();
  uploadTextureImage(image);
  thumb.src = url;
  thumb.hidden = false;
  drop.hidden = true;
  resetCamera();
  autoRotate = true;
  autoBaseYaw = yaw;
  autoBasePitch = pitch;
  autoTime = 0;
  autoPauseUntil = 0;
  updateAutoButton();
  sched();
  analyze()
      .catch(err => {
        console.error(err);
        hideProgress();
      })
      .finally(() => {
        if (revokeUrl) URL.revokeObjectURL(url);
      });
}
function loadImage(file) {
  const url = URL.createObjectURL(file), image = new Image();
  image.onload = () => isGrayscaleImage(image) ?
      activateDepthImage(image, url, true) :
      activateImage(image, url, true);
  image.onerror = () => {
    URL.revokeObjectURL(url);
    console.error('Не удалось открыть изображение');
  };
  image.src = url;
}
function loadTexture(file) {
  const url = URL.createObjectURL(file), image = new Image();
  image.onload = () => {
    uploadTextureImage(image);
    URL.revokeObjectURL(url);
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    console.error('Не удалось открыть текстуру');
  };
  image.src = url;
}
function isGrayscaleImage(image) {
  const side = 64,
        scale = Math.min(
            1, side / Math.max(image.naturalWidth, image.naturalHeight));
  const w = Math.max(1, Math.round(image.naturalWidth * scale)),
        h = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(image, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  let grayPixels = 0;
  for (let p = 0; p < data.length; p += 4)
    if (Math.abs(data[p] - data[p + 1]) <= 2 &&
        Math.abs(data[p + 1] - data[p + 2]) <= 2)
      grayPixels++;
  return grayPixels / Math.max(1, data.length / 4) >= .99;
}
function activateDepthImage(image, url, revokeUrl = false) {
  img = image;
  loadedDepthMap = true;
  updateSourceModeUi();
  analysisJob++;
  if (depthWorker) {
    depthWorker.terminate();
    depthWorker = null;
  }
  if (!textureLoaded) uploadTextureImage(image);
  const canvas = document.createElement('canvas');
  mapW = canvas.width = image.naturalWidth;
  mapH = canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, mapW, mapH).data;
  rawDepth = new Float32Array(mapW * mapH);
  for (let i = 0; i < rawDepth.length; i++) {
    const p = i * 4,
          gray = pixels[p] * .299 + pixels[p + 1] * .587 + pixels[p + 2] * .114;
    rawDepth[i] = gray / 255;
  }
  processedDepth = rawDepth.slice();
  cropX = 0;
  if (revokeUrl)
    thumb.onload = () => {
      URL.revokeObjectURL(url);
      thumb.onload = null;
    };
  thumb.src = url;
  thumb.hidden = false;
  drop.hidden = true;
  resetCamera();
  autoRotate = true;
  updateAutoButton();
  rebuildDepthPreview();
  buildMesh();
  sched();
  hideProgress();
}
async function loadImageUrl(url) {
  const response = await fetch(url, {cache: 'no-store'});
  if (!response.ok) throw new Error(`Изображение: HTTP ${response.status}`);
  loadImage(await response.blob());
}
async function loadRandomStartupImage() {
  try {
    const listUrl = new URL('./src/images/images.json', document.baseURI);
    if (location.protocol === 'file:') {
      throw new Error(
          'Автозагрузка требует запуска страницы через HTTP, а не file://');
    }
    const response = await fetch(listUrl, {cache: 'no-store'});
    if (!response.ok) throw new Error(`images.json: HTTP ${response.status}`);
    const data = await response.json();
    if (!Array.isArray(data))
      throw new Error('images.json должен содержать массив');
    const files = data.filter(v => typeof v === 'string' && v.trim());
    if (!files.length) throw new Error('В images.json нет изображений');
    const selected = files[Math.floor(Math.random() * files.length)].trim();
    await loadImageUrl(new URL(selected, listUrl).href);
  } catch (err) {
    console.error('Не удалось выбрать стартовое изображение:', err);
  }
}

async function analyze() {
  if (!img) return;
  const job = ++analysisJob;
  if (depthWorker) {
    depthWorker.terminate();
    depthWorker = null;
  }
  setProgress(.01, 'Подготовка изображения…');
  await nextFrame();
  const {d, w, h} = getImagePixels(img);
  mapW = w;
  mapH = h;
  rgbaMap = d;
  await recoverDepth(w, h, 0, job);
}

function recoverDepth(w, h, manualPeriod = 0, existingJob = null) {
  const job = existingJob ?? ++analysisJob;
  if (depthWorker) {
    depthWorker.terminate();
    depthWorker = null;
  }
  setProgress(.02, 'Извлечение depth map…');
  depthWorker = createDepthWorker();
  return new Promise((resolve, reject) => {
    const worker = depthWorker;
    worker.onmessage = e => {
      const m = e.data;
      if (m.id !== job) return;
      if (m.type === 'progress') {
        setProgress(m.value, m.label);
        return;
      }
      if (m.type === 'done') {
        if (job !== analysisJob) {
          worker.terminate();
          resolve();
          return;
        }
        rawDepth = new Float32Array(m.rawDepth);
        console.info(`Detected stereogram period: ${m.detectedPeriod}px`);
        periodR.max = String(Math.min(250, w >> 1));
        periodN.max = periodR.max;
        setPair(periodR, periodN, m.detectedPeriod);
        cropX = 0;
        worker.terminate();
        if (depthWorker === worker) depthWorker = null;
        setProgress(.97, 'Постобработка и построение поверхности…');
        reprocess(() => {
          if (job === analysisJob) hideProgress();
          resolve();
        });
      }
    };
    worker.onerror = err => {
      if (depthWorker === worker) depthWorker = null;
      worker.terminate();
      hideProgress();
      reject(err);
    };
    const rgbaCopy = rgbaMap.slice();
    worker.postMessage(
        {id: job, rgbaBuffer: rgbaCopy.buffer, w, h, period: manualPeriod},
        [rgbaCopy.buffer]);
  });
}

function reprocess(onDone = null) {
  clearTimeout(processTimer);
  processTimer = setTimeout(() => {
    if (!rawDepth) {
      if (onDone) onDone();
      return;
    }
    processedDepth = rawDepth.slice();
    rebuildDepthPreview();
    buildMesh();
    sched();
    if (onDone) onDone();
  }, 20);
}

function buildMesh() {
  if (!processedDepth) return;
  const stepPx = MESH_STEP_PX;
  const xs = axisValues(cropX, mapW, stepPx), ys = axisValues(0, mapH, stepPx);
  const nx = xs.length, ny = ys.length;
  const visibleW = Math.max(1, mapW - 1 - cropX),
        centerX = (cropX + mapW - 1) * .5;
  const sign = currentShape(), mode = currentViewMode();

  if (mode === 'layers') {
    // Independent horizontal cut-out planes. Convex and concave use opposite
    // cut rules.
    const layerSource =
        (cleanDepthMap && cleanDepthMap.length === mapW * mapH) ?
        cleanDepthMap :
        processedDepth;
    const depthAt = (x, y) => Math.max(0, layerSource[y * mapW + x]);
    const requested = Math.max(2, +layersR.value || 5);
    let maxLevel = 0;
    for (let gy = 0; gy < ny; gy++)
      for (let gx = 0; gx < nx; gx++)
        maxLevel = Math.max(maxLevel, depthAt(xs[gx], ys[gy]));
    const levels = [];
    for (let i = 0; i < requested; i++)
      levels.push(maxLevel * (i / Math.max(1, requested - 1)));

    const layerCount = levels.length;
    const vertsPerLayer = nx * ny;
    const verts = new Float32Array(layerCount * vertsPerLayer * 5);
    const indices =
        new Uint32Array(Math.max(0, layerCount * (nx - 1) * (ny - 1) * 6));
    const eps = 1e-5, concave = sign < 0, masks = [];
    for (let li = 0; li < layerCount; li++) {
      const level = levels[li], mask = new Uint8Array(nx * ny);
      for (let gy = 0; gy < ny; gy++) {
        const y = ys[gy];
        for (let gx = 0; gx < nx; gx++) {
          const d = depthAt(xs[gx], y);
          if (concave ? d <= level + eps : d >= level - eps)
            mask[gy * nx + gx] = 1;
        }
      }
      masks.push(mask);
    }

    let vo = 0, iq = 0;
    for (let li = 0; li < layerCount; li++) {
      const level = levels[li], baseVertex = li * vertsPerLayer;
      for (let gy = 0; gy < ny; gy++) {
        const y = ys[gy];
        for (let gx = 0; gx < nx; gx++) {
          const x = xs[gx], src = y * mapW + x;
          verts[vo++] = (x - centerX) / visibleW * 2;
          verts[vo++] = (mapH * .5 - y) / visibleW * 2;
          verts[vo++] = level;
          verts[vo++] = 0;
          verts[vo++] = 0;
        }
      }
      const mask = masks[li];
      for (let y = 0; y < ny - 1; y++)
        for (let x = 0; x < nx - 1; x++) {
          const a = baseVertex + y * nx + x, b = a + 1, c = a + nx, d = c + 1;
          const m00 = mask[y * nx + x], m10 = mask[y * nx + x + 1],
                m01 = mask[(y + 1) * nx + x], m11 = mask[(y + 1) * nx + x + 1];
          if (m00 && m01 && m10) {
            indices[iq++] = a;
            indices[iq++] = c;
            indices[iq++] = b;
          }
          if (m10 && m01 && m11) {
            indices[iq++] = b;
            indices[iq++] = c;
            indices[iq++] = d;
          }
        }
    }
    meshIndexCount = iq;
    updateMeshBoundsFromVerts(verts);
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
    gl.bufferData(
        gl.ELEMENT_ARRAY_BUFFER, indices.subarray(0, iq), gl.STATIC_DRAW);
    return;
  }
  // Continuous surface mode: build geometry from the cleaned depth-map result,
  // so the same denoised depth estimation drives both the 2D depth preview and
  // the 3D surface.
  const surfaceMap = (cleanDepthMap && cleanDepthMap.length === mapW * mapH) ?
      cleanDepthMap :
      processedDepth;
  let sampled = new Float32Array(nx * ny);
  const radius = Math.min(2, Math.max(1, Math.floor(stepPx / 2)));
  for (let gy = 0; gy < ny; gy++) {
    const y = ys[gy];
    for (let gx = 0; gx < nx; gx++) {
      sampled[gy * nx + gx] =
          blockMedian(surfaceMap, mapW, mapH, xs[gx], y, radius);
    }
  }
  const meshAmount = MESH_SMOOTH_AMOUNT;
  if (meshAmount > 0) {
    let cur = sampled.slice();
    const passes = 1 + (meshAmount > .45 ? 1 : 0) + (meshAmount > .75 ? 1 : 0);
    const alpha = .12 + meshAmount * .55;
    for (let pass = 0; pass < passes; pass++) {
      const out = cur.slice();
      for (let gy = 1; gy < ny - 1; gy++)
        for (let gx = 1; gx < nx - 1; gx++) {
          const i = gy * nx + gx;
          const avg =
              (cur[i - 1] + cur[i + 1] + cur[i - nx] + cur[i + nx]) * .25;
          out[i] = cur[i] * (1 - alpha) + avg * alpha;
        }
      cur = out;
    }
    sampled = cur;
  }
  const verts = new Float32Array(nx * ny * 5);
  let o = 0;
  for (let gy = 0; gy < ny; gy++) {
    const y = ys[gy];
    for (let gx = 0; gx < nx; gx++) {
      const x = xs[gx], i = y * mapW + x;
      const zm = sampled[gy * nx + gx];
      const xm = Math.max(0, gx - 1), xp = Math.min(nx - 1, gx + 1),
            ym = Math.max(0, gy - 1), yp = Math.min(ny - 1, gy + 1);
      const gradX =
          (sampled[gy * nx + xp] - sampled[gy * nx + xm]) * visibleW * .23;
      const gradY =
          -(sampled[yp * nx + gx] - sampled[ym * nx + gx]) * visibleW * .23;
      verts[o++] = (x - centerX) / visibleW * 2;
      verts[o++] = (mapH * .5 - y) / visibleW * 2;
      verts[o++] = zm;
      verts[o++] = Math.max(-10, Math.min(10, gradX));
      verts[o++] = Math.max(-10, Math.min(10, gradY));
    }
  }
  const indices = new Uint32Array(Math.max(0, (nx - 1) * (ny - 1) * 6));
  let q = 0;
  for (let y = 0; y < ny - 1; y++)
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
      const ia = ys[y] * mapW + xs[x], ib = ys[y] * mapW + xs[x + 1],
            ic = ys[y + 1] * mapW + xs[x], id = ys[y + 1] * mapW + xs[x + 1];
      // In Surface mode keep triangles across depth steps so sharp changes
      // become visible walls, not black holes; ordinary depth jumps stay
      // connected.
      indices[q++] = a;
      indices[q++] = c;
      indices[q++] = b;
      indices[q++] = b;
      indices[q++] = c;
      indices[q++] = d;
    }
  meshIndexCount = q;
  updateMeshBoundsFromVerts(verts);
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
  gl.bufferData(
      gl.ELEMENT_ARRAY_BUFFER, indices.subarray(0, q), gl.STATIC_DRAW);
}

function rebuildDepthPreview() {
  depthPreview = null;
  depthPreviewW = 0;
  depthPreviewH = 0;
  cleanDepthMap = null;
  if (!processedDepth || !mapW || !mapH) return;

  cleanDepthMap = processedDepth.slice();
  depthPreviewW = mapW;
  depthPreviewH = mapH;
  const imageData = depthCtx.createImageData(mapW, mapH);
  for (let i = 0, offset = 0; i < cleanDepthMap.length; i++) {
    const gray = Math.max(0, Math.min(255, Math.round(cleanDepthMap[i] * 255)));
    imageData.data[offset++] = gray;
    imageData.data[offset++] = gray;
    imageData.data[offset++] = gray;
    imageData.data[offset++] = 255;
  }
  depthPreview = imageData;
}
function renderDepthMap() {
  if (!depthPreview || !depthPreviewW || !depthPreviewH) {
    depthCtx.clearRect(0, 0, depthCv.width, depthCv.height);
    return;
  }
  const tmp = document.createElement('canvas');
  tmp.width = depthPreviewW;
  tmp.height = depthPreviewH;
  tmp.getContext('2d').putImageData(depthPreview, 0, 0);
  depthCtx.clearRect(0, 0, depthCv.width, depthCv.height);
  depthCtx.fillStyle = '#090b0f';
  depthCtx.fillRect(0, 0, depthCv.width, depthCv.height);
  const scale =
      Math.min(depthCv.width / depthPreviewW, depthCv.height / depthPreviewH) *
      .92;
  const dw = Math.max(1, depthPreviewW * scale),
        dh = Math.max(1, depthPreviewH * scale);
  const dx = (depthCv.width - dw) / 2, dy = (depthCv.height - dh) / 2;
  depthCtx.imageSmoothingEnabled = true;
  depthCtx.drawImage(tmp, dx, dy, dw, dh);
}

function render() {
  if (currentViewMode() === 'depthmap') {
    renderDepthMap();
    return;
  }
  if (!processedDepth || !meshIndexCount) return;
  const lightVP = computeLightVP();
  renderShadowMap(lightVP);
  renderGBuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, cv.width, cv.height);
  gl.disable(gl.DEPTH_TEST);
  gl.clearColor(.045, .05, .06, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.useProgram(lightProg);
  gl.bindVertexArray(quadVao);
  const lp = lightParams(), ld = lightDirCameraSpace();
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, gAlbedoTex);
  gl.uniform1i(LU.albedo, 0);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, gNormalTex);
  gl.uniform1i(LU.normal, 1);
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, gPositionTex);
  gl.uniform1i(LU.position, 2);
  gl.activeTexture(gl.TEXTURE3);
  gl.bindTexture(gl.TEXTURE_2D, shadowTex);
  gl.uniform1i(LU.shadowTex, 3);
  gl.uniform2f(LU.shadowTexel, 1 / shadowSize, 1 / shadowSize);
  gl.uniformMatrix4fv(LU.lightVP, false, lightVP);
  gl.uniform3f(LU.lightDir, ld.x, ld.y, ld.z);
  gl.uniform3f(LU.cameraPos, 0, 0, 3.2);
  gl.uniform1f(LU.ambient, lp.ambient);
  gl.uniform1f(LU.diffuse, lp.diffuse);
  gl.uniform1f(LU.fill, lp.fill);
  gl.uniform1f(LU.shadowStrength, lp.shadowStrength);
  gl.uniform1f(LU.selfShade, lp.selfShade);
  gl.uniform1f(LU.specular, lp.specular);
  gl.uniform1f(LU.shininess, lp.shininess);
  gl.uniform1f(LU.biasBase, lp.biasBase);
  gl.uniform1f(LU.biasSlope, lp.biasSlope);
  gl.uniform1f(LU.pcfRadius, lp.pcfRadius);
  gl.uniform1f(LU.shadowHardness, lp.shadowHardness);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  gl.bindVertexArray(null);
  gl.enable(gl.DEPTH_TEST);
}

// UI events
syncPair(depthR, depthN);
syncPair(layersR, layersN);
syncPair(periodR, periodN);
setPair(depthR, depthN, 35);
updateAutoButton();
$('#reset').onclick = resetCamera;
autoBtn.onclick = () => {
  autoRotate = !autoRotate;
  updateAutoButton();
  if (autoRotate) {
    autoBaseYaw = yaw;
    autoBasePitch = pitch;
    autoTime = 0;
    autoPauseUntil = 0;
  }
  sched();
};
$('#analyze').onclick = () => {
  if (!loadedDepthMap)
    analyze().catch(err => {
      console.error(err);
      hideProgress();
    });
};
fi.onchange = () => fi.files[0] && loadImage(fi.files[0]);
textureFi.onchange = () =>
    textureFi.files[0] && loadTexture(textureFi.files[0]);
['dragenter', 'dragover'].forEach(
    n => addEventListener(n, e => e.preventDefault()));
addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (!f || !f.type.startsWith('image/')) return;
  const textureTarget =
      e.target instanceof Element && e.target.closest('#textureButton');
  if (textureTarget || e.shiftKey)
    loadTexture(f);
  else
    loadImage(f);
});
periodR.onchange = () => {
  if (loadedDepthMap || !rgbaMap || !mapW || !mapH) return;
  recoverDepth(mapW, mapH, +periodR.value).catch(err => {
    console.error(err);
    hideProgress();
  });
};
depthR.oninput = sched;
layersR.oninput = reprocess;
document.querySelectorAll('input[name="shape"]')
    .forEach(el => el.onchange = () => {
      rebuildDepthPreview();
      if (currentViewMode() !== 'depthmap') buildMesh();
      sched();
    });
document.querySelectorAll('input[name="viewmode"]')
    .forEach(el => el.onchange = () => {
      updateLayerUi();
      reprocess();
      sched();
    });

// pointer controls
cv.addEventListener('pointerdown', e => {
  disableAutoRotate();
  drag = true;
  pauseAuto();
  lastX = e.clientX;
  lastY = e.clientY;
  cv.setPointerCapture(e.pointerId);
});
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  yaw += (e.clientX - lastX) * .008;
  pitch = Math.max(-1.35, Math.min(1.35, pitch + (e.clientY - lastY) * .008));
  lastX = e.clientX;
  lastY = e.clientY;
  sched();
});
['pointerup', 'pointercancel'].forEach(n => cv.addEventListener(n, () => {
  drag = false;
  autoBaseYaw = yaw;
  autoBasePitch = pitch;
  pauseAuto();
}));
cv.addEventListener('wheel', e => {
  e.preventDefault();
  pauseAuto();
  zoom = Math.max(.35, Math.min(4, zoom * Math.exp(-e.deltaY * .001)));
  sched();
}, {passive: false});
cv.addEventListener('touchstart', e => {
  disableAutoRotate();
  if (e.touches.length === 2)
    pinch = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY);
}, {passive: true});
cv.addEventListener('touchmove', e => {
  if (e.touches.length === 2) {
    const p = Math.hypot(
        e.touches[0].clientX - e.touches[1].clientX,
        e.touches[0].clientY - e.touches[1].clientY);
    if (pinch) zoom = Math.max(.35, Math.min(4, zoom * p / pinch));
    pinch = p;
    pauseAuto();
    sched();
  }
}, {passive: true});
cv.addEventListener('touchend', () => {
  pinch = 0;
  autoBaseYaw = yaw;
  autoBasePitch = pitch;
  pauseAuto();
});
cv.ondblclick = resetCamera;

resize();
loadRandomStartupImage();
