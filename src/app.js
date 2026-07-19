import {G_FRAGMENT_SHADER, G_VERTEX_SHADER, LIGHT_FRAGMENT_SHADER, LIGHT_VERTEX_SHADER, SHADOW_FRAGMENT_SHADER, SHADOW_VERTEX_SHADER} from './shaders.js';
import {computeCostVolumeGpu, filterDepthMapGpu} from './depth-gpu.js';

'use strict';
const $ = s => document.querySelector(s);
const cv = $('#view'), depthCv = $('#depthView'),
      depthCtx = depthCv.getContext('2d'), fi = $('#file'),
      textureFi = $('#textureFile'), drop = $('#drop'), thumb = $('#thumb'),
      stage = cv.parentElement,
      progressOverlay = $('#progressOverlay'),
      progressText = $('#progressText'), progressBar = $('#progressBar');
const downloadDepthBtn = $('#downloadDepth');
const depthR = $('#depth'), depthN = $('#depthNum');
const layersR = $('#layers'), layersN = $('#layersNum');
const periodR = $('#period'), periodN = $('#periodNum');
const autoBtn = $('#auto');
const ANALYSIS_SIZE = 512;
const ANALYSIS_SIGMA = 0.24;
const MESH_STEP_PX = 2;
const SPECKLE_SIZE = 200;
const FILL_RADIUS = 8;
const MEDIAN_RADIUS = 8;
const gl = cv.getContext(
    'webgl2',
    {alpha: false, antialias: true, powerPreference: 'high-performance'});
if (!gl) {
  alert('Нужен WebGL2');
  throw new Error('WebGL2 unavailable');
}

let img = null, loadedDepthMap = false, textureLoaded = false;
let mapW = 0, mapH = 0, cropX = 0, bgDepth = .5;
let grayMap = null, confMap = null, rawDepth = null, processedDepth = null,
    cleanDepthMap = null, depthPreview = null, depthPreviewW = 0,
    depthPreviewH = 0, depthCoverage = null, cachedGpuFiltered = null;
let yaw = -.22, pitch = -.16, zoom = 1, drag = false;
let gestureMode = 'none', pinchDistance = 0, gesturePointerId = null,
    lastGestureX = 0, lastGestureY = 0, lastGestureTime = 0,
    inertiaYaw = 0, inertiaPitch = 0;
const activePointers = new Map();
let autoRotate = false, autoPauseUntil = 0, lastFrame = performance.now(),
    dirty = true, autoTime = 0, autoBaseYaw = -.22, autoBasePitch = -.16;
let voidPatternTime = 0, lastVoidPatternFrame = 0;
let meshIndexCount = 0, processTimer = 0, disparityWorker = null,
    analysisJob = 0;
let surfaceGridKey = '', surfaceUsesDepthTexture = false;
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
function disparityWorkerMain() {
  'use strict';
  function postProgress(id, value, label) {
    postMessage({type: 'progress', id, value, label});
  }
  function percentile(arr, p) {
    if (!arr.length) return 0;
    const i =
        Math.max(0, Math.min(arr.length - 1, Math.floor((arr.length - 1) * p)));
    return arr[i];
  }
  function smoothstep(a, b, x) {
    const t = Math.max(0, Math.min(1, (x - a) / Math.max(1e-6, b - a)));
    return t * t * (3 - 2 * t);
  }
  function sort9(a) {
    for (let i = 1; i < 9; i++) {
      const v = a[i];
      let j = i - 1;
      while (j >= 0 && a[j] > v) {
        a[j + 1] = a[j];
        j--;
      }
      a[j + 1] = v;
    }
  }
  function sortN(a, n) {
    for (let i = 1; i < n; i++) {
      const v = a[i];
      let j = i - 1;
      while (j >= 0 && a[j] > v) {
        a[j + 1] = a[j];
        j--;
      }
      a[j + 1] = v;
    }
  }
  function popcount32(v) {
    v -= (v >>> 1) & 0x55555555;
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
  }
  function robustDepthClean(base, conf, w, h, bg, cropX, id) {
    let cur = base.slice();
    let out = new Float32Array(base.length);
    const vals = new Float32Array(9), devs = new Float32Array(9);
    for (let pass = 0; pass < 2; pass++) {
      out.set(cur);
      for (let y = 1; y < h - 1; y++) {
        for (let x = Math.max(cropX + 1, 1); x < w - 1; x++) {
          const i = y * w + x;
          let k = 0;
          for (let yy = -1; yy <= 1; yy++)
            for (let xx = -1; xx <= 1; xx++)
              vals[k++] = cur[(y + yy) * w + x + xx];
          sort9(vals);
          const med = vals[4];
          for (k = 0; k < 9; k++) devs[k] = Math.abs(vals[k] - med);
          sort9(devs);
          const mad = devs[4];
          const center = cur[i], dev = Math.abs(center - med),
                cf = Math.max(0, Math.min(1, conf[i] || 0));
          const th = .010 + 2.8 * mad + (1 - cf) * .025;
          let v = center;
          if (dev > th) v = med * .80 + center * .20;
          const delta = v - bg, sep = Math.abs(delta);
          const keep = smoothstep(.028, .11, sep) *
              (.45 + .55 * smoothstep(.015, .08, cf));
          out[i] = Math.max(0, Math.min(1, bg + delta * (.14 + .86 * keep)));
        }
        if ((y & 63) === 0)
          postProgress(
              id, .88 + pass * .045 + (y / h) * .04, 'Очистка глубины…');
      }
      const swap = cur;
      cur = out;
      out = swap;
    }
    return cur;
  }
  onmessage = e => {
    const startedAt = performance.now();
    const {id, grayBuffer, packedCostsBuffer, packedDisparityBuffer,
           reliabilityBuffer, wordsPerPixel,
           w, h, p} = e.data;
    const gray = new Float32Array(grayBuffer);
    const maxDisp = Math.max(4, Math.min(Math.floor(p * .34), 42));
    const minDisp = -Math.max(2, Math.floor(p * .08));
    const dispCount = maxDisp - minDisp + 1;
    // A single-image stereogram has no unique depth in its first repeat.
    // Start matching at the first position where every disparity candidate
    // has a valid previous repeat, but crop the result by exactly one period.
    const xStart = p - minDisp + 2, xEnd = w - 3,
          validW = Math.max(1, xEnd - xStart);
    const pixels = validW * h;
    let dispIndex, reliability, usedGpuIcm = false, usedGpuSgm = false;
    if (packedDisparityBuffer && reliabilityBuffer) {
      const packed = new Uint32Array(packedDisparityBuffer),
            gpuReliability = new Float32Array(reliabilityBuffer);
      if (packed.length === Math.ceil(pixels / 2) &&
          gpuReliability.length === pixels) {
        dispIndex = new Int16Array(pixels);
        for (let i = 0; i < pixels; i++)
          dispIndex[i] = (packed[i >> 1] >>> ((i & 1) * 16)) & 65535;
        reliability = gpuReliability;
        usedGpuIcm = true;
        usedGpuSgm = true;
        postProgress(id, .78, 'Распаковка WebGPU disparity…');
      }
    }
    let costs, usedGpuCosts = false;
    if (!dispIndex && packedCostsBuffer && wordsPerPixel > 0) {
      const packed = new Uint32Array(packedCostsBuffer);
      costs = new Uint8Array(pixels * dispCount);
      for (let pi = 0; pi < pixels; pi++) {
        const packedOffset = pi * wordsPerPixel,
              costOffset = pi * dispCount;
        for (let di = 0; di < dispCount; di++)
          costs[costOffset + di] =
              (packed[packedOffset + (di >> 2)] >>> ((di & 3) * 8)) & 255;
      }
      function censusAt(x, y) {
        if (x < 2 || x >= w - 2 || y < 2 || y >= h - 2) return 0;
        const center = gray[y * w + x];
        let bits = 0, bit = 0;
        for (let yy = -2; yy <= 2; yy++)
          for (let xx = -2; xx <= 2; xx++) {
            if (xx === 0 && yy === 0) continue;
            if (gray[(y + yy) * w + x + xx] < center) bits |= (1 << bit);
            bit++;
          }
        return bits >>> 0;
      }
      let valid = packed.length === pixels * wordsPerPixel;
      const stepY = Math.max(1, Math.floor(Math.max(1, h - 4) / 12)),
            stepX = Math.max(1, Math.floor(validW / 12));
      for (let y = 2; valid && y < h - 2; y += stepY)
        for (let vx = 0; valid && vx < validW; vx += stepX) {
          const x = xStart + vx, pi = y * validW + vx,
                a = censusAt(x, y);
          for (let di = 0; di < dispCount; di++) {
            const d = minDisp + di, qx = x - (p - d),
                  censusCost = popcount32(a ^ censusAt(qx, y)),
                  intensityCost = Math.min(
                      8, Math.abs(gray[y * w + x] - gray[y * w + qx]) / 12),
                  expected =
                      Math.min(31, Math.round(censusCost + intensityCost));
            if (costs[pi * dispCount + di] !== expected) {
              valid = false;
              break;
            }
          }
        }
      if (valid) {
        usedGpuCosts = true;
        postProgress(id, .28, 'Распаковка WebGPU cost volume…');
      } else {
        costs = null;
        postProgress(id, .10, 'Проверка WebGPU не пройдена, CPU fallback…');
      }
    }
    if (!dispIndex && !costs) {
      const census = new Uint32Array(w * h);
      // 5x5 Census descriptor: local ordering is much less sensitive to the
      // source texture brightness.
      for (let y = 2; y < h - 2; y++)
        for (let x = 2; x < w - 2; x++) {
          const center = gray[y * w + x];
          let bits = 0, bit = 0;
          for (let yy = -2; yy <= 2; yy++)
            for (let xx = -2; xx <= 2; xx++) {
              if (xx === 0 && yy === 0) continue;
              if (gray[(y + yy) * w + x + xx] < center) bits |= (1 << bit);
              bit++;
            }
          census[y * w + x] = bits >>> 0;
        }
      costs = new Uint8Array(pixels * dispCount);
      for (let y = 0; y < h; y++)
        for (let vx = 0; vx < validW; vx++) {
          const x = xStart + vx, pi = y * validW + vx, a = census[y * w + x];
          for (let di = 0; di < dispCount; di++) {
            const d = minDisp + di, qx = x - (p - d), b = census[y * w + qx];
            const censusCost = popcount32(a ^ b);
            const intensityCost =
                Math.min(8, Math.abs(gray[y * w + x] - gray[y * w + qx]) / 12);
            costs[pi * dispCount + di] =
                Math.min(31, Math.round(censusCost + intensityCost));
          }
          if ((y & 63) === 0 && vx === 0)
            postProgress(id, .10 + .18 * y / h, 'Census cost volume…');
        }
    }
    if (!dispIndex) {
    let aggregate;
    if (!aggregate) {
      aggregate = new Uint16Array(costs.length);
      let prev = new Float32Array(dispCount), cur = new Float32Array(dispCount);
      function addPath(startX, startY, dx, dy) {
        let vx = startX, y = startY, first = true;
        while (vx >= 0 && vx < validW && y >= 0 && y < h) {
          const pi = y * validW + vx, off = pi * dispCount;
          if (first) {
            for (let di = 0; di < dispCount; di++) {
              cur[di] = costs[off + di];
              aggregate[off + di] += cur[di];
            }
            first = false;
          } else {
            let prevMin = prev[0];
            for (let di = 1; di < dispCount; di++)
              if (prev[di] < prevMin) prevMin = prev[di];
            const px = xStart + vx - dx, py = y - dy;
            const edge =
                Math.abs(gray[y * w + xStart + vx] - gray[py * w + px]);
            const p1 = 2.2, p2 = Math.max(5, 18 - edge * .20);
            for (let di = 0; di < dispCount; di++) {
              let v = prev[di];
              if (di > 0) v = Math.min(v, prev[di - 1] + p1);
              if (di + 1 < dispCount) v = Math.min(v, prev[di + 1] + p1);
              v = Math.min(v, prevMin + p2);
              cur[di] = costs[off + di] + v - prevMin;
              aggregate[off + di] =
                  Math.min(65535, aggregate[off + di] + Math.round(cur[di]));
            }
          }
          const swap = prev;
          prev = cur;
          cur = swap;
          vx += dx;
          y += dy;
        }
      }
      // Four SGM directions. Each path favours piecewise-smooth disparity but
      // relaxes at image edges.
      for (let y = 0; y < h; y++) {
        addPath(0, y, 1, 0);
        addPath(validW - 1, y, -1, 0);
      }
      postProgress(id, .48, 'SGM по горизонтали…');
      for (let vx = 0; vx < validW; vx++) {
        addPath(vx, 0, 0, 1);
        addPath(vx, h - 1, 0, -1);
      }
      postProgress(id, .70, 'SGM по вертикали…');
    }
    dispIndex = new Int16Array(pixels);
    reliability = new Float32Array(pixels);
    for (let y = 0; y < h; y++)
      for (let vx = 0; vx < validW; vx++) {
        const pi = y * validW + vx, off = pi * dispCount;
        let bestD = 0, best = Infinity, second = Infinity;
        for (let di = 0; di < dispCount; di++) {
          const v = aggregate[off + di];
          if (v < best) {
            best = v;
            bestD = di;
          }
        }
        for (let di = 0; di < dispCount; di++)
          if (Math.abs(di - bestD) > 1 && aggregate[off + di] < second)
            second = aggregate[off + di];
        dispIndex[pi] = bestD;
        reliability[pi] =
            Math.max(0, Math.min(1, (second - best) / Math.max(8, second)));
      }
    // Iterated conditional modes: re-check nearby disparities using the
    // original SGM costs plus support from reliable neighbours. Image edges
    // weaken support.
    let nextDisp = new Int16Array(pixels),
        nextReliability = new Float32Array(pixels);
    for (let pass = 0; pass < 3; pass++) {
      for (let y = 0; y < h; y++)
        for (let vx = 0; vx < validW; vx++) {
          const pi = y * validW + vx, center = dispIndex[pi],
                range = reliability[pi] < .045 ? 3 : 2;
          let bestD = center, bestEnergy = Infinity, secondEnergy = Infinity;
          for (let di = Math.max(0, center - range);
               di <= Math.min(dispCount - 1, center + range); di++) {
            let energy = aggregate[pi * dispCount + di];
            for (let k = 0; k < 4; k++) {
              const nx = vx +
                  (k === 0     ? -1 :
                       k === 1 ? 1 :
                                 0),
                    ny = y +
                  (k === 2     ? -1 :
                       k === 3 ? 1 :
                                 0);
              if (nx < 0 || nx >= validW || ny < 0 || ny >= h) continue;
              const ni = ny * validW + nx, imageA = y * w + xStart + vx,
                    imageB = ny * w + xStart + nx;
              const edgeWeight =
                  Math.exp(-Math.abs(gray[imageA] - gray[imageB]) / 20);
              const neighbourWeight =
                  (.15 + reliability[ni] * .85) * edgeWeight;
              energy += 4.5 * neighbourWeight *
                  Math.min(4, Math.abs(di - dispIndex[ni]));
            }
            if (energy < bestEnergy) {
              secondEnergy = bestEnergy;
              bestEnergy = energy;
              bestD = di;
            } else if (energy < secondEnergy)
              secondEnergy = energy;
          }
          nextDisp[pi] = bestD;
          const localConfidence = Math.max(
              0,
              Math.min(
                  1, (secondEnergy - bestEnergy) / Math.max(8, secondEnergy)));
          nextReliability[pi] = reliability[pi] * .6 + localConfidence * .4;
        }
      let swap = dispIndex;
      dispIndex = nextDisp;
      nextDisp = swap;
      swap = reliability;
      reliability = nextReliability;
      nextReliability = swap;
      postProgress(id, .71 + pass * .025, 'Проверка соседних смещений…');
    }
    }
    const raw = new Float32Array(w * h), conf = new Float32Array(w * h);
    for (let y = 0; y < h; y++)
      for (let vx = 0; vx < validW; vx++) {
        const pi = y * validW + vx, i = y * w + xStart + vx;
        raw[i] = minDisp + dispIndex[pi];
        conf[i] = reliability[pi];
      }
    for (let y = 0; y < h; y++) {
      const src = y * w + xStart;
      for (let x = 0; x < xStart; x++) {
        raw[y * w + x] = raw[src];
        conf[y * w + x] = conf[src] * .5;
      }
      for (let x = xEnd; x < w; x++) {
        raw[y * w + x] = raw[y * w + xEnd - 1];
        conf[y * w + x] = conf[y * w + xEnd - 1] * .5;
      }
    }
    let t = raw, c = conf;
    let out = new Float32Array(t.length), outC = new Float32Array(c.length);
    const vals = new Float32Array(25);
    for (let pass = 0; pass < 2; pass++) {
      out.set(t);
      outC.set(c);
      const radius = pass === 0 ? 2 : 1;
      for (let y = radius; y < h - radius; y++) {
        for (let x = Math.max(xStart, radius); x < w - radius; x++) {
          let k = 0;
          for (let yy = -radius; yy <= radius; yy++)
            for (let xx = -radius; xx <= radius; xx++)
              vals[k++] = t[(y + yy) * w + x + xx];
          sortN(vals, k);
          const median = vals[k >> 1], i = y * w + x;
          let weight = 0, sum = 0, support = 0;
          for (let yy = -radius; yy <= radius; yy++)
            for (let xx = -radius; xx <= radius; xx++) {
              const j = (y + yy) * w + x + xx, delta = Math.abs(t[j] - median);
              if (delta <= 2.25) {
                const ww = .15 + Math.min(1, c[j] * 5);
                sum += t[j] * ww;
                weight += ww;
                support += Math.min(1, c[j] * 5);
              }
            }
          const consensus = weight > 0 ? sum / weight : median;
          const ownConfidence = Math.max(0, Math.min(1, c[i] * 4));
          const blend = .82 - ownConfidence * .62;
          // Do not smear a strong, well-supported depth edge into its
          // neighbour.
          out[i] = ownConfidence > .65 && Math.abs(t[i] - median) > 2.5 ?
              t[i] :
              t[i] * (1 - blend) + consensus * blend;
          outC[i] = Math.min(1, c[i] * .55 + (support / k) * .45);
        }
        if ((y & 63) === 0)
          postProgress(
              id, .74 + pass * .045 + (y / h) * .04,
              'Заполнение ненадёжных областей…');
      }
      let swap = t;
      t = out;
      out = swap;
      swap = c;
      c = outC;
      outC = swap;
    }
    const samples = [];
    for (let y = 4; y < h - 4; y += 3)
      for (let x = 4; x < w - 4; x += 3) samples.push(t[y * w + x]);
    samples.sort((a, b) => a - b);
    const lo = percentile(samples, .05), hi = percentile(samples, .95),
          inv = 1 / Math.max(1e-6, hi - lo);
    let rawDepth = new Float32Array(w * h), confMap = new Float32Array(w * h);
    const bins = new Uint32Array(64);
    for (let i = 0; i < t.length; i++) {
      const v = Math.max(0, Math.min(1, (t[i] - lo) * inv));
      rawDepth[i] = v;
      confMap[i] = Math.max(0, Math.min(1, c[i] * 4));
      bins[Math.max(0, Math.min(63, (v * 63) | 0))]++;
    }
    let bi = 0;
    for (let i = 1; i < 64; i++)
      if (bins[i] > bins[bi]) bi = i;
    const bgDepth = bi / 63, cropX = Math.max(0, Math.min(w - 1, p));
    rawDepth = robustDepthClean(rawDepth, confMap, w, h, bgDepth, cropX, id);
    postMessage(
        {
          type: 'done',
          id,
          bgDepth,
          cropX,
          usedGpuCosts,
          usedGpuSgm,
          usedGpuIcm,
          processingMs: performance.now() - startedAt,
          rawDepth: rawDepth.buffer,
          confMap: confMap.buffer
        },
        [rawDepth.buffer, confMap.buffer]);
  };
}
function createDisparityWorker() {
  const src = '(' + disparityWorkerMain.toString() + ')()';
  return new Worker(
      URL.createObjectURL(new Blob([src], {type: 'text/javascript'})));
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
function smoothstep(a, b, x) {
  const t = Math.max(0, Math.min(1, (x - a) / Math.max(1e-6, b - a)));
  return t * t * (3 - 2 * t);
}
function percentile(arr, p) {
  if (!arr.length) return 0;
  const i =
      Math.max(0, Math.min(arr.length - 1, Math.floor((arr.length - 1) * p)));
  return arr[i];
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
  inertiaYaw = 0;
  inertiaPitch = 0;
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
    shadowStrength: 1.0,
    selfShade: 0.5,
    specular: 0.1,
    shininess: 5.0,
    biasBase: 0.0008,
    biasSlope: 0.003,
    pcfRadius: 2.3,
    shadowHardness: 0.12
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
const quadVao = gl.createVertexArray(), quadVbo = gl.createBuffer();
gl.bindVertexArray(quadVao);
gl.bindBuffer(gl.ARRAY_BUFFER, quadVbo);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
  -1, -1,  1, -1, -1,  1,
  -1,  1,  1, -1,  1,  1,
]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(0);
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
gl.bindVertexArray(null);
gl.bindBuffer(gl.ARRAY_BUFFER, null);

const GU = {
  yaw: gl.getUniformLocation(gProg, 'uYaw'),
  pitch: gl.getUniformLocation(gProg, 'uPitch'),
  depth: gl.getUniformLocation(gProg, 'uDepthScale'),
  sign: gl.getUniformLocation(gProg, 'uSign'),
  zoom: gl.getUniformLocation(gProg, 'uZoom'),
  aspect: gl.getUniformLocation(gProg, 'uAspect'),
  sourceTex: gl.getUniformLocation(gProg, 'uSourceTex'),
  depthTex: gl.getUniformLocation(gProg, 'uDepthTex'),
  useDepthTex: gl.getUniformLocation(gProg, 'uUseDepthTex'),
  sourceCrop: gl.getUniformLocation(gProg, 'uSourceCrop'),
  sourceAspect: gl.getUniformLocation(gProg, 'uSourceAspect'),
  patternTime: gl.getUniformLocation(gProg, 'uPatternTime'),
  backPass: gl.getUniformLocation(gProg, 'uBackPass')
};
const SU = {
  yaw: gl.getUniformLocation(shadowProg, 'uYaw'),
  pitch: gl.getUniformLocation(shadowProg, 'uPitch'),
  depth: gl.getUniformLocation(shadowProg, 'uDepthScale'),
  sign: gl.getUniformLocation(shadowProg, 'uSign'),
  lightVP: gl.getUniformLocation(shadowProg, 'uLightVP'),
  depthTex: gl.getUniformLocation(shadowProg, 'uDepthTex'),
  useDepthTex: gl.getUniformLocation(shadowProg, 'uUseDepthTex'),
  sourceCrop: gl.getUniformLocation(shadowProg, 'uSourceCrop'),
  sourceAspect: gl.getUniformLocation(shadowProg, 'uSourceAspect'),
  patternTime: gl.getUniformLocation(shadowProg, 'uPatternTime'),
  backPass: gl.getUniformLocation(shadowProg, 'uBackPass')
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
const depthSurfaceTex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, depthSurfaceTex);
const floatLinearFiltering = gl.getExtension('OES_texture_float_linear');
const depthTextureFilter = floatLinearFiltering ? gl.LINEAR : gl.NEAREST;
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, depthTextureFilter);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, depthTextureFilter);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(
    gl.TEXTURE_2D, 0, gl.R16F, 1, 1, 0, gl.RED, gl.FLOAT,
    new Float32Array([0]));
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
  const requestedScale = (+depthR.value || 0) / 100;
  const scale = currentViewMode() === 'layers' ?
      Math.max(0.01, requestedScale) : requestedScale;
  const sign = currentShape();
  const b = meshSourceBounds;
  let minDepth, maxDepth;
  if (surfaceUsesDepthTexture) {
    // The texture-driven surface stays in a positive depth range for both
    // shapes. Extreme black/white values can receive up to 10/255 of
    // procedural relief in the vertex shaders.
    minDepth = 0;
    maxDepth = 1 + 10 / 255;
  } else if (sign < 0) {
    // CPU geometry uses the same 1-depth transform as both vertex shaders.
    minDepth = 1 - b.maxD;
    maxDepth = 1 - b.minD;
  } else {
    minDepth = b.minD;
    maxDepth = b.maxD;
  }
  const corners = [];
  for (const x of [b.minX, b.maxX])
    for (const y of [b.minY, b.maxY])
      for (const d0 of [minDepth, maxDepth]) {
        corners.push(rotatePoint({x, y, z: d0 * scale}, yaw, pitch));
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
  gl.useProgram(shadowProg);
  gl.bindVertexArray(vao);
  gl.uniform1f(SU.yaw, yaw);
  gl.uniform1f(SU.pitch, pitch);
  gl.uniform1f(SU.depth, (+depthR.value || 0) / 100);
  gl.uniform1f(SU.sign, currentShape());
  gl.uniform1f(SU.useDepthTex, surfaceUsesDepthTexture ? 1 : 0);
  gl.uniform1f(SU.sourceCrop, cropX / Math.max(1, mapW));
  gl.uniform1f(SU.sourceAspect, mapH / Math.max(1, mapW - cropX));
  gl.uniform1f(SU.patternTime, voidPatternTime);
  gl.uniform1f(SU.backPass, 0);
  gl.activeTexture(gl.TEXTURE5);
  gl.bindTexture(gl.TEXTURE_2D, depthSurfaceTex);
  gl.uniform1i(SU.depthTex, 5);
  gl.uniformMatrix4fv(SU.lightVP, false, lightVP);
  gl.drawElements(gl.TRIANGLES, meshIndexCount, gl.UNSIGNED_INT, 0);
  if (!surfaceUsesDepthTexture && currentViewMode() === 'layers') {
    gl.uniform1f(SU.backPass, 1);
    gl.drawElements(gl.TRIANGLES, meshIndexCount, gl.UNSIGNED_INT, 0);
  }
  gl.bindVertexArray(null);
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
  gl.uniform1f(GU.useDepthTex, surfaceUsesDepthTexture ? 1 : 0);
  gl.uniform1f(GU.patternTime, voidPatternTime);
  gl.uniform1f(GU.backPass, 0);
  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_2D, sourceTex);
  gl.uniform1i(GU.sourceTex, 4);
  gl.activeTexture(gl.TEXTURE5);
  gl.bindTexture(gl.TEXTURE_2D, depthSurfaceTex);
  gl.uniform1i(GU.depthTex, 5);
  gl.drawElements(gl.TRIANGLES, meshIndexCount, gl.UNSIGNED_INT, 0);
  if (!surfaceUsesDepthTexture && currentViewMode() === 'layers') {
    gl.uniform1f(GU.backPass, 1);
    gl.drawElements(gl.TRIANGLES, meshIndexCount, gl.UNSIGNED_INT, 0);
  }
  gl.bindVertexArray(null);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}
function resize() {
  const dpr = Math.min(devicePixelRatio || 1, 2),
        // The WebGL canvas is hidden in depth-map mode and therefore has a
        // zero-sized client rect. The stage remains measurable in every mode.
        r = stage.getBoundingClientRect();
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
  if (!drag && !autoRotate &&
      (Math.abs(inertiaYaw) > .000002 || Math.abs(inertiaPitch) > .000002)) {
    yaw += inertiaYaw * dt;
    pitch = Math.max(-1.35, Math.min(1.35, pitch + inertiaPitch * dt));
    const damping = Math.exp(-dt / 420);
    inertiaYaw *= damping;
    inertiaPitch *= damping;
    if (Math.abs(inertiaYaw) < .000002) inertiaYaw = 0;
    if (Math.abs(inertiaPitch) < .000002) inertiaPitch = 0;
    autoBaseYaw = yaw;
    autoBasePitch = pitch;
    dirty = true;
  }
  if (processedDepth && autoRotate && !drag && now > autoPauseUntil) {
    autoTime += dt * 0.001;
    yaw = autoBaseYaw + Math.sin(autoTime * 0.72) * 0.55;
    pitch = autoBasePitch + Math.sin(autoTime * 0.41) * 0.14;
    dirty = true;
  }
  if (processedDepth && surfaceUsesDepthTexture &&
      now - lastVoidPatternFrame >= 1000 / 24) {
    lastVoidPatternFrame = now;
    voidPatternTime = now * .001;
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
  downloadDepthBtn.hidden = !depthMode;
  downloadDepthBtn.disabled = !depthPreview;
  depthR.parentElement.classList.toggle('disabled', depthMode);
}
updateLayerUi();

function getScaledPixels(image) {
  const scale = Math.min(
      1, ANALYSIS_SIZE / Math.max(image.naturalWidth, image.naturalHeight));
  const w = Math.max(1, Math.round(image.naturalWidth * scale));
  const h = Math.max(1, Math.round(image.naturalHeight * scale));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const x = c.getContext('2d', {willReadFrequently: true});
  x.drawImage(image, 0, 0, w, h);
  return {d: x.getImageData(0, 0, w, h).data, w, h};
}
function estimatePeriod(gray, w, h) {
  const a = Math.max(8, Math.floor(w * .02)),
        b = Math.max(a + 2, Math.min(w - 1, Math.floor(w * .45)));
  let best = a, bestErr = 1e30;
  const sy = Math.max(2, Math.floor((h * .6) / 60));
  for (let p = a; p <= b; p += 2) {
    let err = 0, n = 0;
    for (let y = Math.floor(h * .2); y < h * .8; y += sy) {
      const r = y * w;
      for (let x = p + 6; x < w - 6; x += 4) {
        err += Math.abs(gray[r + x] - gray[r + x - p]);
        n++;
      }
    }
    err /= Math.max(1, n);
    if (err < bestErr) {
      bestErr = err;
      best = p;
    }
  }
  let refined = best;
  for (let p = Math.max(1, best - 3); p <= Math.min(b, best + 3); p++) {
    let err = 0, n = 0;
    for (let y = Math.floor(h * .2); y < h * .8; y += sy) {
      const r = y * w;
      for (let x = p + 6; x < w - 6; x += 3) {
        err += Math.abs(gray[r + x] - gray[r + x - p]);
        n++;
      }
    }
    err /= Math.max(1, n);
    if (err < bestErr) {
      bestErr = err;
      refined = p;
    }
  }
  // Global correlation is biased toward the average foreground spacing. In an
  // autostereogram disparity shortens the repeat, so the base/background period
  // is estimated from the upper robust percentile of local block matches.
  const localShifts = [], searchLo = Math.max(a, refined - 5),
        searchHi =
            Math.min(b, refined + Math.max(10, Math.round(refined * .16)));
  const gridY = Math.max(7, Math.round(h / 26)),
        gridX = Math.max(9, Math.round(w / 30));
  for (let cy = 4; cy < h - 4; cy += gridY)
    for (let cx = searchHi + 4; cx < w - 4; cx += gridX) {
      let localBest = refined, localCost = Infinity, localSecond = Infinity;
      for (let shift = searchLo; shift <= searchHi; shift++) {
        let cost = 0, count = 0;
        for (let yy = -3; yy <= 3; yy += 2)
          for (let xx = -3; xx <= 3; xx++) {
            const i = (cy + yy) * w + cx + xx, j = i - shift;
            const lum = Math.min(50, Math.abs(gray[i] - gray[j]));
            const gradA = gray[i + 1] - gray[i - 1],
                  gradB = gray[j + 1] - gray[j - 1];
            cost += lum * .4 + Math.min(60, Math.abs(gradA - gradB)) * .6;
            count++;
          }
        cost /= count;
        if (cost < localCost) {
          localSecond = localCost;
          localCost = cost;
          localBest = shift;
        } else if (cost < localSecond)
          localSecond = cost;
      }
      const uniqueness = (localSecond - localCost) / Math.max(1, localSecond);
      if (uniqueness > .012) localShifts.push(localBest);
    }
  if (localShifts.length >= 8) {
    localShifts.sort((x, y) => x - y);
    return localShifts[Math.floor((localShifts.length - 1) * .88)];
  }
  return refined;
}
function gaussianKernel(sigma) {
  if (sigma < .05) return {radius: 0, kernel: new Float32Array([1])};
  const radius = Math.min(24, Math.ceil(sigma * 3)),
        kernel = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;
  return {radius, kernel};
}
function gaussianBlur(src, w, h, sigma) {
  const {radius, kernel} = gaussianKernel(sigma);
  if (radius === 0) return src.slice();
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = Math.max(0, Math.min(w - 1, x + k));
        s += src[y * w + xx] * kernel[k + radius];
      }
      tmp[y * w + x] = s;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = Math.max(0, Math.min(h - 1, y + k));
        s += tmp[yy * w + x] * kernel[k + radius];
      }
      out[y * w + x] = s;
    }
  return out;
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
  cachedGpuFiltered = null;
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
  analysisJob++;
  updateSourceModeUi();
  if (disparityWorker) {
    disparityWorker.terminate();
    disparityWorker = null;
  }
  if (!textureLoaded) uploadTextureImage(image);
  const canvas = document.createElement('canvas');
  mapW = canvas.width = image.naturalWidth;
  mapH = canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d', {willReadFrequently: true});
  ctx.drawImage(image, 0, 0);
  const pixels = ctx.getImageData(0, 0, mapW, mapH).data;
  rawDepth = new Float32Array(mapW * mapH);
  cachedGpuFiltered = null;
  grayMap = new Float32Array(mapW * mapH);
  confMap = new Float32Array(mapW * mapH);
  confMap.fill(1);
  for (let i = 0; i < rawDepth.length; i++) {
    const p = i * 4,
          gray = pixels[p] * .299 + pixels[p + 1] * .587 + pixels[p + 2] * .114;
    grayMap[i] = gray;
    rawDepth[i] = gray / 255;
  }
  processedDepth = rawDepth.slice();
  cropX = 0;
  bgDepth = 0;
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
  if (disparityWorker) {
    disparityWorker.terminate();
    disparityWorker = null;
  }
  setProgress(.01, 'Подготовка изображения…');
  await nextFrame();
  const {d, w, h} = getScaledPixels(img);
  mapW = w;
  mapH = h;
  const gray = new Float32Array(w * h);
  const total = w * h;
  for (let j = 0; j < total; j++) {
    const i = j * 4;
    gray[j] = d[i] * .299 + d[i + 1] * .587 + d[i + 2] * .114;
    if ((j & 262143) === 0) {
      setProgress(.02 + .04 * (j / total), 'Подготовка изображения…');
      await nextFrame();
      if (job !== analysisJob) return;
    }
  }
  grayMap = gray;
  setProgress(.065, 'Определение периода…');
  await nextFrame();
  const p = estimatePeriod(gray, w, h);
  periodR.min = 1;
  periodR.max = w;
  setPair(periodR, periodN, p);
  await recoverDepth(gray, w, h, p, job);
}

async function recoverDepth(gray, w, h, p, existingJob = null) {
  const totalStartedAt = performance.now();
  const job = existingJob ?? ++analysisJob;
  if (disparityWorker) {
    disparityWorker.terminate();
    disparityWorker = null;
  }
  let gpuCosts = null;
  try {
    setProgress(.08, 'WebGPU Census и cost volume…');
    gpuCosts = await computeCostVolumeGpu(gray, w, h, p);
    if (gpuCosts)
      console.info(`Depth WebGPU: ${gpuCosts.processingMs.toFixed(0)} ms`);
  } catch (err) {
    console.warn('WebGPU depth prepass unavailable, using CPU:', err);
    gpuCosts = null;
  }
  if (job !== analysisJob) return;
  setProgress(gpuCosts ? .28 : .08, 'Расчёт смещения…');
  disparityWorker = createDisparityWorker();
  return new Promise((resolve, reject) => {
    const worker = disparityWorker;
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
        confMap = new Float32Array(m.confMap);
        bgDepth = m.bgDepth;
        cropX = m.cropX;
        console.info(
            `Depth worker: ${m.processingMs.toFixed(0)} ms ` +
            `(${m.usedGpuIcm ? 'WebGPU ICM' :
                               m.usedGpuSgm ? 'WebGPU SGM' :
                               m.usedGpuCosts ? 'WebGPU costs' : 'CPU'})`);
        console.info(
            `Depth total: ${(performance.now() - totalStartedAt).toFixed(0)} ms`);
        worker.terminate();
        if (disparityWorker === worker) disparityWorker = null;
        setProgress(.97, 'Постобработка и построение поверхности…');
        chooseDefaults();
        reprocess(() => {
          if (job === analysisJob) hideProgress();
          resolve();
        });
      }
    };
    worker.onerror = err => {
      if (disparityWorker === worker) disparityWorker = null;
      worker.terminate();
      hideProgress();
      reject(err);
    };
    const hasGpuDisparity = Boolean(
        gpuCosts?.packedDisparity && gpuCosts?.reliability);
    const grayCopy = hasGpuDisparity ? new Float32Array(0) : gray.slice();
    const message = {id: job, grayBuffer: grayCopy.buffer, w, h, p};
    const transfers = [grayCopy.buffer];
    if (gpuCosts) {
      message.packedCostsBuffer = gpuCosts.packedCosts;
      message.packedDisparityBuffer = gpuCosts.packedDisparity;
      message.reliabilityBuffer = gpuCosts.reliability;
      message.wordsPerPixel = gpuCosts.wordsPerPixel;
      if (gpuCosts.packedCosts) transfers.push(gpuCosts.packedCosts);
      if (gpuCosts.packedDisparity) transfers.push(gpuCosts.packedDisparity);
      if (gpuCosts.reliability) transfers.push(gpuCosts.reliability);
    }
    worker.postMessage(message, transfers);
  });
}

function chooseDefaults() {
  // Preserve the user's Convex/Concave selection. HTML default is Convex.
}

function reprocess(onDone = null) {
  clearTimeout(processTimer);
  processTimer = setTimeout(async () => {
    if (!rawDepth) {
      if (onDone) onDone();
      return;
    }
    processedDepth = rawDepth.slice();
    let gpuFiltered = null;
    cachedGpuFiltered = null;
    if (!loadedDepthMap && confMap) {
      try {
        gpuFiltered = await filterDepthMapGpu(
            processedDepth, confMap, mapW, mapH, bgDepth, cropX);
        if (gpuFiltered)
          console.info(
              `Depth WebGPU post: ${gpuFiltered.processingMs.toFixed(0)} ms ` +
              `(${gpuFiltered.precision})`);
      } catch (err) {
        console.warn('WebGPU depth filtering unavailable, using CPU:', err);
      }
    }
    if (!loadedDepthMap && !gpuFiltered)
      processedDepth = gaussianBlur(rawDepth, mapW, mapH, ANALYSIS_SIGMA);
    cachedGpuFiltered = gpuFiltered;
    rebuildDepthPreview(gpuFiltered);
    buildMesh();
    sched();
    if (onDone) onDone();
  }, 20);
}

function buildMesh() {
  if (!processedDepth) return;
  const mode = currentViewMode();
  // Sub-pixel tessellation lets the vertex shader sample the linearly filtered
  // depth texture between source pixels instead of reproducing its staircase.
  // Cap the grid near one million vertices for large imported depth maps.
  const surfacePixels = Math.max(1, (mapW - cropX) * mapH);
  const surfaceStep = Math.max(0.5, Math.sqrt(surfacePixels / 1000000));
  const stepPx = mode === 'surface' ? surfaceStep : MESH_STEP_PX;
  const xs = axisValues(cropX, mapW, stepPx), ys = axisValues(0, mapH, stepPx);
  const nx = xs.length, ny = ys.length;
  const visibleW = Math.max(1, mapW - 1 - cropX),
        centerX = (cropX + mapW - 1) * .5;
  const sign = currentShape();

  if (mode === 'layers') {
    surfaceUsesDepthTexture = false;
    surfaceGridKey = '';
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
          const x = xs[gx];
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
          const a = baseVertex + y * nx + x, b = a + 1,
                c = a + nx, d = c + 1;
          const m00 = mask[y * nx + x], m10 = mask[y * nx + x + 1],
                m01 = mask[(y + 1) * nx + x],
                m11 = mask[(y + 1) * nx + x + 1];
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
  // Surface mode uses a permanent regular grid. Depth and gradients are read
  // in both geometry passes from an R16F texture, so changing depth no longer
  // rebuilds or uploads the vertex buffer.
  const surfaceMap = (cleanDepthMap && cleanDepthMap.length === mapW * mapH) ?
      cleanDepthMap :
      processedDepth;
  gl.bindTexture(gl.TEXTURE_2D, depthSurfaceTex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(
      gl.TEXTURE_2D, 0, gl.R16F, mapW, mapH, 0, gl.RED, gl.FLOAT, surfaceMap);
  surfaceUsesDepthTexture = true;
  const key = `${mapW}:${mapH}:${cropX}:${stepPx}`;
  if (key === surfaceGridKey) return;
  surfaceGridKey = key;
  const topVertexCount = nx * ny;
  const perimeter = [];
  const edgeNormal = (gx, gy) => {
    let x = 0, y = 0;
    if (gx === 0) x--;
    if (gx === nx - 1) x++;
    if (gy === 0) y++;
    if (gy === ny - 1) y--;
    const length = Math.hypot(x, y) || 1;
    return {x: x / length, y: y / length};
  };
  for (let gx = 0; gx < nx; gx++)
    perimeter.push({index: gx, normal: edgeNormal(gx, 0)});
  for (let gy = 1; gy < ny; gy++)
    perimeter.push({index: gy * nx + nx - 1, normal: edgeNormal(nx - 1, gy)});
  for (let gx = nx - 2; gx >= 0; gx--)
    perimeter.push({index: (ny - 1) * nx + gx, normal: edgeNormal(gx, ny - 1)});
  for (let gy = ny - 2; gy > 0; gy--)
    perimeter.push({index: gy * nx, normal: edgeNormal(0, gy)});
  const sideVertexStart = topVertexCount;
  const backVertexStart = sideVertexStart + perimeter.length * 2;
  const verts = new Float32Array((backVertexStart + 4) * 5);
  let o = 0;
  for (let gy = 0; gy < ny; gy++)
    for (let gx = 0; gx < nx; gx++) {
      verts[o++] = (xs[gx] - centerX) / visibleW * 2;
      verts[o++] = (mapH * .5 - ys[gy]) / visibleW * 2;
      verts[o++] = 0;
      verts[o++] = 0;
      verts[o++] = 0;
    }
  for (const edge of perimeter) {
    const source = edge.index * 5;
    for (const marker of [-2, -3]) {
      verts[o++] = verts[source];
      verts[o++] = verts[source + 1];
      verts[o++] = marker;
      verts[o++] = edge.normal.x;
      verts[o++] = edge.normal.y;
    }
  }
  const cornerTopIndices = [0, nx - 1, (ny - 1) * nx + nx - 1, (ny - 1) * nx];
  for (const topIndex of cornerTopIndices) {
    const source = topIndex * 5;
    verts[o++] = verts[source];
    verts[o++] = verts[source + 1];
    verts[o++] = -1;
    verts[o++] = 0;
    verts[o++] = 0;
  }
  const topIndexCount = (nx - 1) * (ny - 1) * 6;
  const indices = new Uint32Array(topIndexCount + perimeter.length * 6 + 6);
  let q = 0;
  for (let y = 0; y < ny - 1; y++)
    for (let x = 0; x < nx - 1; x++) {
      const a = y * nx + x, b = a + 1, c = a + nx, d = c + 1;
      indices[q++] = a;
      indices[q++] = c;
      indices[q++] = b;
      indices[q++] = b;
      indices[q++] = c;
      indices[q++] = d;
    }
  for (let i = 0; i < perimeter.length; i++) {
    const next = (i + 1) % perimeter.length;
    const topA = sideVertexStart + i * 2;
    const baseA = topA + 1;
    const topB = sideVertexStart + next * 2;
    const baseB = topB + 1;
    indices[q++] = topA;
    indices[q++] = baseA;
    indices[q++] = topB;
    indices[q++] = topB;
    indices[q++] = baseA;
    indices[q++] = baseB;
  }
  const backTL = backVertexStart, backTR = backVertexStart + 1,
        backBR = backVertexStart + 2, backBL = backVertexStart + 3;
  indices[q++] = backTL;
  indices[q++] = backTR;
  indices[q++] = backBL;
  indices[q++] = backTR;
  indices[q++] = backBR;
  indices[q++] = backBL;
  meshIndexCount = q;
  meshSourceBounds = {
    minX: -1, maxX: 1,
    minY: (mapH * .5 - ys[ny - 1]) / visibleW * 2,
    maxY: (mapH * .5 - ys[0]) / visibleW * 2,
    minD: 0, maxD: 1,
  };
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ebo);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
}

function morphMask(mask, w, h, r, dilate) {
  if (r <= 0) return mask.slice();
  const tmp = new Uint8Array(mask.length), out = new Uint8Array(mask.length);
  const rowPrefix = new Int32Array(w + 1), colPrefix = new Int32Array(h + 1);
  for (let y = 0; y < h; y++) {
    rowPrefix.fill(0);
    const row = y * w;
    for (let x = cropX; x < w; x++)
      rowPrefix[x + 1] = rowPrefix[x] + mask[row + x];
    for (let x = cropX; x < w; x++) {
      const x0 = Math.max(cropX, x - r), x1 = Math.min(w - 1, x + r),
            sum = rowPrefix[x1 + 1] - rowPrefix[x0];
      tmp[row + x] = dilate ? (sum > 0 ? 1 : 0) : (sum === x1 - x0 + 1 ? 1 : 0);
    }
  }
  for (let x = cropX; x < w; x++) {
    colPrefix[0] = 0;
    for (let y = 0; y < h; y++)
      colPrefix[y + 1] = colPrefix[y] + tmp[y * w + x];
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h - 1, y + r),
            sum = colPrefix[y1 + 1] - colPrefix[y0];
      out[y * w + x] =
          dilate ? (sum > 0 ? 1 : 0) : (sum === y1 - y0 + 1 ? 1 : 0);
    }
  }
  return out;
}
function dilateMask(mask, w, h, r = 1) {
  return morphMask(mask, w, h, r, true);
}
function erodeMask(mask, w, h, r = 1) {
  return morphMask(mask, w, h, r, false);
}
function removeSmallComponents(mask, w, h, minSize) {
  if (minSize <= 1) return mask.slice();
  const out = mask.slice(), seen = new Uint8Array(mask.length),
        queue = new Int32Array(mask.length);
  for (let y = 0; y < h; y++)
    for (let x = cropX; x < w; x++) {
      const start = y * w + x;
      if (!out[start] || seen[start]) continue;
      let head = 0, tail = 0;
      queue[tail++] = start;
      seen[start] = 1;
      while (head < tail) {
        const v = queue[head++], vx = v % w, vy = (v / w) | 0;
        for (let yy = Math.max(0, vy - 1); yy <= Math.min(h - 1, vy + 1); yy++)
          for (let xx = Math.max(cropX, vx - 1); xx <= Math.min(w - 1, vx + 1);
               xx++) {
            const n = yy * w + xx;
            if (out[n] && !seen[n]) {
              seen[n] = 1;
              queue[tail++] = n;
            }
          }
      }
      if (tail < minSize)
        for (let i = 0; i < tail; i++) out[queue[i]] = 0;
    }
  return out;
}
function fillSmallMaskHoles(mask, w, h, maxSize) {
  const out = mask.slice(), seen = new Uint8Array(mask.length),
        queue = new Int32Array(mask.length);
  for (let y = 0; y < h; y++) for (let x = cropX; x < w; x++) {
    const start = y * w + x;
    if (out[start] || seen[start]) continue;
    let head = 0, tail = 0, touchesOutside = false;
    queue[tail++] = start;
    seen[start] = 1;
    while (head < tail) {
      const index = queue[head++], px = index % w, py = (index / w) | 0;
      if (px === cropX || px === w - 1 || py === 0 || py === h - 1)
        touchesOutside = true;
      for (let yy = Math.max(0, py - 1); yy <= Math.min(h - 1, py + 1); yy++) {
        for (let xx = Math.max(cropX, px - 1); xx <= Math.min(w - 1, px + 1); xx++) {
          const neighbour = yy * w + xx;
          if (!out[neighbour] && !seen[neighbour]) {
            seen[neighbour] = 1;
            queue[tail++] = neighbour;
          }
        }
      }
    }
    if (!touchesOutside && tail <= maxSize)
      for (let i = 0; i < tail; i++) out[queue[i]] = 1;
  }
  return out;
}
function buildHysteresisDepthMask(depth, confidence, w, h, highThreshold) {
  const lowThreshold = Math.max(.0015, highThreshold * .30);
  const weak = new Uint8Array(depth.length), out = new Uint8Array(depth.length),
        queue = new Int32Array(depth.length);
  let head = 0, tail = 0;
  for (let y = 0; y < h; y++)
    for (let x = cropX; x < w; x++) {
      const i = y * w + x, v = depth[i];
      if (v > highThreshold) {
        out[i] = 1;
        queue[tail++] = i;
      }
      if (v <= lowThreshold) continue;
      let support = 0;
      for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++)
        for (let xx = Math.max(cropX, x - 1); xx <= Math.min(w - 1, x + 1); xx++)
          if (depth[yy * w + xx] > lowThreshold) support++;
      const reliable = !confidence || confidence[i] >= .025;
      if (support >= 4 && (reliable || v > highThreshold * .55)) weak[i] = 1;
    }
  while (head < tail) {
    const i = queue[head++], x = i % w, y = (i / w) | 0;
    for (let yy = Math.max(0, y - 1); yy <= Math.min(h - 1, y + 1); yy++)
      for (let xx = Math.max(cropX, x - 1); xx <= Math.min(w - 1, x + 1); xx++) {
        const j = yy * w + xx;
        if (weak[j] && !out[j]) {
          out[j] = 1;
          queue[tail++] = j;
        }
      }
  }
  return out;
}
function smoothDepthMask(mask, w, h) {
  const weights = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) weights[i] = mask[i];
  const coverage = gaussianBlur(weights, w, h, 2.35), out = new Uint8Array(mask.length);
  for (let y = 0; y < h; y++)
    for (let x = cropX; x < w; x++) {
      const i = y * w + x;
      out[i] = coverage[i] >= .47 ? 1 : 0;
    }
  return out;
}
function suppressDepthSpikes(src, surfaceMask, w, h, maxSize) {
  if (maxSize < 1) return src.slice();
  const medianMap = new Float32Array(src.length),
        spikeMask = new Uint8Array(src.length);
  const values = new Float32Array(9), deviations = new Float32Array(9);
  for (let y = 1; y < h - 1; y++)
    for (let x = Math.max(cropX + 1, 1); x < w - 1; x++) {
      const i = y * w + x;
      if (!surfaceMask[i]) continue;
      let n = 0;
      for (let yy = -1; yy <= 1; yy++)
        for (let xx = -1; xx <= 1; xx++) {
          const j = (y + yy) * w + x + xx;
          if (surfaceMask[j]) values[n++] = src[j];
        }
      if (n < 4) continue;
      for (let a = 1; a < n; a++) {
        const v = values[a];
        let b = a - 1;
        while (b >= 0 && values[b] > v) {
          values[b + 1] = values[b];
          b--;
        }
        values[b + 1] = v;
      }
      const median = values[n >> 1];
      medianMap[i] = median;
      for (let a = 0; a < n; a++) deviations[a] = Math.abs(values[a] - median);
      for (let a = 1; a < n; a++) {
        const v = deviations[a];
        let b = a - 1;
        while (b >= 0 && deviations[b] > v) {
          deviations[b + 1] = deviations[b];
          b--;
        }
        deviations[b + 1] = v;
      }
      const mad = deviations[n >> 1], threshold = .010 + mad * 3.2;
      if (Math.abs(src[i] - median) > threshold) spikeMask[i] = 1;
    }
  const out = src.slice(), seen = new Uint8Array(src.length),
        queue = new Int32Array(src.length);
  for (let y = 1; y < h - 1; y++)
    for (let x = Math.max(cropX + 1, 1); x < w - 1; x++) {
      const start = y * w + x;
      if (!spikeMask[start] || seen[start]) continue;
      let head = 0, tail = 0;
      queue[tail++] = start;
      seen[start] = 1;
      while (head < tail) {
        const v = queue[head++], vx = v % w, vy = (v / w) | 0;
        for (let yy = Math.max(1, vy - 1); yy <= Math.min(h - 2, vy + 1); yy++)
          for (let xx = Math.max(cropX + 1, vx - 1);
               xx <= Math.min(w - 2, vx + 1); xx++) {
            const j = yy * w + xx;
            if (spikeMask[j] && !seen[j]) {
              seen[j] = 1;
              queue[tail++] = j;
            }
          }
      }
      if (tail <= maxSize)
        for (let k = 0; k < tail; k++) out[queue[k]] = medianMap[queue[k]];
    }
  return out;
}
function smoothDepthBoundary(src, surfaceMask, w, h, radius = 10) {
  const inner = erodeMask(surfaceMask, w, h, radius);
  const weights = new Float32Array(surfaceMask.length);
  for (let i = 0; i < weights.length; i++) weights[i] = surfaceMask[i] ? 1 : 0;
  // Normalized convolution averages only other surface pixels, never the black
  // background.
  const interpolated =
      weightedGaussian(src, weights, w, h, Math.max(1, radius * .55));
  const out = src.slice();
  for (let i = 0; i < out.length; i++)
    if (surfaceMask[i] && !inner[i] && interpolated[i] > 0)
      out[i] = interpolated[i];
  return out;
}
function weightedGaussian(values, weights, w, h, sigma) {
  const num = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) num[i] = values[i] * weights[i];
  const bNum = gaussianBlur(num, w, h, sigma),
        bDen = gaussianBlur(weights, w, h, sigma),
        out = new Float32Array(values.length);
  for (let i = 0; i < out.length; i++)
    out[i] = bDen[i] > .0001 ? bNum[i] / bDen[i] : 0;
  return out;
}
function smoothInsideMask(depth, mask, w, h, sigma = 4, finalSigma = 1.5) {
  const closedMask = mask.slice();
  const weights = new Float32Array(mask.length);
  for (let i = 0; i < weights.length; i++) weights[i] = closedMask[i];

  // Normalized convolution prevents the black background from darkening the
  // object near its contour.
  const normalized = weightedGaussian(depth, weights, w, h, sigma);
  const smoothed = finalSigma > 0 ?
      gaussianBlur(normalized, w, h, finalSigma) : normalized;
  const coverage = gaussianBlur(weights, w, h, 2.25);
  const effectiveCoverage = new Float32Array(coverage.length);
  for (let i = 0; i < smoothed.length; i++) {
    // A soft coverage mask behaves like contour antialiasing: the geometry
    // reaches the background over a couple of pixels instead of ending on a
    // staircase-shaped binary edge.
    const alpha = coverage[i] <= .005 ? 0 : smoothstep(.015, .985, coverage[i]);
    effectiveCoverage[i] = alpha;
    smoothed[i] *= alpha;
    mask[i] = closedMask[i];
  }
  // Keep exactly the same alpha for normalization and compositing. Dividing
  // by the raw Gaussian coverage after multiplying by smoothstep(coverage)
  // artificially amplified the contour and produced a bright halo.
  depthCoverage = effectiveCoverage;
  return smoothed;
}
function swapWeighted(values, weights, a, b) {
  const v = values[a], ww = weights[a];
  values[a] = values[b];
  weights[a] = weights[b];
  values[b] = v;
  weights[b] = ww;
}
function selectWeightedMedian(values, weights, n, targetWeight) {
  let left = 0, right = n - 1, target = targetWeight;
  while (left <= right) {
    const pivot = values[(left + right) >> 1];
    let lt = left, i = left, gt = right;
    while (i <= gt) {
      if (values[i] < pivot) {
        swapWeighted(values, weights, i, lt);
        i++;
        lt++;
      } else if (values[i] > pivot) {
        swapWeighted(values, weights, i, gt);
        gt--;
      } else
        i++;
    }
    let lowerWeight = 0, equalWeight = 0;
    for (i = left; i < lt; i++) lowerWeight += weights[i];
    for (i = lt; i <= gt; i++) equalWeight += weights[i];
    if (target <= lowerWeight)
      right = lt - 1;
    else if (target <= lowerWeight + equalWeight)
      return pivot;
    else {
      target -= lowerWeight + equalWeight;
      left = gt + 1;
    }
  }
  return values[Math.max(0, Math.min(n - 1, left))];
}
function weightedMedianDepth(src, mask, confidence, guide, w, h, radius) {
  const out = src.slice(), sampleStep = Math.max(1, Math.ceil(radius / 3));
  const samplesPerAxis = Math.floor(radius * 2 / sampleStep) + 1,
        maxN = samplesPerAxis ** 2;
  const values = new Float32Array(maxN), weights = new Float32Array(maxN);
  const guideScale = 20;
  for (let y = radius; y < h - radius; y++)
    for (let x = Math.max(cropX + radius, radius); x < w - radius; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      let n = 0, totalWeight = 0;
      for (let yy = -radius; yy <= radius; yy += sampleStep)
        for (let xx = -radius; xx <= radius; xx += sampleStep) {
          const j = (y + yy) * w + x + xx;
          if (!mask[j] || src[j] <= 0) continue;
          const spatial = 1 / (1 + xx * xx + yy * yy);
          const edge =
              guide ? Math.exp(-Math.abs(guide[j] - guide[i]) / guideScale) : 1;
          values[n] = src[j];
          weights[n] =
              spatial * edge * (.15 + (confidence ? confidence[j] : 1));
          totalWeight += weights[n];
          n++;
        }
      if (n)
        out[i] = selectWeightedMedian(values, weights, n, totalWeight * .5);
    }
  return out;
}
function depthDisplayValue(depth, shape) {
  const value = Math.max(0, Math.min(1, depth));
  const normalized = Math.pow(value, .82);
  return shape > 0 ? normalized : 1 - normalized;
}
function rebuildDepthPreview(gpuFiltered = cachedGpuFiltered) {
  depthPreview = null;
  depthPreviewW = 0;
  depthPreviewH = 0;
  cleanDepthMap = null;
  depthCoverage = null;
  if (!processedDepth || !grayMap || !mapW || !mapH) return;

  const w = mapW, h = mapH, shape = currentShape();
  if (gpuFiltered && gpuFiltered.depth.length === w * h) {
    cleanDepthMap = gpuFiltered.depth;
    depthCoverage = null;
    const srcW = Math.max(1, w - cropX);
    depthPreviewW = srcW;
    depthPreviewH = h;
    const imgData = depthCtx.createImageData(srcW, h);
    let o = 0;
    for (let y = 0; y < h; y++)
      for (let x = cropX; x < w; x++) {
        const lin = Math.max(0, Math.min(1, cleanDepthMap[y * w + x]));
        const shown = depthDisplayValue(lin, shape);
        const g = Math.max(0, Math.min(255, Math.round(shown * 255)));
        imgData.data[o++] = g;
        imgData.data[o++] = g;
        imgData.data[o++] = g;
        imgData.data[o++] = 255;
      }
    depthPreview = imgData;
    return;
  }
  if (loadedDepthMap) {
    cleanDepthMap = processedDepth.slice();
    depthPreviewW = w;
    depthPreviewH = h;
    const imgData = depthCtx.createImageData(w, h);
    for (let i = 0, o = 0; i < cleanDepthMap.length; i++) {
      const g = Math.max(
          0, Math.min(255, Math.round(
              depthDisplayValue(cleanDepthMap[i], shape) * 255)));
      imgData.data[o++] = g;
      imgData.data[o++] = g;
      imgData.data[o++] = g;
      imgData.data[o++] = 255;
    }
    depthPreview = imgData;
    return;
  }
  const signed = new Float32Array(w * h), positives = [];
  for (let y = 0; y < h; y++)
    for (let x = cropX; x < w; x++) {
      const i = y * w + x;
      const s = Math.max(0, processedDepth[i] - bgDepth);
      signed[i] = s;
      if ((x & 1) === 0 && (y & 1) === 0 && s > 0) positives.push(s);
    }
  positives.sort((a, b) => a - b);
  const depthNoiseFloor = positives.length ?
      Math.max(.004, percentile(positives, .10) * .65) :
      .008;
  let mask = buildHysteresisDepthMask(
      signed, confMap, w, h, depthNoiseFloor);
  mask = removeSmallComponents(mask, w, h, SPECKLE_SIZE);
  mask = smoothDepthMask(mask, w, h);

  let clean = new Float32Array(w * h);
  for (let i = 0; i < clean.length; i++)
    if (mask[i]) clean[i] = signed[i];
  {
    const radius = FILL_RADIUS;
    const closed = fillSmallMaskHoles(mask, w, h, radius * radius * 8);
    const weights = new Float32Array(mask.length);
    for (let i = 0; i < weights.length; i++)
      weights[i] = mask[i] ? (.1 + (confMap ? confMap[i] : 1)) : 0;
    const filled = weightedGaussian(signed, weights, w, h, Math.max(1, radius));
    for (let i = 0; i < clean.length; i++)
      if (closed[i]) {
        if (!mask[i]) clean[i] = filled[i];
        mask[i] = 1;
      }
  }
  clean = suppressDepthSpikes(clean, mask, w, h, SPECKLE_SIZE);
  clean = smoothDepthBoundary(clean, mask, w, h, 10);
  clean = weightedMedianDepth(clean, mask, confMap, null, w, h, MEDIAN_RADIUS);
  clean = smoothInsideMask(clean, mask, w, h);

  const nz = [];
  for (let y = 0; y < h; y += 2)
    for (let x = cropX; x < w; x += 2) {
      const i = y * w + x;
      const alpha = depthCoverage ? depthCoverage[i] : 1;
      const v = alpha > .01 ? clean[i] / alpha : clean[i];
      if (mask[i] && v > .0005) nz.push(v);
    }
  nz.sort((a, b) => a - b);
  // Robust global autocontrast. A stronger 1% black cutoff removes the long
  // low-depth tail; the upper end remains conservative to preserve highlights.
  const lo = nz.length ? percentile(nz, .01) : 0;
  const hi = nz.length ? Math.max(lo + .02, percentile(nz, .99)) : 1;
  const inv = 1 / Math.max(1e-6, hi - lo);
  cleanDepthMap = new Float32Array(w * h);
  const srcW = Math.max(1, w - cropX), srcH = h;
  const imgData = depthCtx.createImageData(srcW, srcH);
  let o = 0;
  for (let y = 0; y < h; y++)
    for (let x = cropX; x < w; x++) {
      const i = y * w + x, value = clean[i];
      const alpha = depthCoverage ? depthCoverage[i] : 1;
      const base = alpha > .01 ? value / alpha : value;
      let stretched =
          base > 0 ? Math.max(0, Math.min(1, (base - lo) * inv)) : 0;
      if (stretched > 0 && stretched < .28 && alpha > .82) {
        const shadow = stretched / .28;
        stretched = .28 * Math.pow(shadow, .72);
      }
      const lin = stretched * alpha;
      cleanDepthMap[y * w + x] = lin;
      const t = depthDisplayValue(lin, shape);
      const g = Math.max(0, Math.min(255, Math.round(t * 255)));
      imgData.data[o++] = g;
      imgData.data[o++] = g;
      imgData.data[o++] = g;
      imgData.data[o++] = 255;
    }
  depthPreview = imgData;
  depthPreviewW = srcW;
  depthPreviewH = srcH;
}
function renderDepthMap() {
  downloadDepthBtn.disabled = !depthPreview;
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

downloadDepthBtn.onclick = () => {
  if (!depthPreview || !depthPreviewW || !depthPreviewH) return;
  const source = document.createElement('canvas');
  source.width = depthPreviewW;
  source.height = depthPreviewH;
  source.getContext('2d').putImageData(depthPreview, 0, 0);
  const targetHeight = Math.max(
      depthPreviewH, img?.naturalHeight || depthPreviewH);
  const targetWidth = Math.max(
      1, Math.round(depthPreviewW * targetHeight / depthPreviewH));
  const output = document.createElement('canvas');
  output.width = targetWidth;
  output.height = targetHeight;
  const outputContext = output.getContext('2d');
  outputContext.imageSmoothingEnabled = true;
  outputContext.imageSmoothingQuality = 'high';
  outputContext.drawImage(source, 0, 0, targetWidth, targetHeight);
  output.toBlob(blob => {
    if (!blob) return;
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url;
    link.download = 'depth-map.png';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, 'image/png');
};

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
  if (loadedDepthMap || !grayMap || !mapW || !mapH) return;
  recoverDepth(grayMap, mapW, mapH, +periodR.value).catch(err => {
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

// Unified mouse/touch gesture controller. Pointer events are the sole source
// of rotation and pinch state, so two-finger zoom can never also rotate.
function pointerDistance() {
  const points = [...activePointers.values()];
  if (points.length < 2) return 0;
  return Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
}
cv.addEventListener('pointerdown', e => {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  e.preventDefault();
  activePointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
  cv.setPointerCapture(e.pointerId);
  // Mouse down is unambiguously a rotation gesture. A first touch may become
  // a pinch, so defer stopping auto-rotation until it actually moves alone.
  drag = e.pointerType === 'mouse';
  if (e.pointerType === 'mouse') {
    disableAutoRotate();
    pauseAuto();
  }
  inertiaYaw = 0;
  inertiaPitch = 0;
  if (activePointers.size >= 2) {
    gestureMode = 'pinch';
    drag = false;
    disableAutoRotate();
    pauseAuto();
    gesturePointerId = null;
    pinchDistance = pointerDistance();
  } else {
    gestureMode = 'rotate';
    gesturePointerId = e.pointerId;
    lastGestureX = e.clientX;
    lastGestureY = e.clientY;
    lastGestureTime = performance.now();
  }
});
cv.addEventListener('pointermove', e => {
  if (!activePointers.has(e.pointerId)) return;
  e.preventDefault();
  activePointers.set(e.pointerId, {x: e.clientX, y: e.clientY});
  if (activePointers.size >= 2) {
    if (gestureMode !== 'pinch') {
      gestureMode = 'pinch';
      drag = false;
      disableAutoRotate();
      pauseAuto();
      inertiaYaw = inertiaPitch = 0;
      pinchDistance = pointerDistance();
      return;
    }
    const distance = pointerDistance();
    if (pinchDistance > 0 && distance > 0)
      zoom = Math.max(.35, Math.min(4, zoom * distance / pinchDistance));
    pinchDistance = distance;
    sched();
    return;
  }
  // Do not turn the remaining finger into rotation after a pinch. A new
  // rotation gesture starts only after every pointer has been released.
  if (gestureMode !== 'rotate' || e.pointerId !== gesturePointerId) return;
  if (!drag) {
    if (Math.hypot(
        e.clientX - lastGestureX, e.clientY - lastGestureY) < 5) return;
    drag = true;
    disableAutoRotate();
  }
  const now = performance.now(), elapsed = Math.max(4, now - lastGestureTime),
        dx = e.clientX - lastGestureX, dy = e.clientY - lastGestureY;
  yaw += dx * .008;
  pitch = Math.max(-1.35, Math.min(1.35, pitch + dy * .008));
  const instantYaw = dx * .008 / elapsed, instantPitch = dy * .008 / elapsed;
  inertiaYaw = inertiaYaw * .65 + instantYaw * .35;
  inertiaPitch = inertiaPitch * .65 + instantPitch * .35;
  lastGestureX = e.clientX;
  lastGestureY = e.clientY;
  lastGestureTime = now;
  sched();
});
function finishPointer(e) {
  activePointers.delete(e.pointerId);
  if (cv.hasPointerCapture(e.pointerId)) cv.releasePointerCapture(e.pointerId);
  if (activePointers.size === 0) {
    drag = false;
    if (gestureMode === 'pinch') inertiaYaw = inertiaPitch = 0;
    gestureMode = 'none';
    gesturePointerId = null;
    pinchDistance = 0;
    autoBaseYaw = yaw;
    autoBasePitch = pitch;
    pauseAuto();
  }
}
['pointerup', 'pointercancel', 'lostpointercapture']
    .forEach(n => cv.addEventListener(n, finishPointer));
cv.addEventListener('wheel', e => {
  e.preventDefault();
  pauseAuto();
  zoom = Math.max(.35, Math.min(4, zoom * Math.exp(-e.deltaY * .001)));
  sched();
}, {passive: false});
cv.ondblclick = resetCamera;

function setStereogramView(open) {
  thumb.classList.toggle('stereogram-view', open);
  thumb.setAttribute('aria-expanded', String(open));
  thumb.setAttribute(
      'aria-label', open ? 'Закрыть полноэкранную стереограмму' :
                           'Открыть стереограмму на весь экран');
}
thumb.addEventListener('click', () =>
  setStereogramView(!thumb.classList.contains('stereogram-view')));
thumb.addEventListener('keydown', e => {
  if (e.key !== 'Enter' && e.key !== ' ') return;
  e.preventDefault();
  setStereogramView(!thumb.classList.contains('stereogram-view'));
});
addEventListener('keydown', e => {
  if (e.key === 'Escape' && thumb.classList.contains('stereogram-view'))
    setStereogramView(false);
});

resize();
loadRandomStartupImage();
