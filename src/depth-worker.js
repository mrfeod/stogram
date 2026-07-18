'use strict';

const INF = 1e20;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
const report = (id, value, label) => postMessage({type: 'progress', id, value, label});

function toGray(rgba, count) {
  const gray = new Uint8Array(count);
  for (let i = 0, p = 0; i < count; i++, p += 4)
    gray[i] = (77 * rgba[p] + 150 * rgba[p + 1] + 29 * rgba[p + 2]) >> 8;
  return gray;
}

function estimatePeriod(gray, width, height) {
  const minPeriod = 40, maxPeriod = Math.min(250, width >> 1);
  if (maxPeriod < minPeriod) throw new Error('Image is too narrow');
  const yStep = Math.max(2, Math.floor(height * .6 / 60));
  let bestPeriod = minPeriod, bestScore = Infinity;

  // Fast global estimate. It often lands on the average foreground repeat,
  // so it is used only as the centre of a more robust local search below.
  for (let period = minPeriod; period <= maxPeriod; period += 2) {
    let sum = 0, count = 0;
    for (let y = Math.floor(height * .2); y < height * .8; y += yStep) {
      const row = y * width;
      for (let x = period + 6; x < width - 6; x += 4) {
        sum += Math.abs(gray[row + x] - gray[row + x - period]);
        count++;
      }
    }
    const score = sum / Math.max(1, count);
    if (score < bestScore) { bestScore = score; bestPeriod = period; }
  }

  for (let period = Math.max(minPeriod, bestPeriod - 3);
       period <= Math.min(maxPeriod, bestPeriod + 3); period++) {
    let sum = 0, count = 0;
    for (let y = Math.floor(height * .2); y < height * .8; y += yStep) {
      const row = y * width;
      for (let x = period + 6; x < width - 6; x += 3) {
        sum += Math.abs(gray[row + x] - gray[row + x - period]);
        count++;
      }
    }
    const score = sum / Math.max(1, count);
    if (score < bestScore) { bestScore = score; bestPeriod = period; }
  }

  // Background has the longest repeat. Collect only locally unique matches
  // around the global estimate and use their upper robust percentile.
  const shifts = [];
  const searchMin = Math.max(minPeriod, bestPeriod - 5);
  const searchMax = Math.min(maxPeriod, bestPeriod + Math.max(12, Math.round(bestPeriod * .22)));
  const gridY = Math.max(7, Math.round(height / 26));
  const gridX = Math.max(9, Math.round(width / 30));
  for (let cy = 4; cy < height - 4; cy += gridY) {
    for (let cx = searchMax + 4; cx < width - 4; cx += gridX) {
      let localBest = bestPeriod, localCost = Infinity, secondCost = Infinity;
      for (let shift = searchMin; shift <= searchMax; shift++) {
        let cost = 0, count = 0;
        for (let dy = -3; dy <= 3; dy += 2) for (let dx = -3; dx <= 3; dx++) {
          const index = (cy + dy) * width + cx + dx;
          const other = index - shift;
          const luminance = Math.min(50, Math.abs(gray[index] - gray[other]));
          const gradientA = gray[index + 1] - gray[index - 1];
          const gradientB = gray[other + 1] - gray[other - 1];
          cost += luminance * .4 + Math.min(60, Math.abs(gradientA - gradientB)) * .6;
          count++;
        }
        cost /= count;
        if (cost < localCost) {
          secondCost = localCost; localCost = cost; localBest = shift;
        } else if (cost < secondCost) secondCost = cost;
      }
      const uniqueness = (secondCost - localCost) / Math.max(1, secondCost);
      if (uniqueness > .012) shifts.push(localBest);
    }
  }
  if (shifts.length < 8) return bestPeriod;
  shifts.sort((a, b) => a - b);
  return shifts[Math.floor((shifts.length - 1) * .88)];
}

function matchingCost(gray, width, height, period, windowSize = 9) {
  const diffWidth = width - period, integralWidth = diffWidth + 1;
  const integral = new Float64Array((height + 1) * integralWidth);
  for (let y = 0; y < height; y++) {
    let rowSum = 0;
    const source = y * width, target = (y + 1) * integralWidth, previous = y * integralWidth;
    for (let x = 0; x < diffWidth; x++) {
      rowSum += Math.abs(gray[source + x + period] - gray[source + x]);
      integral[target + x + 1] = integral[previous + x + 1] + rowSum;
    }
  }
  const cost = new Float32Array(width * height); cost.fill(INF);
  const radius = windowSize >> 1, midpoint = period >> 1;
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius), y1 = Math.min(height - 1, y + radius);
    const top = y0 * integralWidth, bottom = (y1 + 1) * integralWidth, row = y * width;
    for (let x = 0; x < diffWidth; x++) {
      const x0 = Math.max(0, x - radius), x1 = Math.min(diffWidth - 1, x + radius);
      const sum = integral[bottom + x1 + 1] - integral[top + x1 + 1] -
          integral[bottom + x0] + integral[top + x0];
      cost[row + midpoint + x] = sum / ((x1 - x0 + 1) * (y1 - y0 + 1));
    }
  }
  return cost;
}

function evaluate(previous, current, next, period, bestCost, secondCost, periodMap, refine) {
  for (let i = 0; i < current.length; i++) {
    const cost = current[i];
    if (cost >= INF / 2) continue;
    if (cost >= bestCost[i]) {
      if (cost < secondCost[i]) secondCost[i] = cost;
      continue;
    }
    secondCost[i] = bestCost[i];
    let delta = 0;
    if (refine) {
      const denominator = previous[i] - 2 * cost + next[i];
      if (Number.isFinite(denominator) && Math.abs(denominator) > 1e-6)
        delta = clamp(.5 * (previous[i] - next[i]) / denominator, -1, 1);
    }
    bestCost[i] = cost; periodMap[i] = period + delta;
  }
}

function refineAmbiguousPeriods(periodMap, bestCost, secondCost, gray, width, height) {
  const confidence = new Float32Array(periodMap.length);
  for (let i = 0; i < confidence.length; i++) {
    const second = secondCost[i];
    confidence[i] = Number.isFinite(second) && second < INF / 2 ?
        clamp((second - bestCost[i]) / Math.max(1, second), 0, 1) : 0;
  }
  let current = periodMap;
  for (let pass = 0; pass < 2; pass++) {
    const output = current.slice();
    for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
      const index = y * width + x;
      if (confidence[index] >= .025) continue;
      let weighted = 0, totalWeight = 0, support = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const neighbour = (y + dy) * width + x + dx;
        if (confidence[neighbour] < .035) continue;
        const edgeWeight = Math.exp(-Math.abs(gray[neighbour] - gray[index]) / 18);
        const weight = edgeWeight * (.15 + confidence[neighbour]);
        weighted += current[neighbour] * weight;
        totalWeight += weight;
        support++;
      }
      if (support >= 2 && totalWeight > .05) {
        const neighbourPeriod = weighted / totalWeight;
        output[index] = current[index] * .2 + neighbourPeriod * .8;
      }
    }
    current = output;
  }
  return current;
}

function median3(input, width, height) {
  const output = new Float32Array(input.length), values = new Float32Array(9);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let n = 0;
    for (let dy = -1; dy <= 1; dy++) {
      const row = clamp(y + dy, 0, height - 1) * width;
      for (let dx = -1; dx <= 1; dx++) values[n++] = input[row + clamp(x + dx, 0, width - 1)];
    }
    for (let i = 1; i < 9; i++) { const v = values[i]; let j = i - 1; while (j >= 0 && values[j] > v) { values[j + 1] = values[j]; j--; } values[j + 1] = v; }
    output[y * width + x] = values[4];
  }
  return output;
}

function largestComponent(mask, width, height) {
  const labels = new Int32Array(mask.length), queue = new Int32Array(mask.length);
  let label = 0, largestLabel = 0, largestSize = 0;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || labels[start]) continue;
    label++; let head = 0, tail = 0; queue[tail++] = start; labels[start] = label;
    while (head < tail) {
      const index = queue[head++], x = index % width, y = (index / width) | 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        if (!dx && !dy) continue;
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx >= width || yy < 0 || yy >= height) continue;
        const next = yy * width + xx;
        if (mask[next] && !labels[next]) { labels[next] = label; queue[tail++] = next; }
      }
    }
    if (tail > largestSize) { largestSize = tail; largestLabel = label; }
  }
  for (let i = 0; i < mask.length; i++) mask[i] = labels[i] === largestLabel ? 1 : 0;
  return mask;
}

function binaryPass(input, width, height, radius, maximum, horizontal) {
  const output = new Uint8Array(input.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let value = maximum ? 0 : 1;
    for (let d = -radius; d <= radius; d++) {
      const xx = horizontal ? clamp(x + d, 0, width - 1) : x;
      const yy = horizontal ? y : clamp(y + d, 0, height - 1);
      value = maximum ? Math.max(value, input[yy * width + xx]) : Math.min(value, input[yy * width + xx]);
    }
    output[y * width + x] = value;
  }
  return output;
}

function closeMask(mask, width, height, radius = 2) {
  let result = binaryPass(mask, width, height, radius, true, true);
  result = binaryPass(result, width, height, radius, true, false);
  result = binaryPass(result, width, height, radius, false, true);
  return binaryPass(result, width, height, radius, false, false);
}

function kernel(sigma) {
  const radius = Math.max(1, Math.ceil(3 * sigma)), result = new Float32Array(radius * 2 + 1);
  let sum = 0;
  for (let i = -radius; i <= radius; i++) { const value = Math.exp(-(i * i) / (2 * sigma * sigma)); result[i + radius] = value; sum += value; }
  for (let i = 0; i < result.length; i++) result[i] /= sum;
  return result;
}

function blur(input, width, height, sigma) {
  const weights = kernel(sigma), radius = weights.length >> 1;
  const temp = new Float32Array(input.length), output = new Float32Array(input.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let sum = 0; for (let k = -radius; k <= radius; k++) sum += input[y * width + clamp(x + k, 0, width - 1)] * weights[k + radius];
    temp[y * width + x] = sum;
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let sum = 0; for (let k = -radius; k <= radius; k++) sum += temp[clamp(y + k, 0, height - 1) * width + x] * weights[k + radius];
    output[y * width + x] = sum;
  }
  return output;
}

function smoothInside(depth, mask, width, height) {
  mask = closeMask(mask, width, height);
  const source = new Float32Array(depth.length), maskFloat = new Float32Array(depth.length);
  for (let i = 0; i < depth.length; i++) { maskFloat[i] = mask[i]; source[i] = depth[i] * mask[i]; }
  const blurredDepth = blur(source, width, height, 4), blurredMask = blur(maskFloat, width, height, 4);
  const normalized = new Float32Array(depth.length);
  for (let i = 0; i < depth.length; i++) normalized[i] = blurredMask[i] > 1e-5 ? blurredDepth[i] / blurredMask[i] : 0;
  const result = blur(normalized, width, height, 1.5);
  for (let i = 0; i < result.length; i++) result[i] *= mask[i];
  return result;
}

function fillSmallHoles(depth, mask, width, height, radius) {
  const closed = closeMask(mask, width, height, radius);
  const source = new Float32Array(depth.length), weights = new Float32Array(depth.length);
  for (let i = 0; i < depth.length; i++) {
    weights[i] = mask[i];
    source[i] = depth[i] * mask[i];
  }
  const blurredDepth = blur(source, width, height, Math.max(1, radius * .7));
  const blurredWeights = blur(weights, width, height, Math.max(1, radius * .7));
  const result = depth.slice();
  for (let i = 0; i < result.length; i++) {
    if (closed[i] && !mask[i] && blurredWeights[i] > 1e-5)
      result[i] = blurredDepth[i] / blurredWeights[i];
    mask[i] = closed[i];
  }
  return result;
}

function suppressSpikes(depth, mask, width, height) {
  const result = depth.slice(), values = new Float32Array(9), deviations = new Float32Array(9);
  for (let y = 1; y < height - 1; y++) for (let x = 1; x < width - 1; x++) {
    const index = y * width + x;
    if (!mask[index]) continue;
    let count = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const neighbour = (y + dy) * width + x + dx;
      if (mask[neighbour]) values[count++] = depth[neighbour];
    }
    if (count < 5) continue;
    for (let i = 1; i < count; i++) { const value = values[i]; let j = i - 1; while (j >= 0 && values[j] > value) { values[j + 1] = values[j]; j--; } values[j + 1] = value; }
    const median = values[count >> 1];
    for (let i = 0; i < count; i++) deviations[i] = Math.abs(values[i] - median);
    for (let i = 1; i < count; i++) { const value = deviations[i]; let j = i - 1; while (j >= 0 && deviations[j] > value) { deviations[j + 1] = deviations[j]; j--; } deviations[j + 1] = value; }
    const mad = deviations[count >> 1];
    if (Math.abs(depth[index] - median) > .012 + mad * 3.2)
      result[index] = depth[index] * .15 + median * .85;
  }
  return result;
}

function guidedMedian(depth, mask, gray, width, height, radius) {
  const step = Math.max(1, Math.ceil(radius / 3));
  const side = Math.floor(radius * 2 / step) + 1;
  const values = new Float32Array(side * side), weights = new Float32Array(side * side);
  const result = depth.slice();
  for (let y = radius; y < height - radius; y++) for (let x = radius; x < width - radius; x++) {
    const index = y * width + x;
    if (!mask[index]) continue;
    let count = 0, total = 0;
    for (let dy = -radius; dy <= radius; dy += step) for (let dx = -radius; dx <= radius; dx += step) {
      const neighbour = (y + dy) * width + x + dx;
      if (!mask[neighbour] || depth[neighbour] <= 0) continue;
      const spatial = 1 / (1 + dx * dx + dy * dy);
      const edge = Math.exp(-Math.abs(gray[neighbour] - gray[index]) / 20);
      values[count] = depth[neighbour];
      weights[count] = spatial * edge;
      total += weights[count++];
    }
    for (let i = 1; i < count; i++) {
      const value = values[i], weight = weights[i]; let j = i - 1;
      while (j >= 0 && values[j] > value) { values[j + 1] = values[j]; weights[j + 1] = weights[j]; j--; }
      values[j + 1] = value; weights[j + 1] = weight;
    }
    let accumulated = 0;
    for (let i = 0; i < count; i++) {
      accumulated += weights[i];
      if (accumulated >= total * .5) { result[index] = values[i]; break; }
    }
  }
  return result;
}

onmessage = event => {
  const {id, rgbaBuffer, w, h, period: manualPeriod} = event.data;
  const rgba = new Uint8ClampedArray(rgbaBuffer), count = w * h, gray = toGray(rgba, count);
  const backgroundPeriod = manualPeriod || estimatePeriod(gray, w, h);
  const minPeriod = Math.max(1, backgroundPeriod - 30), maxPeriod = Math.min(w - 1, backgroundPeriod + 5);
  const periods = Array.from({length: maxPeriod - minPeriod + 1}, (_, i) => minPeriod + i);
  const bestCost = new Float32Array(count); bestCost.fill(INF);
  const secondCost = new Float32Array(count); secondCost.fill(INF);
  let periodMap = new Float32Array(count); periodMap.fill(backgroundPeriod);
  let previous = matchingCost(gray, w, h, periods[0]);
  evaluate(previous, previous, previous, periods[0], bestCost, secondCost, periodMap, false);
  if (periods.length > 1) {
    let current = matchingCost(gray, w, h, periods[1]);
    for (let index = 1; index < periods.length - 1; index++) {
      const next = matchingCost(gray, w, h, periods[index + 1]);
      evaluate(previous, current, next, periods[index], bestCost, secondCost, periodMap, true);
      previous = current; current = next;
      report(id, .08 + .55 * (index + 1) / periods.length, 'Сопоставление узора…');
    }
    evaluate(current, current, current, periods[periods.length - 1], bestCost, secondCost, periodMap, false);
  }
  periodMap = refineAmbiguousPeriods(periodMap, bestCost, secondCost, gray, w, h);
  const margin = (maxPeriod >> 1) + 3;
  for (let y = 0; y < h; y++) for (let x = 0; x < Math.min(margin, w); x++) {
    periodMap[y * w + x] = backgroundPeriod;
    periodMap[y * w + w - 1 - x] = backgroundPeriod;
  }
  const filtered = median3(periodMap, w, h), maxDepth = Math.max(1, backgroundPeriod - minPeriod);
  const relative = new Float32Array(count), mask = new Uint8Array(count);
  for (let i = 0; i < count; i++) { relative[i] = clamp(backgroundPeriod - filtered[i], 0, maxDepth); mask[i] = relative[i] >= 2 ? 1 : 0; }
  largestComponent(mask, w, h);
  const scale = Math.max(w, h) / 512;
  const fillRadius = Math.min(16, Math.max(2, Math.round(8 * scale)));
  const medianRadius = Math.min(16, Math.max(2, Math.round(8 * scale)));
  let cleaned = smoothInside(relative, mask, w, h);
  cleaned = fillSmallHoles(cleaned, mask, w, h, fillRadius);
  cleaned = suppressSpikes(cleaned, mask, w, h);
  cleaned = guidedMedian(cleaned, mask, gray, w, h, medianRadius);
  const rawDepth = new Float32Array(count);
  for (let i = 0; i < count; i++) rawDepth[i] = clamp(cleaned[i] / maxDepth, 0, 1);
  postMessage({type: 'done', id, detectedPeriod: backgroundPeriod, rawDepth: rawDepth.buffer}, [rawDepth.buffer]);
};
