let gpuStatePromise = null;

const GPU_SHADER = /* wgsl */ `
struct Params {
  w: i32,
  h: i32,
  period: i32,
  minDisp: i32,
  dispCount: i32,
  xStart: i32,
  validW: i32,
  wordsPerPixel: i32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> census: array<u32>;
@group(0) @binding(3) var<storage, read_write> packedCosts: array<u32>;
@group(0) @binding(4) var<storage, read_write> aggregate: array<u32>;
@group(0) @binding(5) var<uniform> direction: vec4<i32>;
@group(0) @binding(6) var<storage, read_write> packedAggregate: array<u32>;

var<workgroup> pathPrev: array<f32, 96>;
var<workgroup> pathCur: array<f32, 96>;
var<workgroup> pathMin: f32;

fn costAt(pi: i32, di: i32) -> u32 {
  let word = packedCosts[pi * params.wordsPerPixel + di / 4];
  return (word >> u32((di % 4) * 8)) & 255u;
}

@compute @workgroup_size(8, 8)
fn makeCensus(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= params.w || y >= params.h) { return; }
  let index = y * params.w + x;
  if (x < 2 || x >= params.w - 2 || y < 2 || y >= params.h - 2) {
    census[index] = 0u;
    return;
  }
  let center = gray[index];
  var bits = 0u;
  var bit = 0u;
  for (var yy = -2; yy <= 2; yy++) {
    for (var xx = -2; xx <= 2; xx++) {
      if (xx != 0 || yy != 0) {
        if (gray[(y + yy) * params.w + x + xx] < center) {
          bits = bits | (1u << bit);
        }
        bit++;
      }
    }
  }
  census[index] = bits;
}

@compute @workgroup_size(64)
fn makeCosts(@builtin(global_invocation_id) gid: vec3<u32>) {
  let wordIndex = i32(gid.x);
  let pixelCount = params.validW * params.h;
  let totalWords = pixelCount * params.wordsPerPixel;
  if (wordIndex >= totalWords) { return; }
  let pi = wordIndex / params.wordsPerPixel;
  let wordInPixel = wordIndex - pi * params.wordsPerPixel;
  let y = pi / params.validW;
  let vx = pi - y * params.validW;
  let x = params.xStart + vx;
  let imageIndex = y * params.w + x;
  let a = census[imageIndex];
  var packed = 0u;
  for (var lane = 0; lane < 4; lane++) {
    let di = wordInPixel * 4 + lane;
    if (di < params.dispCount) {
      let disparity = params.minDisp + di;
      let qx = x - (params.period - disparity);
      let otherIndex = y * params.w + qx;
      let censusCost = countOneBits(a ^ census[otherIndex]);
      let intensityCost = min(8.0, abs(gray[imageIndex] - gray[otherIndex]) / 12.0);
      let cost = u32(min(31.0, floor(f32(censusCost) + intensityCost + 0.5)));
      packed = packed | (cost << u32(lane * 8));
    }
  }
  packedCosts[wordIndex] = packed;
}

@compute @workgroup_size(96)
fn aggregatePath(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>) {
  let path = i32(group.x);
  let di = i32(local.x);
  let dir = direction.x;
  let horizontal = dir < 2;
  let length = select(params.h, params.validW, horizontal);
  if ((!horizontal && path >= params.validW) ||
      (horizontal && path >= params.h)) { return; }
  for (var step = 0; step < length; step++) {
    var vx = path;
    var y = step;
    if (horizontal) {
      vx = select(params.validW - 1 - step, step, dir == 0);
      y = path;
    } else {
      y = select(params.h - 1 - step, step, dir == 2);
    }
    let pi = y * params.validW + vx;
    let off = pi * params.dispCount;
    if (step > 0) {
      if (di == 0) {
        var minimum = pathPrev[0];
        for (var candidate = 1; candidate < params.dispCount; candidate++) {
          minimum = min(minimum, pathPrev[candidate]);
        }
        pathMin = minimum;
      }
      workgroupBarrier();
    }
    if (di < params.dispCount) {
      let baseCost = f32(costAt(pi, di));
      if (step == 0) {
        pathCur[di] = baseCost;
      } else {
        var best = pathPrev[di];
        if (di > 0) { best = min(best, pathPrev[di - 1] + 2.2); }
        if (di + 1 < params.dispCount) {
          best = min(best, pathPrev[di + 1] + 2.2);
        }
        var prevVx = vx;
        var prevY = y;
        if (horizontal) {
          prevVx = select(vx + 1, vx - 1, dir == 0);
        } else {
          prevY = select(y + 1, y - 1, dir == 2);
        }
        let imageIndex = y * params.w + params.xStart + vx;
        let previousIndex = prevY * params.w + params.xStart + prevVx;
        let edge = abs(gray[imageIndex] - gray[previousIndex]);
        let p2 = max(5.0, 18.0 - edge * 0.20);
        best = min(best, pathMin + p2);
        pathCur[di] = baseCost + best - pathMin;
      }
    }
    workgroupBarrier();
    if (di < params.dispCount) {
      let addition = u32(floor(pathCur[di] + 0.5));
      aggregate[off + di] = min(65535u, aggregate[off + di] + addition);
      pathPrev[di] = pathCur[di];
    }
    workgroupBarrier();
  }
}

@compute @workgroup_size(64)
fn packAggregate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let wordIndex = i32(gid.x);
  let valueCount = params.validW * params.h * params.dispCount;
  let first = wordIndex * 2;
  if (first >= valueCount) { return; }
  let low = min(65535u, aggregate[first]);
  var high = 0u;
  if (first + 1 < valueCount) { high = min(65535u, aggregate[first + 1]); }
  packedAggregate[wordIndex] = low | (high << 16u);
}
`;

async function createGpuState() {
  if (!navigator.gpu) return null;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({code: GPU_SHADER});
  const censusPipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {module, entryPoint: 'makeCensus'},
  });
  const costPipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {module, entryPoint: 'makeCosts'},
  });
  const sgmPipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {module, entryPoint: 'aggregatePath'},
  });
  const packPipeline = await device.createComputePipelineAsync({
    layout: 'auto',
    compute: {module, entryPoint: 'packAggregate'},
  });
  return {device, censusPipeline, costPipeline, sgmPipeline, packPipeline};
}

function gpuState() {
  if (!gpuStatePromise) gpuStatePromise = createGpuState();
  return gpuStatePromise;
}

function storageBuffer(device, size, usage) {
  return device.createBuffer({size: Math.max(4, size), usage});
}

export async function computeCostVolumeGpu(gray, w, h, period) {
  const state = await gpuState();
  if (!state) return null;
  const {device, censusPipeline, costPipeline, sgmPipeline, packPipeline} = state;
  const minDisp = -Math.max(2, Math.floor(period * .08));
  const maxDisp = Math.max(4, Math.min(Math.floor(period * .34), 42));
  const dispCount = maxDisp - minDisp + 1;
  const xStart = period - minDisp + 2;
  const xEnd = w - 3;
  const validW = Math.max(1, xEnd - xStart);
  const pixels = validW * h;
  const wordsPerPixel = Math.ceil(dispCount / 4);
  const packedByteLength = pixels * wordsPerPixel * 4;
  const aggregateValues = pixels * dispCount;
  const packedAggregateByteLength = Math.ceil(aggregateValues / 2) * 4;
  const U = GPUBufferUsage;
  const paramsBuffer = storageBuffer(device, 32, U.UNIFORM | U.COPY_DST);
  const grayBuffer = storageBuffer(device, gray.byteLength, U.STORAGE | U.COPY_DST);
  const censusBuffer = storageBuffer(device, w * h * 4, U.STORAGE);
  const costsBuffer = storageBuffer(
      device, packedByteLength, U.STORAGE | U.COPY_SRC);
  const aggregateBuffer = storageBuffer(device, aggregateValues * 4, U.STORAGE);
  const packedAggregateBuffer = storageBuffer(
      device, packedAggregateByteLength, U.STORAGE | U.COPY_SRC);
  const readBuffer = storageBuffer(
      device, packedByteLength, U.COPY_DST | U.MAP_READ);
  const aggregateReadBuffer = storageBuffer(
      device, packedAggregateByteLength, U.COPY_DST | U.MAP_READ);
  const params = new Int32Array([
    w, h, period, minDisp, dispCount, xStart, validW, wordsPerPixel,
  ]);
  device.queue.writeBuffer(paramsBuffer, 0, params);
  device.queue.writeBuffer(grayBuffer, 0, gray);
  const censusEntries = [
    {binding: 0, resource: {buffer: paramsBuffer}},
    {binding: 1, resource: {buffer: grayBuffer}},
    {binding: 2, resource: {buffer: censusBuffer}},
  ];
  const costEntries = [
    ...censusEntries,
    {binding: 3, resource: {buffer: costsBuffer}},
  ];
  const directionBuffers = [0, 1, 2, 3].map(value => {
    const buffer = storageBuffer(device, 16, U.UNIFORM | U.COPY_DST);
    device.queue.writeBuffer(buffer, 0, new Int32Array([value, 0, 0, 0]));
    return buffer;
  });
  const startedAt = performance.now();
  const encoder = device.createCommandEncoder();
  let pass = encoder.beginComputePass();
  pass.setPipeline(censusPipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: censusPipeline.getBindGroupLayout(0), entries: censusEntries,
  }));
  pass.dispatchWorkgroups(Math.ceil(w / 8), Math.ceil(h / 8));
  pass.end();
  pass = encoder.beginComputePass();
  pass.setPipeline(costPipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: costPipeline.getBindGroupLayout(0), entries: costEntries,
  }));
  pass.dispatchWorkgroups(Math.ceil(pixels * wordsPerPixel / 64));
  pass.end();
  if (dispCount <= 96) {
    for (let dir = 0; dir < 4; dir++) {
      const sgmEntries = [
        {binding: 0, resource: {buffer: paramsBuffer}},
        {binding: 1, resource: {buffer: grayBuffer}},
        {binding: 3, resource: {buffer: costsBuffer}},
        {binding: 4, resource: {buffer: aggregateBuffer}},
        {binding: 5, resource: {buffer: directionBuffers[dir]}},
      ];
      pass = encoder.beginComputePass();
      pass.setPipeline(sgmPipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: sgmPipeline.getBindGroupLayout(0), entries: sgmEntries,
      }));
      pass.dispatchWorkgroups(dir < 2 ? h : validW);
      pass.end();
    }
    const packEntries = [
      {binding: 0, resource: {buffer: paramsBuffer}},
      {binding: 4, resource: {buffer: aggregateBuffer}},
      {binding: 6, resource: {buffer: packedAggregateBuffer}},
    ];
    pass = encoder.beginComputePass();
    pass.setPipeline(packPipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: packPipeline.getBindGroupLayout(0), entries: packEntries,
    }));
    pass.dispatchWorkgroups(Math.ceil(Math.ceil(aggregateValues / 2) / 64));
    pass.end();
    encoder.copyBufferToBuffer(
        packedAggregateBuffer, 0, aggregateReadBuffer, 0,
        packedAggregateByteLength);
  }
  encoder.copyBufferToBuffer(costsBuffer, 0, readBuffer, 0, packedByteLength);
  device.queue.submit([encoder.finish()]);
  await Promise.all([
    readBuffer.mapAsync(GPUMapMode.READ),
    dispCount <= 96 ? aggregateReadBuffer.mapAsync(GPUMapMode.READ) :
                      Promise.resolve(),
  ]);
  const packedCosts = readBuffer.getMappedRange().slice(0);
  const packedAggregate = dispCount <= 96 ?
      aggregateReadBuffer.getMappedRange().slice(0) : null;
  readBuffer.unmap();
  if (dispCount <= 96) aggregateReadBuffer.unmap();
  const processingMs = performance.now() - startedAt;
  for (const buffer of
       [paramsBuffer, grayBuffer, censusBuffer, costsBuffer, aggregateBuffer,
        packedAggregateBuffer, readBuffer, aggregateReadBuffer,
        ...directionBuffers])
    buffer.destroy();
  return {packedCosts, packedAggregate, wordsPerPixel, processingMs};
}
