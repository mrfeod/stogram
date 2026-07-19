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
  return {device, censusPipeline, costPipeline};
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
  const {device, censusPipeline, costPipeline} = state;
  const minDisp = -Math.max(2, Math.floor(period * .08));
  const maxDisp = Math.max(4, Math.min(Math.floor(period * .34), 42));
  const dispCount = maxDisp - minDisp + 1;
  const xStart = period - minDisp + 2;
  const xEnd = w - 3;
  const validW = Math.max(1, xEnd - xStart);
  const pixels = validW * h;
  const wordsPerPixel = Math.ceil(dispCount / 4);
  const packedByteLength = pixels * wordsPerPixel * 4;
  const U = GPUBufferUsage;
  const paramsBuffer = storageBuffer(device, 32, U.UNIFORM | U.COPY_DST);
  const grayBuffer = storageBuffer(device, gray.byteLength, U.STORAGE | U.COPY_DST);
  const censusBuffer = storageBuffer(device, w * h * 4, U.STORAGE);
  const costsBuffer = storageBuffer(
      device, packedByteLength, U.STORAGE | U.COPY_SRC);
  const readBuffer = storageBuffer(
      device, packedByteLength, U.COPY_DST | U.MAP_READ);
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
  encoder.copyBufferToBuffer(costsBuffer, 0, readBuffer, 0, packedByteLength);
  device.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ);
  const packedCosts = readBuffer.getMappedRange().slice(0);
  readBuffer.unmap();
  const processingMs = performance.now() - startedAt;
  for (const buffer of
       [paramsBuffer, grayBuffer, censusBuffer, costsBuffer, readBuffer])
    buffer.destroy();
  return {packedCosts, wordsPerPixel, processingMs};
}
