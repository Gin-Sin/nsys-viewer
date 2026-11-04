import assert from "node:assert/strict";
import test from "node:test";
import { gzipSync } from "fflate";
import { readTorchTrace } from "../src/torch.ts";
import { isTraceJson, unpack } from "../src/unpack.ts";

// Shaped like torch.profiler's export_chrome_trace output (Kineto).
const events = [
  { ph: "X", cat: "user_annotation", name: "## forward ##", pid: 4242, tid: 4242, ts: 1000, dur: 500 },
  { ph: "X", cat: "cpu_op", name: "aten::linear", pid: 4242, tid: 4242, ts: 1010, dur: 100 },
  { ph: "X", cat: "cpu_op", name: "aten::addmm", pid: 4242, tid: 4242, ts: 1020, dur: 80 },
  { ph: "X", cat: "cuda_runtime", name: "cudaLaunchKernel", pid: 4242, tid: 4242, ts: 1030, dur: 5, args: { correlation: 7 } },
  { ph: "X", cat: "cpu_op", name: "aten::softmax", pid: 4242, tid: 4242, ts: 1200, dur: 30 },
  { ph: "X", cat: "cuda_runtime", name: "cudaLaunchKernel", pid: 4242, tid: 4242, ts: 1210, dur: 5, args: { correlation: 8 } },
  { ph: "X", cat: "python_function", name: "nn.Module: Linear_0", pid: 4242, tid: 4242, ts: 1005, dur: 120 },
  {
    ph: "X", cat: "kernel", name: "sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", pid: 0, tid: 7,
    ts: 1040, dur: 50.5,
    args: { device: 0, stream: 7, correlation: 7, grid: [132, 1, 1], block: [384, 1, 1], "registers per thread": 168, "shared memory": 230400, "est. achieved occupancy %": 13 },
  },
  {
    ph: "X", cat: "kernel", name: "void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float, float, float, at::native::(anonymous namespace)::SoftMaxForwardEpilogue>(float*, float const*, int) {\"quoted\"}", pid: 0, tid: 7,
    ts: 1220, dur: 4,
    args: { device: 0, stream: 7, correlation: 8 },
  },
  { ph: "X", cat: "kernel", name: "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)", pid: 1, tid: 20, ts: 1100, dur: 30, args: { device: 1, stream: 20 } },
  { ph: "X", cat: "gpu_memcpy", name: "Memcpy HtoD (Pageable -> Device)", pid: 0, tid: 7, ts: 900, dur: 10, args: { device: 0, stream: 7 } },
  { ph: "f", id: 7, pid: 0, tid: 7, ts: 1040, cat: "ac2g", name: "ac2g", bp: "e" },
  { ph: "M", name: "process_name", pid: 0, args: { name: "GPU 0" } },
];

const encode = (value) => new TextEncoder().encode(JSON.stringify(value, null, 1));
const trace = encode({ schemaVersion: 1, deviceProperties: [{ id: 0, name: "NVIDIA H100" }], traceEvents: events, traceName: "x" });

test("detects JSON traces", () => {
  assert.ok(isTraceJson(trace));
  assert.ok(isTraceJson(new TextEncoder().encode("﻿  [{}]")));
  assert.ok(!isTraceJson(new TextEncoder().encode("SQLite format 3\0")));
});

test("reads kernels, devices, streams and timing", () => {
  const data = readTorchTrace(trace);
  assert.equal(data.format, "torch");
  assert.equal(data.kernels.length, 3);
  assert.deepEqual(data.stages.map((s) => [s.deviceId, s.kernelCount, s.streamIds, s.pid]), [
    [0, 2, [7], 4242],
    [1, 1, [20], null],
  ]);
  const [gemm, nccl, softmax] = data.kernels;
  assert.equal(gemm.startMs, 0);
  assert.ok(Math.abs(gemm.endMs - 0.0505) < 1e-9);
  assert.ok(Math.abs(nccl.startMs - 0.06) < 1e-9);
  assert.ok(Math.abs(data.durationMs - 0.184) < 1e-9);
  assert.match(data.names[softmax.nameId], /cunn_SoftMaxForward.*\{"quoted"\}$/);
});

test("links kernels to the CPU ops that launched them", () => {
  const data = readTorchTrace(trace);
  const [gemm, nccl, softmax] = data.kernels;
  assert.equal(data.ops[gemm.opId], "## forward ## › aten::linear › aten::addmm");
  assert.equal(data.ops[softmax.opId], "## forward ## › aten::softmax");
  assert.equal(nccl.opId, undefined);
  assert.deepEqual(data.configs[gemm.configId], {
    grid: [132, 1, 1], block: [384, 1, 1], registers: 168, sharedMemory: 230400, occupancy: 13,
  });
  assert.equal(softmax.configId, undefined);
});

test("reads the bare-array form and compressed files", async () => {
  const bare = readTorchTrace(encode(events));
  assert.equal(bare.kernels.length, 3);
  const unpacked = await unpack(gzipSync(trace), "rank0.pt.trace.json.gz");
  assert.equal(readTorchTrace(unpacked).kernels.length, 3);
});

test("explains traces without kernels or traceEvents", () => {
  assert.throws(() => readTorchTrace(encode({ traceEvents: events.filter((e) => e.cat !== "kernel") })), /no GPU kernels/);
  assert.throws(() => readTorchTrace(encode({ foo: 1 })), /no traceEvents/);
  assert.throws(() => readTorchTrace(trace.subarray(0, trace.length / 2)), /truncated/);
});
