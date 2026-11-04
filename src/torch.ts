import type { KernelConfig, KernelEvent, Stage, TraceData } from "./types";
import { firstNonSpace } from "./unpack.ts";

// PyTorch profiler (Kineto) traces are Chrome-trace JSON. They are often
// hundreds of MB, so instead of JSON.parse on the whole file we walk the
// traceEvents array byte by byte and only parse the events we need.

const QUOTE = 0x22;
const BACKSLASH = 0x5c;
const OPEN_BRACE = 0x7b;
const CLOSE_BRACE = 0x7d;
const OPEN_BRACKET = 0x5b;
const CLOSE_BRACKET = 0x5d;

const KERNEL_CATS = new Set(["kernel"]);
const LAUNCH_CATS = new Set(["cuda_runtime", "cuda_driver", "runtime", "driver"]);
const OP_CATS = new Set(["cpu_op", "operator", "user_annotation"]);
// Byte patterns that must appear in an event worth parsing.
const WANTED = ["kernel", "Kernel", "runtime", "Runtime", "driver", "Driver", "cpu_op", "operator", "Operator", "user_annotation"]
  .map((text) => new TextEncoder().encode(text));

function indexOf(bytes: Uint8Array, pattern: Uint8Array, from = 0, to = bytes.length): number {
  const window = bytes.subarray(from, to);
  const first = pattern[0];
  const last = window.length - pattern.length;
  for (let i = window.indexOf(first); i !== -1 && i <= last; i = window.indexOf(first, i + 1)) {
    let match = true;
    for (let j = 1; j < pattern.length; j++) {
      if (window[i + j] !== pattern[j]) {
        match = false;
        break;
      }
    }
    if (match) return from + i;
  }
  return -1;
}

/** Returns the index just past the value (object/array) starting at `start`. */
function skipContainer(bytes: Uint8Array, start: number): number {
  let depth = 0;
  for (let i = start; i < bytes.length; i++) {
    const byte = bytes[i];
    if (byte === QUOTE) {
      for (i++; i < bytes.length && bytes[i] !== QUOTE; i++) {
        if (bytes[i] === BACKSLASH) i++;
      }
    } else if (byte === OPEN_BRACE || byte === OPEN_BRACKET) {
      depth++;
    } else if (byte === CLOSE_BRACE || byte === CLOSE_BRACKET) {
      depth--;
      if (depth === 0) return i + 1;
    }
  }
  throw new Error("The trace JSON is truncated.");
}

/** Finds the `[` that starts the event list. */
function eventsStart(bytes: Uint8Array): number {
  const start = firstNonSpace(bytes);
  if (bytes[start] === OPEN_BRACKET) return start; // Bare array form.
  const key = indexOf(bytes, new TextEncoder().encode('"traceEvents"'));
  if (key < 0) throw new Error("This JSON file has no traceEvents. Open a PyTorch profiler trace (export_chrome_trace).");
  let index = firstNonSpace(bytes, key + 13);
  if (bytes[index] !== 0x3a) throw new Error("Could not read traceEvents in this JSON file.");
  index = firstNonSpace(bytes, index + 1);
  if (bytes[index] !== OPEN_BRACKET) throw new Error("traceEvents is not an array.");
  return index;
}

interface RawEvent {
  ph?: string;
  cat?: string;
  name?: string;
  pid?: number | string;
  tid?: number | string;
  ts?: number | string;
  dur?: number | string;
  args?: Record<string, unknown>;
}

/** Calls `visit` for each event in traceEvents that might be useful. */
function forEachEvent(
  bytes: Uint8Array,
  visit: (event: RawEvent) => void,
  progress: (fraction: number) => void,
): void {
  const decoder = new TextDecoder();
  let index = eventsStart(bytes) + 1;
  let nextReport = 0;
  for (;;) {
    index = firstNonSpace(bytes, index);
    while (bytes[index] === 0x2c) index = firstNonSpace(bytes, index + 1);
    if (index >= bytes.length) throw new Error("The trace JSON is truncated.");
    if (bytes[index] === CLOSE_BRACKET) return;
    if (bytes[index] !== OPEN_BRACE) throw new Error("Unexpected content in traceEvents.");
    const end = skipContainer(bytes, index);
    if (WANTED.some((pattern) => indexOf(bytes, pattern, index, end) !== -1)) {
      visit(JSON.parse(decoder.decode(bytes.subarray(index, end))) as RawEvent);
    }
    index = end;
    if (index >= nextReport) {
      progress(index / bytes.length);
      nextReport = index + 32 * 1024 * 1024;
    }
  }
}

function num(value: unknown): number | null {
  const n = typeof value === "string" ? Number(value) : value;
  return typeof n === "number" && Number.isFinite(n) ? n : null;
}

function dims(value: unknown): [number, number, number] | undefined {
  if (!Array.isArray(value)) return undefined;
  const [x = 1, y = 1, z = 1] = value.map(Number);
  return [x, y, z];
}

interface Launch {
  ts: number;
  thread: string;
  pid: number | null;
  opId?: number;
}

interface Op {
  ts: number;
  end: number;
  name: string;
}

// Innermost ops shown for a launch, outermost first.
const STACK_DEPTH = 4;

export function readTorchTrace(
  bytes: Uint8Array,
  progress: (message: string) => void = () => {},
): TraceData {
  const names: Record<number, string> = {};
  const nameIds = new Map<string, number>();
  const configs: KernelConfig[] = [];
  const configIds = new Map<string, number>();
  const kernels: (KernelEvent & { correlation: number | null; tsUs: number; endUs: number })[] = [];
  const launches = new Map<number, Launch>();
  const opsByThread = new Map<string, Op[]>();

  forEachEvent(bytes, (event) => {
    if (event.ph !== "X") return;
    const cat = String(event.cat ?? "").toLowerCase();
    const ts = num(event.ts);
    const dur = num(event.dur) ?? 0;
    if (ts == null) return;
    const args = event.args ?? {};
    if (KERNEL_CATS.has(cat)) {
      const name = String(event.name ?? "Unknown kernel");
      let nameId = nameIds.get(name);
      if (nameId == null) {
        nameId = nameIds.size;
        nameIds.set(name, nameId);
        names[nameId] = name;
      }
      const config: KernelConfig = {
        grid: dims(args.grid),
        block: dims(args.block),
        registers: num(args["registers per thread"]) ?? undefined,
        sharedMemory: num(args["shared memory"]) ?? undefined,
        occupancy: num(args["est. achieved occupancy %"]) ?? undefined,
      };
      let configId: number | undefined;
      if (Object.values(config).some((value) => value != null)) {
        const key = JSON.stringify(config);
        configId = configIds.get(key);
        if (configId == null) {
          configId = configs.length;
          configs.push(config);
          configIds.set(key, configId);
        }
      }
      kernels.push({
        startMs: 0,
        endMs: 0,
        tsUs: ts,
        endUs: ts + dur,
        deviceId: num(args.device) ?? num(event.pid) ?? 0,
        streamId: num(args.stream) ?? num(event.tid) ?? 0,
        nameId,
        configId,
        correlation: num(args.correlation),
      });
    } else if (LAUNCH_CATS.has(cat)) {
      const correlation = num(args.correlation);
      if (correlation == null) return;
      launches.set(correlation, { ts, thread: `${event.pid}:${event.tid}`, pid: num(event.pid) });
    } else if (OP_CATS.has(cat)) {
      const thread = `${event.pid}:${event.tid}`;
      let ops = opsByThread.get(thread);
      if (!ops) opsByThread.set(thread, ops = []);
      ops.push({ ts, end: ts + dur, name: String(event.name ?? "") });
    }
  }, (fraction) => progress(`Reading trace… ${Math.round(fraction * 100)}%`));

  if (!kernels.length) {
    throw new Error("This trace has no GPU kernels. Profile with ProfilerActivity.CUDA enabled.");
  }

  // Find the CPU ops that were running when each kernel was launched.
  const ops: string[] = [];
  const opIds = new Map<string, number>();
  const launchesByThread = new Map<string, Launch[]>();
  for (const launch of launches.values()) {
    let list = launchesByThread.get(launch.thread);
    if (!list) launchesByThread.set(launch.thread, list = []);
    list.push(launch);
  }
  for (const [thread, list] of launchesByThread) {
    const threadOps = (opsByThread.get(thread) ?? []).sort((a, b) => a.ts - b.ts || b.end - a.end);
    list.sort((a, b) => a.ts - b.ts);
    // Ops that are still open, outermost first. Dropping every finished op
    // (not just the top) keeps this small even when ops overlap untidily.
    let open: Op[] = [];
    let next = 0;
    for (const launch of list) {
      while (next < threadOps.length && threadOps[next].ts <= launch.ts) {
        const op = threadOps[next++];
        open = open.filter((outer) => outer.end >= op.ts);
        open.push(op);
      }
      open = open.filter((op) => op.end >= launch.ts);
      if (!open.length) continue;
      const text = open.slice(-STACK_DEPTH).map((op) => op.name).join(" › ");
      let opId = opIds.get(text);
      if (opId == null) {
        opId = ops.length;
        ops.push(text);
        opIds.set(text, opId);
      }
      launch.opId = opId;
    }
  }

  let startUs = Infinity;
  let endUs = -Infinity;
  for (const kernel of kernels) {
    startUs = Math.min(startUs, kernel.tsUs);
    endUs = Math.max(endUs, kernel.endUs);
  }
  const counts = new Map<number, number>();
  const streams = new Map<number, Set<number>>();
  const pidByDevice = new Map<number, number>();
  const result: KernelEvent[] = kernels
    .sort((a, b) => a.tsUs - b.tsUs)
    .map((kernel) => {
      const launch = kernel.correlation == null ? undefined : launches.get(kernel.correlation);
      const device = kernel.deviceId;
      counts.set(device, (counts.get(device) ?? 0) + 1);
      if (!streams.has(device)) streams.set(device, new Set());
      streams.get(device)?.add(kernel.streamId);
      if (launch?.pid != null && !pidByDevice.has(device)) pidByDevice.set(device, launch.pid);
      const event: KernelEvent = {
        startMs: (kernel.tsUs - startUs) / 1e3,
        endMs: (kernel.endUs - startUs) / 1e3,
        deviceId: device,
        streamId: kernel.streamId,
        nameId: kernel.nameId,
      };
      if (kernel.configId != null) event.configId = kernel.configId;
      if (launch?.opId != null) event.opId = launch.opId;
      return event;
    });

  const stages: Stage[] = [...counts.entries()]
    .map(([deviceId, kernelCount]) => ({
      deviceId,
      pid: pidByDevice.get(deviceId) ?? null,
      kernelCount,
      streamIds: [...(streams.get(deviceId) ?? [])].sort((a, b) => a - b),
    }))
    .sort((a, b) => a.deviceId - b.deviceId);

  return {
    format: "torch",
    schemaVersion: "PyTorch profiler",
    durationMs: Math.max((endUs - startUs) / 1e3, 1e-6),
    stages,
    names,
    kernels: result,
    ops,
    configs,
  };
}
