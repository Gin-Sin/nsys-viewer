export interface KernelEvent {
  startMs: number;
  endMs: number;
  deviceId: number;
  streamId: number;
  nameId: number;
  overlapLane?: number;
  /** Index into TraceData.configs (PyTorch traces only). */
  configId?: number;
  /** Index into TraceData.ops: the CPU ops that launched the kernel. */
  opId?: number;
}

export interface KernelConfig {
  grid?: [number, number, number];
  block?: [number, number, number];
  registers?: number;
  sharedMemory?: number;
  occupancy?: number;
}

export interface Stage {
  deviceId: number;
  pid: number | null;
  kernelCount: number;
  streamIds: number[];
}

export interface TraceData {
  format: "nsys" | "torch";
  schemaVersion: string;
  durationMs: number;
  stages: Stage[];
  names: Record<number, string>;
  kernels: KernelEvent[];
  ops?: string[];
  configs?: KernelConfig[];
}

export type WorkerResponse =
  | { type: "loaded"; trace: TraceData }
  | { type: "progress"; message: string }
  | { type: "error"; message: string };
