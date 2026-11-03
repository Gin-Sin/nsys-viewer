export interface KernelEvent {
  startMs: number;
  endMs: number;
  deviceId: number;
  streamId: number;
  nameId: number;
  overlapLane?: number;
}

export interface Stage {
  deviceId: number;
  pid: number | null;
  kernelCount: number;
  streamIds: number[];
}

export interface TraceData {
  schemaVersion: string;
  durationMs: number;
  stages: Stage[];
  names: Record<number, string>;
  kernels: KernelEvent[];
}

export type WorkerResponse =
  | { type: "loaded"; trace: TraceData }
  | { type: "progress"; message: string }
  | { type: "error"; message: string };
