// Plain-language descriptions of common CUDA kernels, derived only from the
// kernel name. These are heuristics: names follow library conventions, not a
// guaranteed schema, so unrecognized kernels fall back to a readable name.

export interface KernelExplanation {
  /** Function name without return type, namespaces, templates, or arguments. */
  short: string;
  category: string;
  summary: string;
  details: string[];
  tags: string[];
}

interface Rule {
  test: RegExp;
  category: string;
  summary: string | ((name: string) => string);
  details?: (name: string) => string[];
}

// Wrapper kernels whose real identity is their first template argument.
const GENERIC_WRAPPERS = new Set(["device_kernel", "kernel", "Kernel", "run_kernel", "launch_kernel"]);

/** Strips "void ns::foo<...>(...)" down to "foo". */
export function shortKernelName(name: string): string {
  const clean = name.replace(/\(anonymous namespace\)/g, "anon");
  let end = clean.length;
  for (let i = 0; i < clean.length; i++) {
    if (clean[i] === "<" || clean[i] === "(") {
      end = i;
      break;
    }
  }
  const head = clean.slice(0, end).trim().split(/\s+/).at(-1) ?? clean;
  const short = head.split("::").at(-1) || name;
  if (GENERIC_WRAPPERS.has(short) && clean[end] === "<") {
    const inner = shortKernelName(clean.slice(end + 1));
    if (inner && !/^\d/.test(inner)) return inner;
  }
  return short;
}

const ARCHES: [RegExp, string][] = [
  [/sm_?120|sm_?100|blackwell/, "Blackwell (B200/GB200)"],
  [/sm_?90|hopper/, "Hopper (H100/H200)"],
  [/sm_?89|\bada\b/, "Ada (L40/RTX 40)"],
  [/sm_?8[06]|ampere/, "Ampere (A100/A10)"],
  [/sm_?75|turing/, "Turing (T4)"],
  [/sm_?70|volta/, "Volta (V100)"],
];

function archOf(lower: string): string | null {
  return ARCHES.find(([pattern]) => pattern.test(lower))?.[1] ?? null;
}

function commonTags(lower: string): string[] {
  const tags: string[] = [];
  const token = (pattern: string) => new RegExp(`(^|[^a-z0-9])(${pattern})($|[^a-z0-9])`).test(lower);
  if (/bf16|bfloat16/.test(lower)) tags.push("BF16");
  if (/fp16|hgemm/.test(lower) || token("f16|half|__half|c10::half")) tags.push("FP16");
  if (token("fp8|e4m3|e5m2|float8_e4m3fn|float8_e5m2|f8")) tags.push("FP8");
  if (token("tf32")) tags.push("TF32");
  // f32 alongside a low-precision type is just the accumulator.
  if (!tags.length && token("fp32|f32|sgemm")) tags.push("FP32");
  if (token("int8|i8|s8")) tags.push("INT8");
  if (token("int4|u4|s4|w4a16|w4a8|uint4b8")) tags.push("4-bit");
  if (/(^|_)(bwd|backward)(_|$)|backward/.test(lower)) tags.push("Backward pass");
  const arch = archOf(lower);
  if (arch) tags.push(arch);
  return tags;
}

const ELEMENTWISE_OPS: [RegExp, string][] = [
  [/silu/, "SiLU activation"],
  [/gelu/, "GELU activation"],
  [/relu/, "ReLU activation"],
  [/sigmoid/, "sigmoid"],
  [/tanh/, "tanh"],
  [/rsqrt/, "reciprocal square root"],
  [/sqrt/, "square root"],
  [/\bexp|_exp/, "exponential"],
  [/\blog|_log/, "logarithm"],
  [/pow/, "power"],
  [/clamp/, "clamp"],
  [/where/, "torch.where (select)"],
  [/masked_fill|maskedfill/, "masked fill"],
  [/compare|ceq|cne|clt|cgt|_eq_|_ne_/, "comparison"],
  [/fillfunctor|fill_kernel/, "fill (e.g. zeros_/fill_)"],
  [/copy_kernel|copy_device_to_device|direct_copy/, "copy / dtype conversion"],
  [/addfunctor|_add|add_kernel/, "addition"],
  [/mulfunctor|_mul|mul_kernel/, "multiplication"],
  [/divfunctor|_div|div_kernel|div_true/, "division"],
  [/sub_kernel|_sub/, "subtraction"],
  [/neg/, "negation"],
  [/abs/, "absolute value"],
];

function elementwiseOps(lower: string): string[] {
  return ELEMENTWISE_OPS.filter(([pattern]) => pattern.test(lower)).map(([, op]) => op);
}

const NCCL_COLLECTIVES: Record<string, string> = {
  allreduce: "AllReduce combines a tensor across all GPUs (e.g. sums partial results) so every GPU ends with the same full result. In tensor-parallel inference this runs after the attention output and MLP down projections.",
  allgather: "AllGather: each GPU contributes one shard and every GPU receives all shards.",
  reducescatter: "ReduceScatter combines tensors across GPUs and leaves each GPU with one shard of the result.",
  broadcast: "Broadcast copies a tensor from one GPU to all others.",
  reduce: "Reduce combines tensors from all GPUs onto a single GPU.",
  sendrecv: "Point-to-point send/receive between GPUs, used for pipeline parallelism and all-to-all (e.g. expert-parallel MoE).",
  send: "Point-to-point send to another GPU (pipeline parallelism).",
  recv: "Point-to-point receive from another GPU (pipeline parallelism).",
  alltoall: "AllToAll: every GPU sends a different slice to every other GPU (expert-parallel MoE token dispatch).",
};
const NCCL_ALGOS: Record<string, string> = {
  ring: "Ring algorithm: data passes around a ring of GPUs (bandwidth-efficient).",
  tree: "Tree algorithm: lower latency at large GPU counts.",
  nvls: "NVLS (NVLink SHARP): the reduction happens inside the NVSwitch.",
  nvlstree: "NVLS + tree across nodes.",
  collnet: "CollNet: reduction offloaded to the network switches.",
  pat: "PAT algorithm, tuned for one GPU per node.",
};
const NCCL_PROTOCOLS: Record<string, string> = {
  ll: "LL (low-latency) protocol, chosen for small messages; latency matters more than bandwidth.",
  ll128: "LL128 protocol for small-to-medium messages over NVLink.",
  simple: "Simple protocol, chosen for large messages to maximize bandwidth.",
};

function ncclDetails(name: string): string[] {
  const tokens = name.toLowerCase().split(/[^a-z0-9]+/);
  const details: string[] = [];
  const collective = tokens.find((token) => token in NCCL_COLLECTIVES);
  if (collective) details.push(NCCL_COLLECTIVES[collective]);
  const op = tokens.find((token) => ["sum", "prod", "max", "min", "avg", "premulsum", "sumpostdiv"].includes(token));
  if (op) details.push(`Reduction operator: ${op}.`);
  const algo = tokens.find((token) => token in NCCL_ALGOS);
  if (algo) details.push(NCCL_ALGOS[algo]);
  const protocol = tokens.find((token) => token in NCCL_PROTOCOLS);
  if (protocol) details.push(NCCL_PROTOCOLS[protocol]);
  details.push("Time here is communication (or waiting for other GPUs to arrive), not math. Long NCCL kernels often mean another GPU was late.");
  return details;
}

function gemmDetails(name: string): string[] {
  const lower = name.toLowerCase();
  const details = [
    "In transformer models GEMMs are the linear layers: QKV projection, attention output projection, and the MLP gate/up/down projections. They usually take most of the GPU time.",
  ];
  const tile = /(?:_|tilesize)(\d{2,3})x(\d{2,3})(?:x(\d{2,3}))?(?:_|$)/.exec(lower);
  if (tile) details.push(`Thread-block tile ${tile[1]}×${tile[2]}${tile[3] ? `×${tile[3]}` : ""}: each block computes that chunk of the output matrix.`);
  const stages = /stages?_?(\d+)x(\d+)|(\d+)stage/.exec(lower);
  if (stages) details.push("Multi-stage pipeline: loads of the next tiles overlap with math on the current one.");
  const layout = /_(tn|nt|nn|tt)(_|$)/.exec(lower);
  if (layout) details.push(`Layout "${layout[1]}": tells which input matrices are stored transposed (BLAS convention).`);
  if (/split_?k|splitk/.test(lower)) details.push("Split-K: the inner dimension is split across blocks and summed afterwards; common for small-batch (decode) matmuls.");
  if (/s16816|s1688|hmma|imma|wgmma|tensorop|tensor_op|xmma|nvjet|tcgen/.test(lower)) details.push("Runs on tensor cores.");
  if (/gemv/.test(lower)) details.push("GEMV: matrix × vector, typical for batch size 1 decode.");
  return details;
}

function tritonDetails(name: string): string[] {
  const kind = /triton_(poi|red|per|tem)_/.exec(name)?.[1];
  const kinds: Record<string, string> = {
    poi: "Pointwise kernel: elementwise ops fused into one pass over memory.",
    red: "Reduction kernel (e.g. sum, mean, max, softmax denominator, norm).",
    per: "Persistent reduction: the whole reduced row fits in one block.",
    tem: "Template kernel, usually a matmul generated from a Triton template.",
  };
  const details = kind ? [kinds[kind]] : [];
  const fused = /fused_(.+?)(?:_\d+)?$/.exec(shortKernelName(name))?.[1];
  if (fused) details.push(`Fused PyTorch ops: ${fused.split("_").filter((part) => part && !/^\d+$/.test(part)).join(", ")}.`);
  details.push("Generated by torch.compile (Inductor). The name lists the ops it fuses.");
  return details;
}

const RULES: Rule[] = [
  // vLLM / LLM serving kernels
  { test: /reshape_and_cache|concat_and_cache/, category: "KV cache",
    summary: "Writes this step's new key and value vectors into the paged KV cache so later tokens can attend to them.",
    details: (name) => /mla/i.test(name) ? ["MLA variant: caches the compressed latent vector (DeepSeek-style attention)."] : [] },
  { test: /paged_attention_v2_reduce/, category: "Attention",
    summary: "Second step of paged attention v2: merges the partial results computed for each chunk of a long sequence." },
  { test: /paged_attention/, category: "Attention",
    summary: "vLLM paged attention for decode: each new token attends to its sequence's keys/values stored in KV-cache blocks.",
    details: () => ["Decode attention is memory-bound: it reads the whole KV cache for every generated token."] },
  { test: /flash_?mla|flashmla/, category: "Attention",
    summary: "FlashMLA: decode attention for multi-head latent attention (DeepSeek V2/V3) over a paged KV cache." },
  { test: /splitkv_combine|combine_kernel|merge_attn|mergestates|merge_state/, category: "Attention",
    summary: "Combines partial attention results from split-KV (flash-decoding) into the final output." },
  { test: /batchdecode|decode_with_paged|batch_decode/, category: "Attention",
    summary: "FlashInfer decode attention: new tokens attend to their paged KV cache." },
  { test: /batchprefill|prefill_with|batch_prefill|singleprefill/, category: "Attention",
    summary: "FlashInfer prefill attention: processes prompt tokens attending to each other (and any cached prefix)." },
  { test: /flash_?(fwd|bwd|attn)|flashattn|fmha|flash_attention|attention_kernel|attn_fwd|_attn_/, category: "Attention",
    summary: (name) => /bwd|backward/i.test(name)
      ? "FlashAttention backward pass: computes gradients for Q, K, V in a fused kernel."
      : "Fused attention (FlashAttention-style): computes softmax(Q·Kᵀ)·V in one kernel without writing the full attention matrix to memory.",
    details: (name) => [
      ...(/splitkv|split_kv/i.test(name) ? ["Split-KV: long sequences are split across blocks for decode parallelism; a combine kernel merges them."] : []),
      "Prefill (long prompts) is compute-bound; decode (one new token) is memory-bound because it reads the whole KV cache.",
    ] },
  { test: /cross_device_reduce|custom_?all_?reduce|allreduce_fusion|one_shot_all_reduce|two_shot/, category: "Communication",
    summary: "vLLM custom all-reduce: sums tensors across GPUs directly over NVLink/P2P memory. Faster than NCCL for the small tensors in decode.",
    details: (name) => /2stage|two_shot/i.test(name)
      ? ["Two-stage: reduce-scatter then all-gather, used for larger tensors."]
      : /1stage|one_shot/i.test(name) ? ["One-stage: every GPU reads all peers' data, used for small tensors."] : [] },
  { test: /fused_add_rms_norm/, category: "Normalization",
    summary: "Adds the residual connection and applies RMSNorm in one kernel (between transformer sublayers)." },
  { test: /rms_?norm/, category: "Normalization",
    summary: "RMSNorm: scales each token's hidden vector by its root-mean-square (LLaMA-style layer normalization)." },
  { test: /layer_?norm|layernorm/, category: "Normalization",
    summary: "LayerNorm: normalizes each token's hidden vector to zero mean and unit variance, then scales and shifts." },
  { test: /rotary|rope/, category: "Position embedding",
    summary: "Applies rotary position embeddings (RoPE) to queries and keys so attention knows token positions." },
  { test: /act_and_mul|silu_and_mul|gelu_and_mul|swiglu|geglu/, category: "Activation",
    summary: "Gated MLP activation (SwiGLU/GeGLU): activation(gate) × up, the middle step of the transformer MLP." },
  { test: /moe_align|align_block_size|count_and_sort|sort_tokens|expert_tokens|moe_permute|permute_rows/, category: "Mixture of Experts",
    summary: "Groups tokens by the expert they were routed to, so the expert GEMM can process each expert's tokens together." },
  { test: /topk_?softmax|topkgating|moe_gate|grouped_topk|router/, category: "Mixture of Experts",
    summary: "MoE router: computes expert scores and picks the top-k experts for each token." },
  { test: /moe_sum|moe_unpermute|finalize_moe|moe_reduce/, category: "Mixture of Experts",
    summary: "Combines each token's outputs from its selected experts (weighted sum)." },
  { test: /fused_moe|moe_gemm|grouped_gemm|group_gemm|moe_mm/, category: "Mixture of Experts",
    summary: "Mixture-of-Experts grouped GEMM: runs each token through the MLP weights of the experts it was routed to.",
    details: gemmDetails },
  { test: /marlin/, category: "Matrix multiply (quantized)",
    summary: "Marlin GEMM: multiplies FP16/BF16 activations by 4- or 8-bit quantized weights (GPTQ/AWQ models), dequantizing on the fly.",
    details: gemmDetails },
  { test: /gptq|awq|exllama|w4a16|machete/, category: "Matrix multiply (quantized)",
    summary: "Quantized-weight GEMM: weights stored in low precision and dequantized on the fly during the matmul.",
    details: gemmDetails },
  { test: /quant|scaled_fp8|per_token_group|fp8_cast|to_fp8|dynamic_scaled/, category: "Quantization",
    summary: "Quantizes activations (usually to FP8 or INT8): computes scales and casts values before a low-precision GEMM." },
  { test: /cutlass_scaled_mm|scaled_mm|fp8_gemm|fp8gemm|deep_gemm|fp8_blockwise/, category: "Matrix multiply (FP8)",
    summary: "FP8 matrix multiply with scaling factors (quantized model or FP8 linear layer).",
    details: gemmDetails },
  // Communication
  { test: /nccl/, category: "Communication (NCCL)",
    summary: "NCCL collective: moves or combines data across GPUs.", details: ncclDetails },
  { test: /reduce_kernel/, category: "Reduction",
    summary: (name) => {
      const lower = name.toLowerCase();
      const op = /mean/.test(lower) ? "mean" : /argmax|argmin/.test(lower) ? "arg max/min"
        : /max/.test(lower) ? "max" : /min/.test(lower) ? "min" : /norm/.test(lower) ? "norm" : /sum/.test(lower) ? "sum" : null;
      return `PyTorch reduction${op ? ` (${op})` : ""}: combines values along a dimension.`;
    },
    details: (name) => /argmax/i.test(name) ? ["Argmax over logits is greedy sampling: picking the most likely next token."] : [] },
  // Generated kernels
  { test: /triton_(poi|red|per|tem)_/, category: "torch.compile (Triton)",
    summary: "Triton kernel generated by torch.compile, fusing several PyTorch operations.", details: tritonDetails },
  // Matrix multiply
  { test: /gemm|gemv|cublas|cutlass|xmma|nvjet|_mma_|matmul|s16816|hmma|wgmma|sm\d+_xmma|implicit_?gemm|bmm/, category: "Matrix multiply (GEMM)",
    summary: (name) => /gemv/i.test(name) ? "Matrix × vector product (GEMV)."
      : /conv|implicit_?gemm/i.test(name) ? "Convolution implemented as an implicit GEMM."
      : "Matrix multiplication (GEMM) from cuBLAS/CUTLASS.",
    details: gemmDetails },
  { test: /conv|cudnn|winograd/, category: "Convolution",
    summary: "Convolution (cuDNN)." },
  // PyTorch generic kernels
  { test: /softmax/, category: "Softmax",
    summary: "Softmax: turns scores into probabilities (e.g. logits → token probabilities for sampling)." },
  { test: /radixsort|radix_sort|sort|segmented_?sort|bitonic/, category: "Sort",
    summary: "Sorting. In LLM serving this is usually top-k/top-p sampling over the vocabulary." },
  { test: /topk|top_k|gathertopk|argmax|multinomial|sampling|sampler/, category: "Sampling",
    summary: "Picks the next token: top-k/top-p filtering, argmax, or random sampling over the vocabulary." },
  { test: /embedding|embed/, category: "Embedding",
    summary: "Embedding lookup: converts token IDs into hidden vectors." },
  { test: /catarraybatchedcopy|cat_array|_cat_/, category: "Memory",
    summary: "torch.cat: copies several tensors into one contiguous tensor." },
  { test: /index_select|indexselect|index_elementwise|gather|scatter|index_put|indexing/, category: "Indexing",
    summary: "Gather/scatter/indexing: reads or writes elements at given indices (memory-bound)." },
  { test: /transpose|permute|nchwtonhwc|nhwctonchw/, category: "Memory",
    summary: "Transposes or permutes tensor dimensions (copies data into a new layout)." },
  { test: /reduce|reduction/, category: "Reduction",
    summary: "Reduction: combines values along a dimension (sum, max, mean…)." },
  { test: /elementwise|unrolled_elementwise|vectorized_elementwise|launch_kernel|_functor|functor/, category: "Elementwise",
    summary: (name) => {
      const ops = elementwiseOps(name.toLowerCase());
      return `PyTorch elementwise op${ops.length ? ` (${ops.join(", ")})` : ""}: applies the same operation to every element.`;
    },
    details: () => ["Memory-bound: runtime is mostly reading and writing the tensor. Many of these back to back are candidates for fusion (e.g. torch.compile)."] },
  { test: /fill|memset|zero/, category: "Memory",
    summary: "Fills a tensor with a constant (e.g. zeros)." },
  { test: /copy|memcpy/, category: "Memory",
    summary: "Copies tensor data (possibly converting dtype or layout)." },
  { test: /^cub::|cub::|devicescan|devicereduce|deviceselect|histogram/, category: "CUB primitive",
    summary: "CUB library primitive (scan, reduce, select, or histogram) used inside a higher-level op." },
];

export function explainKernel(name: string): KernelExplanation {
  const short = shortKernelName(name);
  const lower = name.toLowerCase();
  const tags = commonTags(lower);
  const rule = RULES.find((item) => item.test.test(lower));
  if (!rule) {
    return {
      short, tags, details: [],
      category: "Unrecognized",
      summary: "No built-in description for this kernel. It is probably a custom or library kernel; the readable name above is the function that was launched.",
    };
  }
  return {
    short, tags,
    category: rule.category,
    summary: typeof rule.summary === "function" ? rule.summary(name) : rule.summary,
    details: rule.details?.(name) ?? [],
  };
}
