import assert from "node:assert/strict";
import test from "node:test";
import { explainKernel, shortKernelName } from "../src/explain.ts";

test("shortens demangled C++ names", () => {
  assert.equal(shortKernelName("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>>(int, T2)"), "vectorized_elementwise_kernel");
  assert.equal(shortKernelName("void at::native::(anonymous namespace)::cunn_SoftMaxForward<4, float>(T4 *, int)"), "cunn_SoftMaxForward");
  assert.equal(shortKernelName("void cutlass::device_kernel<flash::FlashAttnFwdSm90<2, float>>(T1::Params)"), "FlashAttnFwdSm90");
  assert.equal(shortKernelName("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn"), "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn");
});

test("categorizes common LLM inference kernels", () => {
  const cases = {
    "ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn": "Matrix multiply (GEMM)",
    "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096>)": "Communication (NCCL)",
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128>>(Flash_fwd_params)": "Attention",
    "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16>(const T1 *)": "KV cache",
    "void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>(c10::BFloat16 *)": "Normalization",
    "void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, true>(c10::BFloat16 *)": "Activation",
    "void vllm::cross_device_reduce_1stage<__nv_bfloat16, 8>(vllm::RankData *)": "Communication",
    "void vllm::moe::moe_align_block_size_kernel<int>(const int *)": "Mixture of Experts",
    "triton_poi_fused_add_mul_silu_0": "torch.compile (Triton)",
    "void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>>>(T3)": "Reduction",
    "my_custom_kernel(float*, int)": "Unrecognized",
  };
  for (const [name, category] of Object.entries(cases)) {
    assert.equal(explainKernel(name).category, category, name);
  }
});

test("decodes NCCL, GEMM, and Triton name parts", () => {
  const nccl = explainKernel("ncclDevKernel_AllReduce_Sum_bf16_RING_LL").details.join(" ");
  assert.match(nccl, /AllReduce/);
  assert.match(nccl, /Ring/);
  assert.match(nccl, /low-latency/);
  const gemm = explainKernel("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_64x3_tn");
  assert.deepEqual(gemm.tags, ["BF16", "Ampere (A100/A10)"]);
  assert.match(gemm.details.join(" "), /128×128/);
  assert.match(explainKernel("triton_poi_fused_add_mul_silu_0").details.join(" "), /add, mul, silu/);
  assert.match(explainKernel("void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>>(int)").summary, /fill/);
});
