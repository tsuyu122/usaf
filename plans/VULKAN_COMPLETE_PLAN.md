Vulkan Complete Integration Plan — USAF
=======================================
Generated: 2026-07-04

Current State Scorecard
-----------------------

INFRASTRUCTURE (complete):
  Buffer API       ✓ create_buf / upload / download / destroy_buf
  Pipeline dispatches ✓ rmsnorm_pipe / gemm_pipe / rope_pipe / dequant_pipe / attn_softmax_pipe
  Shader compilation ✓ all 7 .comp → .spv via glslc
  Pybind module    ✓ usaf_vk.cp311-win_amd64.pyd (524 lines C++)
  Vulkan Core      ✓ vulkan_core.cpp (330 lines, init/buffer/dispatch/download/destroy)

CORRECTNESS (verified):
  Dequant Q4 GPU   ✓ 0.05% error vs CPU reference
  GEMM fp16        ✓ 0.07% error vs PyTorch
  RMSNorm          ✓ 0.22% error (original validation)
  Monkey-patch     ✓ VK QKV + native DML attention → loss 1.8057 (identical to DML)

NOT WORKING:
  Full VK forward  ✗ Loss 7.6 vs expected 1.8 — attention pipeline broken
  VK backward      ✗ Not implemented
  VK speedup       ✗ Slower than DML due to barrier() serialization + PCIe downloads

================================================================================
PHASE 1 — DEBUG THE ATTENTION SOFTMAX PIPELINE
================================================================================

Root cause analysis (ranked by likelihood):

1. MISSING ATTENTION SCALE (most likely)
   The native attention computes:
     scores = Q @ K^T / sqrt(hd)
   Our Vulkan Q@K^T GEMM computes raw dot products WITHOUT dividing by sqrt(hd).
   Without this scale, softmax produces near-one-hot distributions → wrong output.

   Fix: Multiply scores by 1/sqrt(hd) before uploading to attn_softmax_pipe.
   Or: Include scale in the push constants and apply in the kernel.

2. V BUFFER LAYOUT MISMATCH
   The attn_softmax kernel reads V as [nKV, S, hd] = [4, 512, 128].
   But the uploaded buffer has shape (nKV, S, hd) which in row-major is
   [4, 65536] elements. The kernel reads v[v_off + c * pc.hd + d].
   v_off = head_kv * S * hd = head_kv * 65536.
   Total offset = head_kv * 65536 + c * 128 + d.
   In the numpy buffer v_heads with shape [4, 512, 128]: element [kv, pos, d]
   is at offset kv*65536 + pos*128 + d. The layout matches.

3. LOCAL ARRAY SIZE (shared memory overflow)
   float exp_vals[512] = 512 * 4 bytes = 2048 bytes per thread.
   256 threads = 512KB of local memory. Exceeds typical GPU limits.
   On AMD RX 6750 XT, private memory spills to global memory (very slow).
   
   Fix: Use shared memory instead of local array. Process scores in tiles.

4. Q@K^T GEMM WITH GQA — HEAD MAPPING
   The GEMM produces scores for 8 query heads against 1 KV head.
   Need to verify that the head indices in the scores matrix correctly
   map to the query head indices in the kernel.

================================================================================
PHASE 1 — IMPLEMENTATION PLAN
================================================================================

Step 1.1: Add attention scale (10 minutes)
  File: usaf/vk_layer.py, ~line (after Q@K^T)
  Insert:
    scores_kv = (scores_kv.astype(np.float32) * (1.0 / np.sqrt(hd))).astype(np.float16)
  This applies 1/sqrt(hd) to the Q@K^T scores after the GEMM download.

Step 1.2: Fix local array in attention.comp (30 minutes)
  Replace float exp_vals[512] with shared memory:
    shared float s_exp_vals[512];  // shared across workgroup
  Each thread writes its 2 rows of exp values, synchronizes, reads back.
  This eliminates 512KB of private memory per thread.

  Alternative: process in 2 passes. Pass 1: find max. Pass 2: compute exp + sum.
  Pass 3: weighted V sum. No large local arrays needed.

Step 1.3: Add intermediate value dump (15 minutes)
  Write a debug function that runs one layer forward and dumps:
    - Q after projection (vs DML)
    - K after RoPE (vs DML)
    - Scores after Q@K^T (vs DML)
    - Attn output after softmax (vs DML)
    - Post-norm output (vs DML)
  Compare each stage to find where values diverge.

Step 1.4: Fix the stage that diverges
  The dump from Step 1.3 will pinpoint exactly which stage produces wrong values.
  Fix that specific stage. If it's the softmax, fix the kernel.
  If it's the GEMM, check dimensions. If it's the reshape, fix the numpy logic.

================================================================================
PHASE 2 — RESIDUAL ADD KERNEL
================================================================================

Current: hidden + O-proj output is computed on CPU (costs 2×2MB download).
Goal: Compute on Vulkan GPU, download only final result.

Step 2.1: Write residual_add.comp shader (15 minutes)
  Simple element-wise add: C[i] = A[i] + B[i]
  - Input: A [N] (hidden), B [N] (O-proj output)
  - Output: C [N] (post-attention hidden)
  - One workgroup, N threads

Step 2.2: Add pybind binding (10 minutes)
  residual_add_pipe(a_handle, b_handle, out_handle, N)

Step 2.3: Compile and integrate (10 minutes)
  glslc → spirv, cmake build, add to VKLayer.

================================================================================
PHASE 3 — POST-ATTENTION RMSNORM KERNEL
================================================================================

Current: post-norm computed on CPU.
Goal: Compute on Vulkan GPU.

The existing rmsnorm_pipe already does RMSNorm! Just need to apply it with
post_attention_layernorm.weight instead of input_layernorm.weight.

Step 3.1: Integrate into VKLayer.forward() (5 minutes)
  After residual add, call:
    usaf_vk.rmsnorm_pipe(h_residual, bufs["post_attention_layernorm.weight"], 
                         h_post_norm, B*S, H, 1e-6)
  Download only h_post_norm (2MB).

================================================================================
PHASE 4 — INTEGRATE FULL PIPELINE INTO FWD_BWD
================================================================================

Current state: fwd_bwd uses monkey-patch (VK QKV → native attention).
Goal: Replace with full VK forward (attention block on Vulkan, MLP on DML).

Step 4.1: Update VKLayer.forward() signature (5 minutes)
  Current: returns (q_np, k_np, v_np) — raw projections
  New: returns (post_norm_np) — post-attention norm hidden
  
  Rename current forward() → forward_qkv()
  New forward() does the full pipeline from Phase 1-3.

Step 4.2: Update fwd_bwd in train.py (10 minutes)
  Replace monkey-patch block with:
    post_norm_np = VK_LAYERS[i].forward(h_np, cos_np, sin_np)
    post_norm_t = torch.from_numpy(post_norm_np.astype(np.float32)).to(device).half()
    mlp_out = model.model.layers[i].mlp(post_norm_t)
    hidden = hidden + mlp_out  # residual connection
    xs.append(hidden)  # stash for backward

Step 4.3: Remove monkey-patch code (5 minutes)
  Delete VKProj class and the _orig_q/_orig_k/_orig_v patching.

================================================================================
PHASE 5 — PERFORMANCE OPTIMIZATION
================================================================================

Current bottleneck: barrier() = wait_idle() after every dispatch.
Every kernel call blocks the CPU until GPU finishes.

Impact analysis:
- 3 GEMMs (QKV) × barrier = 3 × (5ms dispatch + negligible wait) = 15ms
- 1 RMSNorm × barrier = 0.5ms
- 4 Q@K^T GEMMs per layer × barrier = 4 × 5ms = 20ms
- 1 attn_softmax × barrier = 2ms  
- 1 O-proj GEMM × barrier = 5ms
- Total per layer: ~152ms (VK) vs ~45ms (DML)

VK is 3.4× SLOWER due to barriers alone!

Step 5.1: Pipeline without barriers (30 minutes)
  Currently barrier() = ctx.device.waitIdle(). This waits for ALL GPU work.
  
  Change: remove barrier() calls. Since all dispatches go to the same
  Vulkan queue, they execute in order. The last operation (download) will
  implicitly wait for all prior dispatches.
  
  Only keep barrier() before the FIRST download (scores after Q@K^T).
  All intermediate dispatches (RMSNorm, GEMM, RoPE) queued without wait.

Step 5.2: Measure speedup (15 minutes)
  Compare tok/s with and without barriers.
  Expected: 3-4× faster, approaching DML speed.

Step 5.3: Persistent temp buffers (20 minutes)
  Currently every forward() call allocates + frees temp buffers.
  Allocate once in __init__, reuse across forward() calls.
  
  For each VKLayer, pre-allocate:
    - hx, hrms, hq, hk, hv: [B*S, dim] shapes
    - hscores, h_attn, ho: fixed shapes
  Total pre-allocated: ~50MB per layer. 8 layers = 400MB. Fits in VRAM.

================================================================================
PHASE 6 — BACKWARD KERNELS (GRADIENTS VIA VULKAN)
================================================================================

Current: DML handles all backward computation.
Goal: VK computes input gradient dx = dy @ W^T for each trainable layer.

Step 6.1: dx = dy @ W GEMM (10 minutes)
  The backward for a linear layer y = x @ W^T is:
    dx = dy @ W  (gradient w.r.t. input)
    dW = x^T @ dy  (gradient w.r.t. weights)
  
  dx can be computed via existing gemm_pipe:
    dx [B*S, H] = dy [B*S, N] @ W^T [N, H]
  Where N = nH*hd for attention, H = hidden_size.
  
  Already have GEMM kernel. Just need to arrange buffers correctly.

Step 6.2: Integrate into fwd_bwd (20 minutes)
  After VK forward produces hidden output:
    1. Cache dy (from DML backward)
    2. For each layer, compute dx via VK gemm_pipe
    3. Pass dx to previous layer
    4. dW gradients already captured by sparse_store hooks

  This saves the DML backward GEMM time (~15ms per layer on DML).
  VK GEMM: ~5ms. Saving: 10ms per layer × 8 layers = 80ms per step.
  Step: ~8000ms. Gain: ~1%.

  Negligible speedup because backward compute is a small fraction of total time.

Step 6.3: Expert weight gradients (not worth implementing)
  Expert weight gradients are already captured by sparse_store hooks during
  the DML forward pass. Computing them via VK would require running the
  MoE forward on VK — a massive undertaking with minimal return.

  Decision: SKIP. Not worth the engineering effort.

================================================================================
PHASE 7 — END-TO-END VALIDATION
================================================================================

Step 7.1: Unit test — one layer, compare VK vs DML (15 minutes)
  Write test_vk_layer.py that:
    1. Loads layer 40 weights
    2. Creates VKLayer
    3. Runs forward with same random input
    4. Compares VK forward output vs DML layer output
    5. Verifies max error < 1% of peak value

Step 7.2: Smoke test — 1 step, compare loss (10 minutes)
  Run train.py with USE_VK=1, SMOKE_N=1, SMOKE_STEPS=1
  Verify loss matches DML baseline (1.8057 ± 0.01)

Step 7.3: Speed benchmark (10 minutes)
  Run train.py with USE_VK=1 vs USE_VK=0, SMOKE_STEPS=5
  Measure tok/s for each.
  VK should be FASTER than DML (target: >9 tok/s baseline).

Step 7.4: Multi-step stability (20 minutes)
  Run train.py with USE_VK=1, SMOKE_STEPS=10
  Verify: zero NaN steps, loss decreases monotonically.

================================================================================
SUMMARY — EFFORT ESTIMATES
================================================================================

Phase  Dependency   Effort   Impact
─────  ──────────   ──────   ──────────────────────────────────
  1    standalone    2-3h    Critical — fixes loss 7.6 → 1.8
  2    after P1      0.5h    Removes CPU compute from critical path
  3    after P2      0.2h    Trivial — reuses existing rmsnorm_pipe
  4    after P1-3    0.5h    Integrates full pipeline into training
  5    after P4      1-2h    Potential 3-4× speedup over DML
  6    after P4      0.5h    Negligible gain (~1%) — optional
  7    after P5      0.5h    Final validation

TOTAL Phase 1-7:  5-7 hours of focused work.

The critical path is:
  P1 (fix attention) → P2 (residual add) → P3 (post-norm) 
  → P4 (integrate) → P5 (performance) → P7 (validate)

P6 (backward) is optional and can be skipped with minimal loss.

================================================================================
KNOWN BUGS TO FIX IN PHASE 1
================================================================================

Bug 1: Missing 1/sqrt(hd) scale in Q@K^T scores
  Location: vk_layer.py, after GEMM download of scores
  Fix: scores *= 1/sqrt(hd)
  Impact: Without this, softmax is effectively one-hot → loss ~7.6
  Likelihood: HIGH — this alone could explain the entire discrepancy

Bug 2: float exp_vals[512] local array (512KB per thread)
  Location: attention.comp, line 58
  Fix: Use shared memory or tiled computation
  Impact: Causes register spilling → correct but very slow
  Likelihood: HIGH for performance, LOW for correctness

Bug 3: V buffer layout for attn_softmax_pipe
  Location: vk_layer.py, hv_all buffer upload
  The kernel expects V as [nKV, S, hd] flat.
  The upload is hv_all = alloc((nKV, S, hd)) which creates [4, 512, 128].
  In row-major: element [kv, pos, d] at kv*65536 + pos*128 + d.
  The kernel reads v[v_off + c * pc.hd + d] = v[kv*65536 + c*128 + d].
  Layout matches ✓ — not a bug.

================================================================================
FILES THAT WILL BE MODIFIED (Phase 1-7)
================================================================================

New files:
  usaf/vulkan/src/kernels/residual_add.comp  — Phase 2
  test_vk_layer.py                           — Phase 7

Modified files:
  usaf/vk_layer.py              — Phase 1-4 (forward pipeline)
  usaf/vulkan/src/kernels/attention.comp    — Phase 1 (fix local array)
  usaf/vulkan/src/pybind_module.cpp         — Phase 2 (residual_add binding)
  usaf/vulkan/src/vulkan_core.cpp           — Phase 5 (pipeline barriers)
  train.py                        — Phase 4 (integration)

================================================================================
