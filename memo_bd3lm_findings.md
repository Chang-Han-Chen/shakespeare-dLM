# BD3-LM on TinyShakespeare: Findings and Methodology Memo

**Date:** March 2026
**Dataset:** TinyShakespeare (~1.1MB, character-level tokenization)
**Repo:** shakespeare-dLM

---

## Main Findings

**Claim 1: Block diffusion scales better than standard diffusion at equal FLOPs.**
At IsoFLOP budget C3 (6e14 FLOPs), the three non-degenerate block models (bl=4) outperform their non-block counterparts by 0.05-0.16 nats at their respective optima. Block MDLM achieves 1.8241 (at 0.5M) vs standard remasked at 1.9814 (at 1M), a gap of 0.16 nats. Block Edit-1 achieves 1.8601 (gap 0.12), and Block Edit-2 achieves 1.9358 (gap 0.05). The block formulation is more FLOP-efficient despite the 2x input length overhead.

**Claim 2: The optimal model size for block diffusion is smaller; large models overfit at fixed compute.**
At C3, block MDLM's IsoFLOP minimum is at 0.5M (val=1.8241), while standard MDLM's is at 1M (val=1.9861). Block models at 1M–3M show rising val loss (1M=1.8640, 2M=1.8572, 3M=1.9189), indicating they are undertrained (too few steps at the fixed FLOP budget due to the 2x cost per step). The ranking among block models is: block_mdlm = block_remasked > block_edit_one_pass > block_edit_two_pass.

**Claim 3: AR pretraining helps block diffusion, with a sweet spot at p_ar=0.3.**
Training AR for 30% of total steps then switching to block MDLM (bl=4) beats pure block diffusion at 0.3M and 1M. At 1200 steps the improvement is -0.077 nats (1M) and -0.088 nats (0.3M). At 4000 steps the improvement is -0.043 nats (1M) and -0.049 nats (0.3M). The effect is U-shaped: too little AR (p=0.1) gives insufficient initialization, too much (p=0.7) starves the block diffusion phase. The sweet spot at p=0.3 is stable at 0.3M and 1M across training durations. At 3M with 1200 steps the curriculum effect is negligible (p=0.1 achieves 1.7320 vs p=0.0 at 1.7323), likely because the larger model already converges well within the budget.

**Observation: block_mdlm and block_remasked are identical during training.**
These two models share the exact same `get_batch`, `compute_loss`, `compute_eval_loss`, and `get_eval_batch` functions. The only difference is in `generate_from()` (sampling strategy): MDLM uses progressive unmasking, remasked re-masks low-confidence tokens. With a fixed seed, they produce byte-for-byte identical loss.pkl files. In the original sweep (parallel execution with process scheduling nondeterminism), they differed by 0.001-0.01 nats — pure noise. Practically, this means only one needs to be trained; the checkpoint can be used with either generation strategy at inference time. For future experiments, we can drop block_remasked from all training sweeps, reducing the block model count from 4 to 3.

**Note: t_min and t_max were not swept.**
The current defaults are t_min=0.45, t_max=0.95, which clip the noise schedule so the model never sees nearly-clean (t < 0.45) or nearly-fully-masked (t > 0.95) inputs. The BD3-LM paper claims that tuning these reduces variance and helps block diffusion match AR quality. This is a high-priority hyperparameter to sweep for OpenWebText.

---

## Architecture

All 9 model variants (4 standard diffusion + 4 block diffusion + AR) share a single DiffusionBackbone transformer. The only differences are attention masking and training objective.

The backbone uses: RoPE positional encoding, QK-norm, ReGLU MLP, RMSNorm, n_head=4 throughout.

Block models use a dual-stream 2L input during training: the input is `[x_t | x_0]` where x_t is the noisy sequence and x_0 is the clean sequence. The attention mask has four quadrants: block-diagonal on x_t, block-causal on x_0, strict one-block-behind cross-attention from x_t to x_0, and zeros preventing x_0 from seeing x_t. During generation, block models use a single-stream block-causal mask where token i attends to token j iff block(j) <= block(i).

When block_len equals block_size (the full sequence length), the model degenerates to standard diffusion and the dual-stream machinery is skipped via a `_is_single_block` flag in the backbone.

Model variants:
- **remasked / block_remasked**: Re-masks low-confidence tokens at each diffusion step
- **MDLM / block_mdlm**: Masked Diffusion LM; carries unmasked tokens forward, no remasking
- **edit_one_pass / block_edit_one_pass**: Adds visible-token corruption during training
- **edit_two_pass / block_edit_two_pass**: Two forward-backward passes per step (denoising + correction); 2x FLOP cost

---

## Phase 1: Hyperparameter Sweep

**Purpose:** Find best (LR, block_len) per (model, size) for all models (both block and non-block).

**Methodology:** Two-stage LR search. First swept by 10x to find the right order of magnitude, then zoomed in by 3x within the promising range. Also grid-searched batch size.

**Final sweep grid (block models):**
- Models: 4 block models
- Sizes: 0.1M, 0.3M, 1M, 3M
- LRs: 3e-4, 1e-3, 3e-3, 1e-2
- Block lengths: 1, 4, 16
- Steps per run: 1000 (200 warmup, cosine decay to min_lr = lr/10)
- Batch size: 128, block size: 256
- Total: 192 runs

**Key findings:**
- bl=1 always achieves the lowest loss (trivially, since bl=1 is AR-like)
- The bl=4 parallelism penalty shrinks with scale: +0.36 nats at 0.1M down to +0.29 nats at 3M
- Optimal LR is block_len-independent (same LR wins regardless of bl)
- LR transition: 1e-2 for models <= 1M, 3e-3 for models >= 2M-3M
- Edit models (one_pass, two_pass) shift to 3e-3 earlier (at 1M)
- All four block models are remarkably similar at the same config; edit_two_pass slightly best at 3M (1.4488 vs 1.4874 for remasked)

---

## Phase 2: IsoFLOP Experiments

**Purpose:** Compare block vs non-block diffusion at equal FLOPs; rank block models against each other.

**Setup:**
- FLOP budget: C3 = 6e14 (single budget; sufficient for TinyShakespeare scale)
- Models: 2 non-block (mdlm, remasked) + 4 block models at bl=4
- Sizes: 0.1M through 3M (6 sizes), filtered by feasibility window [150, 10000] steps
- Total: 31 runs
- Steps per run: budget / (C * N_params * 32768), where C is model-dependent FLOP multiplier
- FLOP multipliers: C=6 (standard diffusion), C=12 (block models and edit_two_pass), C=24 (block_edit_two_pass)
- LRs: from sweep-derived OPTIMAL_LR table
- Warmup: min(100, max(10, steps // 20))
- Eval interval: max(steps // 12, 20)

**IsoFLOP Step Matrix (block_remasked, block_mdlm, block_edit_one_pass; C=12):**

| Budget | 0.1M | 0.3M | 0.5M | 1M | 2M | 3M |
|--------|------|------|------|----|----|-----|
| C3     | ---  | 4422 | 2852 | 1522 | 782 | 479 |

**IsoFLOP Step Matrix (block_edit_two_pass; C=24):**

| Budget | 0.1M | 0.3M | 0.5M | 1M | 2M | 3M |
|--------|------|------|------|----|----|-----|
| C3     | 7130 | 2211 | 1426 | 761 | 391 | 239 |

**Best size per model at C3:**
- block_mdlm: 0.5M (val=1.8241)
- block_remasked: 0.5M (val=1.8241)
- block_edit_one_pass: 1M (val=1.8601)
- block_edit_two_pass: 0.3M (val=1.9358)
- Standard mdlm: 1M (val=1.9861)
- Standard remasked: 1M (val=1.9814)

---

## Phase 3: Curriculum Learning (AR Warm-Start)

**Purpose:** Test whether AR pretraining improves block diffusion quality.

**Mechanism:** Train AR for p_ar * total_steps, save checkpoint, load backbone weights into block_mdlm (bl=4), reset optimizer and LR schedule, continue training block diffusion for the remaining steps. The weight transfer works because AR and block_mdlm share the same DiffusionBackbone; only the attention masking differs.

**LR schedule detail:** The full schedule is: linear warmup -> cosine decay to min_lr (AR phase) -> optimizer reset -> linear warmup -> cosine decay to min_lr (BD phase). So the LR profile is: ramp up, curve down, ramp up again, curve down again. AR warmup is 20% of AR steps; BD warmup is fixed at 50 steps.

**1200-step sweep:**
- p_ar: [0.0, 0.1, 0.3, 0.5, 0.7]
- Sizes: 0.3M, 1M, 3M
- Total: 15 runs
- eval_interval=0 (final loss only)

**1200-step results:**

| Size | p=0.0 | p=0.1 | p=0.3 | p=0.5 | p=0.7 |
|------|-------|-------|-------|-------|-------|
| 0.3M | 2.0447 | 1.9727 | **1.9566** | 1.9660 | 2.0059 |
| 1M   | 1.8912 | 1.8710 | **1.8140** | 1.8390 | 1.8649 |
| 3M   | 1.7323 | **1.7320** | 1.7339 | 1.7473 | 1.7885 |

**4000-step validation (centered on p=0.3):**
- p_ar: [0.0, 0.3, 0.5]
- Sizes: 0.3M, 1M (chosen because 0.3M is IsoFLOP-optimal, 1M showed strongest curriculum effect)
- Total: 6 runs
- eval_interval=50 (dense loss curves for Figure 4)

**4000-step results:**

| Size | p=0.0 | p=0.3 | p=0.5 |
|------|-------|-------|-------|
| 0.3M | 1.8604 | **1.8119** (-0.0485) | 1.8295 (-0.0309) |
| 1M   | 1.7329 | **1.6903** (-0.0427) | 1.6915 (-0.0415) |

---

## Fixed Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch size | 128 |
| Block size (seq len) | 256 |
| n_head | 4 |
| Dropout | 0.1 (diffusion), 0.2 (AR) |
| min_lr | lr / 10 |
| LR schedule | Linear warmup + cosine decay to min_lr |
| Optimizer | AdamW (fused when available) |
| Dataset | TinyShakespeare (~1.1MB, char-level) |
| IsoFLOP step bounds | [150, 10000] |
| Tokens per step | 32768 (= 128 * 256) |

**Model size registry:**

| Label | n_embd | n_layer | Params |
|-------|--------|---------|--------|
| 0.1M  | 64     | 2       | 107K   |
| 0.3M  | 96     | 3       | 345K   |
| 0.5M  | 120    | 3       | 535K   |
| 1M    | 128    | 5       | 1.002M |
| 2M    | 200    | 4       | 1.949M |
| 3M    | 256    | 4       | 3.182M |

---

## Figures

The notebook `reproduce.ipynb` generates 4 figures:

1. **IsoFLOP C3: block vs standard diffusion** - Shows 4 lines: Remasked/MDLM (dashed, merged since identical during training), Block Remasked/MDLM (solid, merged), Block Edit-1, Block Edit-2. Block models sit below standard diffusion across all sizes at equal FLOPs.
2. **Block model ranking at C3** - Shows 3 block model curves (Block Remasked/MDLM merged, Block Edit-1, Block Edit-2) with minima marked, and the overfitting region shaded at 1M–3M.
3. **Curriculum sweet spot** - Left: U-shaped val CE vs p_ar at 1200 steps for 0.3M and 1M. Right: same pattern persists at 4000 steps for 0.3M and 1M. Both panels use the same sizes.
4. **Validation loss curves for p=0.3 curriculum** - Val CE dynamics for 0.3M and 1M showing AR phase (blue) transitioning to BD phase (green) with pure BD baseline (gray).

---

## Notes for OpenWebText Scale-Up

Things that change:
- BPE tokenization (GPT-2 tokenizer, 50257 vocab) via new prepare.py
- Block size: 2048 (up from 256)
- Model sizes: ~10M to ~300M range, targeting ~110M
- Multiple IsoFLOP budgets (C1-C4+) for proper Chinchilla-style scaling curves
- New LR sweep at larger sizes (two-stage: 10x then 3x zoom)
- Batch size sweep at new scale
- Sweep t_min and t_max (noise clipping); BD3-LM paper claims this reduces variance and helps match AR
- Re-validate curriculum p=0.3 sweet spot at scale
- Explore alternative LR schedules for curriculum: WSD (warmup-stable-decay) with AR-to-diffusion switch during the stable phase, instead of double cosine
