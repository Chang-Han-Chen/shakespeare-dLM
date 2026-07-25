# Block diffusion scaling on ClimbMix

This is a clean, self-contained IsoFLOP experiment for block diffusion on
NVIDIA ClimbMix. It imports no code from the TinyShakespeare experiment.

## Data and tokenizer

The official NVIDIA release is a 400B-token, CC BY-NC 4.0 research dataset
stored as GPT-2 token sequences. For practical custom-tokenizer training, this
experiment uses `karpathy/climbmix-400b-shuffle`, a shuffled raw-text
conversion of that release. Shard 0 is held out for validation and shards
1–90 are prepared as training data. The completed \(10^{14}\)–\(10^{16}\)
study used shards 1–25; the extension supplies 5.776B tokens for adaptive
\(10^{18}\) size bracketing. A run reads a deterministic prefix exactly once
and never wraps around the training set.

The tokenizer is byte-level BPE:

- 8,192 base tokens, including `<|endoftext|>` as the document delimiter;
- one additional diffusion-only `<|mask|>` token (ID 8192);
- total input/output vocabulary size 8,193;
- trained on raw training shards 1–4.

## Model table

All primary-grid models are bias-free Llama-2-style transformers with
RMSNorm, RoPE, full multi-head attention, SwiGLU, no dropout, and head
dimension 16. The input
embedding is excluded from counted `N`; the untied 8,193-way LM head and all
learned norms are included.

| label | layers | width | heads | head dim | FFN | counted N |
|---|---:|---:|---:|---:|---:|---:|
| 0.14M | 2 | 16 | 1 | 16 | 48 | 137,824 |
| 0.29M | 2 | 32 | 2 | 16 | 88 | 287,424 |
| 0.5M | 4 | 48 | 3 | 16 | 128 | 504,288 |
| 1M | 4 | 80 | 5 | 16 | 216 | 965,920 |
| 2M | 7 | 112 | 7 | 16 | 304 | 1,985,536 |
| 4M | 11 | 144 | 9 | 16 | 384 | 3,920,256 |
| 8M | 12 | 208 | 13 | 16 | 560 | 7,979,296 |
| 10M* | 13 | 224 | 14 | 16 | 600 | 9,692,032 |

The sub-0.5M points were added adaptively after the first completed profiles
showed that the low-compute minima were below 0.5M. They are the two smallest
valid head-dimension-16 configurations and are run only where they satisfy the
step limits. The requested 1/2/4/8M grid is otherwise unchanged.

`10M*` is an experimental extension used only for the fixed-`D/N=40`
block-length comparison; it is not added retroactively to the completed
IsoFLOP grid.

## Objective and compute

The BD3-style dual stream is `[x_t | x_0]`. A noisy block sees itself
bidirectionally and clean preceding blocks, but never its clean target block
or future blocks. Blocks contain four tokens. The noise probability is uniform
and independently sampled by block; masked CE is importance weighted by
`1/t` to estimate diffusion NELBO.

For sequence length `L=256`, leading training FLOPs per clean token are

```text
F_token = 12*P_layers + 48*L*n_layer*d_model + 6*d_model*8193
```

This counts forward/backward transformer matrices on both streams, dense
attention over the full `2L` sequence, and the LM head on noisy positions.
Pointwise operations are omitted. Runs use dense attention, so masked pairs
are not discounted.

## Experiment

- Compute: `1e14, 3e14, 1e15, 3e15, 1e16` FLOPs.
- Fixed batch: 64 sequences = 16,384 clean tokens/update.
- Sequence length 256; diffusion block length 4.
- Runs outside 150–25,000 optimizer steps are skipped.
- AdamW, betas `(0.9, 0.95)`, matrix weight decay 0.1, no dropout.
- WSD: 5% warmup, 80% stable, 15% linear decay to zero.
- Peak LR sweep at every point: `3e-4, 9e-4, 2.7e-3, 8.1e-3`.
- Boundary LR winners are extended geometrically by 3x until the selected
  full run has worse immediate neighbors on both sides.
- BF16, gradient clipping 1.0, seed 1337, one H100.

Feasible step counts:

| C | 0.14M | 0.29M | 0.5M | 1M | 2M | 4M | 8M |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1e14 | 4,845 | 2,294 | 1,009 | 526 | 218 | — | — |
| 3e14 | 14,536 | 6,882 | 3,028 | 1,580 | 655 | 308 | 157 |
| 1e15 | — | 22,940 | 10,095 | 5,269 | 2,184 | 1,027 | 525 |
| 3e15 | — | — | — | 15,809 | 6,554 | 3,083 | 1,576 |
| 1e16 | — | — | — | — | 21,849 | 10,277 | 5,255 |

## Commands

```bash
python -m pip install -r scale_to_climbmix/requirements.txt
python scale_to_climbmix/download_data.py --workers 4
python scale_to_climbmix/prepare_data.py --workers 4
python scale_to_climbmix/test_scale.py
python scale_to_climbmix/run_sweep.py --dry-run
python scale_to_climbmix/run_sweep.py --workers 4 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --compile --attention-backend cudnn \
  --wandb-project climbmix-isoflop-scaleup
python scale_to_climbmix/bracket_lr.py --workers 4
python scale_to_climbmix/analyze.py
```

Completed runs are restart-safe. Final selections use complete runs only.
`analyze.py` refuses to fit a profile until every LR choice has a positive,
two-sided discrete curvature.

## Pure-BD IsoFLOP result

The initial 1/2/4/8M grid was adaptively extended downward because the first
two minima lay below 0.5M. All 25 final `(C,N)` points have full-run,
two-sided, positive-curvature LR brackets. Five points required one upper 3x
extension; none required a second extension.

The primary fit is an L2 quadratic in `log10(N)` through the three
lowest-validation-loss points on each profile. Curves are drawn only across
the span of those three support points; the remaining measurements are shown
as faded points rather than treated as part of the local parabola.

| C | optimal counted N | optimal clean tokens D | D/N | minimum NELBO |
|---|---:|---:|---:|---:|
| 1e14 | 0.237M | 45.7M | 192.5 | 5.6784 |
| 3e14 | 0.366M | 79.3M | 217.0 | 5.4310 |
| 1e15 | 0.875M | 95.3M | 108.8 | 5.0239 |
| 3e15 | 1.152M | 208.9M | 181.3 | 4.7254 |
| 1e16 | 2.561M | 270.0M | 105.4 | 4.4448 |

With compute normalized at `1e15` FLOPs:

```text
N_opt(C)     = 7.5e5 * (C / 1e15)^0.514
N_eff,opt(C) = 7.2e5 * (C / 1e15)^0.608
D_opt(C)     = 1.2e8 * (C / 1e15)^0.392
L_min(C)     = 5.03  * (C / 1e15)^-0.0547
```

The counted-parameter exponent is now close to the familiar square-root
allocation. Full-profile L1 gives `N_opt ~ C^0.513`, while full-profile L2
gives `C^0.536`; the allocation exponent is therefore insensitive to this
fit choice even though individual low-compute minima move. The
additive-loss-floor sensitivity fit collapses to a zero floor, so it does not
improve on the pure power law over these five budgets.

Outputs:

- `figures/isoflop_scaling.{png,pdf}`: final four-panel IsoFLOP figure.
- `figures/optimization_diagnostics.{png,pdf}`: selected LR trends and WSD
  train traces.
- `results/best_runs.csv`, `results/optimal_allocation.csv`, and
  `results/summary.json`: machine-readable selections, fits, diagnostics, and
  L2 sensitivity results.

## Adaptive IsoFLOP extension through 1e18

The batch-128 scale-up is complete through `3e17`; the final `1e18` wave is
running. After every locally bracketed minimum, the five highest-compute
optima are refit before the next neighborhood is chosen. The fit uses the
measured minimum and its immediate lower and upper model-size neighbors, not
the three globally lowest losses, so both sides of the local basin determine
the quadratic.

| C | forecast N | fitted N | fitted clean tokens D | D/N | fitted NELBO | forecast error |
|---:|---:|---:|---:|---:|---:|---:|
| 3e16 | 4.303M | 4.416M | 0.463B | 104.9 | 4.14035 | -2.6% |
| 1e17 | 8.346M | 9.338M | 0.729B | 78.1 | 3.88064 | -10.6% |
| 3e17 | 15.536M | 13.087M | 1.568B | 119.8 | 3.66372 | +18.7% |
| 1e18 | 28.229M | running | running | running | running | running |

The first two predictions landed inside their initial neighborhoods. At
`3e17`, the initial 12.3M/15.5M/19.1M/24.3M wave was still descending at its
low edge. Parallel 10.3M and 8.5M extensions bracketed the measured minimum;
the fitted local support is 10.3M/12.3M/15.5M. Thus the predict-next strategy
did not prevent a miss, but the boundary check prevented an inaccurate
edge fit and corrected the next forecast.

Using the five highest-compute completed profiles gives

```text
N_opt(C) = 8.23M * (C / 1e17)^0.535
D_opt(C) = 0.829B * (C / 1e17)^0.436
```

The fit through only the three batch-128 profiles gives
\(N_{\mathrm{opt}}\propto C^{0.474}\) and
\(D_{\mathrm{opt}}\propto C^{0.527}\), close to the expected square-root
allocation. The corresponding all-budget sensitivities are 0.518 and 0.423.
No exponent is constrained to 0.5.

### Learning-rate decisions

The scale-up keeps a candidate LR only for at least a 1% relative validation
NELBO improvement. The batch transition tested a 3x increase; later checks
test a 3x decrease only after anchor model size or compute grows by roughly
4x from the previous check.

| C | anchor | incumbent | probe | relative improvement | decision |
|---:|---:|---:|---:|---:|---|
| 3e16 | 4.4M | 9e-4 | 2.7e-3 | -0.99% | retain 9e-4 |
| 1e17 | 8.5M | 9e-4 | 3e-4 | -1.99% | retain 9e-4 |
| 3e17 | — | 9e-4 | not triggered | — | retain 9e-4 |
| 1e18 | 28.1M | 9e-4 | 3e-4 | running | pending |

The initial final wave used 22.3M/28.1M/35.3M at `9e-4` plus the 28.1M
`3e-4` probe. Adaptive extensions down through 12.3M use at most 5.542B
clean tokens, within the prepared 5.776B-token training set, so every run
remains below one effective epoch.

### Parallelism, efficiency, and runtime

Four independent one-GPU jobs minimize wave latency. Four-way DDP divides
global batch 128 into local batches of 32 and measured only 230k aggregate
clean tokens/s in the 10M smoke test, versus 349k for a comparable cold
single-GPU run and 603k for a cache-warm long single-GPU run. DDP is therefore
not used for the main waves. Independent speculative size jobs can consume
extra GPU-hours if an LR probe wins, but they reduce best-case wall time and
do not increase the worst-case dependency path.

Long runs use BF16 autocast, TF32, fused AdamW, foreach gradient clipping,
TorchInductor compilation, and dense cuDNN SDPA. FlashAttention cannot
represent the experiment's non-null boolean BD mask on this PyTorch stack.
All runs upload to the `climbmix-isoflop-scaleup` W&B project, and an upload
initialization failure fails training.

Observed one-GPU accounted MFU rises from 5.9–6.7% at `3e16`, to 8.0–8.8%
at `1e17`, to 9.6–11.4% at `3e17`. The corresponding per-run ranges were
7.5–8.6 minutes, 19.1–21.0 minutes, and 44.3–60.2 minutes. The `1e18` wave
currently projects to about 2.8–3.0 hours.

Interim outputs are:

- `figures_scaleup/isoflop_scaling_to_1e18.{png,pdf}`
- `figures_scaleup/isoflop_profile_details.{png,pdf}`
- `figures_scaleup/scaleup_diagnostics.{png,pdf}`
- `results_scaleup/summary.json`, `optimal_allocation.csv`,
  `best_runs.csv`, `forecast_history.json`, and `lr_decisions.json`

### Historical-curve refinement

The five historical profiles were revisited with targeted batch-64 points
while the final scale-up extensions trained. Existing measurements supplied
the initial neighbors, and roughly 1.25x points were added only in the
direction of a lower loss. LR probes were limited to historical transition
regions and used the same 1% acceptance threshold. The 0.21M interpolation
uses width 24 with one 24-dimensional head; every other selected refinement
model retains head dimension 16.

| C | refined N* | refined D* | D/N | refined NELBO |
|---:|---:|---:|---:|---:|
| 1e14 | 0.249M | 43.6M | 174.9 | 5.63461 |
| 3e14 | 0.379M | 84.9M | 224.0 | 5.26011 |
| 1e15 | 0.841M | 99.2M | 118.0 | 4.99174 |
| 3e15 | 1.178M | 203.5M | 172.8 | 4.69197 |
| 1e16 | 1.973M | 360.1M | 182.5 | 4.46028 |

The refitted historical laws are `N_opt ~ C^0.459` and `D_opt ~ C^0.442`.
All five vertices lie inside the measured minimum's immediate lower and upper
neighbors. Outputs are in `results_refinement/` and `figures_refinement/`;
the final combined scale-up analysis consumes this refined summary.

### Batch-128 pure-AR and matched-step scale-up plan

The next two adaptive studies use batch size 128 at every budget from `1e14`
through `1e18`. Pure AR starts from the exact architecture nearest `D/N=20`
and measures roughly 1.25x neighbors on both sides. Its compute accounting
uses the triangular causal work actually executed by FlashAttention,
including `L(L+1)/2` attended pairs, rather than the historical dense causal
upper bound. Each locally bracketed curve updates the allocation law before
the next budget.

AR LR checks occur only after the anchor model grows by more than 4x from the
previous check. A 3x-lower LR is accepted only if validation cross-entropy
improves by at least 1%; otherwise the incumbent is propagated. The exact
initial neighborhoods and probe triggers are recorded in
`followup_isoflop_plan.json`.

The subsequent fixed `p_AR=0.4` study matches the step count and clean-token
count of a hypothetical batch-128 pure-BD run at each nominal `(C,N)`. It
uses causal FlashAttention for the first 40% of steps, resets AdamW, and uses
cuDNN masked SDPA for the final 60% BD phase. Both the nominal pure-BD
equivalent compute and the lower realized mixed compute are reported. AR and
BD receive separate phase-local 5%/80%/15% WSD schedules and separately
selected peak LRs.

## AR-to-BD curriculum follow-up

After the pure-BD LRs are frozen, the follow-up evaluates
`p_AR = 0.1, ..., 0.6` at every feasible `(C,N)` without another LR sweep.
Every branch inherits the selected pure-BD peak LR at the identical point.

To share AR work honestly under WSD, each `(C,N)` uses one common absolute
warmup equal to 5% of its shortest (`p=0.1`) AR phase, followed by a shared
stable AR trunk. For each requested transition, training branches 15% before
the transition and performs that phase's linear AR decay. AdamW state is then
discarded and the BD phase receives a fresh 5%/80%/15% WSD schedule. Thus all
branches have a real AR decay, while the expensive warmup/stable prefix is
trained once.

The shared trunk also continues to a pure-AR endpoint at the same compute and
LR. Model and optimizer checkpoints are saved at every required branch point.
AR and BD training-only seconds per step are recorded and compared with their
architecture-aware FLOP ratio.

```bash
python scale_to_climbmix/run_ar_trunks.py --workers 4
python scale_to_climbmix/run_curriculum.py --workers 4
python scale_to_climbmix/curriculum_analyze.py
```

The mixed-run step ceiling is 35,000 rather than 25,000. AR is cheaper per
step, and 35,000 includes every requested `p=0.6` point; the actual maximum is
30,887. All mixed and pure-AR runs remain below one pass over the 1.606B-token
training split.

### Curriculum result

All 150 mixed branches and all 25 shared pure-AR endpoints completed. The
curriculum comparison retains a full-profile L1 quadratic for every `p_AR` so
that all seven envelopes use the same robust rule. The table reports both
that fitted envelope and the more conservative best point actually measured
on the discrete model grid:

| C | fitted best p_AR | fitted gain | fitted optimal N | measured best p_AR | measured gain | measured N |
|---|---:|---:|---:|---:|---:|---:|
| 1e14 | 0.4 | 0.3798 | 0.341M | 0.5 | 0.1667 | 0.504M |
| 3e14 | 0.6 | 0.2473 | 0.638M | 0.2 | 0.1745 | 0.504M |
| 1e15 | 0.4 | 0.2843 | 1.004M | 0.3 | 0.0853 | 0.966M |
| 3e15 | 0.3 | 0.0081 | 1.331M | 0.6 | 0.0015 | 1.986M |
| 1e16 | 0.5 | 0.0291 | 4.713M | 0.4 | 0.0439 | 3.920M |

The fitted and measured columns differ materially at low compute because the
robust pure-BD L1 quadratic discounts a low measured point, while some
curriculum quadratics dip below their observed samples. The summary figure
therefore overlays the best measured points as `x` marks rather than silently
presenting the larger fitted gains as direct observations.

At fixed model size, curriculum is most useful for a model that is too large
for its compute budget. For example, at `C=1e16`, the 2M model is best with
pure BD, the 4M model has an interior optimum at `p_AR=0.4`, and the 8M model
continues improving through `p_AR=0.6`. Re-optimizing model size absorbs much
of that fixed-size gain. By `3e15` FLOPs the compute-optimal gain is
effectively zero, although deliberately oversized models still benefit
strongly. The gain reappears weakly at `1e16`, so five budgets and one seed
are not enough to claim a monotone law for curriculum benefit.

Data reuse is not a confounder here: the largest mixed run consumes 0.315
training epochs and the largest pure-AR endpoint consumes 0.435 epochs.

### AR versus block-diffusion wall time

These are medians across available compute budgets. Measurements came from
the concurrent four-worker sweep, so the machine-readable table retains the
per-run values and their scheduling variance.

| model | theoretical BD/AR | measured median BD/AR | median AR ms/step | median BD ms/step |
|---|---:|---:|---:|---:|
| 0.14M | 1.362 | 1.175 | 20.9 | 24.4 |
| 0.29M | 1.386 | 1.060 | 20.9 | 22.6 |
| 0.5M | 1.673 | 1.229 | 27.5 | 33.8 |
| 1M | 1.710 | 1.435 | 29.9 | 42.9 |
| 2M | 1.952 | 1.685 | 44.5 | 76.5 |
| 4M | 2.094 | 1.844 | 73.6 | 142.8 |
| 8M | 2.092 | 1.978 | 99.1 | 204.7 |

Dense FLOP accounting predicts roughly a 2x BD/AR ratio at the larger sizes,
and the measured 8M median is 1.98x. Ratios are lower for tiny models because
fixed kernel/launch overhead makes the one-stream AR step relatively
inefficient. The range bars in the figure expose concurrent-run timing
variance; they are not confidence intervals.

Curriculum outputs:

- `figures_curriculum/curriculum_summary.{png,pdf}`: fitted and observed
  compute-optimal envelopes, optimal-size shifts, and wall-clock comparison.
- `figures_curriculum/fixed_model_l_vs_p/fixed_N_*.{png,pdf}`: one figure per
  model size with every available compute budget. Exactly one subplot title
  is red: the measured budget closest in log space to the continuous
  `C_opt(N)` obtained by inverting the pure-BD allocation law.
- `results_curriculum/curriculum_runs.csv`: all 150 mixed results.
- `results_curriculum/optimal_envelopes.csv` and
  `results_curriculum/observed_envelopes.csv`: fitted and measured envelopes.
- `results_curriculum/phase_timing.csv`: per-point AR and BD step times.
- `results_curriculum/pure_ar_runs.csv`: all 25 pure-AR endpoint losses,
  timings, token counts, and epoch counts.

## 10M fixed-token block-length follow-up

A separate matched-step experiment extends the model table to the
experimental 10M configuration and fixes `D/N=40`:

```text
N = 9,692,032
D = 387,678,208 clean tokens
steps = 23,662
p_AR = 0.0, 0.1, 0.3, 0.5, 0.7
block length = 4, 32
peak LR = 9e-4
weight decay = 0.1
```

The two block-length sweeps use identical total steps and clean tokens.
Curriculum branches reuse one shared AR trunk, reset AdamW at transition, and
give the BD phase a fresh 5%/80%/15% WSD schedule. The AR prefix is
block-independent; only the BD phase and validation mask use the selected
block length.

```bash
python scale_to_climbmix/run_fixed_dn40_10m_blocks.py --dry-run
python scale_to_climbmix/run_fixed_dn40_10m_blocks.py --workers 4
python scale_to_climbmix/analyze_fixed_dn40_10m_blocks.py
```

Runs live in `results_fixed_dn40_10M/`. The final comparison is written to
`figures_fixed_dn40_10M/fixed_dn40_10M_blocks.{png,pdf}`.
