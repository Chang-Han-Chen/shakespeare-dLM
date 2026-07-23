# Block diffusion scaling on TinyShakespeare

This folder is a clean, self-contained IsoFLOP experiment. It imports no code
from the legacy repository; its only external input is `../data.txt`.

The follow-up AR-warm-start sweep, including its p-dependent compute formula
and exact feasible-step tables, is documented in
[`CURRICULUM.md`](CURRICULUM.md).

## Question

For character-level block diffusion on TinyShakespeare:

1. What model size minimizes validation diffusion NELBO at fixed compute?
2. How do compute-optimal model size and training-token count scale?
3. What is the optimal parameter-to-token ratio?

The primary fit is a quadratic in `log10(N)` using least absolute deviations
(L1), so an isolated loss does not receive quadratically larger influence.
With three quadratic coefficients and at most eight observations per profile,
the exact L1 solution is found by enumerating its three-point support sets.
Full-profile L2 and lowest-three-point L2 fits are saved as sensitivity checks.

The quadratic vertices are fit to power laws for counted model size
`N_opt(C)`, compute-effective model size `N_eff,opt(C)`, and clean training
tokens `D_opt(C)`. The minimum-loss panel shows both a primary two-parameter
pure power law and a dashed three-parameter fit with an additive asymptotic
floor. The latter is a sensitivity curve: five compute budgets do not reliably
identify its fitted floor.

## Model

The backbone is a bias-free Llama-2-style transformer: learned RMSNorm, RoPE,
full multi-head attention, and SwiGLU. Training uses the BD3 dual stream
`[x_t | x_0]`: a noisy block sees itself bidirectionally and clean preceding
blocks, never its clean target block or future blocks. Mask probability is
sampled independently per block and the masked CE is weighted by `1/t`.

Input token embeddings are excluded from counted `N`; the untied LM head and
all learned norms are included. The two extra-small models were added only at
`C=1e13` after the corrected profile showed that its minimum was below the
original 0.01M grid. They make that minimum an interpolation rather than a
boundary extrapolation.

| label | layers | width | heads | head dim | FFN | counted N |
|---|---:|---:|---:|---:|---:|---:|
| 0.002M | 2 | 8 | 1 | 8 | 24 | 2,232 |
| 0.005M | 2 | 12 | 2 | 6 | 32 | 4,308 |
| 0.01M | 3 | 16 | 2 | 8 | 48 | 11,152 |
| 0.02M | 3 | 24 | 3 | 8 | 64 | 22,488 |
| 0.04M | 3 | 32 | 4 | 8 | 88 | 39,968 |
| 0.1M | 4 | 48 | 3 | 16 | 128 | 114,192 |
| 0.2M | 4 | 64 | 4 | 16 | 176 | 205,504 |
| 0.4M | 5 | 80 | 5 | 16 | 216 | 393,360 |
| 0.8M | 7 | 96 | 6 | 16 | 256 | 781,920 |
| 1.6M | 8 | 128 | 8 | 16 | 344 | 1,591,680 |

Every model has at least two layers. Models above the 0.04M scale use head
dimension 16 as requested.

## Compute accounting

The original `12ND` rerun treated all counted parameters as though they were
dual-stream transformer matrices. That is not accurate for these very small
models: dense attention is material, norms do not contribute leading matmuls,
and the LM head runs on only the noisy stream.

The corrected training FLOPs per clean token are

```text
F_token = 12*P_layers + 48*L*n_layer*d_model + 6*d_model*V
```

where `P_layers = n_layer*(4*d_model^2 + 3*d_model*d_ff)`, `L=256`, and
`V=66`. The three terms count:

1. forward and backward for attention projections and SwiGLU matrices on both
   streams;
2. dense `QK^T` and `AV` attention over the full `2L` sequence;
3. the LM head on the noisy `L` positions only.

Pointwise operations are omitted. The current PyTorch path uses dense cuDNN
scaled-dot-product attention; the structured mask does not make it
block-sparse. It would therefore be misleading to discount masked attention
pairs without implementing and profiling a sparse kernel.

Define the compute-effective model size

```text
N_eff = F_token / 12
```

so the budget identity is `C = 12*N_eff*D`. `N_eff` is not a parameter count;
it is a convenient representation of all leading per-token compute. Across
the model table, `N_eff/N` falls from 8.20× at 0.002M to 1.65× at 1.6M. This
size-dependent overhead is central to interpreting the scaling exponents.

## Experiment

- Compute budgets: `1e13, 3e13, 1e14, 3e14, 1e15` FLOPs.
- Sequence length 256; diffusion block length 4.
- Fixed batch size 128 sequences = 32,768 clean tokens/update.
- Runs outside 150–25,000 optimizer steps are skipped.
- Initial peak-LR sweep at every feasible `(C,N)`:
  `0.001, 0.003, 0.009, 0.027`, with exact 3× spacing.
- If a boundary LR wins, full-budget trials continue geometrically until the
  winner has worse immediate neighbors at `LR/3` and `3*LR`.
- LR selection uses final validation NELBO from full runs, not partial curves.
- AdamW, `betas=(0.9, 0.95)`, weight decay 0.1 on matrix weights, no dropout.
- WSD schedule: 5% linear warmup, 80% stable, 15% linear decay to zero.
- One seed (`1337`), BF16 on one NVIDIA H100 80GB, gradient clipping at 1.0.
- Two experiments were run concurrently to improve utilization.

Feasible optimizer steps (dashes are deliberately skipped):

| C | .002M | .005M | .01M | .02M | .04M | .1M | .2M | .4M | .8M | 1.6M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e13 | 1,388 | 894 | 426 | 266 | 185 | — | — | — | — | — |
| 3e13 | — | — | 1,278 | 800 | 556 | 247 | 164 | — | — | — |
| 1e14 | — | — | 4,262 | 2,669 | 1,856 | 823 | 546 | 318 | 173 | — |
| 3e14 | — | — | 12,787 | 8,009 | 5,569 | 2,470 | 1,640 | 954 | 520 | 289 |
| 1e15 | — | — | — | — | 18,564 | 8,235 | 5,469 | 3,181 | 1,735 | 965 |

This gives 31 feasible `(C,N)` points, 124 initial LR runs, and 10 boundary
extensions: 134 complete full runs in total. Every selected LR is a strict
local winner against completed runs at one-third and three times its value.

## Run

```bash
python scaling/test_scaling.py
python scaling/run_sweep.py --dry-run
python scaling/run_sweep.py --workers 2
python scaling/bracket_lr.py --workers 2
python scaling/analyze.py
```

Completed runs are restart-safe: `run_sweep.py` skips valid result JSON files.
`bracket_lr.py` extends boundary LR searches. `analyze.py` refuses to proceed
unless all selected LRs have completed worse immediate neighbors.

The corrected outputs are kept separate from the legacy proxy-compute run:

- `figures_dense_attention/isoflop_scaling.{png,pdf}`: headline L1 IsoFLOP
  profiles and scaling-law figure, including pure-power and fitted-floor
  minimum-loss curves.
- `figures_dense_attention/compute_accounting_comparison.{png,pdf}`:
  old `12ND` proxy versus corrected allocation laws.
- `figures_dense_attention/isoflop_fit_comparison.{png,pdf}`: full-profile L1
  versus L2 quadratic fits.
- `figures_dense_attention/isoflop_scaling_l2.{png,pdf}`: full-profile L2
  result.
- `figures_dense_attention/isoflop_scaling_lowest3_l2.{png,pdf}`: local
  quadratic through the three lowest observations at each compute budget.
- `figures_dense_attention/optimization_diagnostics.{png,pdf}`: selected LR
  grid, local LR neighborhoods, and normalized WSD train traces.
- `results_dense_attention/best_runs.csv`: selected full run at every feasible
  `(C,N)`.
- `results_dense_attention/optimal_allocation*.csv`: fitted optima under all
  three quadratic methods.
- `results_dense_attention/summary*.json`: coefficients, diagnostics, and
  machine-readable results.

The old proxy results remain in `results/` and `figures/` for auditability.

## Corrected results

All five primary L1 quadratics are convex, and every fitted vertex lies inside
the measured model range. Mean absolute residuals range from 0.0054 to 0.0273
NELBO.

| C | counted N optimum | effective N optimum | clean tokens D | D / counted N | minimum NELBO | L1 MAE |
|---|---:|---:|---:|---:|---:|---:|
| 1e13 | 7,316 | 42,956 | 19,399,754 | 2,651.6 | 2.2987 | 0.0102 |
| 3e13 | 23,129 | 96,962 | 25,783,246 | 1,114.8 | 2.0779 | 0.0054 |
| 1e14 | 49,974 | 162,860 | 51,168,651 | 1,023.9 | 1.8946 | 0.0124 |
| 3e14 | 124,617 | 328,160 | 76,182,349 | 611.3 | 1.7403 | 0.0273 |
| 1e15 | 332,479 | 694,749 | 119,947,337 | 360.8 | 1.6310 | 0.0094 |

With compute normalized at `1e14` FLOPs, the primary L1 laws are

```text
counted N_opt(C) = 5.20e4 * (C / 1e14)^0.809
N_eff,opt(C)     = 1.75e5 * (C / 1e14)^0.589
D_opt(C)         = 4.76e7 * (C / 1e14)^0.411
counted N_opt/D  = 1.09e-3 * (C / 1e14)^0.398
N_eff,opt/D      = 3.68e-3 * (C / 1e14)^0.178
L_min(C)         = 1.911 * (C / 1e14)^-0.0749
```

Log-space R-squared is 0.996 for counted `N`, 0.995 for `N_eff`, 0.990 for
tokens, and 0.992 for minimum loss.

The dashed additive-floor sensitivity fit is

```text
L_min(C) = 1.181 + 0.707 * (C / 1e14)^-0.199
```

Its in-sample mean absolute error is 0.0041 NELBO, versus 0.0200 for the pure
power law, and its raw-loss R-squared is 0.9995 versus 0.9925. This is visibly
better over the measured interval, but it uses three parameters for only five
fitted minima, leaving two residual degrees of freedom. The floor and exponent
should therefore not be interpreted as reliable long-range extrapolations.

### Interpretation

The raw parameter result is not Chinchilla-like: counted `N` scales as
`C^0.809`, while tokens scale as `C^0.411`. But counted `N` is not proportional
to per-token compute in this sweep. Attention and head overhead dominate the
smallest architectures and become relatively cheaper as width grows.

In compute units, the allocation is much closer to familiar dense-transformer
behavior:

```text
N_eff,opt scales as C^0.589
D_opt     scales as C^0.411
```

The exponents sum to one because `C=12*N_eff*D`. Thus the corrected experiment
supports a roughly 59%/41% model-compute/data-compute allocation, not an
81%/41% allocation. It is still not exactly 50/50. Plausible reasons include
the character-level objective, block-diffusion corruption, the very small
architecture range, repeated passes over the roughly one-million-character
training corpus, and one-seed noise. The sweep alone cannot distinguish those
causes.

The counted parameter-to-token ratio is not constant: `D/N` declines from
about 2,652 to 361 tokens per counted parameter over the measured budgets.
That trend is expected once the compute overhead per counted parameter itself
changes strongly with model size.

### Quadratic sensitivity

| quadratic method | counted-N exponent | effective-N exponent | token exponent |
|---|---:|---:|---:|
| full-profile L1 (primary) | 0.809 | 0.589 | 0.411 |
| full-profile L2 | 0.816 | 0.595 | 0.405 |
| lowest-three L2 | 0.912 | 0.672 | 0.328 |

L1 and full-profile L2 agree closely on the compute allocation. The
lowest-three fit is substantially more model-heavy, especially at the two
largest budgets. Three points determine a quadratic exactly, leaving no
residual degrees of freedom, so the local result is best treated as a
sensitivity bound rather than the primary estimate.

### Optimization diagnostics

All 31 selected LRs are strict local winners using full runs. For all 31
selected runs, the mean logged train loss in the second half of the stable WSD
phase is below that in the first half; the median reduction is 13.43% and the
smallest is 2.01%. Every run drops further during decay, with a median
additional reduction of 6.89% and a minimum of 0.63%. No selected trace shows
a sustained U-shaped train loss.

These are one-seed estimates on a small, repeatedly sampled character dataset.
They establish the result for this controlled sweep, but do not measure
seed-to-seed uncertainty or establish a universal block-diffusion scaling law.
