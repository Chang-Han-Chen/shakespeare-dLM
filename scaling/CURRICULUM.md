# AR warm-start curriculum IsoFLOP sweep

This extension keeps the architecture, dataset, batch size, model table,
compute budgets, and block-diffusion objective from the main experiment. It
sweeps

```text
p_ar = 0.1, 0.2, 0.3, 0.4, 0.5
```

The first `p_ar` fraction of whole optimizer steps uses single-stream causal
next-token prediction. The remaining steps use the dual-stream block-diffusion
objective with block length 4. The model weights transfer directly because
both phases use the same Llama-2-style backbone. AdamW moments and step
counters are discarded at the transition.

## Compute accounting

For model matrices `P_layers`, sequence length `L=256`, width `d`, layers `n`,
and vocabulary size `V=66`, the two phase costs per clean token are

```text
F_AR =  6*P_layers + 12*L*n*d + 6*d*V
F_BD = 12*P_layers + 48*L*n*d + 6*d*V
```

The AR transformer matrices process one stream and its dense causal attention
kernel is `L` by `L`. Block diffusion processes two streams and the current
dense attention kernel is `2L` by `2L`. Both phases evaluate the LM head at
`L` positions.

With `D` total clean-token presentations, the nominal curriculum budget is

```text
C = D * [p_ar*F_AR + (1-p_ar)*F_BD]
  = D * [(12-6*p_ar)*P_layers
         + (48-36*p_ar)*L*n*d
         + 6*d*V]
```

The actual transition must fall on an optimizer-step boundary. The launcher
rounds the requested AR share to the nearest step, computes the exact sum of
AR-step and BD-step FLOPs, and chooses the largest total step count that does
not exceed `C`.

## Optimization policy

- Batch size: 128 sequences, or 32,768 clean tokens per optimizer step.
- The peak LR at each `(C,N)` is copied from the full pure-BD run selected
  previously against completed neighbors at one-third and three times its LR.
  There is no new LR sweep.
- AdamW uses `betas=(0.9,0.95)`, matrix weight decay 0.1, no dropout, and
  gradient clipping at 1.0.
- Each optimizer lifetime gets its own WSD schedule: 5% linear warmup, 80%
  stable, and 15% linear decay to zero. The BD schedule restarts with the
  optimizer.
- Warmup lengths are fractions of their own phases, not the old fixed 50-step
  restart. Among feasible runs the shortest AR and BD phases have 18 and 120
  steps, respectively; rounding makes the largest realized warmup share
  6.45%.
- Runs outside 150–25,000 total optimizer steps are skipped.

### Restart-warmup ablation

A paired `p_ar=0.3` ablation held the seed, phase split, inherited peak LR,
optimizer reset, and AR schedule fixed while changing only the BD restart
warmup from 5% to zero.

| point | C | N | peak LR | no warmup | 5% warmup | improvement |
|---|---:|---:|---:|---:|---:|---:|
| tiny/high-LR | 1e13 | 0.01M | 0.027 | 2.24500 | 2.22005 | 0.02495 |
| short | 3e13 | 0.1M | 0.009 | 2.26355 | 2.23685 | 0.02670 |
| medium | 1e14 | 0.04M | 0.009 | 1.91397 | 1.90402 | 0.00995 |
| wider | 3e14 | 0.2M | 0.003 | 1.76127 | 1.75223 | 0.00904 |

The 5% restart warmup wins all four pairs, by 0.01766 NELBO on average.
Therefore the main sweep retains it. These are one-seed paired checks, so they
support the schedule choice rather than estimating a precise universal effect
size.

## Feasible total optimizer steps

### p_ar=0.1

| C | 0.002M | 0.005M | 0.01M | 0.02M | 0.04M | 0.1M | 0.2M | 0.4M | 0.8M | 1.6M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e13 | 1,496 | 962 | 458 | 286 | 199 | — | — | — | — | — |
| 3e13 | — | — | 1,375 | 860 | 597 | 264 | 175 | — | — | — |
| 1e14 | — | — | 4,583 | 2,867 | 1,990 | 881 | 584 | 339 | 185 | — |
| 3e14 | — | — | 13,751 | 8,602 | 5,972 | 2,644 | 1,752 | 1,018 | 555 | 308 |
| 1e15 | — | — | — | — | 19,909 | 8,815 | 5,842 | 3,393 | 1,849 | 1,027 |

### p_ar=0.2

| C | 0.002M | 0.005M | 0.01M | 0.02M | 0.04M | 0.1M | 0.2M | 0.4M | 0.8M | 1.6M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e13 | 1,620 | 1,042 | 495 | 309 | 214 | — | — | — | — | — |
| 3e13 | — | — | 1,487 | 929 | 644 | 284 | 188 | — | — | — |
| 1e14 | — | — | 4,958 | 3,096 | 2,146 | 948 | 626 | 363 | 198 | — |
| 3e14 | — | — | 14,874 | 9,290 | 6,439 | 2,844 | 1,881 | 1,090 | 593 | 329 |
| 1e15 | — | — | — | — | 21,464 | 9,480 | 6,270 | 3,636 | 1,979 | 1,096 |

### p_ar=0.3

| C | 0.002M | 0.005M | 0.01M | 0.02M | 0.04M | 0.1M | 0.2M | 0.4M | 0.8M | 1.6M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e13 | 1,769 | 1,136 | 539 | 336 | 232 | — | — | — | — | — |
| 3e13 | — | — | 1,619 | 1,009 | 698 | 307 | 203 | — | — | — |
| 1e14 | — | — | 5,398 | 3,366 | 2,327 | 1,025 | 676 | 392 | 212 | — |
| 3e14 | — | — | 16,195 | 10,097 | 6,985 | 3,076 | 2,029 | 1,175 | 638 | 353 |
| 1e15 | — | — | — | — | 23,282 | 10,256 | 6,766 | 3,917 | 2,129 | 1,176 |

### p_ar=0.4

| C | 0.002M | 0.005M | 0.01M | 0.02M | 0.04M | 0.1M | 0.2M | 0.4M | 0.8M | 1.6M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e13 | 1,946 | 1,249 | 592 | 369 | 254 | — | — | — | — | — |
| 3e13 | — | — | 1,777 | 1,105 | 762 | 335 | 220 | — | — | — |
| 1e14 | — | — | 5,925 | 3,685 | 2,544 | 1,117 | 734 | 424 | 230 | — |
| 3e14 | — | — | 17,775 | 11,059 | 7,630 | 3,350 | 2,204 | 1,273 | 690 | 380 |
| 1e15 | — | — | — | — | — | 11,169 | 7,346 | 4,244 | 2,303 | 1,269 |

### p_ar=0.5

| C | 0.002M | 0.005M | 0.01M | 0.02M | 0.04M | 0.1M | 0.2M | 0.4M | 0.8M | 1.6M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1e13 | 2,163 | 1,386 | 657 | 407 | 280 | — | — | — | — | — |
| 3e13 | — | — | 1,969 | 1,222 | 841 | 367 | 241 | — | — | — |
| 1e14 | — | — | 6,565 | 4,073 | 2,803 | 1,226 | 803 | 463 | 251 | — |
| 3e14 | — | — | 19,695 | 12,221 | 8,409 | 3,678 | 2,411 | 1,389 | 752 | 413 |
| 1e15 | — | — | — | — | — | 12,261 | 8,035 | 4,631 | 2,508 | 1,378 |

There are 153 curriculum datapoints in total. The `0.04M, C=1e15` run becomes
longer than 25,000 steps at `p_ar=0.4` and `0.5`, so those two configurations
are deliberately omitted.

## Results

All 153 fixed-LR curriculum runs completed. All 30 plotted IsoFLOP
quadratics—five pure-BD controls and 25 curriculum profiles—are convex, their
vertices are interior to the measured model range, and their maximum
full-profile L1 mean absolute error is 0.0327 NELBO.
The largest compute undershoot from whole-step rounding is 0.567%.

| C | best measured p_ar | optimal counted N | optimal clean tokens D | fitted minimum NELBO | improvement over pure BD |
|---|---:|---:|---:|---:|---:|
| 1e13 | 0.4 | 11,350 | 19,185,953 | 2.2311 | 0.0676 |
| 3e13 | 0.4 | 31,547 | 29,129,087 | 1.9955 | 0.0823 |
| 1e14 | 0.3 | 73,700 | 47,309,331 | 1.8673 | 0.0274 |
| 3e14 | 0.2 | 150,653 | 76,692,633 | 1.7348 | 0.0055 |
| 1e15 | 0.3 | 497,657 | 104,181,786 | 1.6120 | 0.0190 |

There is no single best curriculum fraction at every compute budget.
`p_ar=0.3` is a reasonable center, while the fitted discrete optimum ranges
from 0.2 to 0.4. The curriculum beats the pure-BD control at 26 of the 31
directly matched `(C,N)` points. The 0.0055 improvement at `C=3e14` is small
enough to treat cautiously given single-seed and kernel-level run noise.

The allocation exponents from the L1 profile minima are:

| p_ar | counted N exponent | effective N exponent | D exponent |
|---:|---:|---:|---:|
| 0.0 | 0.809 | 0.589 | 0.411 |
| 0.1 | 0.787 | 0.574 | 0.426 |
| 0.2 | 0.823 | 0.610 | 0.390 |
| 0.3 | 0.795 | 0.593 | 0.407 |
| 0.4 | 0.816 | 0.614 | 0.386 |
| 0.5 | 0.907 | 0.694 | 0.306 |

As before, the effective-model and data exponents sum to one because
`C=12*N_eff*D`. The `p_ar=0.5` law is visibly more model-heavy and should not
be extrapolated beyond this small architecture range.

The phase training curves remain healthy. In both phases, the later stable
window has lower mean training loss for all 153 runs. Median stable-phase
drops are 17.07% for AR and 9.73% for BD. Comparing smoothed windows just
before decay and at the end, all 153 runs improve further; median decay drops
are 5.25% and 5.14%, respectively.

## AR-phase weight-decay ablation at 0.8M

Because the AR curriculum benefit shrinks when a fixed model receives more
compute, a targeted ablation tested whether long AR phases need stronger
regularization. The model was fixed at 781,920 counted parameters. For every
feasible `(C, p_ar)` point, the inherited peak LR, phase lengths, WSD
schedules, and BD weight decay of 0.1 were unchanged. Only AR-phase AdamW
weight decay was tuned, starting from the original 0.1 run and doubling
upward until the measured validation NELBO winner had a higher-decay
neighbor. A 0.1 winner is reported as a lower-boundary result because values
below 0.1 were not searched.

| C | p=0.1 | p=0.2 | p=0.3 | p=0.4 | p=0.5 |
|---:|---:|---:|---:|---:|---:|
| 1e14 | 0.1 (+0.000) | 0.1 (+0.000) | 0.1 (+0.000) | 0.4 (+0.011) | 0.1 (+0.000) |
| 3e14 | 0.4 (+0.014) | 0.1 (+0.000) | 0.1 (+0.000) | 0.2 (+0.007) | 0.8 (+0.016) |
| 1e15 | 0.2 (+0.004) | 0.2 (+0.007) | 0.1 (+0.000) | 1.6 (+0.029) | 0.4 (+0.018) |

Each cell gives the selected AR weight decay followed by its NELBO improvement
over the original AR/BD weight decay of 0.1. Eight of the 15 points have an
interior bracket; seven select the 0.1 lower boundary. The strongest effect is
at `C=1e15, p_ar=0.4`, where the bracket is `(0.8, 1.6, 3.2)` and tuning lowers
NELBO from 1.63661 to 1.60763.

For the 0.8M model at `C=1e15`, the original best curriculum point was
`p_ar=0.2` with NELBO 1.63294, worse than the pure-BD control at 1.62365.
After tuning AR decay, the best point is `p_ar=0.4, wd_AR=1.6` at 1.60763, a
0.01602 improvement over pure BD. This supports excessive AR training or
AR-objective specialization as a real contributor, but the irregular winners
across `p_ar` and the single training seed do not isolate classical
train/validation overfitting.

## Fixed-token comparison at D/N = 60

A second targeted experiment removes the extra-update advantage from the
fixed-FLOP curriculum comparison. Model size and total clean-token
presentations are fixed within each panel, so every `p_ar` value receives the
same total optimizer steps. The experiment uses the existing 0.4M and 0.8M
models, compares block lengths 4 and 32, and sweeps

```text
p_ar = 0.0, 0.1, 0.3, 0.5, 0.7
wd_AR = 0.1, 0.2, 0.4, 0.8
```

Pure BD is run once per `(block length, N)`. For mixed runs, BD weight decay remains
0.1 and only AR weight decay changes. The inherited peak LRs are 3e-3 for
0.4M and 1e-3 for 0.8M. Each phase retains its own 5%/80%/15% WSD schedule,
and AdamW state is reset at transition.

Whole-step rounding gives:

| model | counted N | steps | realized D/N |
|---:|---:|---:|---:|
| 0.4M | 393,360 | 720 | 59.978 |
| 0.8M | 781,920 | 1,432 | 60.011 |

Each table entry below selects the best AR weight decay from the fixed grid.
Gain is `pure-BD NELBO - curriculum NELBO`, so positive values favor the
curriculum.

### Block length 4

| model | p_ar | best wd_AR | validation NELBO | gain |
|---:|---:|---:|---:|---:|
| 0.4M | 0.0 | — | 1.88757 | +0.00000 |
| 0.4M | 0.1 | 0.1 | 1.85618 | +0.03139 |
| 0.4M | 0.3 | 0.2 | 1.83971 | +0.04786 |
| 0.4M | 0.5 | 0.2 | 1.83921 | +0.04836 |
| 0.4M | 0.7 | 0.8 | 1.88940 | -0.00183 |
| 0.8M | 0.0 | — | 1.65706 | +0.00000 |
| 0.8M | 0.1 | 0.8 | 1.68336 | -0.02631 |
| 0.8M | 0.3 | 0.2 | 1.68233 | -0.02528 |
| 0.8M | 0.5 | 0.8 | 1.69994 | -0.04288 |
| 0.8M | 0.7 | 0.4 | 1.75706 | -0.10001 |

### Block length 32

| model | p_ar | best wd_AR | validation NELBO | gain |
|---:|---:|---:|---:|---:|
| 0.4M | 0.0 | — | 1.99050 | +0.00000 |
| 0.4M | 0.1 | 0.2 | 1.98346 | +0.00705 |
| 0.4M | 0.3 | 0.2 | 1.99352 | -0.00302 |
| 0.4M | 0.5 | 0.2 | 2.01292 | -0.02242 |
| 0.4M | 0.7 | 0.1 | 2.10096 | -0.11046 |
| 0.8M | 0.0 | — | 1.81148 | +0.00000 |
| 0.8M | 0.1 | 0.2 | 1.82859 | -0.01711 |
| 0.8M | 0.3 | 0.2 | 1.83468 | -0.02320 |
| 0.8M | 0.5 | 0.8 | 1.86854 | -0.05706 |
| 0.8M | 0.7 | 0.8 | 1.94921 | -0.13773 |

Curriculum therefore does not help both model sizes. With block length 4,
the 0.4M model improves by 0.04836 at `p_ar=0.5`, but every nonzero
curriculum fraction hurts the 0.8M model. Increasing the block length to 32
nearly removes the small-model effect: 0.4M improves by only 0.00705 at
`p_ar=0.1`, while all larger fractions hurt. Every nonzero fraction again
hurts 0.8M.

The selected weight decays are nonmonotone and come from one seed on a fixed
finite grid; they should be treated as nuisance-parameter tuning rather than
a scaling law for regularization.

Outputs:

- `results_fixed_dn60/` and `results_fixed_dn60_bl32/`: all runs and summaries;
- `figures_fixed_ratio/fixed_dn60_bl{4,32}_ar_wd.{png,pdf}`: individual sweeps;
- `figures_fixed_ratio/compare_bl4_bl32_dn60.{png,pdf}`: direct gain comparison;
- `results_fixed_dn60_block_comparison.{csv,json}`: combined selected points.

## Measured AR versus BD step time

The compute formula counts leading matmul FLOPs, but the very small models do
not convert those FLOPs into proportional H100 wall time. A dedicated
single-process benchmark used 20 warmup steps followed by the median of three
synchronized 50-step measurements. Times include batch construction,
corruption where applicable, forward, loss, backward, clipping, and fused
AdamW.

| model | AR ms/step | BD ms/step | theoretical AR/BD FLOPs | measured AR/BD time |
|---|---:|---:|---:|---:|
| 0.002M | 5.471 | 5.951 | 0.284 | 0.919 |
| 0.005M | 6.022 | 7.568 | 0.291 | 0.796 |
| 0.01M | 7.822 | 8.369 | 0.298 | 0.935 |
| 0.02M | 7.555 | 8.269 | 0.311 | 0.914 |
| 0.04M | 7.487 | 7.897 | 0.324 | 0.948 |
| 0.1M | 10.091 | 10.453 | 0.343 | 0.965 |
| 0.2M | 9.987 | 10.982 | 0.361 | 0.909 |
| 0.4M | 12.057 | 15.196 | 0.374 | 0.793 |
| 0.8M | 16.211 | 23.679 | 0.384 | 0.685 |
| 1.6M | 18.357 | 35.789 | 0.401 | 0.513 |

For the smallest models, fixed launch, optimizer, pointwise, and data-pipeline
costs dominate, so removing roughly 60–70% of the leading matmul FLOPs saves
little wall time. The gap narrows with model size: at 1.6M, theory predicts an
AR/BD ratio of 0.401 and the measured ratio is 0.513. IsoFLOP budgets continue
to use mathematical FLOPs rather than wall time; the benchmark explains why
equal-FLOP curriculum runs do not take equal wall-clock time.

## Commands and outputs

```bash
python scaling/test_scaling.py
python scaling/curriculum_warmup_ablation.py
python scaling/curriculum_run_sweep.py --dry-run
python scaling/curriculum_run_sweep.py --workers 2
python scaling/curriculum_analyze.py
python scaling/curriculum_bracket_ar_wd.py --workers 4
```

Required runs live under `results_p_ar/runs/`. Analysis reads only the one
inherited-LR path expected for each datapoint, so unrelated diagnostic trials
cannot enter the reported sweep. Final tables are written to `results_p_ar/`
and figures to `figures_p_ar/`. `phase_timing.{csv,json}` contains the timing
measurements, and `phase_timing.{png,pdf}` visualizes theoretical versus
measured phase-cost ratios. The 0.8M AR-decay brackets are under
`results_p_ar/ar_wd_sweep/0.8M/`. Their selected points replace the original
0.8M curriculum points in the fixed-model `L(p_ar)` profile at
`figures_p_ar/fixed_model_l_vs_p/fixed_N_0p8M.{png,pdf}`.
