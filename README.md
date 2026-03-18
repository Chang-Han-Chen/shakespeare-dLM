# shakespeare-dllm

This repo explores scaling laws and training dynamics for tiny diffusion language models on Shakespeare-like toy data.

We currently study four decoding/training variants:

- `model_MDLM.py`: vanilla MDLM. Train on masked-token prediction only. At inference, progressively unmask tokens, carry over revealed tokens, and never remask.
- `model_remasked.py`: remasking baseline. Predict all non-prompt positions each step, then remask low-confidence tokens based on logits and refine iteratively.
- `model_edit_one_pass.py`: one-pass self-correcting variant. Train on both masked tokens and visible corrupted tokens in a single pass, so the model learns to fill blanks and repair wrong visible tokens together.
- `model_edit_two_pass.py`: two-pass self-correcting variant. First do standard denoising, then build a draft from model predictions and run a second correction pass to edit visible generated tokens.

## Goal

The main question is simple:

> What's the most scalable pretraining paradigm of diffusion language models?

We care about:
- optimization behavior,
- sample quality,
- compute vs quality tradeoffs,
- how correction/remasking changes scaling trends.
