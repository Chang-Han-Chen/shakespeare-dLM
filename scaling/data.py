"""TinyShakespeare character data and blockwise mask corruption."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from config import BLOCK_LEN, DATA_PATH, MASK_EPS, SEQ_LEN, VOCAB_SIZE


@dataclass
class ShakespeareData:
    train: torch.Tensor
    val: torch.Tensor
    chars: tuple[str, ...]
    mask_id: int = 0

    @classmethod
    def load(cls, device: torch.device) -> "ShakespeareData":
        text = DATA_PATH.read_text(encoding="utf-8")
        chars = tuple(sorted(set(text)))
        if len(chars) + 1 != VOCAB_SIZE:
            raise ValueError(f"Expected vocab size {VOCAB_SIZE}, found {len(chars) + 1}")
        stoi = {ch: i + 1 for i, ch in enumerate(chars)}
        tokens = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
        split = int(0.9 * len(tokens))
        return cls(tokens[:split].to(device), tokens[split:].to(device), chars)

    def batch(self, split: str, batch_size: int) -> torch.Tensor:
        source = self.train if split == "train" else self.val
        starts = torch.randint(0, source.numel() - SEQ_LEN, (batch_size,), device=source.device)
        offsets = torch.arange(SEQ_LEN, device=source.device)
        return source[starts[:, None] + offsets[None, :]]

    def autoregressive_batch(
        self,
        split: str,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return L inputs and their L one-token-ahead targets."""
        source = self.train if split == "train" else self.val
        starts = torch.randint(
            0,
            source.numel() - SEQ_LEN - 1,
            (batch_size,),
            device=source.device,
        )
        offsets = torch.arange(SEQ_LEN, device=source.device)
        inputs = source[starts[:, None] + offsets[None, :]]
        targets = source[starts[:, None] + offsets[None, :] + 1]
        return inputs, targets


def sample_mask_probabilities(batch_size: int, device: torch.device) -> torch.Tensor:
    n_blocks = SEQ_LEN // BLOCK_LEN
    return MASK_EPS + (1.0 - MASK_EPS) * torch.rand(batch_size, n_blocks, device=device)


def stratified_mask_probabilities(
    batch_size: int,
    device: torch.device,
    batch_index: int,
) -> torch.Tensor:
    """Low-variance deterministic coverage of the uniform noise-time integral."""
    n_blocks = SEQ_LEN // BLOCK_LEN
    count = batch_size * n_blocks
    u = (torch.arange(count, device=device, dtype=torch.float32) + 0.5) / count
    u = torch.remainder(u + batch_index * 0.6180339887498949, 1.0)
    return (MASK_EPS + (1.0 - MASK_EPS) * u).view(batch_size, n_blocks)


def corrupt(x0: torch.Tensor, probabilities: torch.Tensor, mask_id: int = 0):
    token_prob = probabilities.repeat_interleave(BLOCK_LEN, dim=1)
    masked = torch.rand(x0.shape, device=x0.device) < token_prob
    xt = x0.masked_fill(masked, mask_id)
    return xt, masked, token_prob
