"""Memory-mapped one-pass token data and blockwise diffusion corruption."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from config import (
    BLOCK_LEN,
    DATA_MANIFEST_PATH,
    MASK_EPS,
    MASK_ID,
    SEQ_LEN,
)


class TokenSplit:
    def __init__(self, root: Path, entries: list[dict]):
        if not entries:
            raise ValueError("Token split is empty")
        self.arrays = [
            np.memmap(root / entry["path"], dtype=np.uint16, mode="r")
            for entry in entries
        ]
        self.lengths = np.array([array.size for array in self.arrays], dtype=np.int64)
        self.ends = np.cumsum(self.lengths)
        self.total_tokens = int(self.ends[-1])

    def read(self, offset: int, count: int) -> np.ndarray:
        if offset < 0 or count < 0 or offset + count > self.total_tokens:
            raise IndexError(
                f"Requested [{offset}, {offset + count}) from {self.total_tokens} tokens"
            )
        output = np.empty(count, dtype=np.uint16)
        written = 0
        shard = int(np.searchsorted(self.ends, offset, side="right"))
        while written < count:
            shard_start = 0 if shard == 0 else int(self.ends[shard - 1])
            local_start = offset + written - shard_start
            take = min(count - written, int(self.lengths[shard]) - local_start)
            output[written : written + take] = self.arrays[shard][
                local_start : local_start + take
            ]
            written += take
            shard += 1
        return output


@dataclass
class ClimbMixData:
    train: TokenSplit
    val: TokenSplit
    device: torch.device

    @classmethod
    def load(
        cls,
        device: torch.device,
        manifest_path: Path = DATA_MANIFEST_PATH,
    ) -> "ClimbMixData":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = manifest_path.parent
        if manifest["dtype"] != "uint16":
            raise ValueError(f"Unsupported token dtype: {manifest['dtype']}")
        return cls(
            train=TokenSplit(root, manifest["train"]),
            val=TokenSplit(root, manifest["val"]),
            device=device,
        )

    @staticmethod
    def _tensor(tokens: np.ndarray, batch_size: int) -> torch.Tensor:
        tensor = torch.from_numpy(tokens.astype(np.int64, copy=False))
        return tensor.view(batch_size, SEQ_LEN)

    def train_batch(self, step: int, batch_size: int) -> torch.Tensor:
        count = batch_size * SEQ_LEN
        tokens = self.train.read(step * count, count)
        return self._tensor(tokens, batch_size).to(self.device, non_blocking=True)

    def autoregressive_train_batch(
        self,
        step: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = batch_size * SEQ_LEN
        tokens = self.train.read(step * count, count + 1).astype(np.int64, copy=False)
        inputs = torch.from_numpy(tokens[:-1]).view(batch_size, SEQ_LEN)
        targets = torch.from_numpy(tokens[1:]).view(batch_size, SEQ_LEN)
        return (
            inputs.to(self.device, non_blocking=True),
            targets.to(self.device, non_blocking=True),
        )

    def autoregressive_val_batch(
        self,
        batch_index: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        count = batch_size * SEQ_LEN
        tokens = self.val.read(batch_index * count, count + 1).astype(np.int64, copy=False)
        inputs = torch.from_numpy(tokens[:-1]).view(batch_size, SEQ_LEN)
        targets = torch.from_numpy(tokens[1:]).view(batch_size, SEQ_LEN)
        return (
            inputs.to(self.device, non_blocking=True),
            targets.to(self.device, non_blocking=True),
        )

    def val_batch(self, batch_index: int, batch_size: int) -> torch.Tensor:
        count = batch_size * SEQ_LEN
        # Fixed non-overlapping validation batches make LR comparisons paired.
        tokens = self.val.read(batch_index * count, count)
        return self._tensor(tokens, batch_size).to(self.device, non_blocking=True)


def sample_mask_probabilities(batch_size: int, device: torch.device) -> torch.Tensor:
    n_blocks = SEQ_LEN // BLOCK_LEN
    return MASK_EPS + (1.0 - MASK_EPS) * torch.rand(
        batch_size,
        n_blocks,
        device=device,
    )


def stratified_mask_probabilities(
    batch_size: int,
    device: torch.device,
    batch_index: int,
) -> torch.Tensor:
    n_blocks = SEQ_LEN // BLOCK_LEN
    count = batch_size * n_blocks
    u = (torch.arange(count, device=device, dtype=torch.float32) + 0.5) / count
    u = torch.remainder(u + batch_index * 0.6180339887498949, 1.0)
    return (MASK_EPS + (1.0 - MASK_EPS) * u).view(batch_size, n_blocks)


def corrupt(x0: torch.Tensor, probabilities: torch.Tensor):
    token_probability = probabilities.repeat_interleave(BLOCK_LEN, dim=1)
    masked = torch.rand(x0.shape, device=x0.device) < token_probability
    return x0.masked_fill(masked, MASK_ID), masked, token_probability
