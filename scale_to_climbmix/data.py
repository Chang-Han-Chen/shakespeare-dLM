"""Memory-mapped one-pass token data and blockwise diffusion corruption."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
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

    def train_batch(
        self,
        step: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> torch.Tensor:
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank {rank}/{world_size}")
        count = batch_size * SEQ_LEN
        # With local_batch * world_size equal to the historical global batch,
        # concatenating ranks recovers exactly the same one-pass token prefix.
        tokens = self.train.read((step * world_size + rank) * count, count)
        return self._tensor(tokens, batch_size).to(self.device, non_blocking=True)

    def autoregressive_train_batch(
        self,
        step: int,
        batch_size: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank {rank}/{world_size}")
        count = batch_size * SEQ_LEN
        offset = (step * world_size + rank) * count
        tokens = self.train.read(offset, count + 1).astype(np.int64, copy=False)
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


@dataclass(eq=False)
class FixedTokenEpochData:
    """Repeat a fixed token prefix, reshuffling contiguous batches each epoch.

    Keeping batches contiguous preserves efficient memory-mapped reads while
    changing their order deterministically from epoch to epoch.  Every clean
    token in the prefix is consumed exactly once per epoch.
    """

    source: ClimbMixData
    unique_tokens: int
    batch_size: int
    seed: int

    def __post_init__(self) -> None:
        tokens_per_batch = self.batch_size * SEQ_LEN
        if self.unique_tokens < tokens_per_batch:
            raise ValueError("unique_tokens must contain at least one batch")
        if self.unique_tokens % tokens_per_batch:
            raise ValueError(
                "unique_tokens must be divisible by batch_size * sequence length"
            )
        # AR targets read one lookahead token at the end of the fixed prefix.
        if self.unique_tokens + 1 > self.source.train.total_tokens:
            raise ValueError("fixed token prefix exceeds the training split")

    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * SEQ_LEN

    @property
    def steps_per_epoch(self) -> int:
        return self.unique_tokens // self.tokens_per_batch

    def local_batch_size(self, world_size: int) -> int:
        if world_size < 1 or self.batch_size % world_size:
            raise ValueError(
                f"Global batch {self.batch_size} is not divisible by "
                f"world size {world_size}"
            )
        return self.batch_size // world_size

    @lru_cache(maxsize=128)
    def epoch_order(self, epoch: int) -> tuple[int, ...]:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        generator = np.random.default_rng(self.seed + epoch)
        return tuple(
            int(index) for index in generator.permutation(self.steps_per_epoch)
        )

    def source_batch_index(self, step: int) -> int:
        if step < 0:
            raise ValueError("step must be nonnegative")
        epoch, within_epoch = divmod(step, self.steps_per_epoch)
        return self.epoch_order(epoch)[within_epoch]

    def train_batch(
        self,
        step: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> torch.Tensor:
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank {rank}/{world_size}")
        source_index = self.source_batch_index(step)
        local_batch_size = self.local_batch_size(world_size)
        local_tokens = local_batch_size * SEQ_LEN
        tokens = self.source.train.read(
            source_index * self.tokens_per_batch + rank * local_tokens,
            local_tokens,
        )
        return self.source._tensor(tokens, local_batch_size).to(
            self.source.device,
            non_blocking=True,
        )

    def autoregressive_train_batch(
        self,
        step: int,
        rank: int = 0,
        world_size: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed rank {rank}/{world_size}")
        source_index = self.source_batch_index(step)
        local_batch_size = self.local_batch_size(world_size)
        local_tokens = local_batch_size * SEQ_LEN
        tokens = self.source.train.read(
            source_index * self.tokens_per_batch + rank * local_tokens,
            local_tokens + 1,
        ).astype(np.int64, copy=False)
        inputs = torch.from_numpy(tokens[:-1]).view(local_batch_size, SEQ_LEN)
        targets = torch.from_numpy(tokens[1:]).view(local_batch_size, SEQ_LEN)
        return (
            inputs.to(self.source.device, non_blocking=True),
            targets.to(self.source.device, non_blocking=True),
        )


def sample_mask_probabilities(
    batch_size: int,
    device: torch.device,
    block_len: int = BLOCK_LEN,
) -> torch.Tensor:
    if SEQ_LEN % block_len:
        raise ValueError("block_len must divide sequence length")
    n_blocks = SEQ_LEN // block_len
    return MASK_EPS + (1.0 - MASK_EPS) * torch.rand(
        batch_size,
        n_blocks,
        device=device,
    )


def stratified_mask_probabilities(
    batch_size: int,
    device: torch.device,
    batch_index: int,
    block_len: int = BLOCK_LEN,
) -> torch.Tensor:
    if SEQ_LEN % block_len:
        raise ValueError("block_len must divide sequence length")
    n_blocks = SEQ_LEN // block_len
    count = batch_size * n_blocks
    u = (torch.arange(count, device=device, dtype=torch.float32) + 0.5) / count
    u = torch.remainder(u + batch_index * 0.6180339887498949, 1.0)
    return (MASK_EPS + (1.0 - MASK_EPS) * u).view(batch_size, n_blocks)


def corrupt(x0: torch.Tensor, probabilities: torch.Tensor):
    if x0.shape[1] % probabilities.shape[1]:
        raise ValueError("probability count must divide sequence length")
    block_len = x0.shape[1] // probabilities.shape[1]
    token_probability = probabilities.repeat_interleave(block_len, dim=1)
    masked = torch.rand(x0.shape, device=x0.device) < token_probability
    return x0.masked_fill(masked, MASK_ID), masked, token_probability
