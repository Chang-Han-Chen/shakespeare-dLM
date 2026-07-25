"""Train the 8K byte-level BPE and encode one-pass train/validation shards."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from config import (
    BASE_VOCAB_SIZE,
    DATA_MANIFEST_PATH,
    EOT_TOKEN,
    MASK_ID,
    MASK_TOKEN,
    MIN_PREPARED_TRAIN_TOKENS,
    RAW_DIR,
    TOKENIZED_DIR,
    TOKENIZER_META_PATH,
    TOKENIZER_PATH,
    TOKENIZER_TRAIN_SHARDS,
    TRAIN_SHARDS,
    VALIDATION_SHARDS,
    VOCAB_SIZE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-tokenizer", action="store_true")
    parser.add_argument("--force-encode", action="store_true")
    parser.add_argument("--batch-rows", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def parquet_text(path: Path, batch_rows: int) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    if "text" not in parquet.schema.names:
        raise ValueError(f"{path} has no text column: {parquet.schema.names}")
    for batch in parquet.iter_batches(batch_size=batch_rows, columns=["text"]):
        for value in batch.column(0).to_pylist():
            if value:
                yield value


def tokenizer_training_text(batch_rows: int) -> Iterator[str]:
    for index in TOKENIZER_TRAIN_SHARDS:
        path = RAW_DIR / f"shard_{index:05d}.parquet"
        print(f"tokenizer training input: {path.name}", flush=True)
        yield from parquet_text(path, batch_rows)


def train_tokenizer(batch_rows: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=BASE_VOCAB_SIZE,
        min_frequency=2,
        show_progress=True,
        special_tokens=[EOT_TOKEN],
        initial_alphabet=ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(tokenizer_training_text(batch_rows), trainer=trainer)
    if tokenizer.get_vocab_size() != BASE_VOCAB_SIZE:
        raise RuntimeError(
            f"Expected {BASE_VOCAB_SIZE} base tokens, got {tokenizer.get_vocab_size()}"
        )
    added = tokenizer.add_special_tokens([MASK_TOKEN])
    if added != 1 or tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError("Failed to add exactly one mask token")
    if tokenizer.token_to_id(MASK_TOKEN) != MASK_ID:
        raise RuntimeError(
            f"Expected mask ID {MASK_ID}, got {tokenizer.token_to_id(MASK_TOKEN)}"
        )

    TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = TOKENIZER_PATH.with_suffix(".json.tmp")
    tokenizer.save(str(temporary))
    os.replace(temporary, TOKENIZER_PATH)
    metadata = {
        "algorithm": "byte_level_bpe",
        "base_vocab_size": BASE_VOCAB_SIZE,
        "total_vocab_size": VOCAB_SIZE,
        "eot_token": EOT_TOKEN,
        "eot_id": tokenizer.token_to_id(EOT_TOKEN),
        "mask_token": MASK_TOKEN,
        "mask_id": tokenizer.token_to_id(MASK_TOKEN),
        "training_shards": list(TOKENIZER_TRAIN_SHARDS),
    }
    temporary_meta = TOKENIZER_META_PATH.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_meta, TOKENIZER_META_PATH)
    return tokenizer


def encode_shard(
    tokenizer: Tokenizer,
    raw_path: Path,
    output_path: Path,
    batch_rows: int,
    force: bool,
) -> dict:
    metadata_path = output_path.with_suffix(".json")
    if output_path.exists() and metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if output_path.stat().st_size == metadata["token_count"] * 2:
            return metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".bin.tmp")
    started = time.monotonic()
    token_count = 0
    document_count = 0
    eot_id = tokenizer.token_to_id(EOT_TOKEN)
    parquet = pq.ParquetFile(raw_path)
    with temporary.open("wb") as handle:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=["text"]):
            documents = [text for text in batch.column(0).to_pylist() if text]
            encodings = tokenizer.encode_batch(documents, add_special_tokens=False)
            pieces = []
            for encoding in encodings:
                pieces.append(np.asarray(encoding.ids + [eot_id], dtype=np.uint16))
            if pieces:
                np.concatenate(pieces).tofile(handle)
                token_count += sum(piece.size for piece in pieces)
                document_count += len(pieces)
    os.replace(temporary, output_path)
    metadata = {
        "source": raw_path.name,
        "path": str(output_path.relative_to(TOKENIZED_DIR)),
        "token_count": token_count,
        "document_count": document_count,
        "elapsed_seconds": time.monotonic() - started,
    }
    temporary_meta = metadata_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_meta, metadata_path)
    return metadata


def encode_configured_shard(
    split: str,
    index: int,
    batch_rows: int,
    force: bool,
) -> tuple[str, int, dict]:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    raw_path = RAW_DIR / f"shard_{index:05d}.parquet"
    output_path = TOKENIZED_DIR / split / f"shard_{index:05d}.bin"
    metadata = encode_shard(
        tokenizer,
        raw_path,
        output_path,
        batch_rows,
        force,
    )
    return split, index, metadata


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    raw_manifest = RAW_DIR / "manifest.json"
    if not raw_manifest.exists():
        raise FileNotFoundError("Run download_data.py first")

    if args.force_tokenizer or not TOKENIZER_PATH.exists():
        tokenizer = train_tokenizer(args.batch_rows)
    else:
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise RuntimeError("Tokenizer vocabulary does not match config")

    split_indices = (("val", VALIDATION_SHARDS), ("train", TRAIN_SHARDS))
    jobs = [
        (split, index)
        for split, indices in split_indices
        for index in indices
    ]
    completed = []
    if args.workers == 1:
        for position, (split, index) in enumerate(jobs, start=1):
            print(
                f"[{position}/{len(jobs)}] encoding "
                f"{split}/shard_{index:05d}.parquet",
                flush=True,
            )
            completed.append(
                encode_configured_shard(
                    split,
                    index,
                    args.batch_rows,
                    args.force_encode,
                )
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(
                    encode_configured_shard,
                    split,
                    index,
                    args.batch_rows,
                    args.force_encode,
                ): (split, index)
                for split, index in jobs
            }
            for position, future in enumerate(as_completed(futures), start=1):
                split, index, metadata = future.result()
                completed.append((split, index, metadata))
                print(
                    f"[{position}/{len(jobs)}] {split}/shard_{index:05d}: "
                    f"{metadata['token_count'] / 1e6:.2f}M tokens",
                    flush=True,
                )

    splits = {"train": [], "val": []}
    for split, index, metadata in sorted(
        completed,
        key=lambda row: (row[0] != "val", row[1]),
    ):
        splits[split].append(metadata)
        if args.workers == 1:
            print(
                f"  {metadata['document_count']:,} docs, "
                f"{metadata['token_count'] / 1e6:.2f}M tokens",
                flush=True,
            )

    manifest = {
        "dtype": "uint16",
        "tokenizer": str(TOKENIZER_PATH.relative_to(TOKENIZED_DIR.parent)),
        "vocab_size": VOCAB_SIZE,
        "base_vocab_size": BASE_VOCAB_SIZE,
        "mask_id": MASK_ID,
        "train": splits["train"],
        "val": splits["val"],
        "train_tokens": sum(row["token_count"] for row in splits["train"]),
        "val_tokens": sum(row["token_count"] for row in splits["val"]),
    }
    if manifest["train_tokens"] < MIN_PREPARED_TRAIN_TOKENS:
        raise RuntimeError(
            f"Prepared only {manifest['train_tokens']:,} training tokens; "
            f"need at least {MIN_PREPARED_TRAIN_TOKENS:,}"
        )
    DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATA_MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, DATA_MANIFEST_PATH)
    print(
        f"complete: train={manifest['train_tokens'] / 1e9:.3f}B tokens, "
        f"val={manifest['val_tokens'] / 1e6:.1f}M tokens"
    )


if __name__ == "__main__":
    main()
