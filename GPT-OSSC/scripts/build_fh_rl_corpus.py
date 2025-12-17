#!/usr/bin/env python3
"""Builds the byte-level corpus used by FH-RL training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

DEFAULT_TEXTS = [
    "Recursive loops rehearse their own conclusions.",
    "Thought bends back upon itself to ask what it currently holds.",
    "Signals circulate until they stabilize in a reflective band.",
]


def load_sources(paths: Iterable[str]) -> str:
    buffers: List[str] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        buffers.append(path.read_text(encoding="utf-8"))
    if not buffers:
        return "\n".join(DEFAULT_TEXTS)
    return "\n".join(buffers)


def chunk_bytes(data: bytes, seq_len: int) -> List[List[int]]:
    sequences: List[List[int]] = []
    for idx in range(0, len(data) - seq_len):
        chunk = data[idx : idx + seq_len + 1]
        if len(chunk) < seq_len + 1:
            continue
        sequences.append(list(chunk))
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="*", default=[], help="Input text files")
    parser.add_argument("--output", type=str, default="data/fh_rl_corpus.jsonl")
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    text = load_sources(args.sources)
    byte_data = text.encode("utf-8")
    sequences = chunk_bytes(byte_data, args.seq_len)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for seq in sequences:
            record = {"input_ids": seq[:-1], "labels": seq[1:]}
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(sequences)} sequences to {out_path}")


if __name__ == "__main__":
    main()
