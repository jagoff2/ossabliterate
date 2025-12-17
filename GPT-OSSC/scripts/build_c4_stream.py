#!/usr/bin/env python3
"""Stream a capped number of C4 docs, tokenize, and write JSONL without caching full shards."""

from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def stream_c4(split: str, limit_docs: int, tokenizer_name: str, seq_len: int):
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tok.pad_token = tok.eos_token
    for example in islice(ds, limit_docs):
        text = example["text"]
        tokens = tok.encode(text)
        # chunk into seq_len windows
        for i in range(0, len(tokens) - 1, seq_len):
            window = tokens[i : i + seq_len + 1]
            if len(window) < seq_len + 1:
                continue
            yield window[:-1], window[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", help="HF split, e.g., train or validation")
    ap.add_argument("--limit-docs", type=int, default=100000)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--output", required=True)
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for inputs, labels in stream_c4(args.split, args.limit_docs, args.tokenizer, args.seq_len):
            f.write(json.dumps({"input_ids": inputs, "labels": labels}) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"wrote {count} sequences", flush=True)
    print(f"Done. wrote {count} sequences to {out_path}")


if __name__ == "__main__":
    main()
