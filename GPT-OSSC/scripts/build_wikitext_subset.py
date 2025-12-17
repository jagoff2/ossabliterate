#!/usr/bin/env python3
"""Tokenize a HuggingFace dataset split into GPT-2 LM JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def chunk(tokens, seq_len):
    for start in range(0, len(tokens) - seq_len - 1, seq_len):
        window = tokens[start : start + seq_len + 1]
        yield window[:-1], window[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--subset", default="wikitext-103-v1")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=500000, help="Max tokens")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--text-file", default=None, help="Optional plain text file path")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    if args.text_file:
        joined = Path(args.text_file).read_text(encoding="utf-8")
    else:
        ds = load_dataset(
            args.dataset, args.subset, split=args.split, trust_remote_code=args.trust_remote_code
        )
        joined = "\n\n".join(x["text"] for x in ds)
    enc = tokenizer(joined)
    tokens = enc["input_ids"]
    if args.limit:
        tokens = tokens[: args.limit]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for inputs, labels in chunk(tokens, args.seq_len):
            record = {"input_ids": inputs, "labels": labels}
            f.write(json.dumps(record) + "\n")
            count += 1
    print(f"Wrote {count} sequences to {out_path}")


if __name__ == "__main__":
    main()
