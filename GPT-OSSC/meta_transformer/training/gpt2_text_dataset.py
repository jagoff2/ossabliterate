from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class Gpt2TextDataset(Dataset):
    """Tokenized text dataset for GPT-2-style models using a HF tokenizer."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        text_paths: Iterable[str],
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride or seq_len
        texts: List[str] = []
        for path in text_paths:
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(f"Text corpus path not found: {path}")
            texts.append(p.read_text(encoding="utf-8"))
        joined = "\n".join(texts)
        tokens = tokenizer.encode(joined, add_special_tokens=False)
        if len(tokens) < seq_len + 1:
            raise ValueError("Corpus too small for requested sequence length.")
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.starts = list(range(0, len(self.tokens) - seq_len - 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[idx]
        window = self.tokens[start : start + self.seq_len + 1]
        return window[:-1], window[1:]


def create_gpt2_text_dataloaders(
    tokenizer_name: str,
    train_paths: Iterable[str],
    val_paths: Iterable[str],
    seq_len: int,
    batch_size: int,
    stride: int | None = None,
) -> Tuple[DataLoader[Tuple[torch.Tensor, torch.Tensor]], DataLoader[Tuple[torch.Tensor, torch.Tensor]]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    train_dataset = Gpt2TextDataset(tokenizer, train_paths, seq_len=seq_len, stride=stride)
    val_dataset = Gpt2TextDataset(tokenizer, val_paths, seq_len=seq_len, stride=stride)
    train_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader: DataLoader[Tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader

