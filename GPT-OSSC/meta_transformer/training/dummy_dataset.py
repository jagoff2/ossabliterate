from __future__ import annotations

import torch
from pathlib import Path
from typing import Iterable, List, Tuple

from torch.utils.data import DataLoader, Dataset


class DummyLanguageModelingDataset(Dataset):
    """Simple dataset emitting random token sequences for sanity checks."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:  # noqa: ARG002
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        targets = (tokens + 1) % self.vocab_size
        return tokens, targets


def create_dummy_dataloader(
    batch_size: int,
    num_samples: int,
    seq_len: int,
    vocab_size: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = DummyLanguageModelingDataset(num_samples, seq_len, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


class RealTextDataset(Dataset):
    """Character-level dataset built from a text corpus."""

    def __init__(self, text: str, seq_len: int, vocab_size: int = 256) -> None:
        super().__init__()
        if vocab_size < 2 or vocab_size > 256:
            raise ValueError("vocab_size must be within [2, 256] for byte-level data")
        encoded = text.encode("utf-8")
        if len(encoded) < seq_len + 1:
            raise ValueError("corpus is too small for the requested sequence length")
        self.tokens = torch.tensor(list(encoded), dtype=torch.long) % vocab_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.tokens.numel() - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.tokens[idx : idx + self.seq_len + 1]
        return window[:-1], window[1:]


def _load_corpus(paths: Iterable[str]) -> str:
    texts: List[str] = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Corpus path not found: {path}")
        texts.append(p.read_text(encoding="utf-8"))
    return "\n".join(texts)


def create_real_data_dataloader(
    corpus_paths: Iterable[str],
    batch_size: int,
    seq_len: int,
    vocab_size: int = 256,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    text = _load_corpus(corpus_paths)
    dataset = RealTextDataset(text=text, seq_len=seq_len, vocab_size=vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
