import json
from pathlib import Path
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler


def _extract_ids(rec: dict):
    if "input_ids" in rec and "labels" in rec:
        ids, labs = rec["input_ids"], rec["labels"]
    elif "completion_ids" in rec:
        ids = rec["completion_ids"]
        ids, labs = ids[:-1], ids[1:]
    elif "completion" in rec and isinstance(rec["completion"], list):
        ids = rec["completion"]
        ids, labs = ids[:-1], ids[1:]
    else:
        raise KeyError
    if len(ids) < 2 or len(labs) < 1:
        raise KeyError
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labs, dtype=torch.long)


class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        for line in Path(path).open("r", encoding="utf-8"):
            try:
                rec = json.loads(line)
                ids, labs = _extract_ids(rec)
                self.samples.append((ids, labs))
            except KeyError:
                continue
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def make_simple_loader(paths: List[str], batch_size: int, num_workers: int = 0) -> DataLoader:
    # concatenate all datasets
    datasets = [JsonlDataset(p) for p in paths]
    class Concat(Dataset):
        def __len__(self):
            return sum(len(d) for d in datasets)
        def __getitem__(self, idx):
            for d in datasets:
                if idx < len(d):
                    return d[idx]
                idx -= len(d)
            raise IndexError
    ds = Concat()
    sampler = RandomSampler(ds, replacement=True)
    return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=True)
