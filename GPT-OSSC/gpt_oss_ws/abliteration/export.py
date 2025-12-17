from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class GgufExportConfig:
  model_dir: str
  llama_cpp_repo: str
  outfile: str
  quantize: Optional[str] = None
  outtype: str = "f16"
  extra_args: List[str] = field(default_factory=list)
  python_executable: str = sys.executable


def run_gguf_export(config: GgufExportConfig) -> None:
  convert_script = _resolve_convert_script(Path(config.llama_cpp_repo))
  outfile_path = Path(config.outfile)
  outfile_path.parent.mkdir(parents=True, exist_ok=True)
  cmd = [
    config.python_executable,
    str(convert_script),
    config.model_dir,
    "--outfile",
    str(outfile_path),
    "--outtype",
    config.outtype,
  ]
  if config.quantize:
    cmd.extend(["--quantize", config.quantize])
  cmd.extend(config.extra_args)
  subprocess.run(cmd, check=True)


def _resolve_convert_script(repo_path: Path) -> Path:
  candidates = [
    repo_path / "convert.py",
    repo_path / "convert_hf_to_gguf.py",
    repo_path / "tools" / "convert-hf-to-gguf.py",
  ]
  for candidate in candidates:
    if candidate.exists():
      return candidate
  raise FileNotFoundError(
    f"Could not locate convert script under {repo_path}; expected convert.py or tools/convert-hf-to-gguf.py"
  )
