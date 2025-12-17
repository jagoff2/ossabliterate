import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from gpt_oss_ws.abliteration import (
    AnalyzerConfig,
    analyze_measurements,
    load_ablation_config,
    load_prompt_file,
    run_ablation,
)


def test_load_prompt_file_formats(tmp_path: Path) -> None:
    txt_path = tmp_path / "sample.txt"
    txt_path.write_text("hello\nworld\n", encoding="utf-8")
    assert load_prompt_file(str(txt_path)) == ["hello", "world"]

    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps(["foo", {"text": "bar"}]), encoding="utf-8")
    assert load_prompt_file(str(json_path)) == ["foo", "bar"]

    jsonl_path = tmp_path / "sample.jsonl"
    jsonl_path.write_text("{\"text\": \"alpha\"}\n{\"text\": \"beta\"}\n", encoding="utf-8")
    assert load_prompt_file(str(jsonl_path)) == ["alpha", "beta"]


def test_analyze_measurements_outputs(tmp_path: Path) -> None:
    measure_path = tmp_path / "measure.pt"
    payload = {
        "layers": 2,
        "harmful_0": torch.ones(4),
        "harmless_0": torch.ones(4) * 0.5,
        "refuse_0": torch.tensor([1.0, -1.0, 0.5, -0.5]),
        "harmful_1": torch.ones(4) * 0.8,
        "harmless_1": torch.ones(4) * 0.2,
        "refuse_1": torch.tensor([-0.2, 0.4, -0.6, 0.8]),
    }
    torch.save(payload, measure_path)
    cfg = AnalyzerConfig(input_path=str(measure_path), emit_chart=False)
    result = analyze_measurements(cfg)
    assert result.layers == 2
    assert len(result.signal_to_noise) == 2
    assert all(isinstance(val, float) for val in result.signal_quality)


def test_sharded_ablation_changes_weights(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    weights = {
        "model.layers.0.self_attn.o_proj.weight": torch.eye(4, dtype=torch.float32),
        "model.layers.0.mlp.down_proj.weight": torch.ones((4, 4), dtype=torch.float32),
    }
    save_file(weights, str(model_dir / shard_name))
    index = {
        "metadata": {},
        "weight_map": {key: shard_name for key in weights},
    }
    index_path = model_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    measure_path = tmp_path / "measure.pt"
    torch.save(
        {
            "layers": 1,
            "harmful_0": torch.ones(4),
            "harmless_0": torch.zeros(4),
            "refuse_0": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        },
        measure_path,
    )

    yaml_path = tmp_path / "ablate.yaml"
    yaml_path.write_text(
        """
model: {model}
measurements: {measure}
output: {output}
ablate:
  - layer: 0
    measurement: 0
    scale: 0.5
    sparsity: 0.0
""".format(
            model=model_dir,
            measure=measure_path,
            output=tmp_path / "ablated",
        ),
        encoding="utf-8",
    )

    cfg = load_ablation_config(yaml_path, norm_preserve=False, projected=False)
    run_ablation(cfg)
    output_shard = load_file(str((tmp_path / "ablated") / shard_name))
    assert not torch.equal(
        output_shard["model.layers.0.self_attn.o_proj.weight"],
        weights["model.layers.0.self_attn.o_proj.weight"],
    )
