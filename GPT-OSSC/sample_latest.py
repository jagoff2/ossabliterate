import json, torch
from pathlib import Path
from transformers import AutoTokenizer
from meta_transformer.models.hf_gpt2_fh_rl import load_fh_rl_gpt2, FHRLGPT2Config

prompt = "User: Explain your last mistake.\nAssistant:"
max_new_tokens = 60
config = json.loads(Path("configs/fh_rl_dialogpt_blend.json").read_text())
tok = AutoTokenizer.from_pretrained(config["model"]["base_model_name"])
tok.pad_token = tok.eos_token
inputs = tok(prompt, return_tensors="pt").to("cuda")

ckpts = sorted(Path("runs/fh_rl_dialogpt_blend").glob("fh_rl_step*.pt"))
if not ckpts:
    print("No checkpoint found")
else:
    latest = ckpts[-1]
    cfg = FHRLGPT2Config(
        base_model_name=config["model"]["base_model_name"],
        fh_rank=config["model"]["fh_rank"],
        fh_alpha=config["model"]["fh_alpha"],
        fh_beta=config["model"]["fh_beta"],
        fh_gamma=config["model"]["fh_gamma"],
        noise_std=config["model"]["noise_std"],
        detach_feedback=config["model"]["detach_feedback"],
    )
    model = load_fh_rl_gpt2(cfg).to("cuda")
    model.load_state_dict(torch.load(latest, map_location="cuda"))
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
    print(tok.decode(out[0], skip_special_tokens=True))
