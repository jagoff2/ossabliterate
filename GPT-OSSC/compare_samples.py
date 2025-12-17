import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from meta_transformer.models.hf_gpt2_fh_rl import load_fh_rl_gpt2, FHRLGPT2Config
import json
from pathlib import Path

prompt = 'User: How does the reflective loop help you reason about your own answers?\nAssistant:'
max_new_tokens = 60
config = json.loads(Path('configs/fh_rl_dialogpt_blend.json').read_text())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
tok.pad_token = tok.eos_token
inputs = tok(prompt, return_tensors='pt').to(device)

baseline = GPT2LMHeadModel.from_pretrained(config['model']['base_model_name']).to(device)

cfg = FHRLGPT2Config(
    base_model_name=config['model']['base_model_name'],
    fh_rank=48,
    fh_alpha=config['model']['fh_alpha'],
    fh_beta=0.04,
    fh_gamma=0.18,
    noise_std=config['model']['noise_std'],
    detach_feedback=False,
)
fh_model = load_fh_rl_gpt2(cfg).to(device)
for block in fh_model.transformer.h:
    if hasattr(block, 'fh_rl'):
        block.fh_rl.gamma = 0.18

for i in range(1, 4):
    with torch.no_grad():
        fh_out = fh_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
    fh_text = tok.decode(fh_out[0], skip_special_tokens=True)

    with torch.no_grad():
        base_out = baseline.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
    base_text = tok.decode(base_out[0], skip_special_tokens=True)

    print(f"Sample {i} FH-RL:")
    print(fh_text)
    print()
    print(f"Sample {i} Baseline:")
    print(base_text)
    print('\n---\n')
