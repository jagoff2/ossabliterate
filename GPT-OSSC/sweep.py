import torch
from transformers import AutoTokenizer
from meta_transformer.models.hf_gpt2_fh_rl import load_fh_rl_gpt2, FHRLGPT2Config
import json
from pathlib import Path

prompt = "User: How does the reflective loop help you reason about your own answers?\nAssistant:"
max_new_tokens = 60
config = json.loads(Path('configs/fh_rl_dialogpt_blend.json').read_text())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tok = AutoTokenizer.from_pretrained(config['model']['base_model_name'])
tok.pad_token = tok.eos_token
inputs = tok(prompt, return_tensors='pt').to(device)

rank_values = [32, 48, 64]
beta_values = [0.04, 0.05, 0.06]
gamma_values = [0.15, 0.18, 0.22, 0.25]

outputs = []
for rank in rank_values:
    for beta in beta_values:
        for gamma in gamma_values:
            cfg = FHRLGPT2Config(
                base_model_name=config['model']['base_model_name'],
                fh_rank=rank,
                fh_alpha=config['model']['fh_alpha'],
                fh_beta=beta,
                fh_gamma=gamma,
                noise_std=config['model']['noise_std'],
                detach_feedback=False,
            )
            model = load_fh_rl_gpt2(cfg).to(device)
            for block in model.transformer.h:
                if hasattr(block, 'fh_rl'):
                    block.fh_rl.gamma = gamma
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
            text = tok.decode(out[0], skip_special_tokens=True)
            outputs.append((rank, beta, gamma, text))

for rank, beta, gamma, text in outputs:
    print(f"rank={rank} beta={beta} gamma={gamma}")
    print(text)
    print('---')
