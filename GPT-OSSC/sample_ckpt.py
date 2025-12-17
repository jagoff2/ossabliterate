from pathlib import Path
import torch
import json
from transformers import AutoTokenizer
from meta_transformer.models.hf_gpt2_fh_rl import load_fh_rl_gpt2, FHRLGPT2Config

tok = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
tok.pad_token = tok.eos_token
max_new_tokens = 60
prompts = [
    ('plan', 'User: How does the reflective loop help you reason about your own answers?\nAssistant:'),
    ('diagnosis', 'User: Explain your last mistake.\nAssistant:')
]
ckpts = sorted(Path('runs/fh_rl_dialogpt_blend').glob('fh_rl_step*.pt'))
if not ckpts:
    print('No checkpoint found')
else:
    latest = ckpts[-1]
    config = json.loads(Path('configs/fh_rl_dialogpt_blend.json').read_text())
    fh_cfg = FHRLGPT2Config(**config['model'])
    model = load_fh_rl_gpt2(fh_cfg).to('cuda')
    model.load_state_dict(torch.load(latest, map_location='cuda'))
    model.eval()
    with torch.no_grad():
        for name, prompt in prompts:
            inputs = tok(prompt, return_tensors='pt').to('cuda')
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.8)
            text = tok.decode(out[0], skip_special_tokens=True)
            print(f'Prompt {name}:\n{text}\n')
