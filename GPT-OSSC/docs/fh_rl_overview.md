# Fast-Weights Homeostatic Reentry Layer (FH-RL) Integration Plan

## Objective
Recreate the FH-RL architecture from *Recursive Dynamics in Fast-Weights Homeostatic Reentry Networks* (arXiv:2511.06798) on top of a GPT-2 style transformer. The goal is to faithfully implement the low-rank fast-weight memory, homeostatic normalization, and γ-controlled reentrant feedback while keeping the rest of the GPT-2 stack intact. Training will follow the paper’s byte-level synthetic setup so we can reproduce IRR/ESRI/RDP diagnostics before scaling further.

## Block Placement
For each transformer block:
1. Apply LayerNorm + self-attention + residual (standard GPT-2).
2. Feed the post-attention hidden state, along with that layer’s Q/K/V projections, into FH-RL.
3. FH-RL updates fast-weight factors `(U_t, V_t)` (rank-r) per sequence + token, computes fast-weight output `y_t`, homeostatically normalizes it, and injects reentry via `x <- x + gamma * W_r @ y_t`.
4. Pass the updated state into the block’s MLP + residual as usual.

Gradients through the reentry addition are detached, matching the paper’s stabilization trick.

## Fast-Weight State Handling
- Maintain `U` and `V` tensors of shape `(batch, seq_len?, rank, dim)`; in practice we update per token during autoregressive scanning (causal LM). Implementation will cache per-layer state objects inside the module and reset via an explicit `reset_state(batch_size)` call between sequences.
- Update rule per token `t`:
  - `U_t = (1 - alpha) * U_{t-1} + alpha * normalize(Q_t + eps_u)`
  - `V_t = (1 - alpha) * V_{t-1} + alpha * normalize(K_t + eps_v)`
  - Gaussian noise `eps` is optional but implemented for fidelity; default σ = 1e-4.
- Fast weight operator `W_eff = U_t^T @ V_t` (rank-r, low-r).
- Value projection `V(v)_t = W_v x_t` (use same value projection from attention block).
- Output `y_t = W_eff @ V(v)_t` then apply homeostatic scaling `y_t <- y_t / (1 + beta * (||y_t|| - 1))`.
- Reentry uses learnable `W_r` (d×d) and scalar gain `gamma` (per layer, configurable). Feedback signal `gamma * W_r y_t` is detached before residual addition.

## Training Configuration
- Model: GPT-2 small variant (n_layer=3, n_head=3, d_model=192) to mirror paper; configs kept modular.
- Dataset: byte-level corpus builder that chunks provided seed texts into 128-token sequences.
- Objective: standard next-token cross entropy (byte LM), teacher forcing.
- Optimizer: AdamW, lr=3e-4, betas=(0.9,0.95), weight_decay=0.01.
- Batch size: 32 sequences of length 128, trained for ≥400 steps per γ sweep.
- Sweep reentry gain `gamma` over {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30} with identical seeds.
- Detach gradients through feedback path to avoid recursive BPTT.
- Checkpoint per sweep, log IRR/ESRI/RDP each eval interval.

## Metrics (to implement later steps)
- **Information Reentry Ratio (IRR):** ratio of L2 norms `||gamma * W_r y|| / ||x_pre||` averaged over tokens/layers.
- **Eigen-Spectrum Recursion Index (ESRI):** cosine distance between eigenvalue spectra of consecutive token covariances (per layer, averaged).
- **Representational Drift Periodicity (RDP):** dominant Fourier magnitude of cosine similarities `sim(h_t, h_{t+1})` sequences.

## Files to Create
- `meta_transformer/fh_rl_layer.py` – FH-RL module (parameters + state + forward logic).
- `meta_transformer/gpt2_fh_rl_block.py` – GPT-2 block variant with FH-RL inserted.
- `scripts/build_fh_rl_corpus.py` – deterministic byte-level dataset builder.
- `configs/fh_rl_tiny.json` – baseline hyperparameters for Tiny FH-RL GPT-2.
- `training/train_fh_rl.py` – training entry point with logging + metric hooks.
- `scripts/eval_fh_rl_metrics.py` – offline IRR/ESRI/RDP computation.
- `docs/fh_rl_training.md` – user guide for training/eval.

## Next Steps
Follow the TODO list in `FH_RL_TODO.md`, updating status after each file or milestone to keep progress auditable.
