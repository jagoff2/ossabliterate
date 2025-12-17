# Next-run TODO (no warmup)
[x] 1. Disable LR/γ warm-up in config; set gamma target outright and rely on lower initial LR.
[x] 2. Increase introspection presence: bump weight in blend config + add staged supervised pre-pass.
[ ] 3. Add plan/report supervision head (reuse meta pipeline) to provide explicit reflective loss.
[ ] 4. Improve logging: per-eval samples saved in structured JSON plus plan/report coverage metrics.
[ ] 5. Launch new run with revised configuration (rank 48, β 0.04, γ fixed) and monitor until eval checkpoint.