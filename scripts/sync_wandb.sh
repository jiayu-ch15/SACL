conda activate mappo-sc
ls | grep -E "run|offline" | grep -v latest | xargs wandb sync --no-mark-synced