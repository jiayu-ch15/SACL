conda activate mappo-sc
ls -R | grep wandb/run | grep -v files | grep -v logs | xargs wandb sync --no-mark-synced