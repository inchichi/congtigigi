#!/usr/bin/env bash
set -euo pipefail

stamp="$(date +%Y%m%d_%H%M%S)"

python train_improved.py \
  --content_dir input/content/PNG \
  --style_dir input/style \
  --style_recursive \
  --holdout_style_ratio 0.2 \
  --best_metric_target holdout_style \
  --use_ema_eval \
  --eval_interval 500 \
  --eval_batches 30 \
  --save_model_interval 1000 \
  --gram_style_weight 1.0 \
  --recon_weight 5.0 \
  --max_iter 10000 \
  --batch_size 8 \
  --save_dir "experiments_improved/generalize_${stamp}" \
  --log_dir "logs_improved/generalize_${stamp}" \
  "$@"
