#!/usr/bin/env bash
set -euo pipefail

BASE=/vqatom
DTI="$BASE/dti"
DATASET=kiba
COLD_TYPE=scaffold_cold
SPLIT=1
SEED="${SEED:-0}"
EPOCHS=20
Y_THR=12.1

DATA="$BASE"
TRAIN="$DATA/kiba_scaffold_cold_split_1_train_cb5k.csv"
VALID="$DATA/kiba_scaffold_cold_split_1_valid_cb5k.csv"
TEST="$DATA/kiba_scaffold_cold_split_1_test_cb5k.csv"

CKPT="$BASE/data/model_epoch_6_5k.pt"
RUN="kiba_scaffold_cold_vqatom_cb5k_split1_seed${SEED}"
OUT_DIR="$DTI/$RUN"

for f in "$TRAIN" "$VALID" "$TEST" "$CKPT"; do
    [[ -s "$f" ]] || { echo "Missing: $f" >&2; exit 1; }
done

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$DTI"
python dti.py \
  --attn_activation sigmoid \
  --attn_temp 0.7 \
  --batch_size 4 \
  --cls_lambda 0.05 \
  --contact_lambda 0 \
  --cross_lr 1e-4 \
  --dropout 0.3 \
  --epochs "$EPOCHS" \
  --final_eval_csv "$TEST" \
  --finetune_esm \
  --finetune_lig \
  --freeze_esm_bottom 28 \
  --lig_lr 1e-5 \
  --ligand_mode vqatom \
  --ligand_token_dropout 0.1 \
  --llrd \
  --llrd_decay 0.97 \
  --n_heads 8 \
  --out_dir "$OUT_DIR" \
  --pl_lp_overlap both \
  --protein_lr 5e-6 \
  --protein_token_dropout 0.1 \
  --rank_lambda 0.05 \
  --rank_loss pairwise \
  --rank_margin 0.0 \
  --reg_lambda 1.0 \
  --seed "$SEED" \
  --select_on auroc \
  --train_csv "$TRAIN" \
  --use_train_valid_csv \
  --valid_csv "$VALID" \
  --weight_decay 0.05 \
  --y_thr "$Y_THR" \
  --vq_ckpt "$CKPT"

echo "Completed: $RUN"
