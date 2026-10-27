#!/usr/bin/env bash
set -euo pipefail

DATASET="${DATASET:-kiba}"
COLD_TYPE="${COLD_TYPE:-protein_scaffold_cold}"
SEED="${SEED:-0}"
SPLIT="${SPLIT:-1}"
# Dataset-specific binary affinity threshold (can be overridden with Y_THR).
if [[ "$DATASET" == "metz" ]]; then
  Y_THR="${Y_THR:-6.0}"
else
  Y_THR="${Y_THR:-12.1}"
fi

case "$DATASET" in kiba|metz) ;; *) echo "Invalid DATASET" >&2; exit 1;; esac
case "$COLD_TYPE" in protein_cold|ligand_cold|double_cold|scaffold_cold|protein_scaffold_cold) ;; *) echo "Invalid COLD_TYPE" >&2; exit 1;; esac
EPOCHS="${EPOCHS:-20}"
FTP_USER="chicappa.jp-wakou"
FTP_HOST="ftp.lolipop.jp"
FTP_REMOTE_DIR="${FTP_REMOTE_DIR:-dti_data}"
REMOTE="$FTP_REMOTE_DIR/$DATASET/$COLD_TYPE/split_$SPLIT"
FTP_RESULTS_DIR="${FTP_RESULTS_DIR:-dti_results}/${DATASET}"
BASE="/vqatom"
DATA="$BASE/data/${DATASET}/${COLD_TYPE}/split_${SPLIT}"
DTI="$BASE/dti"
TRAIN="$DATA/${DATASET}_${COLD_TYPE}_split_${SPLIT}_train_morgan_cb5k_cb10k_cb15k_cb3k_cb4k.csv"
VALID="$DATA/${DATASET}_${COLD_TYPE}_split_${SPLIT}_valid_morgan_cb5k_cb10k_cb15k_cb3k_cb4k.csv"
TEST="$DATA/${DATASET}_${COLD_TYPE}_split_${SPLIT}_test_morgan_cb5k_cb10k_cb15k_cb3k_cb4k.csv"
RUN="${DATASET}_${COLD_TYPE}_vqatom_cb5k_split${SPLIT}_seed${SEED}"
OUT_DIR="$DTI/$RUN"

: "${FTP_PASS:?FTP_PASS environment variable is required}"
if [[ ! -d "$BASE/.git" ]]; then
  git clone https://github.com/takahh/vqatom.git "$BASE"
else
  git -C "$BASE" pull --ff-only
fi
pip install ogb googledrivedownloader category_encoders einops pyyaml gdown \
  matplotlib entmax transformers accelerate torch_geometric
if ! command -v lftp >/dev/null 2>&1; then
  apt-get update
  apt-get install -y lftp
fi
mkdir -p "$DATA" "$BASE/data"
fetch_file() {
  local remote="$1" local_name="$2" force="${3:-0}"
  if [[ "$force" != "1" && -s "$local_name" ]]; then
    echo "[FTP SKIP] $local_name"
    return
  fi
  echo "[FTP GET] $remote"
  local tmp="${local_name}.download.$$"
  rm -f "$tmp"
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; set xfer:clobber on; get \"$remote\" -o \"$tmp\"; bye"
  [[ -s "$tmp" ]] || { echo "ERROR: Download failed: $remote" >&2; rm -f "$tmp"; exit 1; }
  mv -f "$tmp" "$local_name"
  [[ -s "$local_name" ]] || { echo "ERROR: Missing $local_name" >&2; exit 1; }
}
# Refresh dataset CSVs by default; set FORCE_REFRESH_CSV=0 to reuse cached copies.
FORCE_REFRESH_CSV="${FORCE_REFRESH_CSV:-0}"
for kind in train valid test; do
  name="${DATASET}_${COLD_TYPE}_split_${SPLIT}_${kind}_morgan_cb5k_cb10k_cb15k_cb3k_cb4k.csv"
  fetch_file "$REMOTE/$name" "$DATA/$name" "$FORCE_REFRESH_CSV"
done
CKPT="$BASE/data/model_epoch_6_5k.pt"
fetch_file "$FTP_REMOTE_DIR/model_epoch_6_5k.pt" "$CKPT"
# Verify again AFTER pip installs, because dependencies may replace PyTorch.
python - <<'PYCUDACHECK'
import torch
print(f"[CUDA CHECK] torch={torch.__version__}, build={torch.version.cuda}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("[CUDA CHECK FAILED] CUDA unavailable after dependency installation")
x = torch.ones((32, 32), device="cuda")
torch.cuda.synchronize()
print(f"[CUDA CHECK OK] {torch.cuda.get_device_name(0)}", flush=True)
PYCUDACHECK
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$DTI"
[[ -f dti.py ]] || { echo 'ERROR: dti.py missing' >&2; exit 1; }
python dti.py \
  --attn_activation sigmoid \
  --attn_temp 0.7 \
  --batch_size 8 \
  --cls_lambda 0.05 \
  --contact_lambda 0 \
  --cross_lr 5e-5 \
  --dropout 0.3 \
  --epochs "$EPOCHS" \
  --final_eval_csv "$TEST" \
  --finetune_esm \
  --finetune_lig \
  --freeze_esm_bottom 28 \
  --lig_lr 5e-6 \
  --ligand_mode vqatom_cb5k \
  --ligand_token_dropout 0.1 \
  --llrd \
  --llrd_decay 0.97 \
  --n_heads 8 \
  --out_dir "$OUT_DIR" \
  --pl_lp_overlap both \
  --protein_lr 2.5e-6 \
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

shopt -s nullglob
json_files=( "$RUN"/*.json )
(( ${#json_files[@]} > 0 )) || { echo "ERROR: No JSON files in $OUT_DIR" >&2; exit 1; }
JSON_ARCHIVE="${RUN}.tar.gz"
QK_ARCHIVE="${RUN}_qkmaps.tar.gz"
tar czf "$JSON_ARCHIVE" "${json_files[@]}" &
PID_JSON=$!
PID_QK=''
if [[ -d "$RUN/qk_maps" ]]; then
  tar czf "$QK_ARCHIVE" "$RUN/qk_maps" &
  PID_QK=$!
else
  echo '[WARN] qk_maps missing; uploading JSON only'
fi
wait "$PID_JSON"
if [[ -n "$PID_QK" ]]; then
  wait "$PID_QK"
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; set xfer:clobber on; cd \"$FTP_RESULTS_DIR\"; put \"$JSON_ARCHIVE\" -o \"$JSON_ARCHIVE\" & put \"$QK_ARCHIVE\" -o \"$QK_ARCHIVE\" & wait; bye"
else
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; set xfer:clobber on; cd \"$FTP_RESULTS_DIR\"; put \"$JSON_ARCHIVE\" -o \"$JSON_ARCHIVE\"; bye"
fi
echo "Completed: $RUN"
