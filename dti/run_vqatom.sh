#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
SPLIT="${SPLIT:-1}"
EPOCHS=20
FTP_USER="chicappa.jp-wakou"
FTP_HOST="ftp.lolipop.jp"
FTP_REMOTE_DIR="${FTP_REMOTE_DIR:-dti_data}"
BASE="/vqatom"
DATA="$BASE/data"
DTI="$BASE/dti"
TRAIN="$DATA/train_split${SPLIT}.csv"
VALID="$DATA/valid_split${SPLIT}.csv"
TEST="$DATA/test_split${SPLIT}.csv"
RUN="vqatom_split${SPLIT}_seed${SEED}"
OUT_DIR="$DTI/$RUN"

if [[ -z "${FTP_PASS:-}" ]]; then
  read -r -s -p "FTP Password: " FTP_PASS </dev/tty
  echo
fi
[[ -n "$FTP_PASS" ]] || { echo "ERROR: FTP password empty" >&2; exit 1; }

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
mkdir -p "$DATA"
fetch_file() {
  local remote="$1" local_name="$2"
  if [[ -s "$local_name" ]]; then
    echo "[FTP SKIP] $local_name"
    return
  fi
  echo "[FTP GET] $remote"
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; set xfer:clobber on; get \"$remote\" -o \"$local_name\"; bye"
  [[ -s "$local_name" ]] || { echo "ERROR: Missing $local_name" >&2; exit 1; }
}
for name in "train_split${SPLIT}.csv" "valid_split${SPLIT}.csv" "test_split${SPLIT}.csv"; do
  fetch_file "$FTP_REMOTE_DIR/$name" "$DATA/$name"
done
CKPT="$DATA/model_epoch_3.pt"
fetch_file "$FTP_REMOTE_DIR/model_epoch_3.pt" "$CKPT"
cd "$DTI"
[[ -f dti.py ]] || { echo 'ERROR: dti.py missing' >&2; exit 1; }
python dti.py \
  --attn_activation sigmoid \
  --attn_temp 0.7 \
  --batch_size 16 \
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
  --y_thr 12.1 \
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
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; set xfer:clobber on; put \"$JSON_ARCHIVE\" -o \"$JSON_ARCHIVE\" & put \"$QK_ARCHIVE\" -o \"$QK_ARCHIVE\" & wait; bye"
else
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; set xfer:clobber on; put \"$JSON_ARCHIVE\" -o \"$JSON_ARCHIVE\"; bye"
fi
echo "Completed: $RUN"
