#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-0}"
SPLIT="${SPLIT:-1}"
EPOCHS=40

# FTP password prompt (hidden input)
FTP_USER="chicappa.jp-wakou"
FTP_HOST="ftp.lolipop.jp"

read -r -s -p "FTP Password: " FTP_PASS </dev/tty
echo
export FTP_PASS
[[ -n "$FTP_PASS" ]] || {
  echo "ERROR: FTP password is empty" >&2
  exit 1
}
FTP_REMOTE_DIR="${FTP_REMOTE_DIR:-dti_data}"
FTP_RESULTS_DIR="${FTP_RESULTS_DIR:-dti_results}"
: "${FTP_PASS:?Set FTP_PASS in the environment before running}"

BASE="/vqatom"
DATA="$BASE/data"
DTI="$BASE/dti"
TRAIN="$DATA/train_split${SPLIT}.csv"
VALID="$DATA/valid_split${SPLIT}.csv"
TEST="$DATA/test_split${SPLIT}.csv"
VOCAB="$DATA/morgan_vocab.csv"
RUN="morgan_split${SPLIT}_seed${SEED}"
OUT_DIR="$DTI/$RUN"

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
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set net:max-retries 10; set net:timeout 30; set xfer:clobber on; get \"$remote\" -o \"$local_name\"; bye"
  [[ -s "$local_name" ]] || { echo "ERROR: Missing $local_name" >&2; exit 1; }
}

for name in "train_split${SPLIT}.csv" "valid_split${SPLIT}.csv" "test_split${SPLIT}.csv"; do
  fetch_file "$FTP_REMOTE_DIR/$name" "$DATA/$name"
done
fetch_file "$FTP_REMOTE_DIR/morgan_vocab.csv" "$VOCAB"

# Inline Morgan input verification; no external Python script or heredoc.
python -c '
import csv, sys
vocab_path, *paths = sys.argv[1:]
with open(vocab_path, newline="") as f:
    vocab_rows = list(csv.DictReader(f))
n = len(vocab_rows)
if n == 0:
    raise SystemExit("Morgan vocabulary is empty")
if {int(row["mapped_id"]) for row in vocab_rows} != set(range(1, n + 1)):
    raise SystemExit("Morgan mapped_id must be exactly 1..N")
for path in paths:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "lig_tok" not in (reader.fieldnames or []):
            raise SystemExit(f"{path}: missing lig_tok")
        for line, row in enumerate(reader, 2):
            ids = [int(x) for x in (row.get("lig_tok") or "").replace(",", " ").split()]
            if not ids or any(x < 1 or x > n for x in ids):
                raise SystemExit(f"{path}:{line}: invalid Morgan IDs (expected 1..{n})")
print(f"[Morgan] Dense ID range validated: 1..{n}; confirm CSVs use this vocabulary")
' "$VOCAB" "$TRAIN" "$VALID" "$TEST"

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
  --ligand_mode morgan \
  --morgan_vocab_path "$VOCAB" \
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
  --y_thr 12.1

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
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set net:max-retries 10; set net:timeout 30; set xfer:clobber on; mkdir -p \"$FTP_RESULTS_DIR\"; cd \"$FTP_RESULTS_DIR\"; put \"$JSON_ARCHIVE\" -o \"$JSON_ARCHIVE\" & put \"$QK_ARCHIVE\" -o \"$QK_ARCHIVE\" & wait; bye"
else
  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set net:max-retries 10; set net:timeout 30; set xfer:clobber on; mkdir -p \"$FTP_RESULTS_DIR\"; cd \"$FTP_RESULTS_DIR\"; put \"$JSON_ARCHIVE\" -o \"$JSON_ARCHIVE\"; bye"
fi

echo "Completed: $RUN"
