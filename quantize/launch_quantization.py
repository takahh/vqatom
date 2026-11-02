#!/usr/bin/env python3
"""Vast.ai VQAtom quantization launcher. Requires dti_launcher_base.py alongside.

Dry run by default. --execute rents new machines after confirmation.
--auto-destroy removes ONLY matching owned instances after remote FTP readback hash verification.
"""
import argparse, concurrent.futures, getpass, hashlib, json, os, re, shlex, subprocess, sys, threading, time
from pathlib import Path
import dti_launcher_base as base

HERE = Path(__file__).resolve().parent
STATE = HERE / 'quantization_jobs.json'
LOCK = threading.RLock()

def log(*a):
    with LOCK: print(*a, flush=True)

def read_state():
    with LOCK:
        return json.loads(STATE.read_text()) if STATE.exists() else {}

def update(name, **kwargs):
    with LOCK:
        data=read_state(); data[name]={**data.get(name, {}), **kwargs}
        tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)); tmp.replace(STATE)

def job_name(cb, dim, tag): return f'quant_cb{cb}_d{dim}_{tag}'

# Remote setup script, invoked with CB, DIM, TAG, DATA_ID, and FTP_PASS in environment.
# Uses existing vqatom repository baked into the selected image or staged separately.
REMOTE_SCRIPT = r'''#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
export PATH=/venv/main/bin:$PATH
export DGLBACKEND=pytorch CUBLAS_WORKSPACE_CONFIG=:4096:8
OUT="/vqatom/quantize/outputs_cb${CB}_d${DIM}_${TAG}"
FILE="model_kmeans_only_cb${CB}.pt"
REMOTE="model_kmeans_only_cb${CB}_d${DIM}_${TAG}.pt"
mkdir -p "$OUT" /vqatom/data
# Make NVIDIA pip CUDA libraries visible to DGL. This must precede import dgl.
CUDA_PATHS=$(python - <<'PYC'
import glob,site,os
for root in site.getsitepackages():
 for path in glob.glob(os.path.join(root,'nvidia','*','lib')):
  if os.path.isdir(path): print(path)
PYC
)
while IFS= read -r path; do
  test -z "$path" || export LD_LIBRARY_PATH="$path:${LD_LIBRARY_PATH:-}"
done <<< "$CUDA_PATHS"
# Install compatible torch only if the current image cannot run CUDA.
if ! python - <<'PYTORCH'
import torch
assert torch.cuda.is_available()
a=torch.ones(8,8,device='cuda'); torch.cuda.synchronize()
PYTORCH
then
  GPU_MAJOR=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | cut -d. -f1)
  if (( GPU_MAJOR >= 10 )); then
    python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  else
    python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
  fi
fi
# DGL is installed only when absent; missing shared libraries are a separate error.
if ! python -m pip show dgl >/dev/null 2>&1; then
  python -m pip install --no-deps 'dgl==2.4.0' -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
fi
python - <<'PYGPU'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
a=torch.randn(32,32,device='cuda'); torch.cuda.synchronize()
print('GPU:',torch.cuda.get_device_name(0),'Torch:',torch.__version__,flush=True)
PYGPU
# Reuse working DGL. Do not reinstall DGL when a shared library is missing.
# Install Python dependencies required by vqatom_train.py.
python -m pip install \
  numpy \
  psutil \
  pytz \
  requests \
  PyYAML \
  einops \
  ogb \
  torch-geometric
python - <<'PYDGL'
import torch,dgl
g=dgl.graph(([0,1],[1,2]),device='cuda')
g.ndata['h']=torch.ones(3,16,device='cuda')
g.update_all(dgl.function.copy_u('h','m'),dgl.function.sum('m','h'))
torch.cuda.synchronize()
print('DGL GPU TEST SUCCESS:',dgl.__version__,flush=True)
PYDGL
cd /vqatom/quantize
test -s vqatom_train.py || { echo '[ERROR] quantization source missing'; exit 21; }
if ! test -d /vqatom/data/discret_final; then
  command -v gdown >/dev/null || python -m pip install gdown
  cd /vqatom/data
  gdown "$DATA_ID" -O discret_final.tar.gz
  tar -tzf discret_final.tar.gz >/dev/null
  tar -xzf discret_final.tar.gz
fi
test -d /vqatom/data/discret_final
cd /vqatom/quantize
python -u vqatom_train.py \
  --exp_setting ind --teacher SAGE --train_or_infer kmeans_only \
  --dataset molecules --output_path "$OUT" --seed 0 --max_epoch 1 \
  --patience 5 --device 0 --get_umap_data --codebook_size "$CB" \
  --hidden_dim "$DIM" --gnn_hops 2 --chem_fusion_alpha 0 \
  --chunk_size 1000 --chunk_size2 1000 --ss_max_total_latent_count 40000 \
  --semantic_atom_features --semantic_hop_mode separate \
  --semantic_hop1_weight 1.0 --semantic_hop2_weight 1.0 \
  --fixed_encoder_ema --kmeans_iters 20
test -s "$OUT/$FILE"
command -v lftp >/dev/null || { apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq lftp; }
# FTP destination must already exist. Never echo the password.
lftp -u "chicappa.jp-wakou","$FTP_PASS" ftp.lolipop.jp -e "set cmd:fail-exit yes; set net:max-retries 3; set net:timeout 30; cd dti_data; put \"$OUT/$FILE\" -o \"$REMOTE\"; bye"
VERIFY="/tmp/${REMOTE}.verify"
rm -f "$VERIFY"
lftp -u "chicappa.jp-wakou","$FTP_PASS" ftp.lolipop.jp -e "set cmd:fail-exit yes; set net:max-retries 3; set net:timeout 30; cd dti_data; get \"$REMOTE\" -o \"$VERIFY\"; bye"
test -s "$VERIFY"
sha256sum "$OUT/$FILE" "$VERIFY"
test "$(sha256sum "$OUT/$FILE" | cut -d' ' -f1)" = "$(sha256sum "$VERIFY" | cut -d' ' -f1)"
echo '[FTP VERIFIED] SHA256 MATCH'
rm -f "$VERIFY"
'''
def quant_gpu_supported(offer, mode):
    import re

    name = str(offer.get("gpu_name") or "").strip()

    blackwell = bool(re.search(
        r"^(?:RTX\s+5090|RTX\s+PRO\s+(?:5000|6000))\b",
        name, re.I
    ))

    standard = bool(re.search(
        r"^(?:RTX\s+4090|RTX\s+4080S?|A100|H100|H200|L40S|A40)\b",
        name, re.I
    ))

    if mode == "blackwell":
        return blackwell

    if mode == "standard":
        return blackwell or standard

    raise ValueError(f"Unknown GPU mode: {mode}")

def choose(args, excluded):
    with LOCK:
        blacklist = base.bad_hosts_read()

        for _ in range(100):
            offer = base.find_offer(args, excluded)
            excluded.add(str(offer["id"]))

            if not quant_gpu_supported(offer, args.gpu_mode):
                log("[SKIP GPU]", offer.get("gpu_name"))
                continue

            key = base.host_key(offer)

            if not key:
                continue

            if key in blacklist or key in base.RESERVED_HOSTS:
                continue

            if offer.get("verification") != "verified" and offer.get("verified") is not True:
                continue

            if float(offer.get("reliability") or 0) < args.min_reliability:
                continue

            if int(offer.get("num_gpus") or 0) != 1:
                continue

            base.RESERVED_HOSTS.add(key)
            return offer

        raise RuntimeError("No eligible quantization GPU offers")


def wait_ssh(iid, timeout): return base.wait_ready_with_auth_diagnostics(iid, timeout)

def remote(iid, cmd, timeout=60): return base.remote_check(iid,cmd,timeout=timeout)

def launch(cb,args,password,excluded):
    name=job_name(cb,args.hidden_dim,args.tag); iid=None
    try:
        offer=choose(args,excluded)
        log('[OFFER]',name,offer.get('gpu_name'),f'${base.offer_price(offer):.3f}/h', 'host',base.host_key(offer))
        if not args.execute: return True
        result=base.cli('create','instance',offer['id'],'--image',args.image,'--disk',args.disk,'--ssh','--direct','--label',name,'--cancel-unavail','--raw',timeout=120)
        if not result.get('success') or not result.get('new_contract'): raise RuntimeError(f'Vast create failed: {result}')
        iid=int(result['new_contract']); update(name,id=iid,status='created',offer_id=offer['id'],host_key=base.host_key(offer),created_at=time.time())
        conn=wait_ssh(iid,args.startup_timeout)
        # Upload local quantization source; exclude previous outputs and large artifacts.
        source = Path(args.source_dir).expanduser().resolve()
        tar = subprocess.Popen(['tar','-C',str(source),'--exclude=outputs*','--exclude=.git',
                                '--exclude=__pycache__','--exclude=*.pt','-czf','-','.'],
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        ssh = subprocess.run(base.ssh_cmd(conn,'mkdir -p /vqatom/quantize /vqatom/data && tar -xzf - -C /vqatom/quantize'),
                             stdin=tar.stdout,capture_output=True,timeout=240)
        tar.stdout.close(); tar.wait(timeout=30)
        if tar.returncode or ssh.returncode: raise RuntimeError(f'Source upload failed: {ssh.stderr[-300:]!r}')
        # molecules.npz is required by the molecules dataset loader on fresh machines.
        asset = Path(args.molecules_file).expanduser().resolve()
        with asset.open('rb') as f:
            up = subprocess.run(base.ssh_cmd(conn,'cat > /vqatom/data/molecules.npz'),stdin=f,capture_output=True,timeout=120)
        if up.returncode: raise RuntimeError(f'molecules.npz upload failed: {up.stderr[-300:]!r}')
        rc,out,err=remote(iid,'test -s /vqatom/quantize/vqatom_train.py && test -s /vqatom/data/molecules.npz',timeout=45)
        if rc: raise RuntimeError('Source/data verification failed')
        run_script='\n'.join([
            '#!/usr/bin/env bash','set -Eeuo pipefail','umask 077',
            "trap 'rc=$?; rm -f /tmp/quant_runner.sh; if ((rc==0)); then touch /tmp/quant_success; else echo \"$rc\" >/tmp/quant_failed; fi' EXIT",
            f'export CB={cb} DIM={args.hidden_dim} TAG={shlex.quote(args.tag)}',
            f'export DATA_ID={shlex.quote(args.data_id)}',
            f'export FTP_PASS={shlex.quote(password)}',
            REMOTE_SCRIPT,
        ])
        p=subprocess.run(base.ssh_cmd(conn,'umask 077; cat > /tmp/quant_runner.sh; chmod 700 /tmp/quant_runner.sh'),input=run_script,text=True,capture_output=True,timeout=90)
        if p.returncode: raise RuntimeError(f'Runner upload failed: {p.stderr[-300:]}')
        rc,out,err=remote(iid,'nohup setsid bash /tmp/quant_runner.sh </dev/null >/tmp/quant_run.log 2>&1 & echo STARTED',timeout=45)
        if rc or 'STARTED' not in out: raise RuntimeError(f'Start failed: {err[-300:]} {out[-300:]}')
        update(name,status='running',started_at=time.time())
        log('[STARTED]',name,'instance',iid)
        return True
    except Exception as e:
        log('[LAUNCH ERROR]',name,str(e))
        if iid is not None: update(name,status='manual_required',error=str(e))
        return False

def monitor(args):
    while True:
        jobs=read_state()
        active=[(n,j) for n,j in jobs.items() if j.get('status') in ('running','created')]
        if not active: log('[MONITOR] no active jobs'); return
        for name,j in active:
            iid=j['id']
            try:
                rc,out,err=remote(iid,'if test -f /tmp/quant_success; then echo SUCCESS; elif test -f /tmp/quant_failed; then echo FAILED:$(cat /tmp/quant_failed); elif test -f /tmp/quant_run.log; then echo RUNNING; else echo WAITING; fi',timeout=45)
                if rc: log('[MONITOR RETRY]',name,err[-200:]); continue
                if out.startswith('SUCCESS'):
                    log('[VERIFIED SUCCESS]',name)
                    if args.auto_destroy:
                        matches=[i for i in base.instances() if str(i.get('id'))==str(iid)]
                        if len(matches)!=1 or str(matches[0].get('label'))!=name:
                            raise RuntimeError('Instance ID/label mismatch: destroy blocked')
                        base.cli('destroy','instance',iid,'-y','--raw',timeout=90)
                        update(name,status='destroyed',finished_at=time.time())
                        log('[DESTROYED]',name,iid)
                    else: update(name,status='success_retained')
                elif out.startswith('FAILED'):
                    update(name,status='failed',error=out)
                    log('[FAILED / INSTANCE RETAINED]',name,iid,out)
            except Exception as e: log('[MONITOR ERROR]',name,str(e))
        time.sleep(args.poll_seconds)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codebook-sizes',nargs='+',type=int,default=[1000,5000,10000])
    p.add_argument('--hidden-dim',type=int,default=64)
    p.add_argument('--tag',default='semantic147_w1')
    p.add_argument('--data-id',default='17vq0OVvV1zks2Q2_zBlExYe7sX91n68F',help='Google Drive dataset ID; confirm before renting')
    p.add_argument('--min-price',type=float,default=0.0)
    p.add_argument('--max-price',type=float,default=1.5)
    p.add_argument('--min-vram',type=int,default=24)
    p.add_argument('--min-reliability',type=float,default=.97)
    p.add_argument('--gpu-mode',choices=['blackwell','standard'],default='blackwell')
    p.add_argument('--disk',type=int,default=60)
    p.add_argument('--image',default='pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
    p.add_argument('--source-dir',default=str(HERE.parent/'quantize'),help='Local VQAtom quantize source directory')
    p.add_argument('--molecules-file',default=str(HERE.parent/'data'/'molecules.npz'),help='Local molecules.npz dataset file')
    p.add_argument('--startup-timeout',type=int,default=1200)
    p.add_argument('--max-parallel',type=int,default=3)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--auto-destroy',action='store_true')
    p.add_argument('--monitor-only',action='store_true')
    p.add_argument('--poll-seconds',type=int,default=60)
    args=p.parse_args()
    if args.monitor_only: monitor(args); return
    if args.max_parallel<1 or args.disk<45 or args.min_price>args.max_price or any(c<=0 for c in args.codebook_sizes): p.error('Invalid parameters')
    if len(set(args.codebook_sizes))!=len(args.codebook_sizes): p.error('Duplicate codebook size')
    # Never duplicate existing instances or completed jobs.
    existing=read_state(); live_labels={str(i.get('label')) for i in base.instances()}
    tasks=[cb for cb in args.codebook_sizes if job_name(cb,args.hidden_dim,args.tag) not in live_labels and existing.get(job_name(cb,args.hidden_dim,args.tag),{}).get('status') not in ('running','created','success_retained','destroyed','manual_required')]
    log('[PLAN]',tasks,'CB; GPU mode',args.gpu_mode,'price <=',args.max_price,'disk',args.disk)
    log('[SOURCE]',args.source_dir,'[MOLECULES]',args.molecules_file)
    if not args.execute:
        log('[DRY RUN] No rental; source and data paths must exist before --execute.')
        return
    if not (Path(args.source_dir).expanduser()/'vqatom_train.py').is_file(): p.error('Local quantize/vqatom_train.py missing; pass --source-dir')
    if not Path(args.molecules_file).expanduser().is_file(): p.error('Local molecules.npz missing; pass --molecules-file')
    if not sys.stdin.isatty(): p.error('Interactive terminal required')
    if input(f'Type RENT {len(tasks)}: ')!=f'RENT {len(tasks)}': p.error('Cancelled')
    password=os.environ.get('FTP_PASS') or getpass.getpass('FTP password: ')
    if not password: p.error('FTP password required')
    # Share existing DTI blacklist but keep separate quantization job state.
    base.RESERVED_HOSTS.clear()
    for inst in base.instances():
        key=base.host_key(inst)
        if key: base.RESERVED_HOSTS.add(key)
    excluded=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        results=list(pool.map(lambda cb:launch(cb,args,password,excluded),tasks))
    log('[LAUNCH RESULT]',sum(results),'/',len(tasks))
    if args.auto_destroy: monitor(args)

if __name__=='__main__': main()
