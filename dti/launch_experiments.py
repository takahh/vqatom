#!/usr/bin/env python3
"""Vast.ai detached DTI launcher. Requires launch_experiments_fast.py in same folder.

Remote training survives SSH disconnection. Mac monitor deletes ONLY after
training exits 0 AND uploaded archives are read back and SHA-256 verified.
Use --monitor-only to resume after Mac restarts.
"""
import argparse
import concurrent.futures
import csv
import re
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
PRINT_LOCK = threading.Lock()

def log(*args):
    with PRINT_LOCK:
        print(*args, flush=True)

def cli(*args, timeout=90):
    p = subprocess.run(['vastai', *map(str, args)], text=True, capture_output=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"vastai {' '.join(map(str,args))}: {p.stderr or p.stdout}")
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f'Expected JSON from vastai: {p.stdout[:600]}')

def offer_price(offer):
    for field in ('dph_total', 'dph'):
        if offer.get(field) is not None:
            return float(offer[field])
    return None

def find_offer(args, excluded):
    # Apply price limits server-side: filtering only the first 100 cheapest
    # unrestricted offers can incorrectly yield zero eligible results.
    query = (f'rentable=true verified=true '
             f'reliability>={args.min_reliability} '
             f'gpu_ram>={args.min_vram} num_gpus=1 '
             f'dph>={args.min_price} dph<={args.max_price}')
    offers = cli('search', 'offers', query, '--raw', '--limit', '1000')
    if isinstance(offers, dict):
        offers = offers.get('offers', offers.get('results', []))
    if not isinstance(offers, list):
        raise RuntimeError(f'Unexpected Vast.ai offers response: {type(offers).__name__}')
    candidates = [o for o in offers if isinstance(o, dict)
                  and str(o.get('id')) not in excluded
                  and offer_price(o) is not None
                  and args.min_price <= offer_price(o) <= args.max_price
                  and float(o.get('disk_space') or o.get('disk_space_total') or args.disk) >= args.disk]

    candidates.sort(key=lambda o: (
        -int(float(o.get('cpu_cores_effective') or 0) >= 32),
        offer_price(o),
        -float(o.get('dlperf') or 0),
        -float(o.get('total_flops') or 0),
    ))
    if not candidates:
        raise RuntimeError(
            f'No offers after filtering: Vast returned {len(offers)} offers '
            f'for query={query!r}; {len(excluded)} offer IDs excluded. '
            'Check current availability and disk requirements.')
    return candidates[0]

def ssh_args(instance_id, timeout=60):
    p = subprocess.run(['vastai', 'ssh-url', str(instance_id)], capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    output = p.stdout.strip()
    match = re.search(r'ssh://[^\s\'"]+', output)
    if match:
        url = urlsplit(match.group(0))
        return ['-p', str(url.port or 22), f'{url.username or "root"}@{url.hostname}']
    match = re.search(r'ssh\s+-p\s+(\d+)\s+([^\s]+@[^\s]+)', output)
    if match:
        return ['-p', match.group(1), match.group(2)]
    raise RuntimeError(f'Cannot parse SSH URL: {output}')

def ssh_cmd(connection, remote_cmd):
    port, target = connection[1], connection[2]
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=6', '-p', port, target, remote_cmd]
STATE = HERE / 'detached_jobs.json'
BAD_HOSTS = HERE / 'failed_vast_hosts.json'
FAILED_CSV = HERE / 'detached_start_failures.csv'
LOCK = threading.Lock()
# Reserved physical hosts for this launch, including already-running instances.
RESERVED_HOSTS = set()


def state_read():
    if not STATE.exists():
        return {}
    return json.loads(STATE.read_text())


def state_put(name, entry):
    with LOCK:
        jobs = state_read()
        jobs[name] = {**jobs.get(name, {}), **entry}
        temp = STATE.with_suffix('.tmp')
        temp.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
        temp.replace(STATE)


def host_key(offer):
    """Stable physical host identifier when Vast exposes one."""
    for key in ('machine_id', 'host_id'):
        if offer.get(key) is not None:
            return f'{key}:{offer[key]}'
    return None


def bad_hosts_read():
    if not BAD_HOSTS.exists():
        return {}
    return json.loads(BAD_HOSTS.read_text())


def bad_host_record(offer, reason):
    key = host_key(offer)
    if not key:
        log(f'[HOST UNKNOWN] offer={offer.get("id")}; cannot blacklist physical host')
        return
    with LOCK:
        data = bad_hosts_read()
        data[key] = {'offer_id': offer.get('id'), 'reason': str(reason), 'at': time.time()}
        temp = BAD_HOSTS.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        temp.replace(BAD_HOSTS)
    log(f'[HOST BLACKLISTED] {key}: {reason}')


# Explicit, mutually exclusive GPU families. Unknown GPU names are rejected.
# "standard" is an allowlist, not an unrestricted fallback.
def gpu_supported(offer, mode='blackwell'):
    name = str(offer.get('gpu_name') or '').strip()
    blackwell = bool(re.search(r'\bRTX\s+PRO\s+(?:5000|6000)(?:\s+(?:Blackwell|S|WS|Server|Workstation))?\b', name, re.I))
    if mode == 'blackwell':
        return blackwell
    if mode == 'standard':
        if blackwell:
            return False
        return bool(re.search(
            r'^(?:A100(?:\s|$)|A800(?:\s|$)|H100(?:\s|$)|H200(?:\s|$)|'
            r'L40S?$|A40$|RTX\s+6000\s*Ada(?:\s|$)|RTX\s+5880\s*Ada(?:\s|$)|'
            r'RTX\s+A6000(?:\s|$)|RTX\s+4090(?:\s|$))',
            name, re.I))
    raise ValueError(f'Unknown GPU mode: {mode}')


def choose_offer(args, excluded):
    # Physical host uniqueness, including other active account instances.
    # Reject offers without a stable host ID rather than silently bypassing safety.
    blacklist = bad_hosts_read()
    for _ in range(100):
        offer = find_offer(args, excluded)
        excluded.add(str(offer['id']))
        if not gpu_supported(offer, args.gpu_mode):
            log(f'[SKIP UNSUPPORTED GPU] {offer.get("gpu_name")!r} offer={offer.get("id")}')
            continue
        key = host_key(offer)
        if not key:
            log(f'[SKIP UNKNOWN HOST] offer={offer.get("id")}')
            continue
        if key in blacklist or key in RESERVED_HOSTS:
            continue
        if offer.get('verification') != 'verified' and offer.get('verified') is not True:
            continue
        if float(offer.get('reliability') or 0) < args.min_reliability:
            continue
        if int(offer.get('num_gpus') or 0) != 1:
            continue
        RESERVED_HOSTS.add(key)
        return offer
    raise RuntimeError('No eligible offers with a distinct, verified, reliable physical host')


def resolve_pytorch_vast_template():
    """Find the exact official PyTorch (Vast) template, without guessing a hash.

    Reject missing or ambiguous matches: renting the wrong template is costly.
    """
    data = cli('search', 'templates', 'PyTorch', '--raw', timeout=60)
    if isinstance(data, dict):
        for key in ('templates', 'results', 'offers'):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise RuntimeError(f'Unexpected Vast template search response: {str(data)[:500]}')
    matches = []
    for t in data:
        if not isinstance(t, dict):
            continue
        name = str(t.get('name') or t.get('template_name') or '').strip()
        # Exact template name from the user's screenshot; do not match similarly named images.
        if name.casefold() == 'pytorch (vast)':
            template_hash = t.get('hash_id') or t.get('template_hash')
            if template_hash:
                matches.append((str(template_hash), t))
    unique = {h: t for h, t in matches}
    if len(unique) != 1:
        names = [str(t.get('name') or t.get('template_name')) for t in data]
        raise RuntimeError(
            f'Expected one PyTorch (Vast) template with hash_id; found {len(unique)}. '
            f'Available template names: {names[:30]}. '
            'Supply --template-hash explicitly after inspecting `vastai search templates "PyTorch" --raw`.'
        )
    return next(iter(unique))



def instances():
    data = cli('show', 'instances', '--raw')
    if isinstance(data, dict):
        data = data.get('instances', data.get('results', []))
    return data if isinstance(data, list) else []


def remote_check(instance_id, command, timeout=45):
    conn = ssh_args(instance_id)
    p = subprocess.run(ssh_cmd(conn, command), text=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()



def wait_ready_with_auth_diagnostics(instance_id, seconds):
    """Fail early on persistent publickey rejection, but allow SSH key propagation.

    Does not change or destroy the instance. Retry cleanup handles pre-start
    instances only, after reconciling their IDs and labels with Vast.
    """
    start = time.monotonic()
    deadline = start + seconds
    auth_failures = 0
    last_error = ''
    while time.monotonic() < deadline:
        data = cli('show', 'instance', instance_id, '--raw')
        if isinstance(data, list):
            data = data[0] if data else {}
        state = data.get('actual_status')
        if state in ('exited', 'offline', 'unknown', 'stopped'):
            raise RuntimeError(f'Instance {instance_id} state: {state}')
        if state == 'running':
            try:
                conn = ssh_args(instance_id)
                p = subprocess.run(ssh_cmd(conn, 'true'), capture_output=True,
                                   text=True, timeout=30)
                if p.returncode == 0:
                    return conn
                last_error = p.stderr.strip()[-400:]
                if 'Permission denied (publickey)' in last_error:
                    auth_failures += 1
                else:
                    auth_failures = 0
                # Give the host 90 seconds for initial key propagation.
                # Repeated publickey refusal thereafter is treated as a
                # pre-start infrastructure failure, not a training failure.
                if auth_failures >= 3 and time.monotonic() - start >= 90:
                    raise TimeoutError(
                        f'Instance {instance_id} not SSH-ready: persistent '
                        f'publickey rejection ({auth_failures} attempts); {last_error}'
                    )
            except subprocess.TimeoutExpired:
                last_error = 'SSH connection timeout'
                auth_failures = 0
            except RuntimeError as exc:
                last_error = str(exc)
                auth_failures = 0
        time.sleep(min(15, max(0, deadline - time.monotonic())))
    raise TimeoutError(f'Instance {instance_id} not SSH-ready in {seconds}s; {last_error}')


def training_started(instance_id):
    """Detect actual train steps; separately recognize a live dti.py process."""
    rc, out, err = remote_check(instance_id, r'''python3 - <<'PYREMOTE'
import os, re
p = '/tmp/dti_run.log'
if os.path.isfile('/tmp/dti_success'):
    print('STARTED')
elif os.path.isfile('/tmp/dti_failed'):
    print('FAILED')
else:
    data = ''
    if os.path.isfile(p):
        with open(p, 'rb') as f:
            f.seek(max(0, os.path.getsize(p) - 1048576))
            data = f.read().decode('utf-8', 'replace')
    # tqdm writes carriage-return separated updates and may omit percentage.
    progress = re.search(r'train:[^\r\n]*?\|\s*([1-9]\d*)/\d+', data)
    if progress:
        print('STARTED')
    else:
        live = False
        for pid in os.listdir('/proc'):
            if not pid.isdigit():
                continue
            try:
                raw = open('/proc/' + pid + '/cmdline', 'rb').read()
                parts = [x.decode('utf-8', 'replace') for x in raw.split(b'\0') if x]
                if any(x.endswith('/dti.py') or x == 'dti.py' for x in parts):
                    live = True
                    break
            except (OSError, PermissionError):
                pass
        print('PROCESS_RUNNING' if live else 'WAITING')
PYREMOTE''', timeout=40)
    if rc:
        raise RuntimeError(f'SSH probe failed: {err[-200:]}')
    return out.strip().splitlines()[-1] if out.strip() else 'WAITING'


def save_remote_diagnostics(name, instance_id):
    """Save a bounded, best-effort remote snapshot on the Mac before any cleanup."""
    directory = HERE / 'experiment_logs'
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / f'{name}_instance{instance_id}_startup.log'
    cmd = ("echo '=== UTC ==='; date -u; "
           "echo '=== runner markers ==='; "
           "ls -l /tmp/dti_run.log /tmp/dti_success /tmp/dti_failed 2>&1 || true; "
           "echo '=== log tail ==='; tail -n 300 /tmp/dti_run.log 2>&1 || true; "
           "echo '=== processes ==='; ps auxww | grep -E '[d]ti.py|[p]ip|[l]ftp|[g]it|[d]ti_run' || true")
    try:
        rc, out, err = remote_check(instance_id, cmd, timeout=60)
        dest.write_text(f'SSH exit: {rc}\n{out}\nSTDERR:\n{err}\n')
    except Exception as exc:
        dest.write_text(f'Diagnostic collection unavailable: {exc}\n')
    log(f'[STARTUP LOG SAVED] {dest}')
    return str(dest)


def verify_owned_and_destroy(name, instance_id):
    # Require exact ID and label match.
    matches = [i for i in instances() if str(i.get('id')) == str(instance_id)]
    if not matches:
        log(f'[CLEANUP] {name}: instance {instance_id} already absent')
        return
    if len(matches) != 1 or str(matches[0].get('label')) != name:
        raise RuntimeError(f'unsafe destroy: instance {instance_id} label mismatch')
    cli('destroy', 'instance', instance_id, '-y', '--raw', timeout=90)
    if any(str(i.get('id')) == str(instance_id) for i in instances()):
        raise RuntimeError(f'instance {instance_id} still listed after destroy')
    log(f'[START TIMEOUT DESTROYED] {name}: instance={instance_id}')


def await_training_start(name, instance_id, created_at, args):
    deadline = created_at + args.training_start_timeout
    process_seen = False
    while True:
        try:
            result = training_started(instance_id)
            if result == 'PROCESS_RUNNING':
                if not process_seen:
                    log(f'[DTI PROCESS RUNNING] {name}: instance={instance_id}; awaiting train steps')
                process_seen = True
            if result == 'STARTED' and time.time() <= deadline:
                log(f'[TRAINING CONFIRMED] {name}: instance={instance_id}')
                return
            if result == 'FAILED':
                raise RuntimeError('remote runner failed before training began')
        except RuntimeError as exc:
            if 'remote runner failed' in str(exc):
                raise
            log(f'[START CHECK] {name}: {exc}')
        if time.time() >= deadline:
            if process_seen and training_started(instance_id) == 'PROCESS_RUNNING':
                log(f'[PROCESS CONFIRMED, PROGRESS UNVERIFIED] {name}: instance={instance_id}; retaining for monitoring')
                return
            raise TimeoutError(f'no positive train progress within {args.training_start_timeout}s of creation')
        time.sleep(min(args.start_check_seconds, max(1, deadline - time.time())))


def write_failed_csv(tasks):
    names = {f'{d}_{c}_{m}_split{sp}_seed{sd}' for d,c,m,sp,sd in tasks}
    rows = [(name, j) for name, j in state_read().items()
            if name in names and j.get('status') in ('start_failed', 'manual_required', 'launch_failed')]
    with FAILED_CSV.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['task', 'status', 'attempt', 'instance_id', 'reason'])
        for name, j in sorted(rows):
            w.writerow([name, j.get('status'), j.get('attempt'), j.get('id'), j.get('error', '')])
    log(f'[FAILED TASKS] {len(rows)}; CSV: {FAILED_CSV}')
    for name, j in sorted(rows):
        log(f'  {name} | attempt={j.get("attempt")} | {j.get("status")} | {j.get("error", "")}')


def launch(task, args, password, excluded, attempt=1):
    dataset, cold, model, split, seed = task
    name = f'{dataset}_{cold}_{model}_split{split}_seed{seed}'
    instance_id = None
    offer = None
    try:
        with LOCK:
            offer = choose_offer(args, excluded)
        log(f'[OFFER] {name}: {offer.get("gpu_name")}, ${offer_price(offer):.3f}/h')
        if not args.execute:
            return True
        res = cli('create', 'instance', offer['id'], '--image', args.image,
                       '--disk', args.disk, '--ssh', '--direct', '--label', name,
                       '--cancel-unavail', '--raw', timeout=120)
        if not res.get('success') or not res.get('new_contract'):
            raise RuntimeError(f'create failed: {res}')
        instance_id = int(res['new_contract'])
        created_at = time.time()
        state_put(name, {'id': instance_id, 'status': 'created', 'created_at': created_at, 'attempt': attempt, 'offer_id': offer.get('id'), 'host_key': host_key(offer)})
        log(f'[CREATED] {name}: {instance_id}')
        conn = wait_ready_with_auth_diagnostics(instance_id, min(args.startup_timeout, max(1, int(created_at + args.training_start_timeout - time.time()))))
        script = resolve_run_script(cold, model)
        up = subprocess.run(ssh_cmd(conn, 'umask 077; cat > /tmp/dti_run.sh; chmod 700 /tmp/dti_run.sh'),
                            input=script.read_bytes(), capture_output=True, timeout=90)
        if up.returncode:
            raise RuntimeError(f'script upload: {up.stderr.decode(errors="replace")}')
        # Secret sent only over SSH stdin. Runner is root-readable on the rented VM,
        # deleted by its EXIT trap. Do not share the VM with untrusted users.
        runner = '\n'.join([
            '#!/usr/bin/env bash', 'set -Eeuo pipefail', 'umask 077',
            'trap \'rc=$?; rm -f /tmp/dti_runner.sh; if (( rc == 0 )); then touch /tmp/dti_success; else echo "$rc" > /tmp/dti_failed; fi\' EXIT',
            f'export FTP_PASS={shlex.quote(password)}',
            f'export DATASET={shlex.quote(dataset)} COLD_TYPE={shlex.quote(cold)} SPLIT={split} SEED={seed}',
            f'export EPOCHS={dict(vqatom=20, vqatom_cb3k=20, vqatom_cb4k=20, vqatom_cb5k=20, vqatom_cb10k=20, vqatom_cb15k=20, continuous=40, morgan=40, smiles=5).get(model, args.epochs)}',
            f'export FTP_RESULTS_DIR={shlex.quote(args.ftp_results_dir)}',
            'if ! command -v python >/dev/null 2>&1; then',
            '  PY3=$(command -v python3) || exit 127',
            '  ln -sf "$PY3" /usr/local/bin/python', 'fi',
            'if test -x /venv/main/bin/python; then export PATH=/venv/main/bin:$PATH; fi',
            'python --version',
            '# Fail before data downloads or training if the rented GPU is unusable.',
            'python -c \'import torch; print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda, flush=True); assert torch.cuda.is_available(), "CUDA unavailable"; a=torch.ones((32,32),device="cuda"); torch.cuda.synchronize(); print("CUDA PREFLIGHT OK:",torch.cuda.get_device_name(0),flush=True)\'',
            '# lftp may not exist yet: run_*.sh installs it later, but preflight needs it now.',
            'if ! command -v lftp >/dev/null 2>&1; then',
            '  export DEBIAN_FRONTEND=noninteractive',
            '  apt-get update -qq && apt-get install -y -qq lftp',
            'fi',
            '# Check the existing FTP target without creating it.',
            'FTP_TARGET="${FTP_RESULTS_DIR:-dti_results}/${DATASET}"',
            'lftp -u "chicappa.jp-wakou","$FTP_PASS" "ftp.lolipop.jp" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; cd \"$FTP_TARGET\"; bye"',
            'echo "[FTP DIRECTORY EXISTS] $FTP_TARGET"',
            'bash /tmp/dti_run.sh',
            '# Independently verify uploaded archives by downloading and hashing.',
            'RUN_NAME="${DATASET}_${COLD_TYPE}_' + model + '_split${SPLIT}_seed${SEED}"',
            'FTP_RESULTS_DIR="${FTP_RESULTS_DIR:-dti_results}/${DATASET}"',
            'FTP_USER="chicappa.jp-wakou"',
            'FTP_HOST="ftp.lolipop.jp"',
            'cd /vqatom/dti',
            'verify_archive() {',
            '  local name="$1" local_copy="$2"',
            '  test -s "$name" || { echo "[VERIFY FAILED] local archive missing: $name"; return 1; }',
            '  rm -f "$local_copy"',
            '  lftp -u "$FTP_USER","$FTP_PASS" "$FTP_HOST" -e "set cmd:fail-exit yes; set net:max-retries 10; set net:timeout 30; cd \"$FTP_RESULTS_DIR\"; get \"$name\" -o \"$local_copy\"; bye"',
            '  test -s "$local_copy" || { echo "[VERIFY FAILED] FTP download missing: $name"; return 1; }',
            '  local a b',
            '  a=$(sha256sum "$name"); a=${a%% *}',
            '  b=$(sha256sum "$local_copy"); b=${b%% *}',
            '  test "$a" = "$b" || { echo "[VERIFY FAILED] checksum mismatch: $name"; return 1; }',
            '  echo "[FTP VERIFIED] $name sha256=$a"',
            '  rm -f "$local_copy"',
            '}',
            'verify_archive "${RUN_NAME}.tar.gz" "/tmp/${RUN_NAME}.verify.tar.gz"',
            'if test -f "${RUN_NAME}_qkmaps.tar.gz"; then',
            '  verify_archive "${RUN_NAME}_qkmaps.tar.gz" "/tmp/${RUN_NAME}_qkmaps.verify.tar.gz"',
            'fi',
            'echo "[TRAINING AND FTP VERIFIED] $RUN_NAME"', ''
        ])
        up = subprocess.run(ssh_cmd(conn, 'umask 077; cat > /tmp/dti_runner.sh; chmod 700 /tmp/dti_runner.sh'),
                            input=runner, text=True, capture_output=True, timeout=60)
        if up.returncode:
            raise RuntimeError(f'runner upload: {up.stderr}')
        # setsid+nohup + stdin /dev/null fully detach from SSH session.
        rc, out, err = remote_check(instance_id,
            'nohup setsid bash /tmp/dti_runner.sh </dev/null >/tmp/dti_run.log 2>&1 & echo STARTED', 45)
        if rc or 'STARTED' not in out:
            raise RuntimeError(f'detached start failed: {out} {err}')
        state_put(name, {'status': 'starting', 'started_at': time.time(), 'attempt': attempt})
        log(f'[DETACHED] {name}: instance={instance_id}; waiting for real train progress')
        await_training_start(name, instance_id, created_at, args)
        state_put(name, {'status': 'running', 'training_confirmed_at': time.time()})
        return True
    except Exception as exc:
        reason = str(exc)
        log(f'[START FAILURE] {name} attempt={attempt}: {reason}')
        # A startup timeout is not proof of a bad physical host.
        # Preserve the VM for inspection; never delete a possibly running job.
        if offer is not None and instance_id is not None and ('not SSH-ready' in reason or 'publickey' in reason):
            bad_host_record(offer, reason)
        status = 'launch_failed' if instance_id is None else 'manual_required'
        log_path = None
        if instance_id is not None:
            log_path = save_remote_diagnostics(name, instance_id)
            try:
                if training_started(instance_id) == 'STARTED':
                    state_put(name, {'status': 'running', 'training_confirmed_at': time.time(), 'startup_log': log_path})
                    log(f'[RECOVERED] {name}: training progress confirmed')
                    return True
            except Exception as probe_exc:
                log(f'[START PROBE UNAVAILABLE] {name}: {probe_exc}')
            log(f'[INSTANCE RETAINED] {name}: {instance_id}; inspect logs and destroy manually only if safe')
        state_put(name, {'status': status, 'error': reason,
                         'attempt': attempt, 'failed_at': time.time(),
                         'id': instance_id, 'offer_id': offer.get('id') if offer else None,
                         'host_key': host_key(offer) if offer else None,
                         'startup_log': log_path})
        return False


def launch_with_retry(task, args, password, excluded):
    name = f'{task[0]}_{task[1]}_{task[2]}_split{task[3]}_seed{task[4]}'
    if launch(task, args, password, excluded, attempt=1):
        return True
    if not args.retry_launch_once:
        return False
    if state_read().get(name, {}).get('status') == 'manual_required':
        log(f'[RETRY BLOCKED] {name}: previous instance cleanup uncertain')
        return False
    log(f'[RETRY 2/2] {name}: renting a different physical host')
    return launch(task, args, password, excluded, attempt=2)


def show_manual_summary(tasks):
    names = {f'{d}_{c}_{m}_split{sp}_seed{sd}' for d,c,m,sp,sd in tasks}
    pending = [(n, j) for n, j in state_read().items()
               if n in names and j.get('status') in ('launch_failed', 'manual_required', 'created')]
    if pending:
        log(f'[MANUAL LAUNCH LIST] {len(pending)} task(s):')
        for n, j in pending:
            log(f'  {n} | instance={j.get("id")} | status={j.get("status")} | {j.get("error", "")}')
    else:
        log('[MANUAL LAUNCH LIST] none')


def monitor(args):
    log('[MONITOR] checking remote success markers; Ctrl+C stops monitoring, not training')
    while True:
        jobs = state_read()
        active = [(name, job) for name, job in jobs.items() if job.get('status') in ('running', 'created', 'starting')]
        if not active:
            log('[MONITOR] no active tracked jobs')
            return
        for name, job in active:
            instance_id = job['id']
            try:
                rc, out, err = remote_check(instance_id,
                    'if test -f /tmp/dti_success; then echo SUCCESS; '
                    'elif test -f /tmp/dti_failed; then echo FAILED:$(cat /tmp/dti_failed); '
                    'elif test -f /tmp/dti_run.log; then echo RUNNING; '
                    'else echo WAITING; fi', timeout=35)
                if rc:
                    log(f'[CHECK RETRY] {name}: SSH exit={rc}')
                    continue
                if out.startswith('SUCCESS'):
                    log(f'[SUCCESS] {name}: training exited 0 and FTP archives SHA-256 verified')
                    if args.auto_destroy:
                        cli('destroy', 'instance', instance_id, '-y', '--raw', timeout=90)
                        state_put(name, {'status': 'destroyed', 'finished_at': time.time()})
                        log(f'[DESTROYED] {name}: {instance_id}')
                    else:
                        state_put(name, {'status': 'success_retained'})
                elif out.startswith('FAILED'):
                    state_put(name, {'status': 'failed', 'error': out})
                    log(f'[FAILED] {name}: {out}; instance={instance_id} RETAINED')
            except Exception as exc:
                log(f'[MONITOR RETRY] {name}: {exc}')
        time.sleep(args.poll_seconds)


def resolve_run_script(cold, model):
    """Prefer the cold-specific runner when it exists."""
    if cold == 'protein_scaffold_cold':
        candidate = HERE / f'run_protein_scaffold_{model}.sh'
        return candidate
    return HERE / f'run_{model}.sh'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['kiba', 'metz'], default='metz')
    p.add_argument('--cold-types', default='protein_cold,ligand_cold,scaffold_cold')
    p.add_argument('--models', default='vqatom,continuous,morgan,smiles')
    p.add_argument('--splits', default='1,2,3')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--min-price', type=float, default=0.0)
    p.add_argument('--max-price', type=float, default=1.2)
    p.add_argument('--min-vram', type=float, default=40)
    p.add_argument('--gpu-mode', choices=['blackwell', 'standard'], default='blackwell',
                   help='blackwell: RTX PRO 5000/6000; standard: allowlisted non-Blackwell GPUs')
    p.add_argument('--min-reliability', type=float, default=.97)
    p.add_argument('--max-parallel', type=int, default=25)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--ftp-results-dir', default='dti_results', help='Existing FTP root (dti_results/<dataset>)')
    p.add_argument('--disk', type=int, default=35)
    p.add_argument('--image', default='pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel')
    p.add_argument('--tasks-file', required=False, help='CSV: dataset,cold_type,model,split,seed; required for launches')
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument('--training-start-timeout', type=int, default=1800, help='Seconds from instance creation until positive train progress')
    p.add_argument('--start-check-seconds', type=int, default=20)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--auto-destroy', action='store_true')
    p.add_argument('--monitor-only', action='store_true')
    p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--retry-delay-seconds', type=int, default=600, help='Wait after failed launch before one retry; default 600')
    p.add_argument('--retry-launch-once', action='store_true', help='Retry failed starts once on a different offer')
    p.add_argument('--block-machine-id', action='append', default=[], help='Permanently exclude a physical machine ID (repeatable)')
    args = p.parse_args()
    # Always exclude the confirmed pathological host, even without a CLI flag.
    if '39567' not in args.block_machine_id:
        args.block_machine_id.append('39567')
    if args.block_machine_id:
        with LOCK:
            data = bad_hosts_read()
            for mid in args.block_machine_id:
                if not str(mid).isdigit():
                    p.error(f'Invalid machine ID: {mid}')
                data.setdefault(f'machine_id:{mid}', {
                    'reason': 'Manually excluded: observed severe training slowdown',
                    'at': time.time(),
                })
            temp = BAD_HOSTS.with_suffix('.tmp')
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            temp.replace(BAD_HOSTS)
        log(f'[MANUAL HOST BLOCK] machine IDs: {", ".join(args.block_machine_id)}')
    if args.monitor_only:
        monitor(args)
        return
    if args.epochs < 1 or args.max_parallel < 1 or args.min_price > args.max_price or args.training_start_timeout < 1 or args.start_check_seconds < 1:
        p.error('Invalid concurrency or price range')
    colds = [s.strip() for s in args.cold_types.split(',')]
    models = [s.strip() for s in args.models.split(',')]
    if any(c not in ('protein_cold','ligand_cold','double_cold','scaffold_cold','protein_scaffold_cold') for c in colds):
        p.error('Unknown cold type')
    if any(m not in ('vqatom','vqatom_cb3k','vqatom_cb4k','vqatom_cb5k','vqatom_cb10k','vqatom_cb15k','continuous','morgan','smiles') for m in models):
        p.error('Unknown model')
    for m in models:
        if not all(resolve_run_script(c, m).is_file() for c in colds):
            p.error(f'Missing run script for {m} and selected cold types in {HERE}')
    if not args.tasks_file and not args.monitor_only:
        p.error('--tasks-file is required to avoid accidentally launching a full task grid')
    tasks = []
    with open(args.tasks_file, newline='') as f:
        reader = csv.DictReader(f)
        required = {'dataset', 'cold_type', 'model', 'split', 'seed'}
        if not required.issubset(reader.fieldnames or []):
            p.error(f'Tasks CSV needs columns: {sorted(required)}')
        for row in reader:
            task = (row['dataset'].strip(), row['cold_type'].strip(), row['model'].strip(),
                    int(row['split']), int(row['seed']))
            if task[0] not in ('kiba','metz') or task[1] not in ('protein_cold','ligand_cold','scaffold_cold','double_cold','protein_scaffold_cold') or task[2] not in ('vqatom','vqatom_cb3k','vqatom_cb4k','vqatom_cb5k','vqatom_cb10k','vqatom_cb15k','continuous','morgan','smiles') or task[3] not in (1,2,3) or task[4] not in (0,1,2):
                p.error(f'Invalid task: {task}')
            tasks.append(task)
    if len(tasks) != len(set(tasks)):
        p.error('Duplicate tasks in CSV')
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    # Avoid duplicating active or previously launched jobs. Check Vast.ai account too.
    existing = state_read()
    running_labels = {str(i.get('label') or '') for i in instances()}
    original = len(tasks)
    active_status = {'running', 'created', 'starting', 'manual_required', 'success_retained', 'destroyed'}
    tasks = [t for t in tasks if (name := f'{t[0]}_{t[1]}_{t[2]}_split{t[3]}_seed{t[4]}')
             not in running_labels and existing.get(name, {}).get('status') not in active_status]
    log(f'{len(tasks)} new tasks (skipped {original-len(tasks)} tracked/running); '
             f'concurrency={args.max_parallel}, ${args.min_price:.2f}-${args.max_price:.2f}/GPU-hour')
    for task in tasks:
        log('[PLANNED]', '_'.join(map(str, task)))
    if args.execute and tasks:
        log(f'[IMAGE] {args.image}')
    if not args.execute:
        log('DRY RUN; no rental. Confirm task list and add --execute --auto-destroy for launch.')
        return
    if not sys.stdin.isatty():
        p.error('Interactive confirmation required')
    if input(f'Type RENT {len(tasks)}: ') != f'RENT {len(tasks)}':
        p.error('Cancelled')
    password = os.environ.get('FTP_PASS') or getpass.getpass('FTP password: ')
    if not password:
        p.error('FTP password required')
    excluded = set()
    # Protect existing KIBA and METZ instances: don't rent on their physical hosts.
    for inst in instances():
        key = host_key(inst)
        if key:
            RESERVED_HOSTS.add(key)
    log(f'[HOST RESERVATIONS] {len(RESERVED_HOSTS)} existing physical hosts excluded')
    log(f'[HOST BLACKLIST] {len(bad_hosts_read())} known failed hosts excluded')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        result = list(pool.map(lambda t: launch_with_retry(t, args, password, excluded), tasks))
    log(f'[LAUNCH FINISHED] {sum(result)}/{len(result)} started')
    show_manual_summary(tasks)
    write_failed_csv(tasks)
    if args.auto_destroy:
        monitor(args)


if __name__ == '__main__':
    main()
