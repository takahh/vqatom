#!/usr/bin/env python3
"""Vast.ai detached DTI launcher. Requires launch_experiments_fast.py in same folder.

Remote training survives SSH disconnection. Mac monitor deletes ONLY after
training exits 0 AND uploaded archives are read back and SHA-256 verified.
Use --monitor-only to resume after Mac restarts.
"""
import argparse
import concurrent.futures
import getpass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time

import launch_experiments_fast as base

HERE = Path(__file__).resolve().parent
STATE = HERE / 'detached_jobs.json'
BAD_HOSTS = HERE / 'failed_vast_hosts.json'
LOCK = threading.Lock()
# Reserved physical hosts selected within this launch only.
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
        base.log(f'[HOST UNKNOWN] offer={offer.get("id")}; cannot blacklist physical host')
        return
    with LOCK:
        data = bad_hosts_read()
        data[key] = {'offer_id': offer.get('id'), 'reason': str(reason), 'at': time.time()}
        temp = BAD_HOSTS.with_suffix('.tmp')
        temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        temp.replace(BAD_HOSTS)
    base.log(f'[HOST BLACKLISTED] {key}: {reason}')


def choose_offer(args, excluded):
    # Physical host uniqueness among new tasks in this launch only.
    # Reject offers without a stable host ID rather than silently bypassing safety.
    blacklist = bad_hosts_read()
    for _ in range(100):
        offer = base.find_offer(args, excluded)
        excluded.add(str(offer['id']))
        key = host_key(offer)
        if not key:
            base.log(f'[SKIP UNKNOWN HOST] offer={offer.get("id")}')
            continue
        if key in blacklist or key in RESERVED_HOSTS:
            continue
        if not offer.get('verified'):
            continue
        if float(offer.get('reliability') or 0) < args.min_reliability:
            continue
        if int(offer.get('num_gpus') or 0) != 1:
            continue
        RESERVED_HOSTS.add(key)
        return offer
    raise RuntimeError('No eligible offers with a distinct, verified, reliable physical host')


def instances():
    data = base.cli('show', 'instances', '--raw')
    if isinstance(data, dict):
        data = data.get('instances', data.get('results', []))
    return data if isinstance(data, list) else []


def remote_check(instance_id, command, timeout=45):
    conn = base.ssh_args(instance_id)
    p = subprocess.run(base.ssh_cmd(conn, command), text=True, capture_output=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def launch(task, args, password, excluded, attempt=1):
    dataset, cold, model, split, seed = task
    name = f'{dataset}_{cold}_{model}_split{split}_seed{seed}'
    instance_id = None
    offer = None
    try:
        with LOCK:
            offer = choose_offer(args, excluded)
        base.log(f'[OFFER] {name}: {offer.get("gpu_name")}, ${base.offer_price(offer):.3f}/h')
        if not args.execute:
            return True
        res = base.cli('create', 'instance', offer['id'], '--image', args.image,
                       '--disk', args.disk, '--ssh', '--direct', '--label', name,
                       '--cancel-unavail', '--raw', timeout=120)
        if not res.get('success') or not res.get('new_contract'):
            raise RuntimeError(f'create failed: {res}')
        instance_id = int(res['new_contract'])
        state_put(name, {'id': instance_id, 'status': 'created', 'created_at': time.time(), 'attempt': attempt, 'offer_id': offer.get('id'), 'host_key': host_key(offer)})
        base.log(f'[CREATED] {name}: {instance_id}')
        conn = base.wait_ready(instance_id, args.startup_timeout)
        script = HERE / f'run_{model}.sh'
        up = subprocess.run(base.ssh_cmd(conn, 'umask 077; cat > /tmp/dti_run.sh; chmod 700 /tmp/dti_run.sh'),
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
            f'export EPOCHS={args.epochs}',
            f'export FTP_RESULTS_DIR={shlex.quote(args.ftp_results_dir)}',
            'if ! command -v python >/dev/null 2>&1; then',
            '  PY3=$(command -v python3) || exit 127',
            '  ln -sf "$PY3" /usr/local/bin/python', 'fi',
            'python --version',
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
        up = subprocess.run(base.ssh_cmd(conn, 'umask 077; cat > /tmp/dti_runner.sh; chmod 700 /tmp/dti_runner.sh'),
                            input=runner, text=True, capture_output=True, timeout=60)
        if up.returncode:
            raise RuntimeError(f'runner upload: {up.stderr}')
        # setsid+nohup + stdin /dev/null fully detach from SSH session.
        rc, out, err = remote_check(instance_id,
            'nohup setsid bash /tmp/dti_runner.sh </dev/null >/tmp/dti_run.log 2>&1 & echo STARTED', 45)
        if rc or 'STARTED' not in out:
            raise RuntimeError(f'detached start failed: {out} {err}')
        state_put(name, {'status': 'running', 'started_at': time.time(), 'attempt': attempt})
        base.log(f'[DETACHED] {name}: instance={instance_id}')
        return True
    except Exception as exc:
        # If rental was not created, free its physical-host reservation.
        if offer is not None and instance_id is None:
            with LOCK:
                RESERVED_HOSTS.discard(host_key(offer))
        # Blacklist host only for infrastructure/SSH startup failures, not training or FTP failures.
        if offer is not None and instance_id is not None and (
            'not SSH-ready' in str(exc) or 'script upload:' in str(exc) or
            'runner upload:' in str(exc) or 'detached start failed:' in str(exc)
        ):
            bad_host_record(offer, exc)
        base.log(f'[FAILED TO LAUNCH] {name}: {exc}; instance={instance_id} retained')
        state_put(name, {'status': 'launch_failed', 'error': str(exc),
                         'attempt': attempt, 'failed_at': time.time(),
                         'id': instance_id, 'offer_id': offer.get('id') if offer else None,
                         'host_key': host_key(offer) if offer else None})
        return False


def retry_failed_launches(tasks, args, password, excluded):
    """Retry each launch failure at most once after the configured delay.

    A machine with an ambiguous/known instance is NEVER blindly replaced:
    first reconcile its label, then destroy only an instance that has not
    confirmed a detached start. If cleanup cannot be verified, leave it for
    manual inspection and do not rent a second machine.
    """
    names = {f'{d}_{c}_{m}_split{sp}_seed{sd}': (d,c,m,sp,sd)
             for d,c,m,sp,sd in tasks}
    failed = [(name, job) for name, job in state_read().items()
              if name in names and job.get('status') == 'launch_failed'
              and job.get('attempt') == 1]
    if not failed:
        return
    base.log(f'[RETRY QUEUED] {len(failed)} launch failures; once after {args.retry_delay_seconds}s')
    # All retries are in one wave. Sleep only until the latest needed due time.
    due = max(float(job.get('failed_at', time.time())) + args.retry_delay_seconds
              for _, job in failed)
    while time.time() < due:
        time.sleep(min(10, max(0.1, due-time.time())))

    def retry_one(name):
        job = state_read().get(name, {})
        if job.get('status') != 'launch_failed' or job.get('attempt') != 1:
            return
        iid = job.get('id')
        try:
            matches = [i for i in instances() if i.get('label') == name]
            ids = {int(i['id']) for i in matches if i.get('id') is not None}
            if iid is not None:
                ids.add(int(iid))
            # A response may have been lost despite successful creation.
            # Never delete any instance if there is ambiguity or duplicates.
            if len(ids) > 1:
                raise RuntimeError(f'multiple instances for label: {sorted(ids)}')
            if ids:
                old_id = next(iter(ids))
                if not matches:
                    # Old instance no longer appears in the account listing.
                    base.log(f'[RETRY CLEAN] {name}: old {old_id} already absent')
                else:
                    # launch() only sets launch_failed before DETACHED confirmation.
                    # An ambiguous detached-start SSH response must be checked.
                    try:
                        rc, out, _ = remote_check(old_id,
                            'if test -e /tmp/dti_success || test -e /tmp/dti_failed || '
                            'pgrep -f "[d]ti.py|[d]ti_runner.sh" >/dev/null; then echo STARTED; '
                            'else echo NOT_STARTED; fi', timeout=35)
                    except Exception:
                        rc, out = 255, ''
                    if rc == 0 and out.strip() == 'STARTED':
                        raise RuntimeError(f'old instance {old_id} may be training; not destroying')
                    # If SSH is unavailable, no detached start was confirmed,
                    # but destroying a machine can still lose work. Require
                    # explicit operator intervention rather than assume idle.
                    if rc != 0:
                        # Safe to replace only if startup never reached a remote
                        # script/runner stage (SSH readiness timed out or create
                        # returned an ambiguous/empty response).
                        reason = str(job.get('error', ''))
                        safe_prestart = ('not SSH-ready' in reason or
                                         'Expected JSON from vastai' in reason or
                                         'create failed:' in reason)
                        if not safe_prestart:
                            raise RuntimeError(f'old instance {old_id} SSH inaccessible; manual cleanup required')
                        base.log(f'[RETRY CLEAN] {name}: pre-start instance {old_id} never SSH-ready')
                    base.cli('destroy', 'instance', old_id, '-y', '--raw', timeout=90)
                    if any(int(i.get('id', -1)) == old_id for i in instances()):
                        raise RuntimeError(f'old instance {old_id} still listed after destroy')
                    base.log(f'[RETRY CLEAN] {name}: destroyed unused instance {old_id}')
            state_put(name, {'status': 'retrying', 'attempt': 2, 'id': None})
            base.log(f'[RETRY 2/2] {name}: selecting a different offer')
            launch(names[name], args, password, excluded, attempt=2)
        except Exception as exc:
            state_put(name, {'status': 'manual_required', 'error': str(exc)})
            base.log(f'[MANUAL REQUIRED] {name}: {exc}')

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        list(pool.map(retry_one, [name for name, _ in failed]))


def show_manual_summary(tasks):
    names = {f'{d}_{c}_{m}_split{sp}_seed{sd}' for d,c,m,sp,sd in tasks}
    pending = [(n, j) for n, j in state_read().items()
               if n in names and j.get('status') in ('launch_failed', 'manual_required', 'created')]
    if pending:
        base.log(f'[MANUAL LAUNCH LIST] {len(pending)} task(s):')
        for n, j in pending:
            base.log(f'  {n} | instance={j.get("id")} | status={j.get("status")} | {j.get("error", "")}')
    else:
        base.log('[MANUAL LAUNCH LIST] none')


def monitor(args):
    base.log('[MONITOR] checking remote success markers; Ctrl+C stops monitoring, not training')
    while True:
        jobs = state_read()
        active = [(name, job) for name, job in jobs.items() if job.get('status') in ('running', 'created')]
        if not active:
            base.log('[MONITOR] no active tracked jobs')
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
                    base.log(f'[CHECK RETRY] {name}: SSH exit={rc}')
                    continue
                if out.startswith('SUCCESS'):
                    base.log(f'[SUCCESS] {name}: training exited 0 and FTP archives SHA-256 verified')
                    if args.auto_destroy:
                        base.cli('destroy', 'instance', instance_id, '-y', '--raw', timeout=90)
                        state_put(name, {'status': 'destroyed', 'finished_at': time.time()})
                        base.log(f'[DESTROYED] {name}: {instance_id}')
                    else:
                        state_put(name, {'status': 'success_retained'})
                elif out.startswith('FAILED'):
                    state_put(name, {'status': 'failed', 'error': out})
                    base.log(f'[FAILED] {name}: {out}; instance={instance_id} RETAINED')
            except Exception as exc:
                base.log(f'[MONITOR RETRY] {name}: {exc}')
        time.sleep(args.poll_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['kiba', 'metz'], default='metz')
    p.add_argument('--cold-types', default='protein_cold,ligand_cold,scaffold_cold')
    p.add_argument('--models', default='vqatom,continuous,morgan,smiles')
    p.add_argument('--splits', default='1,2,3')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--min-price', type=float, default=.8)
    p.add_argument('--max-price', type=float, default=1.5)
    p.add_argument('--min-vram', type=float, default=40)
    p.add_argument('--min-reliability', type=float, default=.97)
    p.add_argument('--max-parallel', type=int, default=25)
    p.add_argument('--max-tasks', type=int)
    p.add_argument('--epochs', type=int, default=1, help='Epochs for each job; default 1 for pilot')
    p.add_argument('--ftp-results-dir', default='dti_results', help='Existing FTP root (dti_results/<dataset>)')
    p.add_argument('--disk', type=int, default=35)
    p.add_argument('--image', default='vastai/pytorch:@vastai-automatic-tag')
    p.add_argument('--startup-timeout', type=int, default=1200)
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
                data[f'machine_id:{mid}'] = {
                    'reason': 'Manually excluded: observed severe training slowdown',
                    'at': time.time(),
                }
            temp = BAD_HOSTS.with_suffix('.tmp')
            temp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            temp.replace(BAD_HOSTS)
        base.log(f'[MANUAL HOST BLOCK] machine IDs: {", ".join(args.block_machine_id)}')
    if args.monitor_only:
        monitor(args)
        return
    if args.epochs < 1 or args.max_parallel < 1 or args.min_price > args.max_price:
        p.error('Invalid concurrency or price range')
    colds = [s.strip() for s in args.cold_types.split(',')]
    models = [s.strip() for s in args.models.split(',')]
    if any(c not in ('protein_cold','ligand_cold','double_cold','scaffold_cold') for c in colds):
        p.error('Unknown cold type')
    if any(m not in ('vqatom','continuous','morgan','smiles') for m in models):
        p.error('Unknown model')
    for m in models:
        if not (HERE / f'run_{m}.sh').is_file():
            p.error(f'Missing run_{m}.sh in {HERE}')
    tasks = [(args.dataset, c, m, int(sp), int(sd))
             for c in colds for m in models for sp in args.splits.split(',') for sd in args.seeds.split(',')]
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    # Avoid duplicating active or previously launched jobs. Check Vast.ai account too.
    existing = state_read()
    running_labels = {str(i.get('label') or '') for i in instances()}
    original = len(tasks)
    tasks = [t for t in tasks if (name := f'{t[0]}_{t[1]}_{t[2]}_split{t[3]}_seed{t[4]}')
             not in existing and name not in running_labels]
    base.log(f'{len(tasks)} new tasks (skipped {original-len(tasks)} tracked/running); '
             f'concurrency={args.max_parallel}, ${args.min_price:.2f}-${args.max_price:.2f}/GPU-hour')
    if not args.execute:
        base.log('DRY RUN; no rental. Add --execute --auto-destroy for launch.')
        return
    if not sys.stdin.isatty():
        p.error('Interactive confirmation required')
    if input(f'Type RENT {len(tasks)}: ') != f'RENT {len(tasks)}':
        p.error('Cancelled')
    password = os.environ.get('FTP_PASS') or getpass.getpass('FTP password: ')
    if not password:
        p.error('FTP password required')
    excluded = set()
    # Existing KIBA and METZ hosts are eligible. Only hosts selected in this
    # launch are reserved, and blacklisted hosts remain excluded.
    base.log('[HOST RESERVATIONS] existing hosts allowed; distinct hosts within batch')
    base.log(f'[HOST BLACKLIST] {len(bad_hosts_read())} known failed hosts excluded')
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        result = list(pool.map(lambda t: launch(t, args, password, excluded), tasks))
    base.log(f'[LAUNCH FINISHED] {sum(result)}/{len(result)} started')
    if args.retry_launch_once:
        retry_failed_launches(tasks, args, password, excluded)
    show_manual_summary(tasks)
    if args.auto_destroy:
        monitor(args)


if __name__ == '__main__':
    main()
