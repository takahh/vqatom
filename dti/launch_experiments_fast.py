#!/usr/bin/env python3
"""Launch one Vast.ai SSH instance per DTI task. Dry-run by default."""
import argparse
import concurrent.futures
import getpass
import json
import os
from pathlib import Path
import re
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


def wait_ready(instance_id, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        data = cli('show', 'instance', instance_id, '--raw')
        if isinstance(data, list):
            data = data[0] if data else {}
        state = data.get('actual_status')
        if state == 'running':
            try:
                connection = ssh_args(instance_id)
                test = subprocess.run(ssh_cmd(connection, 'true'), capture_output=True, timeout=30)
                if test.returncode == 0:
                    return connection
            except (RuntimeError, subprocess.TimeoutExpired):
                pass
        if state in ('exited', 'offline', 'unknown', 'stopped'):
            raise RuntimeError(f'Instance {instance_id} state: {state}')
        time.sleep(15)
    raise TimeoutError(f'Instance {instance_id} not SSH-ready in {seconds}s')


def run_task(task, args, password, excluded):
    dataset, cold, model, split, seed = task
    name = f'{dataset}_{cold}_{model}_split{split}_seed{seed}'
    instance_id = None
    try:
        with PRINT_LOCK:
            offer = find_offer(args, excluded)
            excluded.add(str(offer['id']))
        log(f'[OFFER] {name}: offer={offer["id"]}, GPU={offer.get("gpu_name")}, dlperf={offer.get("dlperf", "n/a")}, $/hr={offer_price(offer):.3f}')
        if not args.execute:
            return True
        response = cli('create', 'instance', offer['id'], '--image', args.image,
                       '--disk', args.disk, '--ssh', '--direct', '--label', name,
                       '--cancel-unavail', '--raw', timeout=120)
        if not response.get('success') or not response.get('new_contract'):
            raise RuntimeError(f'Instance creation failed: {response}')
        instance_id = int(response['new_contract'])
        log(f'[CREATED] {name}: instance={instance_id}')
        connection = wait_ready(instance_id, args.startup_timeout)
        script = HERE / f'run_{model}.sh'
        upload = subprocess.run(ssh_cmd(connection, 'cat > /tmp/dti_run.sh && chmod 700 /tmp/dti_run.sh'),
                                input=script.read_bytes(), capture_output=True, timeout=60)
        if upload.returncode:
            raise RuntimeError(f'Script upload failed: {upload.stderr.decode(errors="replace")}')
        # Secret travels only through SSH stdin, not command-line arguments or disk files.
        wrapper = (f'export FTP_PASS={shlex.quote(password)}\n'
                   f'export DATASET={shlex.quote(dataset)} COLD_TYPE={shlex.quote(cold)} '
                   f'SPLIT={split} SEED={seed}\n'
                   'set -o pipefail\n'
                   'if ! command -v python >/dev/null 2>&1; then\n'
                   '  PY3=$(command -v python3) || { echo "ERROR: python3 not found" >&2; exit 127; }\n'
                   '  mkdir -p /usr/local/bin\n'
                   '  ln -s "$PY3" /usr/local/bin/python\n'
                   'fi\n'
                   'python --version\n'
                   'bash /tmp/dti_run.sh 2>&1 | tee /tmp/dti_run.log\n'
                   'exit ${PIPESTATUS[0]}\n')
        log(f'[START] {name}: instance={instance_id}')
        logs = HERE / 'experiment_logs'
        logs.mkdir(exist_ok=True)
        with (logs / f'{name}.log').open('w') as f:
            proc = subprocess.Popen(ssh_cmd(connection, 'bash -s'), stdin=subprocess.PIPE,
                                    stdout=f, stderr=subprocess.STDOUT, text=True)
            try:
                proc.communicate(wrapper, timeout=args.task_timeout_hours * 3600)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()
                raise RuntimeError('Task timed out; instance preserved for investigation')
        if proc.returncode:
            raise RuntimeError(f'SSH/training exited {proc.returncode}; inspect experiment_logs/{name}.log')
        log(f'[SUCCESS] {name}: FTP upload command completed')
        if args.auto_destroy:
            cli('destroy', 'instance', instance_id, '-y', '--raw', timeout=90)
            log(f'[DESTROYED] {name}: instance={instance_id}')
        else:
            log(f'[PRESERVED] {name}: instance={instance_id} (auto-destroy disabled)')
        return True
    except Exception as exc:
        log(f'[FAILED] {name}: {exc}' + (f' | instance={instance_id} retained' if instance_id else ''))
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['kiba', 'metz'], default='metz')
    p.add_argument('--cold-type', choices=['protein_cold', 'ligand_cold', 'double_cold', 'scaffold_cold'], default=None, help='Single cold type (overrides --cold-types)')
    p.add_argument('--cold-types', default='protein_cold,ligand_cold,scaffold_cold', help='Comma-separated cold types')
    p.add_argument('--models', default='vqatom,continuous,morgan,smiles')
    p.add_argument('--splits', default='1,2,3')
    p.add_argument('--seeds', default='0,1,2')
    p.add_argument('--min-price', type=float, default=0.80, help='Minimum GPU hourly price in USD (default 0.80)')
    p.add_argument('--max-price', type=float, default=1.50)
    p.add_argument('--min-vram', type=float, default=40, help='Minimum GPU VRAM in GB (default 40)')
    p.add_argument('--min-reliability', type=float, default=0.97)
    p.add_argument('--max-parallel', type=int, default=108)
    p.add_argument('--max-tasks', type=int, default=None, help='Limit total tasks (e.g. 5 for initial test)')
    p.add_argument('--disk', type=int, default=35)
    p.add_argument('--image', default='vastai/pytorch:@vastai-automatic-tag')
    p.add_argument('--startup-timeout', type=int, default=1200)
    p.add_argument('--task-timeout-hours', type=float, default=24)
    p.add_argument('--execute', action='store_true', help='Actually rent GPUs (default is dry-run)')
    p.add_argument('--auto-destroy', action='store_true', help='Destroy instance only after successful training and FTP upload')
    args = p.parse_args()
    cold_types = [args.cold_type] if args.cold_type else [s.strip() for s in args.cold_types.split(',')]
    if not cold_types or any(c not in ('protein_cold', 'ligand_cold', 'double_cold', 'scaffold_cold') for c in cold_types):
        p.error('Unknown cold type')
    models = [s.strip() for s in args.models.split(',')]
    if not models or any(m not in ('vqatom', 'vqatom_cb5k', 'continuous', 'morgan', 'smiles') for m in models):
        p.error('Unknown model')
    splits = [int(s) for s in args.splits.split(',')]
    seeds = [int(s) for s in args.seeds.split(',')]
    if args.max_parallel < 1 or args.min_price < 0 or args.max_price < args.min_price or args.disk < 10 or (args.max_tasks is not None and args.max_tasks < 1):
        p.error('Invalid parallelism, max-price, or disk')
    for model in models:
        if not (HERE / f'run_{model}.sh').is_file():
            p.error(f'Missing {HERE / f"run_{model}.sh"}')
    tasks = [(args.dataset, cold, m, sp, sd) for cold in cold_types for m in models for sp in splits for sd in seeds]
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    log(f'{len(tasks)} tasks, max parallel={args.max_parallel}, GPU price range ${args.min_price:.2f}-${args.max_price:.2f}/hr, '
        f'max concurrent GPU rental ${args.max_price * args.max_parallel:.2f}/hr (excluding storage/fees)')
    log(f'GPU selection: VRAM >= {args.min_vram:g} GB; lowest price first (then dlperf)')
    log('MODE:', 'EXECUTE' if args.execute else 'DRY-RUN (no instances created)')
    if args.execute:
        if not sys.stdin.isatty():
            p.error('--execute requires interactive terminal for explicit confirmation')
        answer = input(f'Type RENT {len(tasks)} to authorize paid rentals: ')
        if answer != f'RENT {len(tasks)}':
            p.error('Cancelled')
        password = os.environ.get('FTP_PASS') or getpass.getpass('FTP password: ')
        if not password:
            p.error('FTP password required')
    else:
        password = ''
    excluded = set()
    # Search/create tasks are serialized inside the lock to avoid reusing an offer.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        results = list(pool.map(lambda t: run_task(t, args, password, excluded), tasks))
    log(f'Finished: {sum(results)}/{len(tasks)} tasks successful')
    if not all(results):
        sys.exit(1)


if __name__ == '__main__':
    main()

