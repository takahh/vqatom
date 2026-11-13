"""Run a single new-condition task from an immutable matched_v2 bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

METHODS=('morgan','continuous','smiles','vq_cb1000','vq_cb5000','vq_cb10000','vq_cb15000')
EPOCHS={m:50 if m=='continuous' else 10 if m=='smiles' else 20 for m in METHODS}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def command(bundle,task,out_root):
    bundle=Path(bundle).resolve()
    options=dict(attn_activation='sigmoid',attn_temp=.7,batch_size=8,cls_lambda=.05,
        contact_lambda=0,cross_lr=1e-4,dropout=.3,epochs=task['epochs'],
        final_eval_csv=bundle/task['csv']['test'],freeze_esm_bottom=28,lig_lr=1e-5,
        ligand_mode=task['ligand_mode'],ligand_token_dropout=.1,llrd_decay=.97,n_heads=8,
        out_dir=Path(out_root).resolve()/task['name'],pl_lp_overlap='both',protein_lr=5e-6,
        protein_token_dropout=.1,rank_lambda=.05,rank_loss='pairwise',rank_margin=0,
        reg_lambda=1,seed=task['seed'],select_on='auroc',train_csv=bundle/task['csv']['train'],
        valid_csv=bundle/task['csv']['valid'],weight_decay=.05,y_thr=task['y_thr'],
        warmup_ratio=.05,atom_feat_dim=10)
    if task['method']=='morgan':options.update(morgan_vocab_path=bundle/task['morgan_vocab'],morgan_encoder='transformer')
    elif task['method']=='smiles':options['smiles_vocab_path']=bundle/task['smiles_vocab']
    elif task['method']!='continuous':options['vq_ckpt']=bundle/task['checkpoint']
    args=[sys.executable,str(bundle/'dti/dti.py')]
    for key,value in options.items():args.extend(['--'+key,str(value)])
    return args+['--finetune_esm','--finetune_lig','--llrd','--use_train_valid_csv','--matched_ligand_transformer']


def verify(bundle,manifest,task):
    bundle=Path(bundle).resolve()
    required=['dti/dti.py','dti/model.py','dti/matched_ligand_encoder.py','run_matched_v2.py',*task['csv'].values()]
    for key in ('morgan_vocab','smiles_vocab','checkpoint'):
        if task.get(key):required.append(task[key])
    for relative in required:
        path=(bundle/relative).resolve()
        if not path.is_relative_to(bundle) or digest(path)!=manifest['sha256'][relative]:raise ValueError('Bundle file hash mismatch: '+relative)
    if task['epochs']!=EPOCHS[task['method']]:raise ValueError('Epoch policy mismatch')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path);p.add_argument('--dataset',required=True,choices=['kiba','metz'])
    p.add_argument('--cold-type',required=True,choices=['protein_cold','scaffold_cold','ligand_cold'])
    p.add_argument('--method',required=True,choices=METHODS);p.add_argument('--split',type=int,required=True,choices=[1,2,3])
    p.add_argument('--seed',type=int,required=True,choices=[0,1,2]);p.add_argument('--epochs',required=True,type=int)
    p.add_argument('--out-root',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
    manifest=json.loads((args.bundle/'manifest.json').read_text())
    matches=[t for t in manifest['tasks'] if (t['dataset'],t['cold_type'],t['method'],t['split'],t['seed'])==
             (args.dataset,args.cold_type,args.method,args.split,args.seed)]
    if len(matches)!=1:raise ValueError('Task missing or ambiguous')
    task=matches[0]
    if task['epochs']!=args.epochs:raise ValueError('Epochs differ from new-condition manifest')
    verify(args.bundle,manifest,task);cmd=command(args.bundle,task,args.out_root)
    print(shlex.join(cmd),flush=True)
    if args.execute:
        output=args.out_root/task['name']
        if output.exists() and any(output.iterdir()):raise FileExistsError('Existing task output')
        subprocess.run(cmd,cwd=args.bundle/'dti',check=True)


if __name__=='__main__':main()
