VQAtom quantization launcher prototype (2026-11-02)

Place launch_quantization.py and dti_launcher_base.py in ~/PycharmProjects/vqatom/dti.
The second file is a copy of the supplied DTI launcher used for Vast.ai helper functions;
its main() is NOT invoked by the quantization launcher.

First: python launch_quantization.py --help
Then:  python launch_quantization.py --codebook-sizes 1000 5000 10000

The dry-run prints the task plan and does not rent machines. It does not query offers.

IMPORTANT: This is a prototype, NOT validated against a live Vast.ai machine.
Before using --execute:
- Verify local ../quantize/vqatom_train.py and ../data/molecules.npz paths, or pass --source-dir and --molecules-file.
- Confirm that the Google Drive data ID is correct and publicly downloadable.
- Check that the selected image can install DGL and all vqatom_train.py dependencies; initial CUDA/DGL preflight may fail.
- Confirm Vast.ai offer API and source staging against your current environment.
- The user must supply FTP_PASS via environment or secure prompt.
- For safety, do not use --auto-destroy until an end-to-end successful test and checksum verification.
- Jobs that fail are retained and continue to incur rental charges until manually stopped/destroyed.
- Monitoring stops if the Mac is offline; restart with --monitor-only.

Execute (rents GPUs, after interactive confirmation):
python launch_quantization.py --codebook-sizes 1000 --execute
Monitor:
python launch_quantization.py --monitor-only
After validating successful upload/verification and cleanup:
python launch_quantization.py --monitor-only --auto-destroy

The launcher uses the DTI helper's existing failed_vast_hosts.json blacklist,
but quantization_jobs.json is separate from detached_jobs.json.
