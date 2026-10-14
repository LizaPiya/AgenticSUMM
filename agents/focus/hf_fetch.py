"""
Download one Hugging Face model repo with a disk guard (weights, config and tokenizer only; the repo's duplicate
`original/` folder is skipped). Usage: REPO=<hf id> MIN_LEFT_GB=25 python hf_fetch.py
Refuses to start if fewer than MIN_LEFT_GB would remain free on the disk that holds the cache.
"""
import fnmatch
import os
import shutil
import sys

from huggingface_hub import HfApi, snapshot_download

repo = os.environ["REPO"]
keep = int(os.environ.get("MIN_LEFT_GB", "25"))
token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
allow = ["*.safetensors", "*.json", "tokenizer*", "*.model", "*.txt"]
info = HfApi().model_info(repo, files_metadata=True, token=token)
files = [s for s in info.siblings if "/" not in s.rfilename and any(fnmatch.fnmatch(s.rfilename, p) for p in allow)]
need = sum((s.size or 0) for s in files) / 1e9
home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
free = shutil.disk_usage(home).free / 1e9
print(f"{repo}: {len(files)} files, {need:.1f} GB to download; {free:.1f} GB free on the disk", flush=True)
if free - need < keep:
    sys.exit(f"REFUSING: only {free - need:.1f} GB would be left (need at least {keep} GB)")
path = snapshot_download(repo, allow_patterns=allow, token=token)
print("downloaded to", path, flush=True)
print(f"{shutil.disk_usage(home).free / 1e9:.1f} GB free after the download", flush=True)
