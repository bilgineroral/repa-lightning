"""
Download ImageNet-1k (ILSVRC 2012) from the Hugging Face Hub and unpack it into the
ImageFolder layout expected by dataset.py:

    <out>/train/<wnid>/<image>.JPEG    1,281,167 images
    <out>/val/<wnid>/<image>.JPEG         50,000 images

Requires `hf auth login` with an account that accepted the ILSVRC/imagenet-1k terms.
Needs ~2x the dataset size (~310GB) while unpacking; archives are deleted afterwards.

Usage:
    python prepare_imagenet.py --out /path/to/imagenet
"""
import argparse
import os
import shutil
import tarfile
from concurrent.futures import ProcessPoolExecutor

from huggingface_hub import snapshot_download

REPO = "ILSVRC/imagenet-1k"
REVISION = "4603483700ee984ea9debe3ddbfdeae86f6489eb" # tar.gz archives, same revision as the original NEPA code
ARCHIVES = {
    "train": [f"data/train_images_{i}.tar.gz" for i in range(5)],
    "val": ["data/val_images.tar.gz"],
}
NUM_IMAGES = {"train": 1_281_167, "val": 50_000}

def unpack(archive: str, out_dir: str) -> int:
    """ Archive members are named <image>_<wnid>.JPEG; writes them to <out_dir>/<wnid>/<image>.JPEG """
    made, n = set(), 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".JPEG")):
                continue
            image, wnid = os.path.basename(member.name)[:-len(".JPEG")].rsplit("_", 1)
            if wnid not in made:
                os.makedirs(os.path.join(out_dir, wnid), exist_ok=True)
                made.add(wnid)
            with tar.extractfile(member) as src, open(os.path.join(out_dir, wnid, f"{image}.JPEG"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n

def main():
    p = argparse.ArgumentParser(description="Download ImageNet-1k and unpack it into ImageFolder layout")
    p.add_argument("--out", type=str, required=True, help="Output root (will contain train/ and val/)")
    p.add_argument("--keep-archives", action="store_true", help="Keep the downloaded .tar.gz archives")
    args = p.parse_args()

    archive_dir = os.path.join(args.out, "archives")
    files = [f for split in ARCHIVES for f in ARCHIVES[split]]
    print(f"Downloading {len(files)} archives to {archive_dir}", flush=True)
    snapshot_download(
        REPO, repo_type="dataset", revision=REVISION,
        allow_patterns=files, local_dir=archive_dir, max_workers=len(files),
    )

    jobs = [(os.path.join(archive_dir, f), os.path.join(args.out, split)) for split in ARCHIVES for f in ARCHIVES[split]]
    print(f"Unpacking {len(jobs)} archives", flush=True)
    with ProcessPoolExecutor(len(jobs)) as ex:
        counts = list(ex.map(unpack, *zip(*jobs)))

    for split in ARCHIVES:
        split_dir = os.path.join(args.out, split)
        n = sum(c for (_, out_dir), c in zip(jobs, counts) if out_dir == split_dir)
        num_classes = len(os.listdir(split_dir))
        print(f"{split}: {n} images, {num_classes} classes", flush=True)
        if n != NUM_IMAGES[split] or num_classes != 1000:
            raise RuntimeError(f"{split}: expected {NUM_IMAGES[split]} images in 1000 classes")

    if not args.keep_archives:
        shutil.rmtree(archive_dir)

if __name__ == "__main__":
    main()
