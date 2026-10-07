"""
Download ADE20K (the SceneParsing / ADEChallengeData2016 release used by mmsegmentation) and unpack it into
the layout expected by ade20k.py:

    <out>/images/training/*.jpg           20,210 images
    <out>/images/validation/*.jpg          2,000 images
    <out>/annotations/training/*.png      (0 = other, 1..150 = classes)
    <out>/annotations/validation/*.png

Usage:
    python prepare_ade20k.py --out /path/to/ade20k
"""
import argparse
import os
import shutil
import urllib.request
import zipfile

URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
NUM_IMAGES = {"training": 20_210, "validation": 2_000}

def main():
    p = argparse.ArgumentParser(description="Download ADE20K and unpack it")
    p.add_argument("--out", type=str, required=True, help="Output root (will contain images/ and annotations/)")
    p.add_argument("--keep-archive", action="store_true", help="Keep the downloaded .zip archive")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    archive = os.path.join(args.out, "ADEChallengeData2016.zip")
    if not os.path.exists(archive):
        print(f"Downloading {URL}", flush=True)
        urllib.request.urlretrieve(URL, archive + ".part")
        os.rename(archive + ".part", archive)

    print("Unpacking", flush=True)
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            parts = member.filename.split("/")
            if member.is_dir() or len(parts) != 4 or parts[1] not in ("images", "annotations"):
                continue # ADEChallengeData2016/{images,annotations}/{training,validation}/<file>
            dst = os.path.join(args.out, *parts[1:])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(member) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)

    for split, n in NUM_IMAGES.items():
        images = sorted(f[:-4] for f in os.listdir(os.path.join(args.out, "images", split)) if f.endswith(".jpg"))
        labels = sorted(f[:-4] for f in os.listdir(os.path.join(args.out, "annotations", split)) if f.endswith(".png"))
        print(f"{split}: {len(images)} images, {len(labels)} annotations", flush=True)
        if len(images) != n or images != labels:
            raise RuntimeError(f"{split}: expected {n} images with matching annotations")

    if not args.keep_archive:
        os.remove(archive)

if __name__ == "__main__":
    main()
