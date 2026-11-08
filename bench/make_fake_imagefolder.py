"""
Fake ImageNet-layout JPEG folder for testing the real-data path (decode, crop, H2D) without ImageNet.
Image content is random (smooth color fields plus noise), so decode cost is similar but nothing is learnable.

    python bench/make_fake_imagefolder.py --root ~/data/repa_fake --train 8000 --val 200
"""
import argparse
import os

import numpy as np
from PIL import Image

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--train", type=int, default=8000)
    p.add_argument("--val", type=int, default=200)
    p.add_argument("--classes", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, n in (("train", args.train), ("val", args.val)):
        for i in range(n):
            d = os.path.join(args.root, split, f"n{i % args.classes:08d}")
            os.makedirs(d, exist_ok=True)
            h, w = int(rng.integers(300, 500)), int(rng.integers(350, 500)) # around ImageNet's typical sizes
            small = rng.integers(0, 256, (h // 32 + 1, w // 32 + 1, 3)).astype(np.uint8)
            img = np.asarray(Image.fromarray(small).resize((w, h), Image.BILINEAR), dtype=np.int16)
            img = np.clip(img + rng.integers(-20, 21, img.shape), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(d, f"{split}_{i:06d}.JPEG"), quality=90)
    print(f"wrote {args.train} train and {args.val} val JPEGs to {args.root}")

if __name__ == "__main__":
    main()
