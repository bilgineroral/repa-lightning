"""
ADE20K data for UPerNet finetuning: mmsegmentation's ADE20K pipelines (configs/_base_/datasets/ade20k.py with
SegDataPreProcessor), ported with the same OpenCV / numpy operations so that a sample matches mmsegmentation's
bit for bit given the same random numbers.

Training: LoadImageFromFile (cv2, BGR) -> LoadAnnotations(reduce_zero_label) -> RandomResize(img_scale, ratio_range)
    -> RandomCrop(crop_size, cat_max_ratio) -> RandomFlip(0.5) -> PhotoMetricDistortion -> BGR to RGB, normalize,
    pad to crop_size (image 0, label 255)
Validation: LoadImageFromFile -> Resize(img_scale, keep_ratio) -> BGR to RGB, normalize; labels at original size

Augmentations draw from a RandomState seeded by (seed, round, index), in the same order as mmsegmentation's calls to
the global numpy RNG, so every sample is reproducible regardless of rank, dataloader worker or resume.
"""
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
import pytorch_lightning as pl

from config import SegmentationConfig
from dataset import mix_seed

cv2.setNumThreads(0) # as mmsegmentation (opencv_num_threads=0): dataloader workers parallelize instead

IGNORE_INDEX = 255

# class names of the labels 0..149 (mmseg ADE20KDataset.METAINFO, whose "bed " has a stray space)
CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed", "windowpane", "grass", "cabinet", "sidewalk",
    "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water", "painting", "sofa",
    "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard", "chest of drawers", "counter", "sand",
    "sink", "skyscraper", "fireplace", "refrigerator", "grandstand", "path", "stairs", "runway", "case", "pool table",
    "pillow", "screen door", "stairway", "river", "bridge", "bookcase", "blind", "coffee table", "toilet", "flower",
    "book", "hill", "bench", "countertop", "stove", "palm", "kitchen island", "computer", "swivel chair", "boat", "bar",
    "arcade machine", "hovel", "bus", "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight",
    "booth", "television receiver", "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator",
    "ottoman", "bottle", "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer",
    "plaything", "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier",
    "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag",
)

def load_image(path: str) -> np.ndarray:
    """ mmcv LoadImageFromFile: cv2 decode, BGR uint8 """
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

def load_label(path: str) -> np.ndarray:
    """ mmseg LoadAnnotations(reduce_zero_label=True), pillow backend: 0 (other) -> 255, 1..150 -> 0..149 """
    label = np.array(Image.open(path)).squeeze().astype(np.uint8)
    label[label == 0] = 255
    label = label - 1
    label[label == 254] = 255
    return label

def imrescale(img: np.ndarray, scale: tuple, interpolation: int) -> np.ndarray:
    """ mmcv.imrescale: largest size that fits (long, short) = (max(scale), min(scale)), keeping the aspect ratio """
    h, w = img.shape[:2]
    scale_factor = min(max(scale) / max(h, w), min(scale) / min(h, w))
    size = int(w * float(scale_factor) + 0.5), int(h * float(scale_factor) + 0.5)
    return cv2.resize(img, size, interpolation=interpolation)

class PhotoMetricDistortion:
    """ mmseg PhotoMetricDistortion on BGR uint8 images (brightness, contrast, saturation, hue) """
    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    @staticmethod
    def convert(img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img, rs):
        if rs.randint(2):
            return self.convert(img, beta=rs.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img, rs):
        if rs.randint(2):
            return self.convert(img, alpha=rs.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img, rs):
        if rs.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(img[:, :, 1], alpha=rs.uniform(self.saturation_lower, self.saturation_upper))
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img, rs):
        if rs.randint(2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (img[:, :, 0].astype(int) + rs.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, img, rs):
        img = self.brightness(img, rs)
        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = rs.randint(2)
        if mode == 1:
            img = self.contrast(img, rs)
        img = self.saturation(img, rs)
        img = self.hue(img, rs)
        if mode == 0:
            img = self.contrast(img, rs)
        return img

class ADE20KDataset(Dataset):
    """ <root>/{images,annotations}/{training,validation}/ (see prepare_ade20k.py) """
    def __init__(self, root: str, split: str, config: SegmentationConfig, base_seed: Optional[int] = None):
        self.train = split == "training"
        self.config = config
        self.base_seed = base_seed
        img_dir = os.path.join(root, "images", split)
        names = sorted(f[:-len(".jpg")] for f in os.listdir(img_dir) if f.endswith(".jpg"))
        self.samples = [(os.path.join(img_dir, f"{n}.jpg"), os.path.join(root, "annotations", split, f"{n}.png")) for n in names]
        self.mean = torch.tensor(config.mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(config.std, dtype=torch.float32).view(3, 1, 1)
        self.photometric = PhotoMetricDistortion()

    def __len__(self):
        return len(self.samples)

    def _to_input(self, img: np.ndarray) -> torch.Tensor:
        """ HWC BGR uint8 -> CHW RGB float, normalized (PackSegInputs + SegDataPreProcessor) """
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))[[2, 1, 0]].float()
        return (x - self.mean) / self.std

    def _crop_bbox(self, label: np.ndarray, rs: np.random.RandomState) -> tuple:
        """ mmseg RandomCrop: retry up to 10 times while one class covers more than cat_max_ratio of the crop """
        crop = self.config.crop_size
        def generate_crop_bbox():
            margin_h = max(label.shape[0] - crop, 0)
            margin_w = max(label.shape[1] - crop, 0)
            offset_h = rs.randint(0, margin_h + 1)
            offset_w = rs.randint(0, margin_w + 1)
            return offset_h, offset_h + crop, offset_w, offset_w + crop

        bbox = generate_crop_bbox()
        if self.config.cat_max_ratio < 1.0:
            for _ in range(10):
                y1, y2, x1, x2 = bbox
                labels, cnt = np.unique(label[y1:y2, x1:x2], return_counts=True)
                cnt = cnt[labels != IGNORE_INDEX]
                if len(cnt) > 1 and np.max(cnt) / np.sum(cnt) < self.config.cat_max_ratio:
                    break
                bbox = generate_crop_bbox()
        return bbox

    def train_sample(self, img: np.ndarray, label: np.ndarray, rs: np.random.RandomState):
        cfg = self.config
        # RandomResize(scale=img_scale, ratio_range, keep_ratio=True)
        lo, hi = cfg.ratio_range
        ratio = rs.random_sample() * (hi - lo) + lo
        scale = int(cfg.img_scale[0] * ratio), int(cfg.img_scale[1] * ratio)
        img = imrescale(img, scale, cv2.INTER_LINEAR)
        label = imrescale(label, scale, cv2.INTER_NEAREST)
        # RandomCrop(crop_size, cat_max_ratio)
        y1, y2, x1, x2 = self._crop_bbox(label, rs)
        img, label = img[y1:y2, x1:x2, ...], label[y1:y2, x1:x2]
        # RandomFlip(prob=0.5), horizontal
        if rs.choice(["horizontal", None], p=[0.5, 0.5]) == "horizontal":
            img, label = np.flip(img, axis=1), np.flip(label, axis=1)
        img = self.photometric(img, rs)

        # SegDataPreProcessor: normalize, then pad bottom / right to the crop size
        x = self._to_input(img)
        y = torch.from_numpy(np.ascontiguousarray(label).astype(np.int64))
        pad = (0, max(cfg.crop_size - x.shape[-1], 0), 0, max(cfg.crop_size - x.shape[-2], 0))
        return F.pad(x, pad, value=0), F.pad(y, pad, value=IGNORE_INDEX)

    def __getitem__(self, code: int):
        # training indices from InfiniteShardSampler encode the round (pass over the dataset)
        rnd, i = divmod(code, len(self.samples))
        img_path, label_path = self.samples[i]
        img, label = load_image(img_path), load_label(label_path)
        if self.train:
            seed = mix_seed(self.base_seed, rnd, i) % 2**32
            x, y = self.train_sample(img, label, np.random.RandomState(seed))
        else: # Resize(img_scale, keep_ratio=True); labels stay at the original size
            x = self._to_input(imrescale(img, self.config.img_scale, cv2.INTER_LINEAR))
            y = torch.from_numpy(label.astype(np.int64))
        return {"img": x, "label": y, "idx": i}

class InfiniteShardSampler(Sampler):
    """
    mmengine's InfiniteSampler, made resumable: an endless stream of permutations of the dataset (one per round, seeded
    by (seed, round)) whose positions are dealt to ranks in turn. Yields round * len(dataset) + index; `start` skips the
    samples this rank already consumed, and the length stays the full schedule's.
    """
    def __init__(self, size: int, seed: int, rank: int, world_size: int, num_samples: int, start: int = 0):
        self.size, self.seed = size, seed
        self.rank, self.world_size = rank, world_size
        self.num_samples, self.start = num_samples, start

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rnd_cached, perm = None, None
        for pos in range(self.start, self.num_samples):
            rnd, i = divmod(pos * self.world_size + self.rank, self.size)
            if rnd != rnd_cached:
                g = torch.Generator().manual_seed(mix_seed(self.seed, rnd))
                rnd_cached, perm = rnd, torch.randperm(self.size, generator=g).tolist()
            yield rnd * self.size + perm[i]

class ShardSampler(Sampler):
    """ Every rank-th sample, without padding (each validation image is evaluated exactly once) """
    def __init__(self, size: int, rank: int, world_size: int):
        self.indices = list(range(rank, size, world_size))

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)

def _dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1

class ADE20KDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, seed: int, batch_size: int, num_workers: int, num_batches: int):
        """ num_batches: training batches per rank over the whole schedule (max_steps x gradient accumulation) """
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches

    def train_dataloader(self):
        rank, world_size = _dist_info()
        # on resume, continue the sample stream after the batches already trained on
        done = self.trainer.global_step * self.trainer.accumulate_grad_batches if self.trainer is not None else 0
        sampler = InfiniteShardSampler(
            len(self.train_ds), self.seed, rank, world_size,
            num_samples=self.num_batches * self.batch_size, start=done * self.batch_size,
        )
        return DataLoader(
            self.train_ds, sampler=sampler, batch_size=self.batch_size, drop_last=True,
            num_workers=self.num_workers, persistent_workers=(self.num_workers > 0),
        )

    def val_dataloader(self):
        rank, world_size = _dist_info()
        return DataLoader(
            self.val_ds, sampler=ShardSampler(len(self.val_ds), rank, world_size), batch_size=1,
            num_workers=self.num_workers, persistent_workers=(self.num_workers > 0),
        )
