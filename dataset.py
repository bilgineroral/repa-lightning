import os
import random
from contextlib import contextmanager
from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop, Compose, InterpolationMode, Normalize, RandomHorizontalFlip, RandomResizedCrop, Resize, ToTensor
)
from timm.data import create_transform
import pytorch_lightning as pl

from config import ClassificationConfig

# ImageNet RGB normalization on [0, 1] tensors, for both pretraining and finetuning
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

def mix_seed(*keys: int) -> int:
    """ 64-bit seed from integer keys, e.g. (base_seed, epoch, idx, offset) """
    ss = np.random.SeedSequence(list(keys))
    return int(ss.generate_state(1, dtype=np.uint64)[0])

@contextmanager
def seed_rngs(seed: Optional[int]):
    """ Seed the global torch (CPU), numpy and python RNGs within the block and restore them afterwards;
    torchvision / timm transforms and timm's Mixup draw from these """
    if seed is None:
        yield
        return
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed % 2**32)
        try:
            yield
        finally:
            random.setstate(py_state)
            np.random.set_state(np_state)

def seed_training_batch(seed: int, epoch: int, batch_idx: int, rank: int) -> None:
    """ Seed the global python, numpy and torch (CPU and CUDA) RNGs for one training batch, so its random draws (dropout,
    drop path, RoPE rescaling, mixup) only depend on (seed, epoch, batch_idx, rank). Lightning doesn't checkpoint the
    RNG states, so this is what makes a resumed run draw as an uninterrupted one """
    s = mix_seed(seed, epoch, batch_idx, rank, 0) # 5 keys: never one of the per-sample (seed, epoch, idx, offset) seeds
    random.seed(s)
    np.random.seed(s % 2**32)
    torch.manual_seed(s)

def build_pretrain_transform(image_size: int, train: bool) -> Compose:
    """ NEPA pretraining augmentations with ImageNet normalization """
    size = (image_size, image_size)
    normalize = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
    if train:
        return Compose([
            RandomResizedCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
    return Compose([
        Resize(size),
        CenterCrop(size),
        ToTensor(),
        normalize,
    ])

def build_finetune_transform(config: ClassificationConfig, image_size: int, train: bool) -> Compose:
    """ MAE-style finetuning augmentations with ImageNet normalization """
    if train:
        return create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=None,
            auto_augment=config.auto_augment,
            interpolation="bicubic",
            re_prob=config.reprob,
            re_mode=config.remode,
            re_count=config.recount,
            mean=IMAGE_MEAN,
            std=IMAGE_STD,
        )
    return Compose([
        Resize(int(image_size / config.crop_pct), interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        ToTensor(),
        Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])

class ImageNetDataset(ImageFolder):
    """
    ImageNet in ImageFolder layout: <root>/<split>/<wnid>/*.JPEG (see prepare_imagenet.py).
    Random augmentations are seeded by (base_seed, epoch, idx), so every sample is reproducible
    regardless of rank, dataloader worker or resume.
    """
    def __init__(self, root: str, split: str, transform, base_seed: Optional[int] = None):
        super().__init__(os.path.join(root, split), transform=transform)
        self.base_seed = base_seed
        self._epoch = torch.zeros((), dtype=torch.int64).share_memory_()

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def _seed(self, epoch: int, idx: int, off: int) -> Optional[int]:
        return None if self.base_seed is None else mix_seed(self.base_seed, epoch, idx, off)

    def __getitem__(self, i: int):
        return self.get_item(i, int(self._epoch.item()))

    def get_item(self, i: int, epoch: int) -> dict:
        path, label = self.samples[i]
        with seed_rngs(self._seed(epoch, i, off=2)):
            img = self.transform(self.loader(path))
        return {"img": img, "label": label, "idx": i}

class PretrainDataset(ImageNetDataset):
    """ Adds REPA's random generation order (position_ids) and number of context tokens (s) """
    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        patch_size: int = 16,
        sl: int = 1, sh: int = 1,
        base_seed: Optional[int] = None,
        use_permutation: bool = True,
    ):
        super().__init__(root, split, build_pretrain_transform(image_size, train=(split == "train")), base_seed)
        self.sl, self.sh = sl, sh
        self.use_permutation = use_permutation

        self.Hp = self.Wp = image_size // patch_size

        self.s = np.arange(self.sl, self.sh + 1, dtype=np.int64)
        self.probs = np.ones_like(self.s, dtype=np.float64)
        self.probs /= self.probs.sum()

    def _generate_permutation(self, n, seed: Optional[int]) -> torch.Tensor:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        perm = rng.permutation(n)
        return torch.from_numpy(perm).to(torch.int64)

    def _sample_s(self, seed: Optional[int]) -> int:
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        s = int(rng.choice(self.s, p=self.probs))
        return s

    def get_item(self, i: int, epoch: int) -> dict:
        item = super().get_item(i, epoch)
        item["s"] = self._sample_s(self._seed(epoch, i, off=0))

        if self.use_permutation:
            item["position_ids"] = self._generate_permutation(self.Hp * self.Wp, self._seed(epoch, i, off=1))
        else:
            item["position_ids"] = torch.arange(self.Hp * self.Wp, dtype=torch.int64)

        return item

class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, train_ds, val_ds, seed: int, batch_size: int, num_workers: int):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _get_sampler(self, ds, shuffle: bool, drop_last: bool):
        # single replica without DDP, so the order only depends on (seed, epoch) either way
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        return DistributedSampler(
            ds, num_replicas=None if distributed else 1, rank=None if distributed else 0,
            shuffle=shuffle, seed=self.seed, drop_last=drop_last,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_ds,
            sampler=self._get_sampler(self.train_ds, shuffle=True, drop_last=True),
            drop_last=True,
            persistent_workers=(self.num_workers > 0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_ds,
            sampler=self._get_sampler(self.val_ds, shuffle=False, drop_last=False),
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

if __name__ == "__main__":
    seed = 42
    root = "/home/bilginer/imagenet"

    train = PretrainDataset(root, "train", base_seed=seed)
    val = ImageNetDataset(root, "val", build_finetune_transform(ClassificationConfig(), 224, train=False))
    dm = ImageNetDataModule(
        train_ds=train,
        val_ds=val,
        seed=seed,
        batch_size=4,
        num_workers=4,
    )

    for batch in dm.train_dataloader():
        print(batch["img"].shape, batch["position_ids"].shape, batch["s"])
        break
    for batch in dm.val_dataloader():
        print(batch["img"].shape, batch["label"])
        break
