"""
Loss scaling under gradient accumulation, through the real Lightning Trainer (as pretrain.py and the bench run it).
CPU only.

LitREPA.training_step returns loss * hf_accumulation_scale(...) (repa.py); Lightning divides that by
accumulate_grad_batches before backward() (pytorch_lightning/loops/optimization/automatic.py). Checked here:
  - the accumulated gradient at each optimizer step equals the gradient of one batch's mean loss, both for a full
    accumulation group and for the epoch's trailing partial group (HF Trainer semantics: averaged over its own size)
  - the logged train/loss is the mean prediction loss (not summed, not divided by the accumulation)
  - the bench's StopAfter (bench/harness.py) reports the prediction loss itself, not Lightning's divided closure loss

Setup: identical batches, lr = 0 (weights stay fixed between optimizer steps), no RoPE rescale draw, so every batch
has the same loss L and gradient g.
"""
import os
import sys

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from conftest import DIFFS # noqa: E402
from harness import StopAfter, _SyntheticDataModule # noqa: E402
from repa import LitREPA, prediction_loss # noqa: E402
from test_perf_equivalence import make_config, max_rel # noqa: E402

ACCUM, BATCHES, BATCH = 2, 3, 2 # 3 batches: one full group of 2 and a trailing partial group of 1

class GradsAtStep(Callback):
    """ the accumulated gradients just before each optimizer step (before clipping) """
    def __init__(self):
        self.steps = []

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        self.steps.append({n: p.grad.detach().clone() for n, p in pl_module.named_parameters() if p.grad is not None})

class ClosureLoss(Callback):
    """ what Lightning hands on_train_batch_end: the divided closure loss """
    def __init__(self):
        self.losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.losses.append(outputs["loss"].detach().float().item())

def _config():
    world = torch.cuda.device_count() if torch.cuda.is_available() else 1 # the config checks global % (batch * GPUs)
    return make_config("tiny", batch_size=BATCH, global_batch_size=BATCH * ACCUM * world, blr=0.0,
                       pos_embed_rescale=None, grad_clip=0.0)

def _reference(cfg, batch):
    """ loss and gradient of one batch, outside Lightning, from the same initial weights """
    torch.manual_seed(0)
    lit = LitREPA(cfg)
    tgt, pred = lit.model(batch["img"], position_ids=batch["position_ids"])
    loss = prediction_loss(tgt, pred, s=batch["s"] if cfg.use_permutation else None)
    loss.backward()
    return loss.detach(), {n: p.grad.detach().clone() for n, p in lit.named_parameters() if p.grad is not None}

def test_accumulated_gradient_and_logged_loss():
    cfg = _config()
    assert cfg.grad_accum_steps == ACCUM
    dm = _SyntheticDataModule(cfg, BATCHES)
    loss_ref, g_ref = _reference(cfg, dm.train_ds.batch)

    torch.manual_seed(0)
    lit = LitREPA(cfg)
    grads, closure, stop = GradsAtStep(), ClosureLoss(), StopAfter(BATCHES)
    trainer = pl.Trainer(max_epochs=1, accelerator="cpu", devices=1, precision="32-true", logger=False,
                         accumulate_grad_batches=cfg.grad_accum_steps, limit_val_batches=0, num_sanity_val_steps=0,
                         enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
                         use_distributed_sampler=False, callbacks=[grads, closure, stop])
    trainer.fit(lit, datamodule=dm)
    L = loss_ref.item()

    # gradients: one full group (batches 0, 1) and the partial group (batch 2) both equal one batch's gradient
    assert len(grads.steps) == 2, f"expected 2 optimizer steps, got {len(grads.steps)}"
    for i, step in enumerate(grads.steps):
        worst = 0.0
        for n, g in g_ref.items():
            d, r = max_rel(step[n], g)
            assert r <= 1e-5, f"optimizer step {i}, {n}: relative {r:.2e} (gradient not averaged over the group?)"
            worst = max(worst, r)
        DIFFS.append((f"accum {ACCUM}, optimizer step {i}", "grad vs 1 batch (rel)", worst, "rtol 1e-5"))

    # what Lightning passes on: L / accum for the full group, L * (accum / 1) / accum = L for the partial group
    assert all(abs(a - b) <= 1e-6 * L for a, b in zip(closure.losses, [L / ACCUM, L / ACCUM, L])), closure.losses

    # logged train/loss: the epoch mean of the prediction loss, not summed and not divided
    logged = trainer.callback_metrics["train/loss"].item()
    DIFFS.append((f"accum {ACCUM}, logged train/loss", "vs prediction loss", abs(logged - L), "rtol 1e-5"))
    assert abs(logged - L) <= 1e-5 * L, f"logged {logged} vs loss {L}"

    # bench StopAfter: the prediction loss for every batch
    summary = stop.loss_summary()
    assert abs(summary["loss_first"] - L) <= 1e-6 * L and abs(summary["loss_last"] - L) <= 1e-6 * L, summary
