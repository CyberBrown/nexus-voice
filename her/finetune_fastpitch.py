#!/usr/bin/env python3
"""Fine-tune FastPitch for Samantha voice."""

import pytorch_lightning as pl
from nemo.collections.tts.models import FastPitchModel
from nemo.utils.exp_manager import exp_manager
from omegaconf import OmegaConf, open_dict

print("Loading pretrained FastPitch model...")
model = FastPitchModel.from_pretrained("tts_en_fastpitch")

cfg = model.cfg.copy()

with open_dict(cfg):
    cfg.train_ds.manifest_filepath = "/data/her/train_nemo.json"
    cfg.validation_ds.manifest_filepath = "/data/her/train_nemo.json"
    cfg.train_ds.dataloader_params.batch_size = 8
    cfg.validation_ds.dataloader_params.batch_size = 8
    cfg.optim.lr = 1e-4
    cfg.train_ds.sup_data_path = "/data/her/fastpitch_samantha/sup_data"
    cfg.validation_ds.sup_data_path = "/data/her/fastpitch_samantha/sup_data"

model._cfg = cfg
model.setup_training_data(cfg.train_ds)
model.setup_validation_data(cfg.validation_ds)

trainer = pl.Trainer(
    devices=1,
    accelerator="gpu",
    max_epochs=50,
    check_val_every_n_epoch=5,
    log_every_n_steps=10,
    enable_checkpointing=True,
    logger=True,
)

exp_cfg = OmegaConf.create({
    "exp_dir": "/data/her/fastpitch_samantha",
    "name": "samantha_fastpitch",
    "create_tensorboard_logger": True,
    "create_checkpoint_callback": True,
    "checkpoint_callback_params": {
        "save_top_k": 3,
        "monitor": "val_loss",
        "mode": "min",
    }
})
exp_manager(trainer, exp_cfg)

print("Starting training for 50 epochs...")
trainer.fit(model)

print("Training complete!")
model.save_to("/data/her/fastpitch_samantha/samantha_fastpitch_final.nemo")
print("Saved to /data/her/fastpitch_samantha/samantha_fastpitch_final.nemo")
