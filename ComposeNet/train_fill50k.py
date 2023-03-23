import torch

import pytorch_lightning as pl
from transformers import logging
from torch.utils.data import DataLoader

from cldm.logger import ImageLogger
from cldm.util import create_model, load_state_dict
from datasets import CircleDataset

resume_path = './ComposeNet/models/control_sd21_ini.ckpt'
batch_size = 2
logger_freq = 250
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

workers = [1,2,3,4,5,6]
max_steps = 1000

logging.set_verbosity(logging.ERROR)

model = create_model('./ComposeNet/models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

dataset = CircleDataset()

dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=50)

trainer = pl.Trainer(gpus=workers, accelerator="gpu", devices=len(workers), strategy="ddp", precision=32, callbacks=[logger], max_steps=max_steps/len(workers), default_root_dir = './ComposeNet')

trainer.fit(model, dataloader)

torch.save(model.state_dict(), "./ComposeNet/models/{}_sd21_gs-{:06}.ckpt".format(dataset.name, max_steps))
