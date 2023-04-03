import os
import torch
import numpy as np

from PIL import Image
from einops import rearrange
from cldm.util import load_model_from_config

from comldm.comldm import ComposeNet

config_path = './ComposeNet/models/cldm_v21.yaml'
model_path =  './ComposeNet/models/control_sd21_ini.ckpt'
output_path = './ComposeNet'

device = torch.device("cuda")

prompt_i = "plane on descent."
prompt_j = "fail wing slat"

w_i = 0.50
w_j = 0.50

sample_path = os.path.join(output_path, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))

model = load_model_from_config(config_path, model_path)
model = model.to(device)

com = ComposeNet(model, device)
x = com.combine(w_i, prompt_i, None, w_j, prompt_j, None)

for sample in x:
    sample = 255 * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
    img = Image.fromarray(sample.astype(np.uint8))
    img.save(os.path.join(sample_path, f"sample_{base_count:05}.png"))
    base_count += 1