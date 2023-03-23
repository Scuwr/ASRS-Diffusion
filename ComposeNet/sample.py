import os
import sys
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch import autocast
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import CircleDataset
from cldm.util import load_model_from_config
from ldm.models.diffusion.plms import PLMSSampler

config_path = './ComposeNet/models/cldm_v21.yaml'
model_path =  './ComposeNet/models/CircleDataset_sd21_gs-001000.ckpt'
output_path = './ComposeNet'

device = torch.device("cuda")

prompt_i = "blue circle with snow background"
prompt_j = "hogwarts"

w_i = 0.20
w_j = 0.80

n = 1 # Number of samples / batch size
b = n 
ch = 4 # Latent channels
f = 8 # Downsample factor
h = 512 # Image height
w = 512 # Image width

scale = 7.5 # Unconditional guidance scale
ddim_eta = 0.0 # 0.0 corresponds to deterministic sampling
shape = [ch, h // f, w // f]


# Create Control Images
dataset = CircleDataset()
control_i = torch.from_numpy((dataset[21]['hint'])[None, :])
control_i = control_i.to(device)
control_i = rearrange(control_i, 'b h w c -> b c h w')
control_i = control_i.to(memory_format=torch.contiguous_format).float()

# Unconditioned Control Image
control_j = torch.zeros_like(control_i)
control_j = control_j.to(device)
control_j = control_j.to(memory_format=torch.contiguous_format).float()

control_u = control_j


# Load model
model = load_model_from_config(config_path, model_path)
model = model.to(device)
sampler = PLMSSampler(model)
sampler.make_schedule(ddim_num_steps=500, ddim_eta=ddim_eta, verbose=False)

# Get scaling factors from Sampler Schedule
alphas = sampler.ddim_alphas
alphas_prev = sampler.ddim_alphas_prev
sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
sigmas = sampler.ddim_sigmas


# Sampler Functions
@torch.no_grad()
def p_sample(model, x, c, ts, index, old_eps=None, t_next=None):
    x, _, e_t = model.p_sample_plms(x, c, ts, index=index, unconditional_guidance_scale=scale, 
                                    unconditional_conditioning=uc, old_eps=old_eps, t_next=t_next)
    old_eps.append(e_t)
    if len(old_eps) >= 4:
        old_eps.pop(0)

    return x, old_eps 

def sample_x(x, e_t, index):
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * torch.randn(x.shape, device=device)
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    return x_prev

def sample_x_no_sqrta(x, e_t, index):
    # select parameters corresponding to the currently considered timestep
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

    # current prediction for x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    # direction pointing to x_t
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * torch.randn(x.shape, device=device)
    x_prev = pred_x0 + dir_xt + noise
    return x_prev


# Initialize conditioning
with torch.no_grad():
    with autocast('cuda'):
        with model.ema_scope():
            uc = model.get_learned_conditioning(n * [""])
            c_i = model.get_learned_conditioning(n * [prompt_i])
            c_j = model.get_learned_conditioning(n * [prompt_j])

            uc = dict(c_crossattn=[uc], c_concat=[control_u])
            c_i = dict(c_crossattn=[c_i], c_concat=[control_i])
            c_j = dict(c_crossattn=[c_j], c_concat=[control_j])

sample_path = os.path.join(output_path, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))


# Run Diffusion Loop
with torch.no_grad():
    with autocast('cuda'):
        with model.ema_scope():
            # Initialize sample x_T to N(0,I)
            x = torch.randn((n, ch, h // f, w // f)).to(device)

            timesteps = sampler.ddim_timesteps
            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]
            print(total_steps)
            e_ti = []
            e_tj = []
            e_t = []
            for i, step in enumerate(tqdm(time_range, desc='PLMS Sampler', total=total_steps)):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)
                
                # Compute conditional scores for each concept c_i
                x_i, e_ti = p_sample(sampler, x, c_i, ts, index, old_eps=e_ti, t_next=ts_next) 
                x_j, e_tj = p_sample(sampler, x, c_j, ts, index, old_eps=e_tj, t_next=ts_next)
                e_i = e_ti[-1]
                e_j = e_tj[-1]

                # Compute unconditional score
                x_u, e_t = p_sample(sampler, x, uc, ts, index, old_eps=e_t, t_next=ts_next)
                e = e_t[-1]
                
                # Sampling
                e_c = (e + w_i * (e_i - e) + w_j * (e_j - e))
                
                x = sample_x(x, e_c, index)

            x = model.decode_first_stage(x)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

            for sample in x:
                sample = 255 * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"sample_{base_count:05}.png"))
                base_count += 1