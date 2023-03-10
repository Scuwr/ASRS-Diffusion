import os
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch import autocast
from einops import rearrange
from omegaconf import OmegaConf

from stablediffusion.ldm.models.diffusion.plms import PLMSSampler
from stablediffusion.scripts.txt2img import load_model_from_config

device = torch.device("cuda")

prompt_i = "a stone castle surrounded by lakes and trees"
prompt_j = "black and white"

w_i = 0.5
w_j = 0.5

configpath = "stablediffusion/configs/stable-diffusion/v1-inference.yaml"
outpath = "."

config = OmegaConf.load(configpath)

n = 1 # Number of samples / batch size
ch = 4 # Latent channels
f = 8 # Downsample factor
h = 512 # Image height
w = 512 # Image width

scale = 7.5 # Unconditional guidance scale
ddim_eta = 0.0 # 0.0 corresponds to deterministic sampling
shape = [ch, h // f, w // f]

b = n

model = load_model_from_config(config, 'sd-v1-4.ckpt')
model = model.to(device)
sampler = PLMSSampler(model)
sampler.make_schedule(ddim_num_steps=500, ddim_eta=ddim_eta, verbose=False)

alphas = sampler.ddim_alphas
alphas_prev = sampler.ddim_alphas_prev
sqrt_one_minus_alphas = sampler.ddim_sqrt_one_minus_alphas
sigmas = sampler.ddim_sigmas

with torch.no_grad():
    with autocast('cuda'):
        with model.ema_scope():
            uc = model.get_learned_conditioning(n * [""])
            c_i = model.get_learned_conditioning(n * [prompt_i])
            c_j = model.get_learned_conditioning(n * [prompt_j])

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

sample_path = os.path.join(outpath, "samples")
os.makedirs(sample_path, exist_ok=True)
base_count = len(os.listdir(sample_path))
grid_count = len(os.listdir(outpath)) - 1

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
                #mean = x - (e + w_i * (e_i - e) + w_j * (e_j - e))
                #covar = covar = torch.full((b, 1, 1, 1), sigmas[index], device=device)**2
                #ident = torch.eye(h // f, w // f).to(device)
                #x = torch.normal(mean, covar*ident)

            x = model.decode_first_stage(x)
            x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

            for sample in x:
                sample = 255 * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                img = Image.fromarray(sample.astype(np.uint8))
                img.save(os.path.join(sample_path, f"sample_{base_count:05}.png"))
                base_count += 1