
import torch
import numpy as np
from tqdm import tqdm
from torch import autocast
from ldm.models.diffusion.plms import PLMSSampler

class ComposeNet(object):
    def __init__(self, model, device, timesteps=50, ddim_eta=0.0, scale=7.5, samples=1):
        self.model = model
        self.sampler = PLMSSampler(model)
        self.sampler.make_schedule(ddim_num_steps=50, ddim_eta=ddim_eta, verbose=False)

        # Get scaling factors from Sampler Schedule
        self.alphas = self.sampler.ddim_alphas
        self.alphas_prev = self.sampler.ddim_alphas_prev
        self.sqrt_one_minus_alphas = self.sampler.ddim_sqrt_one_minus_alphas
        self.sigmas = self.sampler.ddim_sigmas

        self.scale = scale
        self.device = device
        self.samples = samples

        self.ch = 4
        self.f = 8
        self.h = 512
        self.w = 512

        self.shape = [self.ch, self.h // self.f, self.w // self.f]

    # Sampler Functions
    @torch.no_grad()
    def p_sample(self, sampler, x, c, uc, ts, index, old_eps=None, t_next=None):
        x, _, e_t = sampler.p_sample_plms(x, c, ts, index=index, unconditional_guidance_scale=self.scale, 
                                        unconditional_conditioning=uc, old_eps=old_eps, t_next=t_next)
        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)

        return x, old_eps

    @torch.no_grad()
    def sample_x(self, x, e_t, index):
        b = self.samples

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), self.alphas[index], device=self.device)
        a_prev = torch.full((b, 1, 1, 1), self.alphas_prev[index], device=self.device)
        sigma_t = torch.full((b, 1, 1, 1), self.sigmas[index], device=self.device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), self.sqrt_one_minus_alphas[index],device=self.device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn(x.shape, device=self.device)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    def init_control(self, image=None):
        if not image:
            control = torch.zeros((self.samples, 3, self.h, self.w), device=self.device)
            control = control.to(memory_format=torch.contiguous_format).float()
        else:
            control = torch.from_numpy(image).to(self.device)
            control = control.to(memory_format=torch.contiguous_format).float()
        return control

    def init_cond(self, prompt="", image=None):
        with torch.no_grad():
            with autocast('cuda'):
                with self.model.ema_scope():
                    c = self.model.get_learned_conditioning(self.samples * [prompt])
                    c = dict(c_crossattn=[c], c_concat=[self.init_control(image)])

                    return c

    def combine(self, w_i=0.5, prompt_i="", control_i=None, w_j=0.5, prompt_j="", control_j=None):
        uc = self.init_cond()
        c_i = self.init_cond(prompt_i, control_i)
        c_j = self.init_cond(prompt_j, control_j)

        # Run Diffusion Loop
        with torch.no_grad():
            with autocast('cuda'):
                with self.model.ema_scope():
                    # Initialize sample x_T to N(0,I)
                    b = self.samples
                    x = torch.randn((b, self.ch, self.h // self.f, self.w // self.f)).to(self.device)

                    timesteps = self.sampler.ddim_timesteps
                    time_range = np.flip(timesteps)
                    total_steps = timesteps.shape[0]
                    e_ti = []
                    e_tj = []
                    e_t = []
                    for i, step in enumerate(tqdm(time_range, desc='PLMS Sampler', total=total_steps)):
                        index = total_steps - i - 1
                        ts = torch.full((b,), step, device=self.device, dtype=torch.long)
                        ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=self.device, dtype=torch.long)
                        
                        # Compute conditional scores for each concept c_i
                        x_i, e_ti = self.p_sample(self.sampler, x, c_i, uc, ts, index, old_eps=e_ti, t_next=ts_next) 
                        x_j, e_tj = self.p_sample(self.sampler, x, c_j, uc, ts, index, old_eps=e_tj, t_next=ts_next)
                        e_i = e_ti[-1]
                        e_j = e_tj[-1]

                        # Compute unconditional score
                        x_u, e_t = self.p_sample(self.sampler, x, uc, uc, ts, index, old_eps=e_t, t_next=ts_next)
                        e = e_t[-1]
                        
                        # Sampling
                        e_c = (e + w_i * (e_i - e) + w_j * (e_j - e))
                        
                        x = self.sample_x(x, e_c, index)

                    x = self.model.decode_first_stage(x)
                    x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)

                    return x