from typing import List, Tuple, Union, Callable, Literal

import PIL
from tqdm import tqdm
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as T

from einops import rearrange, repeat
from einops_exts import check_shape

from Unet import Unet
from utils import cast_tuple, default, resize_image_to, normalize_neg_one_to_one, \
    unnormalize_zero_to_one, identity, exists, module_device, right_pad_dims_to, maybe, eval_decorator, null_context
from t5 import t5_encode_text, get_encoded_dim
from diffusion_model import GaussianDiffusion

class Imagen(nn.Module):
    def __init__(self, unets, *, text_encoder_name, image_sizes, text_embed_dim=None, channels=3, timesteps=1000,
                 cond_drop_prob=0.1, loss_type='l2', lowres_sample_noise_level=0.2, auto_normalize_img=True,
                 dynamic_thresholding_percentile=0.9, only_train_unet_number=None 
                ):
        
        super().__init__()

        # Set loss
        self.loss_type = loss_type
        self.loss_fn = self._set_loss_fn(loss_type)

        self.channels = channels

        unets = cast_tuple(unets)
        num_unets = len(unets)

        # Create noise schedulers for each UNet
        self.noise_schedulers = self._make_noise_schedulers(num_unets, timesteps)

        # Lowres augmentation noise schedule
        self.lowres_noise_schedule = GaussianDiffusion(timesteps=timesteps)

        # Text encoder params
        self.text_encoder_name = text_encoder_name
        self.text_embed_dim = default(text_embed_dim, lambda: get_encoded_dim(text_encoder_name))

        # Keep track of which unet is being trained at the moment
        self.unet_being_trained_index = -1

        self.only_train_unet_number = only_train_unet_number

        # Cast the relevant hyperparameters to the input Unets, ensuring that the first Unet does not condition on
        #   lowres images (base unet) while the remaining ones do (super-res unets)
        self.unets = nn.ModuleList([])
        for ind, one_unet in enumerate(unets):
            assert isinstance(one_unet, Unet)
            is_first = ind == 0

            one_unet = one_unet._cast_model_parameters(
                lowres_cond=not is_first,
                text_embed_dim=self.text_embed_dim,
                channels=self.channels,
                channels_out=self.channels,
            )

            self.unets.append(one_unet)

        # Uet image sizes
        self.image_sizes = cast_tuple(image_sizes)
        assert num_unets == len(
            image_sizes), f'you did not supply the correct number of u-nets ({len(self.unets)}) for resolutions' \
                          f' {image_sizes}'

        self.sample_channels = cast_tuple(self.channels, num_unets)

        self.lowres_sample_noise_level = lowres_sample_noise_level

        # Classifier free guidance
        self.cond_drop_prob = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        # Normalize and un-normalize image functions
        self.normalize_img = normalize_neg_one_to_one if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one if auto_normalize_img else identity
        self.input_image_range = (0. if auto_normalize_img else -1., 1.)

        # Dynamic thresholding
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        # one temp parameter for keeping track of device
        self.register_buffer('_temp', torch.tensor([0.]), persistent=False)

        # default to device of unets passed in
        self.to(next(self.unets.parameters()).device)
        
        
    @property
    def device(self) -> torch.device:
        # Returns device of Imagen instance (not writeable)
        return self._temp.device
    
    
    @staticmethod
    def _set_loss_fn(loss_type) -> Callable:
        # loss
        if loss_type == 'l1':
            loss_fn = F.l1_loss
        elif loss_type == 'l2':
            loss_fn = F.mse_loss
        elif loss_type == 'huber':
            loss_fn = F.smooth_l1_loss
        else:
            raise NotImplementedError()
        return loss_fn
    
    
    @staticmethod
    def _make_noise_schedulers(num_unets, timesteps):
        # determine noise schedules per unet
        timesteps = cast_tuple(timesteps, num_unets)

        # construct noise schedulers
        noise_schedulers = nn.ModuleList([])
        for timestep in timesteps:
            noise_scheduler = GaussianDiffusion(timesteps=timestep)
            noise_schedulers.append(noise_scheduler)

        return noise_schedulers
    
    
    def _get_unet(self, unet_number) -> Unet:
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1

        if isinstance(self.unets, nn.ModuleList):
            unets_list = [unet for unet in self.unets]
            delattr(self, 'unets')
            self.unets = unets_list

        # If gotten unet different than one listed as being trained, pl
        if index != self.unet_being_trained_index:
            for unet_index, unet in enumerate(self.unets):
                unet.to(self.device if unet_index == index else 'cpu')

        # Update relevant attribute
        self.unet_being_trained_index = index
        return self.unets[index]
    
    
    def _reset_unets_all_one_device(self, device=None):
        # Creates ModuleList out of the Unets and places on the relevant device.
        device = default(device, self.device)
        self.unets = nn.ModuleList([*self.unets])
        self.unets.to(device)

        # Resets relevant attribute to specify that no Unet is being trained at the moment
        self.unet_being_trained_index = -1

        
    def state_dict(self, *args, **kwargs):
        self._reset_unets_all_one_device()
        return super().state_dict(*args, **kwargs)

    
    def load_state_dict(self, *args, **kwargs):
        self._reset_unets_all_one_device()
        return super().load_state_dict(*args, **kwargs)
    
    @contextmanager
    def _one_unet_in_gpu(self, unet_number=None, unet=None):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.unets[unet_number - 1]

        # Store which device each UNet is on, place them all on CPU except the specified one
        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.to(self.device)

        yield

        # Restore all UNets back to their original devices
        for unet, device in zip(self.unets, devices):
            unet.to(device)    
    
    def _p_mean_variance(self, unet, x, t, *, noise_scheduler, text_embeds=None,
                         text_mask=None, lowres_cond_img=None, lowres_noise_times=None,
                         cond_scale=1., model_output=None
                        ):
        
        assert not (cond_scale != 1. and not self.can_classifier_guidance)

        # Get the prediction from the base unet
        pred = default(model_output, lambda: unet.forward_with_cond_scale(x,
                                                                          t,
                                                                          text_embeds=text_embeds,
                                                                          text_mask=text_mask,
                                                                          cond_scale=cond_scale,
                                                                          lowres_cond_img=lowres_cond_img,
                                                                          lowres_noise_times=lowres_noise_times))

        # Calculate the starting images from the noise
        x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        # DYNAMIC THRESHOLDING
        #   https://www.assemblyai.com/blog/how-imagen-actually-works/#large-guidance-weight-samplers

        # Calculate threshold for each image
        s = torch.quantile(
            rearrange(x_start, 'b ... -> b (...)').abs(),
            self.dynamic_thresholding_percentile,
            dim=-1
        )

        # If threshold is less than 1, simply clamp values to [-1., 1.]
        s.clamp_(min=1.)
        s = right_pad_dims_to(x_start, s)
        # Clamp to +/- s and divide by s to bring values back to range [-1., 1.]
        x_start = x_start.clamp(-s, s) / s

        # Return the forward process posterior parameters given the predicted x_start
        return noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
    
    @torch.no_grad()
    def _p_sample(self, unet, x, t, *, noise_scheduler, text_embeds=None, text_mask=None, 
                  lowres_cond_img=None, lowres_noise_times=None, cond_scale=1.):
        
        b, *_, device = *x.shape, x.device
        
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x=x, t=t,
                                                                  noise_scheduler=noise_scheduler,
                                                                  text_embeds=text_embeds, text_mask=text_mask,
                                                                  cond_scale=cond_scale,
                                                                  lowres_cond_img=lowres_cond_img,
                                                                  lowres_noise_times=lowres_noise_times)
        
        noise = torch.randn_like(x)
        
        is_last_sampling_timestep = (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def _p_sample_loop(self, unet, shape, *,
                       noise_scheduler, text_embeds=None,
                       text_mask=None, lowres_cond_img=None,
                       lowres_noise_times=None, cond_scale=1.
                      ):
        
        device = self.device
        
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)
        
        batch = shape[0]
        timesteps = noise_scheduler._get_sampling_timesteps(batch, device=device)
        
        img = torch.randn(shape, device=device)
        
        for times in tqdm(timesteps, desc='sampling loop time step', total=len(timesteps)):
            img = self.p_sample(unet, img, times, text_embeds=text_embeds, text_mask=text_mask,
                                cond_scale=cond_scale, lowres_cond_img=lowres_cond_img,
                                lowres_noise_times=lowres_noise_times, noise_scheduler=noise_scheduler
                               )
            
        img.clamp_(-1., 1.)
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img
    
    @torch.no_grad()
    @eval_decorator
    def sample(self, texts=None, text_masks=None, text_embeds=None, cond_scale=1., 
               lowres_sample_noise_level=None, return_pil_images=False, device=None
              ):
    
        device = default(device, self.device)
        self._reset_unets_all_one_device(device=device)
        
        # Calculate text embeddings/mask if not passed in
        if exists(texts) and not exists(text_embeds):
            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))
        
        assert exists(text_embeds), 'text or text encodings must be passed into Imagen'
        assert not (exists(text_embeds) and text_embeds.shape[
            -1] != self.text_embed_dim), f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        batch_size = text_embeds.shape[0]
        
        outputs = None
        
        is_cuda = next(self.parameters()).is_cuda
        device = next(self.parameters()).device
        
        lowres_sample_noise_level = default(lowres_sample_noise_level, self.lowres_sample_noise_level)
        
        for unet_number, unet, channel, image_size, noise_scheduler in tqdm(
                zip(range(1, len(self.unets) + 1), self.unets, self.sample_channels, self.image_sizes,
                self.noise_schedulers)):
            
            context = self.one_unet_in_gpu(unet=unet) if is_cuda else null_context()
            
            with context:
                lowres_cond_img = lowres_noise_times = None

                # If on a super-resolution model, noise the previously generated images for conditioning
                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule._get_times(batch_size, lowres_sample_noise_level,
                                                                              device=device)
                    lowres_cond_img = resize_image_to(img, image_size, pad_mode='reflect')
                    lowres_cond_img = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img,
                                                                          t=lowres_noise_times,
                                                                          noise=torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, image_size, image_size)

                # Generate images with the current unet
                img = self._p_sample_loop(
                    unet,
                    shape,
                    text_embeds=text_embeds,
                    text_mask=text_masks,
                    cond_scale=cond_scale,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_times=lowres_noise_times,
                    noise_scheduler=noise_scheduler,
                )

                # Output the image if at the end of the super-resolution chain
                outputs = img if unet_number == len(self.unets) else None

        # Return torch tensors or PIL Images
        if not return_pil_images:
            return outputs

        pil_images = list(map(T.ToPILImage(), img.unbind(dim=0)))

        return pil_images
    
    
    def _p_losses(self, unet, x_start, times, *, noise_scheduler, lowres_cond_img=None, lowres_aug_times=None,
                  text_embeds=None, text_mask=None, noise=None
                 ):
        
        noise = torch.randn_like(x_start)
        
        # normalize x_start to [-1, 1] and so too lowres_cond_img if it exists
        x_start = self.normalize_img(x_start)
        lowres_cond_img = maybe(self.normalize_img)(lowres_cond_img)

        # get x_t (i.e. noise the inputs)
        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # Also noise the lowres conditioning image
        lowres_cond_img_noisy = None
        if exists(lowres_cond_img):
            lowres_aug_times = default(lowres_aug_times, times)
            lowres_cond_img_noisy = self.lowres_noise_schedule.q_sample(x_start=lowres_cond_img, t=lowres_aug_times,
                                                                        noise=torch.randn_like(lowres_cond_img))

        # Predict the noise component of the noised images
        pred = unet.forward(
            x_noisy,
            times,
            text_embeds=text_embeds,
            text_mask=text_mask,
            lowres_noise_times=lowres_aug_times,
            lowres_cond_img=lowres_cond_img_noisy,
            cond_drop_prob=self.cond_drop_prob,
        )

        # Return loss between prediction and ground truth
        return self.loss_fn(pred, noise)
    
    
    def forward(self, images, texts=None, text_embeds=None, text_masks=None, unet_number=None):
        
        unet_number = default(unet_number, 1)
        
        unet_index = unet_number - 1
        unet = self._get_unet(unet_number)
        
        noise_scheduler = self.noise_schedulers[unet_index]
        target_image_size = self.image_sizes[unet_index]
        prev_image_size = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        b, c, h, w, device, = *images.shape, images.device

        # Make sure images have proper number of dimensions and channels.
        check_shape(images, 'b c h w', c=self.channels)
        assert h >= target_image_size and w >= target_image_size

        # Randomly sample a timestep value for each image in the batch.
        times = noise_scheduler._sample_random_times(b, device=device)

        # If text conditioning info supplied as text rather than embeddings, calculate the embeddings/mask
        if exists(texts) and not exists(text_embeds):
            assert len(texts) == len(images), \
                'number of text captions does not match up with the number of images given'

            text_embeds, text_masks = t5_encode_text(texts, name=self.text_encoder_name)
            text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        # Make sure embeddings are not supplied if not conditioning on text and vice versa
        assert exists(text_embeds), \
            'text or text encodings must be passed into decoder'

        # Ensure text embeddings are right dimensionality
        assert not (exists(text_embeds) and text_embeds.shape[-1] != self.text_embed_dim), \
            f'invalid text embedding dimension being passed in (should be {self.text_embed_dim})'

        # Create low-res conditioning information if a super-res model
        lowres_cond_img = lowres_aug_times = None
        if exists(prev_image_size):
            lowres_cond_img = resize_image_to(images, prev_image_size, clamp_range=self.input_image_range,
                                              pad_mode='reflect')
            lowres_cond_img = resize_image_to(lowres_cond_img, target_image_size, clamp_range=self.input_image_range,
                                              pad_mode='reflect')

            lowres_aug_time = self.lowres_noise_schedule._sample_random_times(1, device=device)
            lowres_aug_times = repeat(lowres_aug_time, '1 -> b', b=b)

        # Resize images to current unet size
        images = resize_image_to(images, target_image_size)

        # Calculate and return the loss
        return self._p_losses(unet, images, times, text_embeds=text_embeds, text_mask=text_masks,
                              noise_scheduler=noise_scheduler, lowres_cond_img=lowres_cond_img,
                              lowres_aug_times=lowres_aug_times)