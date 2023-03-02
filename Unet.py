from typing import Union

from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts.torch import EinopsToAndFrom
import torch
from torch import nn
import torch.nn.functional as F

from utils import default, exists, cast_tuple, prob_mask_like
from layers import (
    Attention,
    CrossEmbedLayer,
    Downsample,
    Residual,
    ResnetBlock,
    SinusoidalPosEmb,
    TransformerBlock,
    Upsample, Parallel, Identity
)

from t5 import get_encoded_dim

class Unet(nn.Module):
    def __init__(
            self,
            *,
            dim = 128,
            dim_mults = (1, 2, 4),
            channels = 3,
            channels_out = None,
            cond_dim = None,
            text_embed_dim=get_encoded_dim('t5_small'),
            num_resnet_blocks: Union[int, tuple] = 1,
            layer_attns: Union[bool, tuple] = True,
            layer_cross_attns: Union[bool, tuple] = True,
            attn_heads = 8,
            lowres_cond = False,
            memory_efficient = False,
            attend_at_middle = False
            ):
        
        super().__init__()
        
        # Save arguments locals to take care of some hyperparameters for cascading DDPM
        self._locals = locals()
        self._locals.pop('self', None)
        self._locals.pop('__class__', None)

        # Constants
        ATTN_DIM_HEAD = 64  # Dimensionality for attention.
        NUM_TIME_TOKENS = 2  # Number of time tokens to use in conditioning tensor
        RESNET_GROUPS = 8  # Number of groups in ResNet block GroupNorms

        # Model constants
        init_conv_to_final_conv_residual = False  # Whether to add skip connection between Unet input and output
        final_resnet_block = True  # Whether to add a final resnet block to the output of the Unet

        # TIME CONDITIONING

        # Double conditioning dimensionality for super-res models due to concatenation of low-res images
        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4 * (2 if lowres_cond else 1)

        # Maps time to time hidden state
        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_cond_dim),
            nn.SiLU()
        )
        
        # Maps time hidden state to time conditioning (non-attention)
        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # Maps time hidden states to time tokens for main conditioning tokens (attention)
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * NUM_TIME_TOKENS),
            Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
        )
        
        # TEXT CONDITIONING

        self.norm_cond = nn.LayerNorm(cond_dim)

        # Projection from text embedding dim to cond_dim
        self.text_embed_dim = text_embed_dim
        self.text_to_cond = nn.Linear(self.text_embed_dim, cond_dim)

        # Create null tokens for classifier-free guidance. See
        #   https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance
        max_text_len = 256
        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, time_cond_dim))

        # For injecting text information into time conditioning (non-attention)
        self.to_text_non_attn_cond = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, time_cond_dim),
            nn.SiLU(),
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        # UNET LAYERS

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # Initial convolution that brings input images to proper number of channels for the Unet
        self.init_conv = CrossEmbedLayer(channels if not lowres_cond else channels * 2,
                                         dim_out=dim,
                                         kernel_sizes=(3, 7, 15),
                                         stride=1)
        
        # Determine channel numbers for UNet descent/ascent and then zip into in/out pairs
        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Number of resolutions/layers in the UNet
        num_resolutions = len(in_out)

        # Cast relevant arguments to tuples (with one element for each Unet layer) if a single value rather than tuple
        #   was input for the argument
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_resolutions)
        resnet_groups = cast_tuple(RESNET_GROUPS, num_resolutions)
        layer_attns = cast_tuple(layer_attns, num_resolutions)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_resolutions)

        # Make sure relevant tuples have one elt for each layer in the UNet (if tuples rather than single values passed
        #   in as arguments)
        assert all(
            [layers == num_resolutions for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # Scale for resnet skip connections
        self.skip_connect_scale = 2 ** -0.5

        # Downsampling and Upsampling modules of the Unet
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Parameter lists for downsampling and upsampling trajectories
        layer_params = [num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        reversed_layer_params = list(map(reversed, layer_params))

        # DOWNSAMPLING LAYERS

        # Keep track of skip connection channel depths for concatenation later
        skip_connect_dims = []

        # For each layer in the Unet
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(in_out, *layer_params)):

            is_last = ind == (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn else None

            # Potentially use Transformer encoder at end of layer
            transformer_block_klass = TransformerBlock if layer_attn else Identity

            current_dim = dim_in

            # Whether to downsample at the beginning of the layer - cuts image spatial size-length
            pre_downsample = None
            if memory_efficient:
                pre_downsample = Downsample(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # Downsample at the end of the layer if not `pre_downsample`
            post_downsample = None
            if not memory_efficient:
                post_downsample = Downsample(current_dim, dim_out) if not is_last else Parallel(
                    nn.Conv2d(dim_in, dim_out, 3, padding=1), nn.Conv2d(dim_in, dim_out, 1))

            # Create the layer
            self.downs.append(nn.ModuleList([
                pre_downsample,
                # ResnetBlock that conditions, in addition to time, on the main tokens via cross attention.
                ResnetBlock(current_dim,
                            current_dim,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                # Sequence of ResnetBlocks that condition only on time
                nn.ModuleList(
                    [
                        ResnetBlock(current_dim,
                                    current_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups
                                    )
                        for _ in range(layer_num_resnet_blocks)
                    ]
                ),
                # Transformer encoder for multi-headed self attention
                transformer_block_klass(dim=current_dim,
                                        heads=attn_heads,
                                        dim_head=ATTN_DIM_HEAD),
                post_downsample,
            ]))

        # MIDDLE LAYERS

        mid_dim = dims[-1]

        # ResnetBlock that incorporates cross-attention conditioning on main tokens
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim,
                                      groups=resnet_groups[-1])

        # Optional residual self-attention
        self.mid_attn = EinopsToAndFrom('b c h w', 'b (h w) c',
                                        Residual(Attention(mid_dim, heads=attn_heads,
                                                           dim_head=ATTN_DIM_HEAD))) if attend_at_middle else None

        # ResnetBlock that incorporates cross-attention conditioning on main tokens
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=time_cond_dim,
                                      groups=resnet_groups[-1])

        # UPSAMPLING LAYERS

        # For each layer in the unet
        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_cross_attn) in enumerate(
                zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (num_resolutions - 1)
            layer_cond_dim = cond_dim if layer_cross_attn else None

            # Potentially use Transformer encoder at end of layer
            transformer_block_klass = TransformerBlock if layer_attn else Identity

            skip_connect_dim = skip_connect_dims.pop()

            # Create the layer
            self.ups.append(nn.ModuleList([
                # Same as `downs` except add channels for skip-connect
                ResnetBlock(dim_out + skip_connect_dim,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups),
                # Same as `downs` except add channels for skip-connect
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out + skip_connect_dim,
                                    dim_out,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups)
                        for _ in range(layer_num_resnet_blocks)
                    ]),
                transformer_block_klass(dim=dim_out,
                                        heads=attn_heads,
                                        dim_head=ATTN_DIM_HEAD),
                # Upscale on the final layer too if memory_efficient to make sure get correct output size
                Upsample(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # Whether to do a final residual from initial conv to the final resnet block out
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = dim * (2 if init_conv_to_final_conv_residual else 1)

        # Final optional resnet block and convolution out
        self.final_res_block = ResnetBlock(final_conv_dim, dim, time_cond_dim=time_cond_dim,
                                           groups=resnet_groups[0]) if final_resnet_block else None

        # Final convolution to bring to right num channels
        final_conv_dim_in = dim if final_resnet_block else final_conv_dim
        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, 3,
                                    padding=3 // 2)
        
    def forward(
            self,
            x,
            time,
            *,
            lowres_cond_img = None,
            lowres_noise_times = None,
            text_embeds = None,
            text_mask = None,
            cond_drop_prob: float = 0.
            ):
        
        batch_size, device = x.shape[0], x.device
        
        assert not (self.lowres_cond and not exists(lowres_cond_img)), \
            'low resolution conditioning image must be present'
        assert not (self.lowres_cond and not exists(lowres_noise_times)), \
            'low resolution conditioning noise time must be present'
        
        # time conditioning
        t, time_tokens = self._generate_t_tokens(time, lowres_noise_times)
        
        # text conditioning
        t, c = self._text_condition(text_embeds, batch_size, cond_drop_prob, device, text_mask, t, time_tokens)

        # Concatenate low-res image to input if super-resolution Unet
        if exists(lowres_cond_img):
            x = torch.cat((x, lowres_cond_img), dim=1)

        # Initial convolution
        x = self.init_conv(x)

        # Initial convolution clone for residual
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # DOWNSAMPLING TRAJECTORY

        # To store images for skip connections
        hiddens = []

        # For every layer in the downwards trajectory
        for pre_downsample, init_block, resnet_blocks, attn_block, post_downsample in self.downs:

            # Downsample before processing at this resolution if using efficient UNet
            if exists(pre_downsample):
                x = pre_downsample(x)

            # Initial block. Conditions on `c` via cross attention and conditions on `t` via scale-shift.
            x = init_block(x, t, c)

            # Series of residual blocks that are like `init_block` except they don't condition on `c`.
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            # Transformer encoder
            x = attn_block(x)
            hiddens.append(x)

            # If not using efficient UNet, downsample after processing at this resolution
            if exists(post_downsample):
                x = post_downsample(x)

        # MIDDLE PASS

        # Pass through two ResnetBlocks that condition on `c` and `t`, with a possible residual Attention layer between.
        x = self.mid_block1(x, t, c)
        if exists(self.mid_attn):
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # UPSAMPLING TRAJECTORY

        # Lambda function for skip connections
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim=1)

        for init_block, resnet_blocks, attn_block, upsample in self.ups:
            # Concatenate the skip connection (post Transformer encoder) from the corresponding layer in the
            #   downsampling trajectory and pass through `init_block`, which again conditions on `c` and `t`
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            # For each resnet block, concatenate the corresponding skip connection and then pass through the block.
            #   These blocks again condition only on `t`.
            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            # Transformer encoder and upsampling
            x = attn_block(x)
            x = upsample(x)

        # Final skip connect from the initial conv if used
        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim=1)

        # Potentially one final residual block
        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        # Final convolution to get the proper number of channels.
        return self.final_conv(x)
        
    def _generate_t_tokens(
            self,
            time,
            lowres_noise_times
            ):

        time_hiddens = self.to_time_hiddens(time)
        t = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)

        # If lowres conditioning, add lowres time conditioning to `t` and concat lowres time tokens to `c`
        if self.lowres_cond:
            lowres_time_hiddens = self.to_lowres_time_hiddens(lowres_noise_times)
            lowres_time_tokens = self.to_lowres_time_tokens(lowres_time_hiddens)
            lowres_t = self.to_lowres_time_cond(lowres_time_hiddens)

            t = t + lowres_t
            time_tokens = torch.cat((time_tokens, lowres_time_tokens), dim=-2)

        return t, time_tokens
    
    def _text_condition(
            self,
            text_embeds,
            batch_size,
            cond_drop_prob,
            device,
            text_mask,
            t,
            time_tokens
            ):

        text_tokens = None
        if exists(text_embeds):

            # Project the text embeddings to the conditioning dimension `cond_dim`.
            text_tokens = self.text_to_cond(text_embeds)

            # Truncate the tokens to have the maximum number of allotted words.
            text_tokens = text_tokens[:, :self.max_text_len]

            # Pad the text tokens up to self.max_text_len if needed
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            # Prob. mask for clf-free guidance conditional dropout. Tells which elts in the batch to keep. Size (b,).
            text_keep_mask = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)

            # Combines T5 and clf-free guidance masks
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            if exists(text_mask):
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value=False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')  # (b, self.max_text_len, 1)
                text_keep_mask_embed = text_mask & text_keep_mask_embed  # (b, self.max_text_len, 1)

            # Creates NULL tensor of size (1, self.max_text_len, cond_dim)
            null_text_embed = self.null_text_embed.to(text_tokens.dtype)  # for some reason pytorch AMP not working

            # Replaces masked elements with NULL
            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            # Extra non-attention conditioning by projecting and then summing text embeddings to time (text hiddens)
            # Pool the text tokens along the word dimension.
            mean_pooled_text_tokens = text_tokens.mean(dim=-2)

            # Project to `time_cond_dim`
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)  # (b, cond_dim) -> (b, time_cond_dim)

            null_text_hidden = self.null_text_hidden.to(t.dtype)

            # Drop relevant conditioning info as demanded by clf-free guidance mask
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')
            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )

            # Add this conditioning to our `t` tensor
            t = t + text_hiddens

        # main conditioning tokens `c` - concatenate time/text tokens
        c = time_tokens if not exists(text_tokens) else torch.cat((time_tokens, text_tokens), dim=-2)

        # normalize conditioning tokens
        c = self.norm_cond(c)

        return t, c
    
class Base(Unet):
    """
    Base image generation U-Net with default arguments from original Imagen implementation.
    - dim = 512
    - dim_mults = (1, 2, 3, 4),
    - num_resnet_blocks = 3,
    - layer_attns = (False, True, True, True),
    - layer_cross_attns = (False, True, True, True),
    - memory_efficient = False
    """

    defaults = dict(
        dim=512,
        dim_mults=(1, 2, 3, 4),
        num_resnet_blocks=3,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        memory_efficient=False
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Base.defaults, **kwargs})


class Super(Unet):
    """
    Super-Resolution U-Net with default arguments from original Imagen implementation.
    - dim = 128
    - dim_mults = (1, 2, 4, 8),
    - num_resnet_blocks = (2, 4, 8, 8),
    - layer_attns = (False, False, False, True),
    - layer_cross_attns = (False, False, False, True),
    - memory_efficient = True
    """
    defaults = dict(
        dim=128,
        dim_mults=(1, 2, 4, 8),
        num_resnet_blocks=(2, 4, 8, 8),
        layer_attns=(False, False, False, True),
        layer_cross_attns=(False, False, False, True),
        memory_efficient=True
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Super.defaults, **kwargs})


class BaseTest(Unet):
    """
    Base image generation U-Net with default arguments intended for testing.
    - dim = 8
    - dim_mults = (1, 2)
    - num_resnet_blocks = 1
    - layer_attns = False
    - layer_cross_attns = False
    - memory_efficient = False
    """

    defaults = dict(
        dim=8,
        dim_mults=(1, 2),
        num_resnet_blocks=1,
        layer_attns=False,
        layer_cross_attns=False,
        memory_efficient=False
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Base.defaults, **kwargs})


class SuperTest(Unet):
    """
    Super-Resolution U-Net with default arguments intended for testing.
    - dim = 8
    - dim_mults = (1, 2)
    - num_resnet_blocks = (1, 2)
    - layer_attns = False
    - layer_cross_attns = False
    - memory_efficient = True
    """
    defaults = dict(
        dim=8,
        dim_mults=(1, 2),
        num_resnet_blocks=(1, 2),
        layer_attns=False,
        layer_cross_attns=False,
        memory_efficient=True
    )
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **{**Super.defaults, **kwargs})