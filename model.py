import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from einops import rearrange
from solver import broyden,naive_solver

def sample_uniformly(n1, n2):
    return random.randint(int(n1), int(n2))

def _sample_discrete_laplace_unbounded(center=0, b=2.0):
    """
    Sample Y ~ discrete Laplace on all integers with parameter b (scale).
    Construction: Y = G1 - G2, G_i ~ Geometric(p), p = 1 - exp(-1/b).
    Returns center + Y.
    """
    q = math.exp(-1.0 / float(b))
    p = 1.0 - q
    # geometric with support {1,2,...}
    g1 = np.random.geometric(p)
    g2 = np.random.geometric(p)
    y = int(g1 - g2)  # can be negative
    return y

def sample_uniformly_with_long_tail(n1, n2, b=2.5, mixer_value=0.0, center=None):
    """
    Mixture sampler:
      with prob mixer_value -> Uniform[n1..n2]
      with prob (1-mixer_value) -> center + DiscreteLaplace(b), UNBOUNDED ABOVE
    Only a lower bound is enforced: result = max(n1, ...).
    """
    n1, n2 = int(n1), int(n2)
    if center is None:
        center = (n1 + n2) // 2
    if random.random() < float(1-mixer_value):
        return sample_uniformly(n1, n2)
    k = center + _sample_discrete_laplace_unbounded(center=0, b=b)
    # enforce only a lower bound (no cap above)
    return int(max(n1, k))


class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class Conv2d(torch.nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, up=False, down=False,
        resample_filter=[1,1], fused_resample=False, init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)
        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer('resample_filter', f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=max(f_pad - w_pad, 0))
            x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad+f_pad)
            x = torch.nn.functional.conv2d(x, f.tile([self.out_channels, 1, 1, 1]), groups=self.out_channels, stride=2)
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(x, f.mul(4).tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if self.down:
                x = torch.nn.functional.conv2d(x, f.tile([self.in_channels, 1, 1, 1]), groups=self.in_channels, stride=2, padding=f_pad)
            if w is not None:
                x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class TiedTransposeConv(nn.Module):
    """
    A convolution-transpose operator that is tied to the weights of a given convolution.
    Here we add an output_padding parameter so that the dimensions of the reconstruction
    match those of the target.
    """
    def __init__(self, conv, output_padding=1):
        super(TiedTransposeConv, self).__init__()
        self.conv = conv
        self.output_padding = output_padding

    def forward(self, x):
        return F.conv_transpose2d(
            x,
            self.conv.weight,
            bias=None,
            stride=self.conv.stride,
            padding=self.conv.padding,
            output_padding=self.output_padding,
            groups=self.conv.groups
        )

# Unet
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        # run in float32 then cast back to original dtype
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    else:
        raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def normalization(channels):
    # use 32 groups by default
    return GroupNorm32(32, channels)


# ----------------------------------------
# Attention (unchanged)
# ----------------------------------------

class QKVAttentionLegacy(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = (
            qkv.reshape(bs * self.n_heads, ch * 3, length)
               .split(ch, dim=1)
        )
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)


class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        bs, width, length = qkv.shape
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = F.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct",
            weight,
            v.reshape(bs * self.n_heads, ch, length),
        )
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    Spatial self‑attention block.
    """
    def __init__(self, channels, num_heads=1, num_head_channels=-1,
                 use_new_attention_order=False):
        super().__init__()
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            self.attn = QKVAttention(self.num_heads)
        else:
            self.attn = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)           # (B, C, T)
        qkv = self.qkv(self.norm(x_flat))      # (B, 3C, T)
        h = self.attn(qkv)                     # (B, C, T)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


# ----------------------------------------
# Up/Down‑sampling layers (unchanged)
# ----------------------------------------

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.dims = dims
        self.use_conv = use_conv
        self.out_channels = out_channels or channels
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            out = F.interpolate(
                x, (x.shape[2], x.shape[3]*2, x.shape[4]*2), mode="nearest"
            )
        else:
            out = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            out = self.conv(out)
        return out


class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.dims = dims
        self.use_conv = use_conv
        self.out_channels = out_channels or channels
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, channels, self.out_channels, 3,
                stride=stride, padding=1
            )
        else:
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


# ----------------------------------------
# ResBlock without any timestep conditioning
# ----------------------------------------

class ResBlockNoTime(nn.Module):
    """
    Exactly the same as the original ResBlock *minus* timestep embeddings.
    """
    def __init__(self, channels, dropout,
                 out_channels=None, use_conv=False,
                 dims=2, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.updown = up or down
        self.dims = dims

        # main path
        kernel_size = 3
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, kernel_size, padding=kernel_size//2),
        )

        # up/down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        # output tortuous path
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # skip
        if self.out_channels == channels:
            self.skip = nn.Identity()
        elif use_conv:
            self.skip = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            # split off the last conv in in_layers
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip(x) + h


# ----------------------------------------
# The UNet for blind denoising
# ----------------------------------------
class UNetBlindDenoise(nn.Module):
    """
    A UNet that takes only an image `x` (no timesteps or labels) and returns
    a denoised image of the same shape.
    """
    def __init__(
        self,
        image_size,
        in_channels=3,
        num_res_blocks=0,
        attention_resolutions=(8, 16, 32),
        num_basis = (1, 2, 4, 8),
        dropout=0.1,
        dims=2,
        num_heads=4,
        num_head_channels=64,
        use_conv_resample=True,
        resblock_updown=True,
        use_new_attention_order=True,
        eta_base=None,
        n_iters_inter = None,
        n_iters_intra = None,
        kernel_size = None,
        stride = None,
    ):
        super().__init__()
        self.image_size = image_size
        out_channels = in_channels
        self.out_channels =out_channels
        self.in_channels = in_channels

        # build list of resolutions at which to apply attention
        attn_ds = []
        for r in attention_resolutions:
            attn_ds.append(image_size // r)

        # head for input
        ch = input_ch = int(num_basis[0])
        self.input_blocks = nn.ModuleList([
            nn.Sequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        ds = 1
        # downsampling path
        for level, mult in enumerate(num_basis):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlockNoTime(
                        ch,
                        dropout,
                        out_channels=int(mult),
                        dims=dims,
                        use_conv=False,
                    )
                ]
                ch = int(mult)
                # if ds in attn_ds:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
            if level != len(num_basis) - 1:
                out_ch = ch
                if resblock_updown:
                    down_layer = ResBlockNoTime(
                        ch, dropout, out_channels=out_ch,
                        dims=dims, down=True
                    )
                else:
                    down_layer = Downsample(ch, use_conv_resample, dims=dims, out_channels=out_ch)
                self.input_blocks.append(nn.Sequential(down_layer))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # bottleneck
        self.middle_block = nn.Sequential(
            ResBlockNoTime(ch, dropout, dims=dims),
            # AttentionBlock(
            #     ch,
            #     num_heads=num_heads,
            #     num_head_channels=num_head_channels,
            #     use_new_attention_order=use_new_attention_order,
            # ),
            ResBlockNoTime(ch, dropout, dims=dims),
        )

        # upsampling path
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(num_basis))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockNoTime(
                        ch + ich,
                        dropout,
                        out_channels=int(mult),
                        dims=dims,
                        use_conv=False,
                    )
                ]
                ch = int(mult)
                # if ds in attn_ds:
                #     layers.append(
                #         AttentionBlock(
                #             ch,
                #             num_heads=num_heads,
                #             num_head_channels=num_head_channels,
                #             use_new_attention_order=use_new_attention_order,
                #         )
                #     )
                if level and i == num_res_blocks:
                    if resblock_updown:
                        up_layer = ResBlockNoTime(
                            ch, dropout,
                            out_channels=ch,
                            dims=dims,
                            up=True
                        )
                    else:
                        up_layer = Upsample(ch, use_conv_resample, dims=dims)
                    layers.append(up_layer)
                    ds //= 2
                self.output_blocks.append(nn.Sequential(*layers))

        # final conv
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def forward(self, x):
        """
        :param x: [B, C, ...] image tensor
        :return: [B, C, ...] denoised image
        """
        hs = []
        h = x
        # down
        for blk in self.input_blocks:
            h = blk(h)
            hs.append(h)
        # bottleneck
        h = self.middle_block(h)
        # up
        for blk in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = blk(h)
        return self.out(h)


class OneLayerAE_MinWithNoise(nn.Module):
    """
    x -> enc0 -> E1 -> (+ noise_emb) -> ReLU -> D1 -> dec0

    Embedding path (matches your snippet order):
      emb = map_noise(noise_labels)
      emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)  # swap sin/cos
      if map_label:   emb += map_label( (maybe-dropped) class_labels * sqrt(in_features) )
      if map_augment: emb += map_augment(augment_labels)
      emb = ReLU(map_layer0(emb))
      noise_emb = affine(emb)[:, :, None, None]  # broadcast to h1
    """
    def __init__(
        self,
        in_channels,
        num_basis,
        whiten_dim: int = None,
        kernel_size: int = 7,
        stride: int = 2,
        channel_mult_emb: int = 2,
        channel_mult_noise: int = 1,
        embedding_type: str = "positional",
        init_lambda: float = 0.0,
        noise_embedding: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        # --- Outer linear encoder/decoder (like whitening/unwhitening) ---
        # print(whiten_dim)
        self.noise_embedding=noise_embedding
        if whiten_dim is not None:
            self.enc0 = Conv2d(in_channels=in_channels, out_channels=whiten_dim, kernel=3)
            self.dec0 = Conv2d(in_channels=whiten_dim, out_channels=in_channels, kernel=3)
        else:
            self.enc0 = nn.Identity()
            self.dec0 = nn.Identity()
            whiten_dim = in_channels
        k1 = kernel_size
        # --- One spatial encode/decode (tied) ---
        p1 = (k1 - 1) // 2
        c1 = num_basis[0]
        
        self.E1  = nn.Conv2d(whiten_dim, c1, kernel_size=k1, stride=stride, padding=p1, bias=bias)
        # nn.init.constant_(self.E1.bias, -init_lambda)
        # single_bias_term:
        bias_term = nn.Parameter(torch.zeros(1,1,1,1))
        nn.init.constant_(bias_term, -init_lambda)
        self.register_parameter("bias_term", bias_term)


        # We'll compute output_padding dynamically in forward to match x0's H,W.
        self.D1  = nn.ConvTranspose2d(
            in_channels=c1,
            out_channels=whiten_dim,
            kernel_size=k1,
            stride=stride,
            padding=p1,
            bias=False,
            output_padding=0  # placeholder; we'll override per-batch at runtime
        )
        self.act1 = nn.ReLU(inplace=True)

        # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
        emb_channels   = num_basis[0] * channel_mult_emb
        noise_channels = num_basis[0] * channel_mult_noise
        if self.noise_embedding:
            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.affine     = nn.Linear(emb_channels, c1, bias=True)


    @torch.no_grad()
    def _maybe_dropout_labels(self, class_labels: torch.Tensor) -> torch.Tensor:
        if self.training and self.label_dropout and class_labels is not None:
            keep = (torch.rand([class_labels.shape[0], 1], device=class_labels.device) >= self.label_dropout).to(class_labels.dtype)
            return class_labels * keep
        return class_labels

    def _deconv_match(self, h: torch.Tensor, target_hw: tuple) -> torch.Tensor:
        """
        Do ConvTranspose2d but compute the per-dimension output_padding on the fly
        so the output H,W match `target_hw` exactly, regardless of stride=1 or 2 (even/odd sizes).
        """
        th, tw = target_hw
        sh_h, sh_w = self.D1.stride if isinstance(self.D1.stride, tuple) else (self.D1.stride, self.D1.stride)
        ph_h, ph_w = self.D1.padding if isinstance(self.D1.padding, tuple) else (self.D1.padding, self.D1.padding)
        dh_h, dh_w = self.D1.dilation if isinstance(self.D1.dilation, tuple) else (self.D1.dilation, self.D1.dilation)
        kh_h, kh_w = self.D1.kernel_size if isinstance(self.D1.kernel_size, tuple) else (self.D1.kernel_size, self.D1.kernel_size)

        Hin, Win = h.shape[-2], h.shape[-1]
        # Formula: H_out = (Hin-1)*s - 2p + d*(k-1) + 1 + opad
        base_h = (Hin - 1) * sh_h - 2 * ph_h + dh_h * (kh_h - 1) + 1
        base_w = (Win - 1) * sh_w - 2 * ph_w + dh_w * (kh_w - 1) + 1
        opad_h = th - base_h
        opad_w = tw - base_w

        # Sanity: 0 <= output_padding < stride
        if isinstance(self.D1.stride, tuple):
            max_h, max_w = sh_h, sh_w
        else:
            max_h = max_w = self.D1.stride
        # Clamp-to-valid just in case numerical weirdness appears
        opad_h = int(max(0, min(opad_h, max_h - 1)))
        opad_w = int(max(0, min(opad_w, max_w - 1)))

        return F.conv_transpose2d(
            h, self.D1.weight, bias=None,
            stride=self.D1.stride, padding=self.D1.padding,
            output_padding=(opad_h, opad_w),
            groups=self.D1.groups, dilation=self.D1.dilation
        )

    def forward(self, x,
                 noise_labels=None):
        B = x.shape[0]

        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:    
            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = 0

        # --- One-layer encode/decode with noise injected BEFORE ReLU ---
        x0 = self.enc0(x)
        preact = self.E1(x0)
        h1 = self.act1(preact + noise_emb + self.bias_term)

        # *** Key change: deconv to match x0's H,W with dynamic output_padding ***
        x0_hat = self._deconv_match(h1, target_hw=x0.shape[-2:])

        y = self.dec0(x0_hat)

        return y
        
    def infer(self, x, noise_labels=None,n_iter=1,return_feature=False,return_hist=False):
        B = x.shape[0]
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)
        return self.forward(x, noise_labels=noise_labels)




def UNetBlind64(**kwargs):
    return UNetBlindDenoise(image_size=64, **kwargs)

