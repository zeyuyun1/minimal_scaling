from ast import If
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from einops import rearrange
from solver import broyden,naive_solver
from optimizer import weight_norm

def _gauss_kernel2d(ks: int, sigma: float, device=None, dtype=None):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2.0
    g1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1 = g1 / g1.sum()
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    return g2.view(1, 1, ks, ks)

class GaussianPyramidEncoder(nn.Module):
    """
    Multi-level Gaussian/Laplacian analysis.
    Outputs residuals r_s and final base b_L. Optionally upsamples
    everything to the input resolution and concatenates along channels.
    """
    def __init__(self, levels=2, kernel_size=5, sigma=1.2,
                 concat_to_channels=True, learnable=False, reflect_pad=True):
        super().__init__()
        assert levels >= 1
        self.levels = levels
        self.ks = int(kernel_size)
        self.sigma = float(sigma)
        self.concat = concat_to_channels
        self.reflect_pad = reflect_pad

        # Depthwise blur conv (same weights per channel)
        self.register_buffer('k2d', _gauss_kernel2d(self.ks, self.sigma))
        self.learnable = learnable
        if learnable:
            # Make kernel learnable but normalized at run time
            self.k2d_param = nn.Parameter(self.k2d.clone())
        else:
            self.k2d_param = None

    def _blur(self, x):
        B, C, H, W = x.shape
        k = self.k2d_param if self.learnable else self.k2d
        k = k.to(dtype=x.dtype, device=x.device)
        k = k.repeat(C, 1, 1, 1)  # depthwise
        pad = self.ks // 2
        if self.reflect_pad:
            x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            pad = 0
        return F.conv2d(x, k, padding=pad, groups=x.size(1))

    def forward(self, x):
        """
        x: [B, C, H, W]
        Returns:
          pyr: list [r0, r1, ..., bL] with native resolutions
          feat: [B, C*(levels+1), H, W] if concat_to_channels=True
                (order: r0, up(r1), ..., up(bL))
        """
        B, C, H, W = x.shape
        xs = x
        residuals = []
        bases = []

        # Build Gaussian pyramid bases b_1..b_L and residuals r_0..r_{L-1}
        for s in range(self.levels):
            # blur + decimate
            b_next = F.avg_pool2d(self._blur(xs), kernel_size=2, stride=2)
            # upsample back to current scale
            up_next = F.interpolate(b_next, size=xs.shape[-2:], mode='bilinear', align_corners=False)
            # Laplacian residual at this scale
            r_s = xs - up_next
            residuals.append(r_s)
            bases.append(b_next)
            xs = b_next

        bL = xs  # final base
        pyr = residuals + [bL]

        if not self.concat:
            return pyr

        # Upsample all to input resolution and concat along channels
        ups = []
        for r_s in residuals:
            if r_s.shape[-2:] != (H, W):
                r_s = F.interpolate(r_s, size=(H, W), mode='bilinear', align_corners=False)
            ups.append(r_s)
        bL_up = F.interpolate(bL, size=(H, W), mode='bilinear', align_corners=False)
        ups.append(bL_up)
        feat = torch.cat(ups, dim=1)  # [B, C*(levels+1), H, W]
        return feat

    def decode(self, pyr):
        """
        Invert the Laplacian/ Gaussian pyramid and reconstruct the image.

        Args:
            pyr: list [r0, r1, ..., bL]
                 r_s: residual at scale s (native resolution of that scale)
                 bL: final Gaussian base at coarsest scale

        Returns:
            x_recon: [B, C, H, W] reconstructed image at finest resolution.
        """
        assert isinstance(pyr, (list, tuple)), "pyr must be a list/tuple of tensors"
        assert len(pyr) == self.levels + 1, f"Expected {self.levels+1} tensors, got {len(pyr)}"

        residuals = pyr[:-1]  # [r0, r1, ..., r_{L-1}]
        bL = pyr[-1]          # final base

        # Start from coarsest base
        current = bL  # Gaussian at level L

        # Go from coarsest residual back to finest
        for r_s in reversed(residuals):
            # Upsample current Gaussian to residual's spatial size
            if current.shape[-2:] != r_s.shape[-2:]:
                current_up = F.interpolate(
                    current,
                    size=r_s.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            else:
                current_up = current

            # Reconstruct Gaussian at this finer level:
            # xs = r_s + upsample(b_next)
            current = r_s + current_up

        # `current` is now the finest-level Gaussian, which is the original x
        return current

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

###############################################################################
# Recurrent Convolutional Unit with Output Padding in the Decoder
###############################################################################

class RecurrentConvUnit(nn.Module):
    """
    Convolutional recurrent unit, now with output_padding added to the decoder
    to ensure that the reconstructed feedback has matching spatial dimensions.
    
    The update rule is:
    
        a_{t+1} = ReLU( a_t + η * ( encoder(x) + M(a_t) + top_signal ) )
        
    where:
      - encoder(x) is the feedforward drive (implemented as a conv with stride 2).
      - M is a local (1x1, group) convolution.
      - top_signal is a top-down feedback signal.
    """
    def __init__(self, in_channels, num_basis, kernel_size=7, stride=2,
                 padding=3, eta=0.5, init_lambda=0.1, output_padding=1):
        super(RecurrentConvUnit, self).__init__()
        # Convolutional dictionary encoder.
        self.encoder = nn.Conv2d(
            in_channels, num_basis,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        nn.init.constant_(self.encoder.bias, -init_lambda)
        
        # Horizontal / lateral connection: using a 1x1 grouped convolution.
        self.M = nn.Conv2d(
            num_basis, num_basis,
            kernel_size=3,
            padding =1,
            bias=False,
            # groups=num_basis//4
        )
        # torch.nn.init.dirac_(self.M.weight)
        torch.nn.init.constant_(self.M.weight, 0)
        
        # Tied transpose convolution for decoding. The added output_padding ensures
        # that the reconstructed (decoded) tensor matches the dimensions of a_prev.
        self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        self.eta = eta
        self.relu = nn.ReLU()

    def forward(self, x, a_prev=None, top_signal=None, noise_emb=None):
        """
        Forward pass that updates the latent variable.
        
        Args:
          x: Input activation (or lower-layer feature map).
          a_prev: Previous latent variable (if None, initialize from feedforward drive).
          top_signal: Top-down feedback signal (must match the shape of the latent code).
        """
        # Use zero feedback if none provided.
        feedback = top_signal if top_signal is not None else 0
        noise_emb = noise_emb if noise_emb is not None else 0
        
        if a_prev is None:
            # Initial iteration: use only feedforward drive.
            a = self.relu(self.eta * (self.encoder(x) + feedback + noise_emb))
        else:
            # Otherwise, update the previous state.
            update = self.encoder(x) + self.M(a_prev)
            a = self.relu(a_prev + self.eta * (update + feedback + noise_emb))
        
        # Decode (reconstruct) from the latent representation.
        decoded = self.decoder(a)
        return a, decoded

class RecurrentConvUnit_cc(nn.Module):
    """
    Convolutional recurrent unit, now with output_padding added to the decoder
    to ensure that the reconstructed feedback has matching spatial dimensions.
    
    The update rule is:
    
        a_{t+1} = ReLU( a_t + η * ( encoder(x) + M(a_t) + top_signal ) )
        
    where:
      - encoder(x) is the feedforward drive (implemented as a conv with stride 2).
      - M is a local (1x1, group) convolution.
      - top_signal is a top-down feedback signal.
    """
    def __init__(self, in_channels, num_basis, kernel_size=7, stride=2,
                 padding=3, eta=0.5, init_lambda=0.1, output_padding=1):
        super(RecurrentConvUnit_cc, self).__init__()
        # Convolutional dictionary encoder.
        self.encoder = nn.Conv2d(
            in_channels, num_basis,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        nn.init.constant_(self.encoder.bias, -init_lambda)
        
        # Horizontal / lateral connection: using a 1x1 grouped convolution.
        self.M = nn.Conv2d(
            num_basis, num_basis,
            kernel_size=3,
            padding =1,
            bias=False,
            # groups=num_basis//4
        )
        # init M's weight to 0
        torch.nn.init.constant_(self.M.weight, 0)
        
        # Tied transpose convolution for decoding. The added output_padding ensures
        # that the reconstructed (decoded) tensor matches the dimensions of a_prev.
        self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        self.eta = eta
        self.relu = nn.ReLU()

    def forward(self, x, a_prev=None, top_signal=None):
        """
        Forward pass that updates the latent variable.
        
        Args:
          x: Input activation (or lower-layer feature map).
          a_prev: Previous latent variable (if None, initialize from feedforward drive).
          top_signal: Top-down feedback signal (must match the shape of the latent code).
        """
        # Use zero feedback if none provided.
        feedback = top_signal if top_signal is not None else 0
        
        if a_prev is None:
            # Initial iteration: use only feedforward drive.
            a = self.relu(self.eta * (self.encoder(x) + feedback))
        else:
            # Otherwise, update the previous state.
            update = self.encoder(x) + self.M(a_prev)
            a = self.relu(a_prev + self.eta * (update + feedback))
        
        # Decode (reconstruct) from the latent representation.
        decoded = self.decoder(a)
        return a, decoded

class RecurrentConvUnit_gram(nn.Module):
    """
    Convolutional recurrent unit, now with output_padding added to the decoder
    to ensure that the reconstructed feedback has matching spatial dimensions.
    
    The update rule is:
    
        a_{t+1} = ReLU( a_t + η * ( encoder(x) + M(a_t) + top_signal ) )
        
    where:
      - encoder(x) is the feedforward drive (implemented as a conv with stride 2).
      - M is a local (1x1, group) convolution.
      - top_signal is a top-down feedback signal.
    """
    def __init__(self, in_channels, num_basis, kernel_size=7, stride=2,
                 padding=3, eta=0.5, init_lambda=0.0, output_padding=1,learning_horizontal=True, groups=1, bias=True, relu_6=False, wnorm=True, h_groups =1):
        super(RecurrentConvUnit_gram, self).__init__()
        # Convolutional dictionary encoder.
        self.encoder = nn.Conv2d(
            in_channels, num_basis,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            groups=groups,
        )
        if bias:
            nn.init.constant_(self.encoder.bias, -init_lambda)
        if learning_horizontal:
            # Horizontal / lateral connection: using a 1x1 grouped convolution.
            self.M = nn.Conv2d(
                num_basis, num_basis,
                kernel_size=3,
                padding =1,
                bias=False,
                groups=h_groups,
                # groups=num_basis//4
            )
            # torch.nn.init.dirac_(self.M.weight)
            # torch.nn.init.constant_(self.M.weight, 0)
        
        # Tied transpose convolution for decoding. The added output_padding ensures
        # that the reconstructed (decoded) tensor matches the dimensions of a_prev.
        self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        self.eta = eta
        if relu_6:
            # print("Using ReLU6")
            self.relu = nn.ReLU6()
        else:
            self.relu = nn.ReLU()
        self.learning_horizontal = learning_horizontal
        self._wnorm = bool(wnorm)
        if self._wnorm:
            # normalize per-output-channel: dim=0
            self.encoder, self._enc_wn = weight_norm(self.encoder, names=['weight'], dim=0)
            if self.learning_horizontal:
                self.M, self._M_wn = weight_norm(self.M, names=['weight'], dim=0)

    def reset_wnorm(self):
        """Recompute normalized weights once per batch/forward."""
        if not getattr(self, '_wnorm', False):
            return
        self._enc_wn.reset(self.encoder)
        if self.learning_horizontal:
            self._M_wn.reset(self.M)

    def forward(self, x, a_prev=None, top_signal=None, noise_emb=None, T = 0):
        """
        Forward pass that updates the latent variable.
        
        Args:
          x: Input activation (or lower-layer feature map).
          a_prev: Previous latent variable (if None, initialize from feedforward drive).
          top_signal: Top-down feedback signal (must match the shape of the latent code).
        """
        self.reset_wnorm()
        # Use zero feedback if none provided.
        feedback = top_signal if top_signal is not None else 0
        noise_emb = noise_emb if noise_emb is not None else 0
        
        if a_prev is None:
            # Initial iteration: use only feedforward drive.
            # note I changed this to start at full FF
            a = self.relu(self.eta * (self.encoder(x) + feedback + noise_emb))
        else:
            # Otherwise, update the previous state.
            # update = self.encoder(x) + self.M(a_prev)
            update = self.encoder(x - self.decoder(a_prev)) - (self.M(a_prev) if self.learning_horizontal else 0)
            a = self.relu(a_prev + self.eta * (update + feedback + noise_emb) + np.sqrt(self.eta) * torch.randn_like(a_prev)*T)
        
        # Decode (reconstruct) from the latent representation.
        decoded = self.decoder(a)
        return a, decoded


class RecurrentConvUnit_diverse(nn.Module):
    """
    We define the energy coupling between previous variable and current variable.
    Given the energy coupling, we define the corresponding update rule.

    Convolutional recurrent unit, now with output_padding added to the decoder
    to ensure that the reconstructed feedback has matching spatial dimensions.
    Let input be x, latent state be a
    Several energy function and update mode:
        Positive sparse coding: 
        E(x, a) = ||x - \Phi a||_2^2 + \lambda(\sigma) ||a||_1, a>0
        Positive (diverse) elastic net (I think none-zero make this still convex?): 
        E(x, a) = ||x - \Phi a||_2^2 + \lambda(\sigma)  ||a||_1 + a^T M a, where M is a diagonal matrix, a>0
        Boltzmann machine (non-convex):
        E(x, a) = - a^T \Phi^T x + a^T M a + \lambda(\sigma) ||a||_1 , a>0 where M can be any matrix
        Hybrid energy:
        E(x, a) = ||x - \Phi a||_2^2 + \lambda(\sigma)  ||a||_1 + a^T M a, where M is a diagonal matrix, a>0
    where:
      - encoder(x) is the feedforward drive (implemented as a conv with stride 2).
      - M is a local (1x1, group) convolution.
      - top_signal is a top-down feedback signal.
    """
    def __init__(self, in_channels, num_basis, kernel_size=7, stride=2,
                 padding=3, eta=0.5, init_lambda=0.0, output_padding=1,
                 learning_horizontal=True, groups=1, bias=False, relu_6=True, 
                 wnorm=True, h_groups =1, energy_function = "SC"):
        super(RecurrentConvUnit_diverse, self).__init__()
        # Convolutional dictionary encoder.
        # print(in_channels,groups)
        self.encoder = nn.Conv2d(
            in_channels, num_basis,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )

        self.energy_function = energy_function
        if energy_function  == "elastic":
            self.M = nn.Conv2d(
                num_basis, num_basis,
                kernel_size=1,
                padding =0,
                bias=False,
                groups=num_basis,
                # groups=num_basis//4
            )
        elif energy_function in ["BM","hybrid"]:
            self.M = nn.Conv2d(
                num_basis, num_basis,
                kernel_size=3,
                padding =1,
                bias=False,
                groups=h_groups,
                # groups=num_basis//4
            )
        else:
            self.M = None

        # Tied transpose convolution for decoding. The added output_padding ensures
        # that the reconstructed (decoded) tensor matches the dimensions of a_prev.
        self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        self.eta = eta
        if relu_6:
            # print("Using ReLU6")
            self.relu = nn.ReLU6()
        else:
            self.relu = nn.ReLU()
        self.learning_horizontal = learning_horizontal
        self._wnorm = bool(wnorm)
        if self._wnorm:
            # normalize per-output-channel: dim=0
            self.encoder, self._enc_wn = weight_norm(self.encoder, names=['weight'], dim=0)
            if self.learning_horizontal:
                self.M, self._M_wn = weight_norm(self.M, names=['weight'], dim=0)

    def reset_wnorm(self):
        """Recompute normalized weights once per batch/forward."""
        if not getattr(self, '_wnorm', False):
            return
        self._enc_wn.reset(self.encoder)
        if self.learning_horizontal:
            self._M_wn.reset(self.M)

    def forward(self, x, a_prev=None, top_signal=None, noise_emb=None, T = 0):
        """
        Forward pass that updates the latent variable.
        
        Args:
          x: Input activation (or lower-layer feature map).
          a_prev: Previous latent variable (if None, initialize from feedforward drive).
          top_signal: Top-down feedback signal (must match the shape of the latent code).
        """
        self.reset_wnorm()
        # Use zero feedback if none provided.
        feedback = top_signal if top_signal is not None else 0
        noise_emb = noise_emb if noise_emb is not None else 0
        
        if a_prev is None:
            # Initial iteration: use only feedforward drive.
            # note I changed this to start at full FF
            a = self.relu(self.eta * (self.encoder(x) + feedback + noise_emb))
        else:
            # Otherwise, update the previous state.
            if self.energy_function in ["BM","BM_h_order"]:
                update = self.encoder(x) - self.M(a_prev)
            elif self.energy_function in ["elastic","hybrid"]:
                update = self.encoder(x - self.decoder(a_prev)) - self.M(a_prev)
            else:
                update = self.encoder(x - self.decoder(a_prev))
            a = self.relu(a_prev + self.eta * (update + feedback + noise_emb) + np.sqrt(self.eta) * torch.randn_like(a_prev)*T)
        # Decode (reconstruct) from the latent representation.
        decoded = self.decoder(a)
        # if self.energy_function in ["BM"]:
        #     decoded = self.decoder(a)
        # else:
        #     decoded = x - self.decoder(a)
        return a, decoded

class MultiScaleRecurrentConvUnit_diverse(nn.Module):
    """
    Multi-scale recurrent convolutional unit.

    Inputs:
      x: list of tensors [x_0, x_1, ..., x_{S-1}]
         where x_s has shape [B, C_in, H_s, W_s],
         and typically H_{s+1} ~ H_s / 2, W_{s+1} ~ W_s / 2.
      a_prev: list of latents with same shapes as encoded outputs at each scale,
              or None for initialization.

    We keep:
      - A list of encoders/decoders (one pair per scale).
      - A single M acting on the concatenated, upsampled latents.

    Horizontal term:
      1. Upsample all a_s to the finest scale (s=0).
      2. Concatenate along channel -> [B, S * num_basis, H_0, W_0].
      3. Apply M (group conv or full conv).
      4. Split back into S chunks and downsample each to its original scale.

    If you want horizontal interaction only within each scale:
      - Use groups = num_scales in M (each group sees num_basis channels).
    If you want interaction across scales:
      - Use groups = 1.
    """

    def __init__(
        self,
        in_channels,
        num_basis,
        num_scales,
        num_basis_per_scale = None,
        kernel_size=7,
        stride=2,
        padding=3,
        eta=0.5,
        init_lambda=0.0,           # kept for interface consistency
        output_padding=1,
        learning_horizontal=True,
        groups=1,             # groups for encoders
        bias=False,
        relu_6=True,
        wnorm=True,
        h_groups=1,                # groups for M: 1 (full) or num_scales (per-scale)
        energy_function="SC"       # "SC", "elastic", "BM", "hybrid", ...
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_basis = num_basis
        self.num_scales = num_scales
        self.eta = eta
        self.energy_function = energy_function
        self.learning_horizontal = learning_horizontal
        self._wnorm = bool(wnorm)

        # -------------------------
        # Encoders/decoders per scale
        # -------------------------
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self._enc_wn = []  # weight-norm handles per encoder
        if num_basis_per_scale is None:
            num_basis_per_scale = [num_basis//num_scales] * num_scales

        for s in range(num_scales):
            enc = nn.Conv2d(
                in_channels,
                num_basis_per_scale[s],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
                groups=groups,
            )
            if self._wnorm:
                enc, wn_handle = weight_norm(enc, names=['weight'], dim=0)
                self._enc_wn.append(wn_handle)
            self.encoders.append(enc)

            # tied transpose decoder for each encoder
            dec = TiedTransposeConv(enc, output_padding=output_padding)
            self.decoders.append(dec)

        # -------------------------
        # Horizontal M (single conv in concatenated space)
        # -------------------------
        if energy_function == "elastic":
            # diagonal-ish: 1x1 conv, group = num_scales or 1
            self.M = nn.Conv2d(
                num_basis,
                num_basis,
                kernel_size=1,
                padding=0,
                bias=False,
                groups=num_basis,
            )
        elif energy_function in ["BM", "hybrid", "BM_h_order"]:
            # general 3x3 conv in concatenated space
            self.M = nn.Conv2d(
                num_basis,
                num_basis,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=h_groups,
            )
        else:
            self.M = None

        if self._wnorm and self.M is not None and self.learning_horizontal:
            self.M, self._M_wn = weight_norm(self.M, names=['weight'], dim=0)
        else:
            self._M_wn = None

        # print(self.M)

        # -------------------------
        # Nonlinearity
        # -------------------------
        self.relu = nn.ReLU6() if relu_6 else nn.ReLU()

    # -------------------------
    # Weight-norm reset
    # -------------------------
    def reset_wnorm(self):
        if not self._wnorm:
            return
        # encoders
        for enc, wn_handle in zip(self.encoders, self._enc_wn):
            wn_handle.reset(enc)
        # M
        if self.learning_horizontal and self.M is not None and self._M_wn is not None:
            self._M_wn.reset(self.M)

    # -------------------------
    # Helper: encode / decode / difference
    # -------------------------
    def encode(self, x_list):
        """Apply encoders per scale."""
        return [enc(x_s) for x_s, enc in zip(x_list, self.encoders)]

    def decode(self, a_list):
        """Apply decoders per scale."""
        return [dec(a_s) for a_s, dec in zip(a_list, self.decoders)]

    def difference(self, x_list, decoded_list):
        """Compute x_s - decoded_s at each scale."""
        return [x_s - d_s for x_s, d_s in zip(x_list, decoded_list)]

    # -------------------------
    # Helper: upsample all scales to finest + concat
    # -------------------------
    def _upsample_concat(self, a_list):
        """
        Upsample all a_s to the finest resolution (scale 0) and concatenate.
        Assumes a_list[0] is the finest scale.
        """
        B, C, H0, W0 = a_list[0].shape
        ups = [a_list[0]]
        for s in range(1, self.num_scales):
            ups.append(
                F.interpolate(
                    a_list[s],
                    size=(H0, W0),
                    mode="bilinear",
                    align_corners=False,
                )
            )
        return torch.cat(ups, dim=1)  # [B, S * num_basis, H0, W0]

    # -------------------------
    # Helper: split concatenated tensor and downsample per scale
    # -------------------------
    def _split_downsample(self, tensor, shapes):
        """
        tensor: [B, S * num_basis, H0, W0]
        shapes: list of (H_s, W_s) for each scale's target size.
        """
        B, C, H0, W0 = tensor.shape
        # S = self.num_scales
        S = len(shapes)
        C_per = C // S
        out = []
        c_total = 0
        # print(shapes)
        for s in range(S):
            # print(shapes[s])
            c, H_s, W_s = shapes[s]
            t_s = tensor[:, c_total:c_total+c, :, :]
            if (H_s, W_s) != (H0, W0):
                t_s = F.interpolate(
                    t_s,
                    size=(H_s, W_s),
                    mode="bilinear",
                    align_corners=False,
                )
            out.append(t_s)
        return out

    # -------------------------
    # Helper: horizontal term M(a)
    # -------------------------
    def horizontal(self, a_list):
        """
        Compute horizontal interaction term M(a) at each scale.
        Returns a list of tensors with same shapes as a_list.
        """
        if self.M is None or not self.learning_horizontal:
            print("No horizontal interaction")
            return [0.0 for _ in a_list]

        a_cat = self._upsample_concat(a_list)           # [B, S*C, H0, W0]
        h_cat = self.M(a_cat)                           # same shape
        shapes = [a.shape[1:] for a in a_list]
        # print(shapes)
        h_list = self._split_downsample(h_cat, shapes)  # list of [B, C, H_s, W_s]
        return h_list

    # -------------------------
    # Utility: get per-scale top/noise
    # -------------------------
    def _get_scale_item_or_zero(self, item, s, device):
        if item is None:
            return 0.0
        if isinstance(item, (list, tuple)):
            return item[s] if item[s] is not None else 0.0
        # same top_signal/noise_emb for all scales
        return item

    # -------------------------
    # Forward (one recurrent step)
    # -------------------------
    def forward(self, x_list, a_prev=None, top_signal=None, noise_emb=None, T=0.0):
        """
        x_list: list of inputs [x_0, x_1, ..., x_{S-1}]
        a_prev: list of previous latents of same length, or None
        top_signal: list or tensor, same shapes as encodings
        noise_emb: list or tensor, same shapes as encodings
        """
        self.reset_wnorm()

        # ---------------------
        # Initialization: feedforward only
        # ---------------------
        if a_prev is None:
            ff = self.encode(x_list)  # list of enc(x_s)
            a_list = []
            for s in range(self.num_scales):
                fb_s = self._get_scale_item_or_zero(top_signal, s, ff[s].device)
                nz_s = self._get_scale_item_or_zero(noise_emb, s, ff[s].device)
                a_s = self.relu(self.eta * (ff[s] + fb_s + nz_s[s]))
                a_list.append(a_s)

        # ---------------------
        # Recurrent update
        # ---------------------
        else:
            # horizontal interaction
            # print("Horizontal interaction")
            h_list = self.horizontal(a_prev)  # list of same shapes as a_prev

            # decode previous latent and compute difference if needed
            decoded_prev = self.decode(a_prev)
            diff_list = self.difference(x_list, decoded_prev)  # x_s - dec(a_prev_s)

            a_list = []
            for s in range(self.num_scales):
                x_s = x_list[s]
                a_prev_s = a_prev[s]
                fb_s = self._get_scale_item_or_zero(top_signal, s, x_s.device)
                nz_s = self._get_scale_item_or_zero(noise_emb, s, x_s.device)

                if self.energy_function in ["BM", "BM_h_order"]:
                    # Boltzmann-like update: encoder(x) - M(a_prev)
                    update_s = self.encoders[s](x_s) - h_list[s]
                elif self.energy_function in ["hybrid" or "elastic"]:
                    # hybrid: reconstruction term + M
                    update_s = self.encoders[s](diff_list[s]) - h_list[s]
                else:  # "SC" or default sparse coding
                    update_s = self.encoders[s](diff_list[s])

                a_s = a_prev_s + self.eta * (update_s + fb_s + nz_s)

                if T != 0.0:
                    a_s = a_s + np.sqrt(self.eta) * torch.randn_like(a_s) * T

                a_s = self.relu(a_s)
                a_list.append(a_s)

        # ---------------------
        # Decode from current latent
        # ---------------------
        decoded_list = []
        for s in range(self.num_scales):
            decoded_s = self.decoders[s](a_list[s])
            decoded_list.append(decoded_s)

        return a_list, decoded_list


class RecurrentConvNLayer(nn.Module):
    """
    n‑layer hierarchical convolutional sparse coding model with top‑down feedback.

    Parameters:
      - in_channels:    Number of channels in the input (e.g. 1 for grayscale).
      - num_basis:      Sequence of latent‐channel sizes, length = n.
      - eta:            Step size for all levels.
      - n_iters_inter:  Number of unrolled inference iterations.
      - kernel_size, stride, padding, output_padding: conv params for every level.
    """
    def __init__(self,
                 in_channels: int,
                 num_basis: list[int],
                 eta_base: float = None,
                 n_iters_inter: int = 2,
                 n_iters_intra: int = 1,
                 kernel_size: int = 5,
                 stride: int = 2,
                 output_padding: int = 1,
                 whiten_dim=None,):
        super().__init__()
        self.n_iters_inter = n_iters_inter
        n_iters_train = n_iters_inter*n_iters_intra
        self.n_levels = len(num_basis)
        self.eta = eta_base/n_iters_train
        # assert n_iters_train % n_iters_intra == 0, "n_iters_train must be divisible by n_iters_intra"
        self.n_iters_inter = n_iters_inter
        self.n_iters_intra = n_iters_intra
        self.whiten_dim = whiten_dim
        if whiten_dim is not None:
            self.encoder = Conv2d(
                in_channels=in_channels,
                out_channels=whiten_dim,
                kernel=3)
            
            self.decoder = Conv2d(
                in_channels=whiten_dim,
                out_channels=in_channels,
                kernel=3)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels
        
        # build a ModuleList of RecurrentConvUnit, chaining channels
        levels = []
        for i,nb in enumerate(num_basis):
            levels.append(
                RecurrentConvUnit(
                    in_channels=prev_channels,
                    num_basis=nb,
                    kernel_size=kernel_size if i>0 else 7,
                    stride=stride,
                    padding=kernel_size//2 if i>0 else 3,
                    eta=self.eta,
                    output_padding=output_padding
                )
            )
            prev_channels = nb
        self.levels = nn.ModuleList(levels)

    def forward(self, x, noise_labels=None):
        # initialize latent activations & decodings
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels
        x = self.encoder(x)

        for i_out in range(self.n_iters_inter):
            self.forward_inter(x, a, decoded, i_out)

        # final reconstruction from the first level
        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, i_out=None):
        for i in range(len(self.levels)):
            inp = x if i == 0 else a[i-1]
            top_signal = decoded[i+1] if i < self.n_levels-1 else None
            a_cur = a[i]
            for j in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
                if i_out==0:
                    break
            a[i], decoded[i] = a_cur, decoded_cur

        # top‑down refinement (exclude the topmost level)
        for i in range(self.n_levels-1, -1, -1):
            inp = x if i == 0 else a[i-1]
            top_signal = decoded[i+1] if i < self.n_levels-1 else None
            a_cur = a[i]
            for j in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
                if i_out==0:
                    break
            a[i], decoded[i] = a_cur, decoded_cur


# class RecurrentConvNLayer2(nn.Module):
#     """
#     n‑layer hierarchical convolutional sparse coding model with top‑down feedback.

#     Parameters:
#       - in_channels:    Number of channels in the input (e.g. 1 for grayscale).
#       - num_basis:      Sequence of latent‐channel sizes, length = n.
#       - eta:            Step size for all levels.
#       - n_iters_inter:  Number of unrolled inference iterations.
#       - kernel_size, stride, padding, output_padding: conv params for every level.
#     """
#     def __init__(self,
#                  in_channels: int,
#                  num_basis: list[int],
#                  eta_base: float = None,
#                  n_iters_inter: int = 2,
#                  n_iters_intra: int = 1,
#                  kernel_size: int = 5,
#                  stride: int = 2,
#                  output_padding: int = 1,
#                  whiten_dim=None):
#         super().__init__()
#         self.n_iters_inter = n_iters_inter
#         n_iters_train = n_iters_inter*n_iters_intra
#         self.n_levels = len(num_basis)
#         self.eta = eta_base/n_iters_train
#         # assert n_iters_train % n_iters_intra == 0, "n_iters_train must be divisible by n_iters_intra"
#         self.n_iters_inter = n_iters_inter
#         self.n_iters_intra = n_iters_intra
#         self.whiten_dim = whiten_dim
#         if whiten_dim is not None:
#             self.encoder = Conv2d(
#                 in_channels=in_channels,
#                 out_channels=whiten_dim,
#                 kernel=3)
            
#             self.decoder = Conv2d(
#                 in_channels=whiten_dim,
#                 out_channels=in_channels,
#                 kernel=3)
#             prev_channels = whiten_dim
#         else:
#             self.encoder = nn.Identity()
#             self.decoder = nn.Identity()
#             prev_channels = in_channels
        
#         # build a ModuleList of RecurrentConvUnit, chaining channels
#         levels = []
#         for i,nb in enumerate(num_basis):
#             levels.append(
#                 RecurrentConvUnit(
#                     in_channels=prev_channels,
#                     num_basis=nb,
#                     kernel_size=kernel_size if i>0 else 7,
#                     stride=stride,
#                     padding=kernel_size//2 if i>0 else 3,
#                     eta=self.eta,
#                     output_padding=output_padding
#                 )
#             )
#             prev_channels = nb
#         self.levels = nn.ModuleList(levels)

#     # def forward(self, x):
#     #     # initialize latent activations & decodings
#     #     a = [None] * self.n_levels
#     #     decoded = [None] * self.n_levels
#     #     x = self.encoder(x)

#     #     for i_out in range(self.n_iters_inter):
#     #         # bottom‑up sweep
#     #         for i in range(len(self.levels)):
#     #             inp = x if i == 0 else a[i-1]
#     #             top_signal = decoded[i+1] if i < self.n_levels-1 else None
#     #             a_cur = a[i]
#     #             a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
#     #             a[i], decoded[i] = a_cur, decoded_cur

#     #         # top‑down refinement (exclude the topmost level)
#     #         for i in range(self.n_levels-1, -1, -1):
#     #             inp = x if i == 0 else a[i-1]
#     #             top_signal = decoded[i+1] if i < self.n_levels-1 else None
#     #             a_cur = a[i]
#     #             for j in range(self.n_iters_intra):
#     #                 a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
#     #             a[i], decoded[i] = a_cur, decoded_cur

#     #     # final reconstruction from the first level
#     #     return self.decoder(decoded[0])

#     def forward(self,x):
#         if self.training:
#             early_exit_iter = 1 + np.random.randint(self.n_iters_inter)
#             return self.forward_(x, n_iters = self.n_iters_inter, n_iters_g = early_exit_iter)
#         else:
#             return self.forward_(x, n_iters = int(self.n_iters_inter*1.5))
        
#     def forward_(self, x, a = None, decoded =None, n_iters_g=0, n_iters=1, return_feature = False, T = 0, return_hist = False):
        
#         if return_hist:
#             decoded_hist = [[] for _ in range(self.n_levels)]
#         else:
#             decoded_hist= None
        
#         # initialize latent activations & decodings
#         if a is None:
#             a = [None] * self.n_levels
#         if decoded is None:
#             decoded = [None] * self.n_levels
#         # initialization:
#         x = self.encoder(x)
#         with torch.no_grad():
#             for i in range(len(self.levels)):
#                 inp = x if i == 0 else a[i-1]
#                 top_signal = decoded[i+1] if i < self.n_levels-1 else None
#                 a_cur = a[i]
#                 for j in range(self.n_iters_intra):
#                     a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
#                 a[i], decoded[i] = a_cur, decoded_cur
#             # fix point update
#             self.fix_point_update(x,a,decoded,n_iters=n_iters,decoded_hist=decoded_hist)
#         self.fix_point_update(x,a,decoded,n_iters=n_iters_g,decoded_hist=decoded_hist)
                    
#         # final reconstruction from the first level
#         if return_feature:
#             if return_hist:
#                 return a, decoded, decoded_hist
#             else:
#                 return a, decoded
#         else:
#             return self.decoder(decoded[0])
#             # return decoded[0]

#     def fix_point_update(self, x, a, decoded, n_iters=None, T = 0, decoded_hist=None):
#         if n_iters is None or n_iters <= 0:
#             return
#         for _ in range(n_iters):
#             # top-down sweep
#             for i in range(self.n_levels-1, -1, -1):
#                 inp = x if i == 0 else a[i-1]
#                 top_signal = decoded[i+1] if i < self.n_levels-1 else None
#                 a_cur = a[i]
#                 for j in range(self.n_iters_intra):
#                     a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
#                 a[i], decoded[i] = a_cur, decoded_cur
class RecurrentConvNLayer2(nn.Module):
    def __init__(self, in_channels, num_basis,
                 n_iters_inter=1, n_iters_intra=1,eta_base=0.1,
                 kernel_size=5, stride=2, output_padding=1, whiten_dim=None,
                 # NEW: JFB controls
                 jfb_no_grad_iters=(0, 6),       # n in [0, N]
                 jfb_with_grad_iters=(1, 3),     # m in [1, M]
                 jfb_reuse_solution=False,
                 jfb_ddp_safe=True,
                 ):
        super().__init__()
        self.n_levels = len(num_basis)
        self.n_iters_inter = n_iters_inter           # keep as 1 for DEQ
        self.n_iters_intra = n_iters_intra
        self.eta_base = eta_base
        # Default eta list if not provided; validate length
        self.eta = eta_base / max(1, n_iters_intra)  # decouple from inter iters
        # print(self.eta_ls)

        self.jfb_no_grad_iters = jfb_no_grad_iters
        self.jfb_with_grad_iters = jfb_with_grad_iters
        self.jfb_reuse_solution = jfb_reuse_solution
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None

        if whiten_dim is not None:
            self.encoder = Conv2d(in_channels, whiten_dim, kernel=3)
            self.decoder = Conv2d(whiten_dim, in_channels, kernel=3)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels

        levels = []
        for i, nb in enumerate(num_basis):
            levels.append(
                RecurrentConvUnit(
                    in_channels=prev_channels,
                    num_basis=nb,
                    kernel_size=kernel_size if i > 0 else 7,
                    stride=stride,
                    padding=kernel_size // 2 if i > 0 else 3,
                    eta=self.eta,
                    output_padding=output_padding,
                )
            )
            prev_channels = nb
        self.levels = nn.ModuleList(levels)

    def reset_fp_cache(self):
        self._last_a = None

    def forward(self, x, deq_mode=True):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        x = self.encoder(x)

        if not deq_mode:
            # Legacy unrolling (no DEQ)
            a = [None] * self.n_levels
            decoded = [None] * self.n_levels
            for _ in range(self.n_iters_inter):
                self.forward_inter(x, a, decoded)
            return self.decoder(decoded[0])

        # ----- JFB / DEQ mode -----
        # init hidden state
        if self.jfb_reuse_solution and (self._last_a is not None):
            a = [ai.clone() for ai in self._last_a]
        else:
            a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # n ~ U{0..N}
        n0 = random.randint(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1])
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            if self.jfb_ddp_safe:
                # DDP-safe no-grad: just run and detach later
                for _ in range(n0):
                    self.forward_inter(x, a, decoded)
            else:
                with torch.no_grad():
                    for _ in range(n0):
                        self.forward_inter(x, a, decoded)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded)

        # cache solution if desired
        if self.jfb_reuse_solution:
            self._last_a = [ai.detach() if ai is not None else None for ai in a]

        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded):
        # bottom-up
        for i in range(len(self.levels)):
            inp = x if i == 0 else a[i - 1]
            top_signal = decoded[i + 1] if i < self.n_levels - 1 else None
            a_cur = a[i]
            for _ in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
            a[i], decoded[i] = a_cur, decoded_cur

        # top-down
        for i in range(self.n_levels - 1, -1, -1):
            inp = x if i == 0 else a[i - 1]
            top_signal = decoded[i + 1] if i < self.n_levels - 1 else None
            a_cur = a[i]
            for _ in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
            a[i], decoded[i] = a_cur, decoded_cur

# Unet
class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        # run in float32 then cast back to original dtype
        return super().forward(x.float()).type(x.dtype)


class RecurrentConvNLayer3(nn.Module):
    def __init__(self, in_channels, num_basis,
                 n_iters_inter=1, n_iters_intra=1,eta_base=0.1,
                 kernel_size=5, stride=2, output_padding=1, whiten_dim=None, learning_horizontal = True,
                 # NEW: JFB controls
                jfb_no_grad_iters=None,       # n in [0, N]
                jfb_with_grad_iters=None,     # m in [1, M]
                 jfb_reuse_solution=False,
                 jfb_ddp_safe=True,
                 eta_ls = None):
        super().__init__()
        self.n_levels = len(num_basis)
        self.n_iters_inter = n_iters_inter           # keep as 1 for DEQ
        self.n_iters_intra = n_iters_intra
        self.eta_base = eta_base
        # Default eta list if not provided; validate length
        if eta_ls is None:
            self.eta_ls = [float(eta_base)] * self.n_levels
        else:
            if len(eta_ls) != self.n_levels:
                raise ValueError(f"eta_ls length {len(eta_ls)} must match number of levels {self.n_levels}")
            self.eta_ls = [float(x) for x in eta_ls]
        # print(self.eta_ls)
        # self.eta = eta_base / max(1, n_iters_intra)  # decouple from inter iters
        # print(self.eta_ls)

        # Default JFB tuples if None
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution = jfb_reuse_solution
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None

        if whiten_dim is not None:
            self.encoder = Conv2d(in_channels, whiten_dim, kernel=3)
            self.decoder = Conv2d(whiten_dim, in_channels, kernel=3)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels

        levels = []
        for i, nb in enumerate(num_basis):
            levels.append(
                RecurrentConvUnit(
                    in_channels=prev_channels,
                    num_basis=nb,
                    kernel_size=kernel_size if i > 0 else 7,
                    stride=stride,
                    padding=kernel_size // 2 if i > 0 else 3,
                    eta=self.eta_ls[i],
                    output_padding=output_padding,
                    learning_horizontal=learning_horizontal
                )
            )
            prev_channels = nb
        self.levels = nn.ModuleList(levels)

    def reset_fp_cache(self):
        self._last_a = None

    def forward(self, x, deq_mode=True):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        x = self.encoder(x)

        if not deq_mode:
            # Legacy unrolling (no DEQ)
            a = [None] * self.n_levels
            decoded = [None] * self.n_levels
            for _ in range(self.n_iters_inter):
                self.forward_inter(x, a, decoded)
            return self.decoder(decoded[0])

        # ----- JFB / DEQ mode -----
        # init hidden state
        if self.jfb_reuse_solution and (self._last_a is not None):
            a = [ai.clone() for ai in self._last_a]
        else:
            a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # n ~ U{0..N}
        n0 = random.randint(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1])
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            if self.jfb_ddp_safe:
                # DDP-safe no-grad: just run and detach later
                for _ in range(n0):
                    self.forward_inter(x, a, decoded)
            else:
                with torch.no_grad():
                    for _ in range(n0):
                        self.forward_inter(x, a, decoded)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded)

        # cache solution if desired
        if self.jfb_reuse_solution:
            self._last_a = [ai.detach() if ai is not None else None for ai in a]

        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded):
        # bottom-up
        for i in range(len(self.levels)):
            inp = x if i == 0 else a[i - 1]
            top_signal = decoded[i + 1] if i < self.n_levels - 1 else None
            a_cur = a[i]
            for _ in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
            a[i], decoded[i] = a_cur, decoded_cur

        # top-down
        for i in range(self.n_levels - 1, -1, -1):
            inp = x if i == 0 else a[i - 1]
            top_signal = decoded[i + 1] if i < self.n_levels - 1 else None
            a_cur = a[i]
            for _ in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
            a[i], decoded[i] = a_cur, decoded_cur

class RecurrentConvNLayer_cc(nn.Module):
    """
    n‑layer hierarchical convolutional sparse coding model with top‑down feedback.

    Parameters:
      - in_channels:    Number of channels in the input (e.g. 1 for grayscale).
      - num_basis:      Sequence of latent‐channel sizes, length = n.
      - eta:            Step size for all levels.
      - n_iters_inter:  Number of unrolled inference iterations.
      - kernel_size, stride, padding, output_padding: conv params for every level.
    """
    def __init__(self,
                 in_channels: int,
                 num_basis: list[int],
                 eta_base: float = None,
                 n_iters_inter: int = 2,
                 n_iters_intra: int = 1,
                 kernel_size: int = 5,
                 stride: int = 2,
                 output_padding: int = 1,
                 whiten_dim=None):
        super().__init__()
        self.n_iters_inter = n_iters_inter
        n_iters_train = n_iters_inter*n_iters_intra
        self.n_levels = len(num_basis)
        self.eta = eta_base/n_iters_train
        # assert n_iters_train % n_iters_intra == 0, "n_iters_train must be divisible by n_iters_intra"
        self.n_iters_inter = n_iters_inter
        self.n_iters_intra = n_iters_intra
        self.whiten_dim = whiten_dim
        if whiten_dim is not None:
            self.encoder = Conv2d(
                in_channels=in_channels,
                out_channels=whiten_dim,
                kernel=3)
            
            self.decoder = Conv2d(
                in_channels=whiten_dim,
                out_channels=in_channels,
                kernel=3)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels
        
        # build a ModuleList of RecurrentConvUnit, chaining channels
        levels = []
        for i,nb in enumerate(num_basis):
            levels.append(
                RecurrentConvUnit_cc(
                    in_channels=prev_channels,
                    num_basis=nb,
                    kernel_size=kernel_size if i>0 else 7,
                    stride=stride,
                    padding=kernel_size//2 if i>0 else 3,
                    eta=self.eta,
                    output_padding=output_padding
                )
            )
            prev_channels = nb
        self.levels = nn.ModuleList(levels)

    def forward(self, x):
        # initialize latent activations & decodings
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels
        x = self.encoder(x)

        for i_out in range(self.n_iters_inter):
            self.forward_inter(x, a, decoded)

        # final reconstruction from the first level
        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded):
        for i in range(len(self.levels)):
            inp = x if i == 0 else a[i-1]
            top_signal = decoded[i+1] if i < self.n_levels-1 else None
            a_cur = a[i]
            for j in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
            a[i], decoded[i] = a_cur, decoded_cur

        # top‑down refinement (exclude the topmost level)
        for i in range(self.n_levels-1, -1, -1):
            inp = x if i == 0 else a[i-1]
            top_signal = decoded[i+1] if i < self.n_levels-1 else None
            a_cur = a[i]
            for j in range(self.n_iters_intra):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal)
            a[i], decoded[i] = a_cur, decoded_cur

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

class RecurrentOneLayer(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=7,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution=False,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1):
        super().__init__()
        self.eta_base = eta_base
        # Default JFB tuples if None
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution = jfb_reuse_solution
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None

        if whiten_dim is not None:
            self.encoder = Conv2d(in_channels, whiten_dim, kernel=3)
            self.decoder = Conv2d(whiten_dim, in_channels, kernel=3)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels
        self.levels = nn.ModuleList([
            RecurrentConvUnit_gram(prev_channels,num_basis[0],kernel_size=kernel_size,eta=eta_base,
            stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal)
        ])
        self.n_levels=1

        # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
        emb_channels   = num_basis[0] * channel_mult_emb
        noise_channels = num_basis[0] * channel_mult_noise

        self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
        self.affine     = nn.Linear(emb_channels, num_basis[0], bias=True)
    
    def forward(self, x, deq_mode=True,noise_labels=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]

        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        emb = self.map_noise(noise_labels)                       # [B, Cn]
        emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
        emb = F.relu(self.map_layer0(emb))
        noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]

        x = self.encoder(x)

        if not deq_mode:
            # Legacy unrolling (no DEQ)
            a = [None] * self.n_levels
            decoded = [None] * self.n_levels
            for _ in range(self.n_iters_inter):
                self.forward_inter(x, a, decoded, noise_emb)
            return self.decoder(decoded[0])

        # ----- JFB / DEQ mode -----
        # init hidden state
        # if self.jfb_reuse_solution and (self._last_a is not None):
        #     a = [ai.clone() for ai in self._last_a]
        # else:

        a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        if self.jfb_reuse_solution:
            k0 = random.randint(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1])

        # n ~ U{0..N}
        n0 = random.randint(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1])
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            if self.jfb_ddp_safe:
                # DDP-safe no-grad: just run and detach later
                for _ in range(n0):
                    self.forward_inter(x, a, decoded, noise_emb)
            else:
                with torch.no_grad():
                    if self.jfb_reuse_solution:
                        perm_idx = torch.randperm(B)
                        for _ in range(k0):
                            self.forward_inter(x[perm_idx], a, decoded, noise_emb)
                    for _ in range(n0):
                        self.forward_inter(x, a, decoded, noise_emb)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        # cache solution if desired
        # if self.jfb_reuse_solution:
        #     self._last_a = [ai.detach() if ai is not None else None for ai in a]

        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, noise_emb=None):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb)

    def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False):
        
        B = x.shape[0]
        if a is None:
            a = [None] * self.n_levels
        if decoded is None:
            decoded = [None] * self.n_levels
        
        decoded_hist = {"a":[],"decoded":[]}
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        emb = self.map_noise(noise_labels)                       # [B, Cn]
        emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
        emb = F.relu(self.map_layer0(emb))
        noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        x = self.encoder(x)

        with torch.no_grad():
            for _ in range(n_iter):
                self.forward_inter(x, a, decoded, noise_emb)
                decoded_hist["a"].append([i.cpu().detach() for i in a])
                decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
        decoded_out = self.decoder(decoded[0])
            
        if return_feature:
            if return_hist:
                return a, decoded, decoded_hist
            else:
                return a, decoded
        else:
            return decoded_out


# class RecurrentOneLayer_reuse(nn.Module):
#     def __init__(self,in_channels,num_basis,kernel_size=7,
#     stride=2,output_padding=1,whiten_dim=None,
#     learning_horizontal=True,eta_base=0.1,
#     jfb_no_grad_iters=None,jfb_with_grad_iters=None,
#     jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
#     channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, frequency_groups=None, init_lambda=0.0, whiten_ks = 3):
#         super().__init__()
#         self.eta_base = eta_base
#         # if frequency_groups is None:
#         #     self.frequency_groups = in_channels
#         # else:
#         #     self.frequency_groups = frequency_groups
#         # Default JFB tuples if None
#         self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
#         self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
#         self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
#         self.jfb_ddp_safe = jfb_ddp_safe
#         self._last_a = None
#         self.mixer_value=mixer_value

#         if whiten_dim is not None:
#             self.encoder = Conv2d(in_channels, whiten_dim, kernel=whiten_ks)
#             self.decoder = Conv2d(whiten_dim, in_channels, kernel=whiten_ks)
#             prev_channels = whiten_dim
#             if frequency_groups is not None:
#                 G = prev_channels//frequency_groups
#             else:
#                 G = 1
#         else:
#             G =1
#             self.encoder = nn.Identity()
#             self.decoder = nn.Identity()
#             prev_channels = in_channels
#         self.levels = nn.ModuleList([
#             RecurrentConvUnit_gram(prev_channels,num_basis[0],kernel_size=kernel_size,eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, groups=G, init_lambda=init_lambda)
#         ])
#         self.n_levels=1

#         # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
#         emb_channels   = num_basis[0] * channel_mult_emb
#         noise_channels = num_basis[0] * channel_mult_noise

#         self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
#         # I modified the following line.
#         self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
#         self.affine     = nn.Linear(emb_channels, num_basis[0], bias=True)
#         # init map_layer0 and affine both to 0
#         nn.init.constant_(self.map_layer0.weight, 0)
#         nn.init.constant_(self.affine.weight, 0)
#         nn.init.constant_(self.map_layer0.bias, 0)
#         nn.init.constant_(self.affine.bias, 0)
    
#     def forward(self, x, deq_mode=True,noise_labels=None):
#         """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
#         B = x.shape[0]

#         # --- Build embedding exactly in your order ---
#         if noise_labels is None:
#             noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

#         emb = self.map_noise(noise_labels)                       # [B, Cn]
#         emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
#         emb = F.relu(self.map_layer0(emb))
#         noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
#         # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
#         # noise_emb = 0
#         x = self.encoder(x)

#         # ----- JFB / DEQ mode -----
#         a = [None] * self.n_levels
#         decoded = [None] * self.n_levels

#         # k ~ U{0..N}|dice>jfb_reuse_solution_rate
#         k0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value =self.mixer_value) if self.jfb_reuse_solution_rate>random.random() else 0
#         # n ~ U{0..N}
#         n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value=self.mixer_value)
#         # m ~ U{1..M}
#         m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

#         # ---- no-grad phase ----
#         if n0 > 0:
#             with torch.no_grad():
#                 # either start with zero init, i.e. a, and decoded are zero, or start with random init, 
#                 # i.e. a, and decoded are infered from random image with same noise level
#                 # print(k0)
#                 if k0>0:
#                     perm_idx = torch.randperm(B)
#                     for _ in range(k0):
#                         self.forward_inter(x[perm_idx], a, decoded, noise_emb)
#                 for _ in range(n0):
#                     self.forward_inter(x, a, decoded, noise_emb)

#         # cut graph between phases
#         a = [ai.detach() if ai is not None else None for ai in a]

#         # ---- with-grad phase ----
#         for _ in range(m1):
#             self.forward_inter(x, a, decoded, noise_emb)

#         return self.decoder(decoded[0])

#     def forward_inter(self, x, a, decoded, noise_emb=None):
#         a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb)

#     def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False):
        
#         B = x.shape[0]
#         if a is None:
#             a = [None] * self.n_levels
#         if decoded is None:
#             decoded = [None] * self.n_levels
        
#         decoded_hist = {"a":[],"decoded":[]}
#         # --- Build embedding exactly in your order ---
#         if noise_labels is None:
#             noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

#         emb = self.map_noise(noise_labels)                       # [B, Cn]
#         emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
#         emb = F.relu(self.map_layer0(emb))
#         noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
#         x = self.encoder(x)
        
#         with torch.no_grad():
#             for _ in range(n_iter):
#                 self.forward_inter(x, a, decoded, noise_emb)
#                 decoded_hist["a"].append([i.cpu().detach() for i in a])
#                 decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
#         decoded_out = self.decoder(decoded[0])
            
#         if return_feature:
#             if return_hist:
#                 return a, decoded, decoded_hist
#             else:
#                 return a, decoded
#         else:
#             return decoded_out


class RecurrentOneLayer_reuse(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=7,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
    groups=2,h_groups=2,init_lambda=0.1,whiten_ks=3,noise_embedding=True,bias=True,relu_6=False,T=0.1):
        super().__init__()
        self.eta_base = eta_base
        # if frequency_groups is None:
        #     self.frequency_groups = in_channels
        # else:
        #     self.frequency_groups = frequency_groups
        # Default JFB tuples if None
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None
        self.mixer_value=mixer_value
        self.T=T

        if whiten_dim is not None:
            self.encoder = Conv2d(in_channels, whiten_dim, kernel=whiten_ks)
            self.decoder = Conv2d(whiten_dim, in_channels, kernel=whiten_ks)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels

        self.levels = nn.ModuleList([
            RecurrentConvUnit_gram(prev_channels,num_basis[0],kernel_size=kernel_size,
            eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, groups=groups, bias=bias,relu_6=relu_6, h_groups=h_groups)
        ])

        self.n_levels=1
        self.noise_embedding = noise_embedding
        if noise_embedding:
            # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
            emb_channels   = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.affine     = nn.Linear(emb_channels, num_basis[0], bias=True)
        else:
            self.lambda_bias = torch.nn.Parameter(torch.zeros(1,1,1,1))
            
    def _reset_all_wnorm(self):
        if not getattr(self, '_wnorm', False):
            return
        # Recompute normalized conv weights inside the recurrent block
        for lvl in self.levels:
            if hasattr(lvl, 'reset_wnorm'):
                lvl.reset_wnorm()

    def forward(self, x, deq_mode=True,noise_labels=None,T=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias

        x = self.encoder(x)

        # ----- JFB / DEQ mode -----
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # k ~ U{0..N}|dice>jfb_reuse_solution_rate
        k0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value =self.mixer_value) if self.jfb_reuse_solution_rate>random.random() else 0
        # n ~ U{0..N}
        n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value=self.mixer_value)
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            with torch.no_grad():
                # either start with zero init, i.e. a, and decoded are zero, or start with random init, 
                # i.e. a, and decoded are infered from random image with same noise level
                # print(k0)
                if k0>0:
                    perm_idx = torch.randperm(B)
                    for _ in range(k0):
                        self.forward_inter(x[perm_idx], a, decoded, noise_emb, T=T)
                for _ in range(n0):
                    self.forward_inter(x, a, decoded, noise_emb, T=T)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, noise_emb=None, T=0):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb, T=T)

    def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False,T=None):
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        B = x.shape[0]
        if a is None:
            a = [None] * self.n_levels
        if decoded is None:
            decoded = [None] * self.n_levels
        
        decoded_hist = {"a":[],"decoded":[]}
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias
            
        x = self.encoder(x)
        
        with torch.no_grad():
            for _ in range(n_iter):
                self.forward_inter(x, a, decoded, noise_emb,T=T)
                decoded_hist["a"].append([i.cpu().detach() for i in a])
                decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
        decoded_out = self.decoder(decoded[0])
            
        if return_feature:
            if return_hist:
                return a, decoded, decoded_hist
            else:
                return a, decoded
        else:
            return decoded_out



class RecurrentOneLayer_splitNet(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=7,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
    frequency_groups=None,init_lambda=0.1,whiten_ks=3,noise_embedding=True,bias=True,
    relu_6=False,T=0.1,pyramid=False,groups=2,h_groups=2):
        super().__init__()
        self.eta_base = eta_base
        # if frequency_groups is None:
        #     self.frequency_groups = in_channels
        # else:
        #     self.frequency_groups = frequency_groups
        # Default JFB tuples if None
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None
        self.mixer_value=mixer_value
        self.T=T

        if whiten_dim is not None:
            if pyramid:
                self.encoder = GaussianPyramidEncoder(levels = whiten_dim-1)
            else:
                self.encoder = Conv2d(in_channels, whiten_dim, kernel=whiten_ks)
            # self.decoder = Conv2d(whiten_dim, in_channels, kernel=whiten_ks)
            prev_channels = whiten_dim
            if frequency_groups is not None:
                G = prev_channels//frequency_groups
            else:
                G = 1
        else:
            self.encoder = nn.Identity()
            # self.decoder = nn.Identity()
            prev_channels = in_channels
            G = 1

        self.levels = nn.ModuleList([
            RecurrentConvUnit_gram(prev_channels,num_basis[0],kernel_size=kernel_size,
            eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, groups=groups, bias=bias,relu_6=relu_6, h_groups=h_groups)
        ])

        self.n_levels=1
        self.noise_embedding = noise_embedding
        if noise_embedding:
            # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
            emb_channels   = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.affine     = nn.Linear(emb_channels, num_basis[0], bias=True)
        else:
            self.lambda_bias = torch.nn.Parameter(torch.zeros(1,1,1,1))
            
    def _reset_all_wnorm(self):
        if not getattr(self, '_wnorm', False):
            return
        # Recompute normalized conv weights inside the recurrent block
        for lvl in self.levels:
            if hasattr(lvl, 'reset_wnorm'):
                lvl.reset_wnorm()

    def forward(self, x, deq_mode=True,noise_labels=None,T=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias

        x = self.encoder(x)

        # ----- JFB / DEQ mode -----
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # k ~ U{0..N}|dice>jfb_reuse_solution_rate
        k0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value =self.mixer_value) if self.jfb_reuse_solution_rate>random.random() else 0
        # n ~ U{0..N}
        n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value=self.mixer_value)
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            with torch.no_grad():
                # either start with zero init, i.e. a, and decoded are zero, or start with random init, 
                # i.e. a, and decoded are infered from random image with same noise level
                # print(k0)
                if k0>0:
                    perm_idx = torch.randperm(B)
                    for _ in range(k0):
                        self.forward_inter(x[perm_idx], a, decoded, noise_emb, T=T)
                for _ in range(n0):
                    self.forward_inter(x, a, decoded, noise_emb, T=T)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        return decoded[0].sum(dim=1,keepdim=True)

    def forward_inter(self, x, a, decoded, noise_emb=None, T=0):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb, T=T)

    def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False,T=None):
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        B = x.shape[0]
        if a is None:
            a = [None] * self.n_levels
        if decoded is None:
            decoded = [None] * self.n_levels
        
        decoded_hist = {"a":[],"decoded":[]}
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias
            
        x = self.encoder(x)
        
        with torch.no_grad():
            for _ in range(n_iter):
                self.forward_inter(x, a, decoded, noise_emb,T=T)
                decoded_hist["a"].append([i.cpu().detach() for i in a])
                decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
        # decoded_out = self.decoder(decoded[0])
        decoded_out = decoded[0].sum(dim=1,keepdim=True)
            
        if return_feature:
            if return_hist:
                return a, decoded, decoded_hist
            else:
                return a, decoded
        else:
            return decoded_out



class RecurrentLayers_diverse(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=3,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
    frequency_groups=None,init_lambda=0.1,whiten_ks=3,noise_embedding=True,bias=True,
    relu_6=False,T=0.1,pyramid=False,groups=2,h_groups=2,energy_function="elastic",
    per_dim_threshold=False,positive_threshold=False,multiscale=False,num_basis_per_scale=None):
        super().__init__()
        # This model capture many different energy function between latent variables and data.
        # It starts with latent variable a_0 and data x_0, and the energy function is E(x_0, a_0) = ||x_0 - \Phi_0 a_0||^2
        # From which we can derive x_0 = \Phi_0 a_0, and a_0 = \Phi_0^T x_0, which result in a linear encoder and decoder.
        # This linear encoder and decoder can be multi-scale as well, which make it a little more complicated.
        # For multiscale, we only support 1 layer for now, but will extend to more layers in the future.
        # We also support different energy function between latent variables and data.

        self.eta_base = eta_base
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None
        self.mixer_value=mixer_value
        self.T=T
        self.positive_threshold = positive_threshold
        self.multiscale=multiscale
        self.num_scales = whiten_dim

        if whiten_dim is not None:
            if pyramid:
                if multiscale:
                    self.encoder = GaussianPyramidEncoder(levels = whiten_dim-1,concat_to_channels=False)
                    self.decoder = self.encoder.decode
                else:
                    self.encoder = GaussianPyramidEncoder(levels = whiten_dim-1)
                    self.decoder = lambda x: x.sum(dim=1,keepdim=True)
            else:
                self.encoder = Conv2d(in_channels, whiten_dim, kernel=whiten_ks)
                self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
            # self.decoder = Conv2d(whiten_dim, in_channels, kernel=whiten_ks)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels
        print(self.encoder,self.decoder)

        # if multiscale:
        #     self.levels = nn.ModuleList([
        #         MultiScaleRecurrentConvUnit_diverse(in_channels,num_basis[0],num_scales=whiten_dim,kernel_size=kernel_size,
        #         eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, 
        #         groups=groups, bias=bias,relu_6=relu_6, h_groups=h_groups, energy_function=energy_function)
        #     ])
        # else:
        #     self.levels = nn.ModuleList([
                # RecurrentConvUnit_diverse(prev_channels,num_basis[0],kernel_size=kernel_size,
                # eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, 
                # groups=groups, bias=bias,relu_6=relu_6, h_groups=h_groups, energy_function=energy_function)
        #     ])
        levels = []
        for i, nb in enumerate(num_basis):
            levels.append(
                RecurrentConvUnit_diverse(prev_channels,nb,kernel_size=kernel_size if i > 0 else 7,
                eta=eta_base,stride=stride,padding = kernel_size // 2 if i > 0 else 3, output_padding=output_padding,
                learning_horizontal=learning_horizontal, groups=groups, bias=bias,relu_6=relu_6, 
                h_groups=h_groups, energy_function="hybrid" if i == 0 else "BM")
            )
            prev_channels = nb
        self.levels = nn.ModuleList(levels)

        self.n_levels=1
        self.noise_embedding = noise_embedding
        if noise_embedding:
            # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
            emb_channels   = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.affine     = nn.Linear(emb_channels, b_n if per_dim_threshold else 1, bias=True)
        else:
            self.lambda_bias = torch.nn.Parameter(torch.zeros(1,1,1,1))
            
    def _reset_all_wnorm(self):
        if not getattr(self, '_wnorm', False):
            return
        # Recompute normalized conv weights inside the recurrent block
        for lvl in self.levels:
            if hasattr(lvl, 'reset_wnorm'):
                lvl.reset_wnorm()

    def forward(self, x, deq_mode=True,noise_labels=None,T=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            if self.positive_threshold:
                # print("Using positive threshold")
                noise_emb = F.relu(noise_emb)
                # print(len(noise_emb))
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias

        x = self.encoder(x)

        # ----- JFB / DEQ mode -----
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # k ~ U{0..N}|dice>jfb_reuse_solution_rate
        k0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value =self.mixer_value) if self.jfb_reuse_solution_rate>random.random() else 0
        # n ~ U{0..N}
        n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value=self.mixer_value)
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            with torch.no_grad():
                # either start with zero init, i.e. a, and decoded are zero, or start with random init, 
                # i.e. a, and decoded are infered from random image with same noise level
                # print(k0)
                if k0>0:
                    perm_idx = torch.randperm(B)
                    # print(perm_idx)
                    for _ in range(k0):
                        self.forward_inter(x[perm_idx], a, decoded, noise_emb, T=T)
                for _ in range(n0):
                    self.forward_inter(x, a, decoded, noise_emb, T=T)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        # return decoded[0].sum(dim=1,keepdim=True)
        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, noise_emb=None, T=0):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb, T=T)

    def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False,T=None):
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        B = x.shape[0]
        if a is None:
            a = [None] * self.n_levels
        if decoded is None:
            decoded = [None] * self.n_levels
        
        decoded_hist = {"a":[],"decoded":[]}
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias
            
        x = self.encoder(x)
        
        with torch.no_grad():
            for _ in range(n_iter):
                self.forward_inter(x, a, decoded, noise_emb,T=T)
                decoded_hist["a"].append([i.cpu().detach() for i in a])
                decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
        decoded_out = self.decoder(decoded[0])
        # decoded_out = decoded[0].sum(dim=1,keepdim=True)
            
        if return_feature:
            if return_hist:
                return a, decoded, decoded_hist
            else:
                return a, decoded
        else:
            return decoded_out


class RecurrentOneLayer_diverse(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=7,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
    frequency_groups=None,init_lambda=0.1,whiten_ks=3,noise_embedding=True,bias=True,
    relu_6=False,T=0.1,pyramid=False,groups=2,h_groups=2,energy_function="elastic",
    per_dim_threshold=False,positive_threshold=False,multiscale=False,num_basis_per_scale=None):
        super().__init__()
        # This model capture many different energy function between latent variables and data.
        # It starts with latent variable a_0 and data x_0, and the energy function is E(x_0, a_0) = ||x_0 - \Phi_0 a_0||^2
        # From which we can derive x_0 = \Phi_0 a_0, and a_0 = \Phi_0^T x_0, which result in a linear encoder and decoder.
        # This linear encoder and decoder can be multi-scale as well, which make it a little more complicated.
        # For multiscale, we only support 1 layer for now, but will extend to more layers in the future.
        # We also support different energy function between latent variables and data.

        self.eta_base = eta_base
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None
        self.mixer_value=mixer_value
        self.T=T
        self.positive_threshold = positive_threshold
        self.multiscale=multiscale
        self.num_scales = whiten_dim
        if num_basis_per_scale is not None:
            self.num_basis_per_scale_cum = [0]+[sum(num_basis_per_scale[:i+1]) for i in range(len(num_basis_per_scale))]
        else:
            self.num_basis_per_scale_cum = None

        if whiten_dim is not None:
            if pyramid:
                if multiscale:
                    self.encoder = GaussianPyramidEncoder(levels = whiten_dim-1,concat_to_channels=False)
                    self.decoder = self.encoder.decode
                else:
                    self.encoder = GaussianPyramidEncoder(levels = whiten_dim-1)
                    self.decoder = lambda x: x.sum(dim=1,keepdim=True)
                    
            else:
                self.encoder = Conv2d(in_channels, whiten_dim, kernel=whiten_ks)
                self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
            # self.decoder = Conv2d(whiten_dim, in_channels, kernel=whiten_ks)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels

        if multiscale:
            self.levels = nn.ModuleList([
                MultiScaleRecurrentConvUnit_diverse(in_channels,num_basis[0],num_scales=whiten_dim,kernel_size=kernel_size,
                eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, 
                groups=groups, bias=bias,relu_6=relu_6, h_groups=h_groups, energy_function=energy_function,num_basis_per_scale=num_basis_per_scale)
            ])
        else:
            self.levels = nn.ModuleList([
                RecurrentConvUnit_diverse(prev_channels,num_basis[0],kernel_size=kernel_size,
                eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, 
                groups=groups, bias=bias,relu_6=relu_6, h_groups=h_groups, energy_function=energy_function)
            ])

        self.n_levels=1
        self.noise_embedding = noise_embedding
        if noise_embedding:
            # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
            emb_channels   = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.affine     = nn.Linear(emb_channels, num_basis[0] if per_dim_threshold else 1, bias=True)
        else:
            self.lambda_bias = torch.nn.Parameter(torch.zeros(1,1,1,1))
            
    def _reset_all_wnorm(self):
        if not getattr(self, '_wnorm', False):
            return
        # Recompute normalized conv weights inside the recurrent block
        for lvl in self.levels:
            if hasattr(lvl, 'reset_wnorm'):
                lvl.reset_wnorm()

    def forward(self, x, deq_mode=True,noise_labels=None,T=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            if self.positive_threshold:
                # print("Using positive threshold")
                noise_emb = F.relu(noise_emb)
            if self.multiscale:
                if self.num_basis_per_scale_cum is not None:
                    # print(self.num_basis_per_scale_cum)
                    noise_emb = [noise_emb[:,self.num_basis_per_scale_cum[i]:self.num_basis_per_scale_cum[i+1]]  for i in range(len(self.num_basis_per_scale_cum)-1)]
                    # [print(i.shape) for i in noise_emb]
                else:
                    noise_emb = noise_emb.chunk(self.num_scales, dim=1)

                # print(len(noise_emb))
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias

        x = self.encoder(x)

        # ----- JFB / DEQ mode -----
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # k ~ U{0..N}|dice>jfb_reuse_solution_rate
        k0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value =self.mixer_value) if self.jfb_reuse_solution_rate>random.random() else 0
        # n ~ U{0..N}
        n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value=self.mixer_value)
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            with torch.no_grad():
                # either start with zero init, i.e. a, and decoded are zero, or start with random init, 
                # i.e. a, and decoded are infered from random image with same noise level
                # print(k0)
                if k0>0:
                    perm_idx = torch.randperm(B)
                    # print(perm_idx)
                    for _ in range(k0):
                        if self.multiscale:
                            self.forward_inter([x_s[perm_idx] for x_s in x], a, decoded, noise_emb, T=T)
                        else:
                            self.forward_inter(x[perm_idx], a, decoded, noise_emb, T=T)
                for _ in range(n0):
                    self.forward_inter(x, a, decoded, noise_emb, T=T)

        # cut graph between phases
        if self.multiscale:
            a = [[ai_s.detach() for ai_s in ai] if ai is not None else None for ai in a]
        else:
            a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        # return decoded[0].sum(dim=1,keepdim=True)
        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, noise_emb=None, T=0):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb, T=T)

    def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False,T=None):
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        B = x.shape[0]
        if a is None:
            a = [None] * self.n_levels
        if decoded is None:
            decoded = [None] * self.n_levels
        
        decoded_hist = {"a":[],"decoded":[]}
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias
            
        x = self.encoder(x)
        
        with torch.no_grad():
            for _ in range(n_iter):
                self.forward_inter(x, a, decoded, noise_emb,T=T)
                decoded_hist["a"].append([i.cpu().detach() for i in a])
                decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
        decoded_out = self.decoder(decoded[0])
        # decoded_out = decoded[0].sum(dim=1,keepdim=True)
            
        if return_feature:
            if return_hist:
                return a, decoded, decoded_hist
            else:
                return a, decoded
        else:
            return decoded_out


class RecurrentOneLayer_stable(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=7,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution=False,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1):
        super().__init__()
        self.eta_base = eta_base
        # Default JFB tuples if None
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution = jfb_reuse_solution
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None

        if whiten_dim is not None:
            self.encoder = Conv2d(in_channels, whiten_dim, kernel=3)
            self.decoder = Conv2d(whiten_dim, in_channels, kernel=3)
            prev_channels = whiten_dim
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels
        self.levels = nn.ModuleList([
            RecurrentConvUnit_gram(prev_channels,num_basis[0],kernel_size=kernel_size,eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal)
        ])
        self.n_levels=1

        # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
        emb_channels   = num_basis[0] * channel_mult_emb
        # noise_channels = num_basis[0] * channel_mult_noise

        self.map_noise   = PositionalEmbedding(num_channels=emb_channels, endpoint=True)
        self.map_layer0 = nn.Linear(emb_channels, num_basis[0], bias=True)
        # self.affine     = nn.Linear(emb_channels, num_basis[0], bias=True)
    
    def forward(self, x, deq_mode=True,noise_labels=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]

        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        emb = self.map_noise(noise_labels)                       # [B, Cn]
        emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
        emb = self.map_layer0(emb)
        # noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]

        x = self.encoder(x)

        if not deq_mode:
            # Legacy unrolling (no DEQ)
            a = [None] * self.n_levels
            decoded = [None] * self.n_levels
            for _ in range(self.n_iters_inter):
                self.forward_inter(x, a, decoded, noise_emb)
            return self.decoder(decoded[0])

        # ----- JFB / DEQ mode -----
        # init hidden state
        if self.jfb_reuse_solution and (self._last_a is not None):
            a = [ai.clone() for ai in self._last_a]
        else:
            a = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # n ~ U{0..N}
        n0 = random.randint(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1])
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- no-grad phase ----
        if n0 > 0:
            if self.jfb_ddp_safe:
                # DDP-safe no-grad: just run and detach later
                for _ in range(n0):
                    self.forward_inter(x, a, decoded, noise_emb)
            else:
                with torch.no_grad():
                    for _ in range(n0):
                        self.forward_inter(x, a, decoded, noise_emb)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        # cache solution if desired
        if self.jfb_reuse_solution:
            self._last_a = [ai.detach() if ai is not None else None for ai in a]

        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, noise_emb=None):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb)

    def get_noise_emb(self, noise_labels):
        B = noise_labels.shape[0]
        emb = self.map_noise(noise_labels)                       # [B, Cn]
        emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
        emb = F.relu(self.map_layer0(emb))
        noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(noise_labels.dtype)  # [B, c1, 1, 1]
        return noise_emb

    def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False):
        
        B = x.shape[0]
        if a is None:
            a = [None] * self.n_levels
        if decoded is None:
            decoded = [None] * self.n_levels

        decoded_hist = {"a":[],"decoded":[]}
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        emb = self.map_noise(noise_labels)                       # [B, Cn]
        emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
        emb = F.relu(self.map_layer0(emb))
        noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        x = self.encoder(x)
        with torch.no_grad():
            for _ in range(n_iter):
                self.forward_inter(x, a, decoded, noise_emb)
                decoded_hist["a"].append([i.cpu().detach() for i in a])
                decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
        decoded_out = self.decoder(decoded[0])
            
        if return_feature:
            if return_hist:
                return a, decoded, decoded_hist
            else:
                return a, decoded
        else:
            return decoded_out


class RecurrentOneLayer_DEQ(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=7,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,jfb_with_grad_iters=None,
    jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
    frequency_groups=None,init_lambda=0.1,whiten_ks=3,noise_embedding=True,bias=True,relu_6=False):
        super().__init__()
        self.eta_base = eta_base
        # if frequency_groups is None:
        #     self.frequency_groups = in_channels
        # else:
        #     self.frequency_groups = frequency_groups
        # Default JFB tuples if None
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None
        self.mixer_value=mixer_value

        if whiten_dim is not None:
            self.encoder = Conv2d(in_channels, whiten_dim, kernel=whiten_ks)
            self.decoder = Conv2d(whiten_dim, in_channels, kernel=whiten_ks)
            prev_channels = whiten_dim
            if frequency_groups is not None:
                G = prev_channels//frequency_groups
            else:
                G = 1
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
            prev_channels = in_channels
            G = 1
        self.levels = nn.ModuleList([
            RecurrentConvUnit_gram(prev_channels,num_basis[0],kernel_size=kernel_size,
            eta=eta_base,stride=stride,output_padding=output_padding,learning_horizontal=learning_horizontal, 
            groups=G, bias=bias,relu_6=relu_6)
        ])
        self.n_levels=1
        self.noise_embedding = noise_embedding
        if noise_embedding:
            # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
            emb_channels   = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.affine     = nn.Linear(emb_channels, num_basis[0], bias=True)
        else:
                self.lambda_bias = torch.nn.Parameter(torch.zeros(1,1,1,1))
    # print()
    def a_to_vec(self,a):             # a: [B, C, H, W] -> v: [B, C, H*W], plus shape token
        B, C, H, W = a.shape
        return a.flatten(1).unsqueeze(-1), (C, H, W)

    def vec_to_a(self,v, shape):      # v: [B, C, L] -> a: [B, C, H, W]
        C, H, W = shape
        return rearrange(v.squeeze(-1),"b (c h w) -> b c h w",c = C, h = H)

    def T_fix(self, x, a, noise_emb,shape):
        a = self.vec_to_a(a,shape)
        a_next, _decoded = self.levels[0](x, a_prev=a, top_signal=None, noise_emb=noise_emb)
        a_next,_ = self.a_to_vec(a_next)
        return a_next
    
    def forward(self, x, deq_mode=True,noise_labels=None):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]

        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        else:
            noise_emb = self.lambda_bias

        x = self.encoder(x)

        with torch.no_grad():
            a0, _ = self.levels[0](x, a_prev=None, top_signal=None, noise_emb=noise_emb)
            a0_vec, shape = self.a_to_vec(a0)
            result = naive_solver(lambda a: self.T_fix(x, a, noise_emb, shape),
                                a0_vec, threshold=40,
                                name="forward")
            a_out = self.vec_to_a(result['result'],shape)
            x_hat= self.levels[0].decoder(a_out)

        # cut graph between phases
        a = [a_out.detach()]
        decoded = [x_hat.detach()]

        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, decoded, noise_emb)

        return self.decoder(decoded[0])

    def forward_inter(self, x, a, decoded, noise_emb=None):
        a[0], decoded[0] = self.levels[0](x, a_prev=a[0], top_signal=None, noise_emb=noise_emb)

    # def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False):
        
    #     B = x.shape[0]
    #     if a is None:
    #         a = [None] * self.n_levels
    #     if decoded is None:
    #         decoded = [None] * self.n_levels
        
    #     decoded_hist = {"a":[],"decoded":[]}
    #     # --- Build embedding exactly in your order ---
    #     if noise_labels is None:
    #         noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

    #     if self.noise_embedding:

    #         emb = self.map_noise(noise_labels)                       # [B, Cn]
    #         emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
    #         emb = F.relu(self.map_layer0(emb))
    #         noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
    #         # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
    #     else:
    #         noise_emb = self.lambda_bias
            
    #     x = self.encoder(x)
        
    #     with torch.no_grad():
    #         for _ in range(n_iter):
    #             self.forward_inter(x, a, decoded, noise_emb)
    #             decoded_hist["a"].append([i.cpu().detach() for i in a])
    #             decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
    #     decoded_out = self.decoder(decoded[0])
            
    #     if return_feature:
    #         if return_hist:
    #             return a, decoded, decoded_hist
    #         else:
    #             return a, decoded
    #     else:
    #         return decoded_out

class RecurrentConvNLayer_simple(nn.Module):
    def __init__(self, in_channels, num_basis,
                 n_iters_inter=1, n_iters_intra=4,eta_base=0.1,
                 kernel_size=3, stride=2, output_padding=1, learning_horizontal = True,
                 eta_ls = None,channel_mult_emb=2,channel_mult_noise=1, 
                 num_classes: int = 0, cond_drop_prob: float = 0.0):
        super().__init__()
        self.n_levels = len(num_basis)
        self.n_iters_inter = n_iters_inter           # keep as 1 for DEQ
        self.n_iters_intra = n_iters_intra
        self.eta_base = eta_base

        # noise_embeddings
        emb_channels   = num_basis[0] * channel_mult_emb
        noise_embed_dim = noise_channels = num_basis[0] * channel_mult_noise
        self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
        self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
        # self.affine = nn.Linear(emb_channels, num_basis[0], bias=True)
        self.affines = nn.ModuleList([nn.Linear(emb_channels, nb, bias=True) for nb in num_basis])
        # zero_init(self.map_layer0)

            # ---- Label conditioning ----
        self.num_classes = int(num_classes) if num_classes is not None else 0
        self.noise_embed_dim = int(noise_embed_dim)
        self.cond_drop_prob = float(cond_drop_prob)

        if self.num_classes > 0:
            # +1 for a learned "null/unconditional" label embedding (for classifier-free guidance)
            self.null_class = self.num_classes
            self.label_embed = nn.Embedding(self.num_classes + 1, self.noise_embed_dim)
        else:
            self.null_class = None
            self.label_embed = None
        
        # Default eta list if not provided; validate length
        if eta_ls is None:
            self.eta_ls = [float(eta_base)] * self.n_levels
        else:
            if len(eta_ls) != self.n_levels:
                raise ValueError(f"eta_ls length {len(eta_ls)} must match number of levels {self.n_levels}")
            self.eta_ls = [float(x) for x in eta_ls]

        prev_channels = in_channels
        levels = []
        for i, nb in enumerate(num_basis):
            levels.append(
                RecurrentConvUnit(
                    in_channels=prev_channels,
                    num_basis=nb,
                    kernel_size=kernel_size if i > 0 else 7,
                    stride=stride,
                    padding=kernel_size // 2 if i > 0 else 3,
                    eta=self.eta_ls[i],
                    output_padding=output_padding,
                )
            )
            prev_channels = nb
        self.levels = nn.ModuleList(levels)
        
        
    def _get_label_embedding(self, class_labels, B: int, device, dtype):
        """
        Returns label embedding vector of shape [B, noise_embed_dim] in `dtype`.
        Supports:
          - class_labels is None  -> unconditional
          - class_labels shape [B] (int class indices)
          - class_labels shape [B, num_classes] (one-hot or soft)
        """
        if self.label_embed is None:
            return torch.zeros(B, self.noise_embed_dim, device=device, dtype=dtype)

        # Unconditional path
        if class_labels is None:
            idx = torch.full((B,), self.null_class, device=device, dtype=torch.long)
            return self.label_embed(idx).to(dtype)

        # Integer labels: [B]
        idx = class_labels.to(device=device, dtype=torch.long)

        # Classifier-free guidance dropout during training:
        # randomly replace some labels with the null label.
        if self.training and self.cond_drop_prob > 0.0:
            drop = (torch.rand(B, device=device) < self.cond_drop_prob)
            if drop.any():
                idx = idx.clone()
                idx[drop] = self.null_class

        return self.label_embed(idx).to(dtype)

    def get_embedding(self, x, noise_labels =None, class_labels=None):
        B,_,_,_ = x.shape

        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        emb = self.map_noise(noise_labels)                       # [B, Cn]
        emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos


        # ---- label embedding vector (same dim as emb) ----
        y_emb = self._get_label_embedding(
            class_labels=class_labels,
            B=B,
            device=x.device,
            dtype=emb.dtype,   # keep it compatible with emb before map_layer0
        )                                                         # [B, Cn]

        # Fuse conditioning (typical pattern: additive)
        emb = emb + y_emb

        noise_emb = F.relu(self.map_layer0(emb))
        # noise_emb = self.affine(noise_emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
        noise_emb = [affine(noise_emb).unsqueeze(2).unsqueeze(3).to(x.dtype) for affine in self.affines]
        return noise_emb


    def forward(self, x, deq_mode=True, noise_labels = None, class_labels=None,return_feature=False,CFG_scale=0.0):
        B,_,_,_ = x.shape
        # initalize state and label embddings
        a = [None] * self.n_levels
        decoded = [None] * self.n_levels
        noise_emb = self.get_embedding(x, noise_labels, class_labels)

        if CFG_scale > 0.0:
            a_uncond = [None] * self.n_levels
            decoded_uncond = [None] * self.n_levels
            noise_emb_uncond = self.get_embedding(x, noise_labels, torch.full((B,), self.null_class, device=x.device, dtype=torch.long))

        # ---- with-grad phase ----
        for _ in range(self.n_iters_intra):
            self.forward_inter(x, a, decoded, noise_emb)
            if CFG_scale > 0.0:
                self.forward_inter(x, a_uncond, decoded_uncond, noise_emb_uncond)
                for i in range(self.n_levels):
                    a[i] = a[i] + CFG_scale * (a[i] - a_uncond[i])
                    decoded[i] = decoded[i] + CFG_scale * (decoded[i] - decoded_uncond[i])

        if return_feature:
            return a, decoded
        else:
            return decoded[0]

    def forward_inter(self, x, a, decoded, noise_emb=None):
        if noise_emb is None:
            noise_emb = [None] * self.n_levels

        # bottom-up
        for i in range(len(self.levels)):
            inp = x if i == 0 else a[i - 1]
            top_signal = decoded[i + 1] if i < self.n_levels - 1 else None
            a_cur = a[i]
            for _ in range(self.n_iters_inter):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal, noise_emb= noise_emb[i])
            a[i], decoded[i] = a_cur, decoded_cur

        # top-down
        for i in range(self.n_levels - 1, -1, -1):
            inp = x if i == 0 else a[i - 1]
            top_signal = decoded[i + 1] if i < self.n_levels - 1 else None
            a_cur = a[i]
            for _ in range(self.n_iters_inter):
                a_cur, decoded_cur = self.levels[i](inp, a_prev=a_cur, top_signal=top_signal, noise_emb= noise_emb[i])
            a[i], decoded[i] = a_cur, decoded_cur

class neural_node(nn.Module):
    """
    We define the energy coupling between previous variable and current variable.
    Given the energy coupling, we define the corresponding update rule.

    Convolutional recurrent unit, now with output_padding added to the decoder
    to ensure that the reconstructed feedback has matching spatial dimensions.
    Let input be x, latent state be a
    Several energy function and update mode:
        Positive sparse coding: 
        E(x, a) = ||x - \Phi a||_2^2 + \lambda(\sigma) ||a||_1, a>0
        Positive (diverse) elastic net (I think none-zero make this still convex?): 
        E(x, a) = ||x - \Phi a||_2^2 + \lambda(\sigma)  ||a||_1 + a^T M a, where M is a diagonal matrix, a>0
        Boltzmann machine (non-convex):
        E(x, a) = - a^T \Phi^T x + a^T M a + \lambda(\sigma) ||a||_1 , a>0 where M can be any matrix
        Hybrid energy:
        E(x, a) = ||x - \Phi a||_2^2 + \lambda(\sigma)  ||a||_1 + a^T M a, where M is a diagonal matrix, a>0
    where:
      - encoder(x) is the feedforward drive (implemented as a conv with stride 2).
      - M is a local (1x1, group) convolution.
      - top_signal is a top-down feedback signal.
    """
    def __init__(self, in_channels, num_basis_prev, num_basis, kernel_size=7, stride=2,
                 padding=3, eta=0.5, init_lambda=0.0, output_padding=1,
                 learning_horizontal=True, groups=1, bias=False, relu_6=True, 
                 wnorm=True, h_groups =1, constraint_energy="SC", k_inter = None, down=True,tied_transpose=True):
        super(neural_node, self).__init__()
        # Convolutional dictionary encoder.
        
        M_stride = 2 if down else 1
        output_padding = 1 if down else 0
        if learning_horizontal:
            self.M_inter = nn.Conv2d(
                num_basis, num_basis,
                kernel_size=3,
                padding =1,
                bias=False,
                groups=h_groups,
            )
        else:
            self.M_inter = None
        if num_basis_prev:
            self.M_intra = nn.Conv2d(
                num_basis_prev, num_basis,
                stride = M_stride,
                kernel_size=3,
                padding =1,
                bias=False,
                groups=groups,
            )
            if tied_transpose:
                self.M_intra_T = TiedTransposeConv(self.M_intra, output_padding=output_padding)
            else:
                self.M_intra_T = nn.ConvTranspose2d(
                    num_basis, num_basis_prev,
                    stride = M_stride,
                    kernel_size=3,
                    padding =1,
                    bias=False,
                    groups=groups,
                    output_padding = output_padding,
                )
        else:
            self.M_intra = self.M_intra_T = None
            
        self.tied_transpose = tied_transpose

        if in_channels:
            self.encoder = nn.Conv2d(
                in_channels, num_basis,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
            self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        else:
            self.encoder = self.decoder = None

        # Tied transpose convolution for decoding. The added output_padding ensures
        # that the reconstructed (decoded) tensor matches the dimensions of a_prev.
        # self.decoder = TiedTransposeConv(self.encoder, output_padding=output_padding)
        self.eta = torch.tensor(eta)
        # register eta as a buffer ? 
        # self.register_buffer('eta', self.eta)

        if relu_6:
            # print("Using ReLU6")
            self.relu = nn.ReLU6()
        else:
            self.relu = nn.ReLU()
        self.learning_horizontal = learning_horizontal
        self.constraint_energy=constraint_energy
        self.num_basis_prev=num_basis_prev
        self.num_basis=num_basis
        self._wnorm = bool(wnorm)
        if self._wnorm:
            # normalize per-output-channel: dim=0
            if self.M_inter is not None:
                self.M_inter, self._M_inter_wn = weight_norm(self.M_inter, names=['weight'], dim=0, k = k_inter)
            if self.encoder is not None:
                self.encoder, self._enc_wn = weight_norm(self.encoder, names=['weight'], dim=0)
            if self.M_intra is not None:
                self.M_intra, self._M_intra_wn = weight_norm(self.M_intra, names=['weight'], dim=0)
            if self.M_intra_T is not None and not self.tied_transpose:
                self.M_intra_T, self._M_intra_T_wn = weight_norm(self.M_intra_T, names=['weight'], dim=0)

    def reset_wnorm(self):
        """Recompute normalized weights once per batch/forward."""
        if not getattr(self, '_wnorm', False):
            return
        if self.encoder is not None:
            self._enc_wn.reset(self.encoder)
        if self.M_inter is not None:
            self._M_inter_wn.reset(self.M_inter)
        if self.M_intra is not None:
            self._M_intra_wn.reset(self.M_intra)
        if self.M_intra_T is not None and not self.tied_transpose:
            self._M_intra_T_wn.reset(self.M_intra_T)

    def decompose_M_inter(self,n_components=1,steer_components=None):
        horizontal = self.M_inter.weight.data
        n1,n2,h,w =horizontal.shape
        horizontal_flat = rearrange(horizontal,"n1 n2 h w -> n1 (n2 h w)")
        U,S,V = torch.linalg.svd(horizontal_flat,full_matrices=False)
        print(S.shape)
        if steer_components is not None:
            for key,value in steer_components.items():
                S[key] *= value
        horizontal_flat_approx = U[:,:n_components]@torch.diag(S[:n_components])@V[:n_components]
        V_back =  rearrange(horizontal_flat_approx,"n1 (n2 h w) -> n1 n2 h w",n2=n2,h=h)
        self.M_inter.weight.data = V_back
                # self.M_inter

    def forward(self, a_p_1=None, a_prev=None, x_c = None, 
    top_signal=None, noise_emb=None, T = 0,n_components=None, steer_components=None, eta_emb=None):
        """
        a_i 
        basis energy associated with this term: 
        a_i^T W_inter_i a_i + a_i^T W_intra_i a_{i-1} + a_{i+1}^T W_intra_{i+1} a_i
        basis update for a_i:
        da_i = W_inter_i a_i + W_intra_i a_{i-1} + W_intra_{i+1}^T a_{i+1} (top_signal)
        a_i's contribution to other term:
        upstream_grad = W_intra_i^T a_i
        other term:
        noise_emb
        visible connection: energy: ||x_c - \Phi a_i||^2 or x_c^T \Phi a_i

        At init, we assume a = 0, then we only have: 
        W_intra_i a_{i-1}, noise_emb, and visible connection term

        
        Args:
          x: Input activation (or lower-layer feature map).
          a_prev: Previous latent variable (if None, initialize from feedforward drive).
          top_signal: Top-down feedback signal (must match the shape of the latent code).
        """
        if eta_emb is not None:
            eta = eta_emb
        else:
            eta = self.eta
        self.reset_wnorm()
        if n_components is not None:
            self.decompose_M_inter(n_components,steer_components)
        # Use zero feedback if none provided.
        feedback = top_signal if top_signal is not None else 0
        noise_emb = noise_emb if noise_emb is not None else 0
        feedup = self.M_intra(a_p_1)  if a_p_1 is not None and self.M_intra is not None else 0
        constraint_ff = self.encoder(x_c) if x_c is not None and self.encoder is not None else 0
        a_inter = self.M_inter(a_prev) if self.M_inter is not None and a_prev is not None else 0

        
        if a_prev is None:
            # print(constraint_ff.shape if constraint_ff is not 0 else None , feedup.shape if feedup is not 0 else None)
            if x_c is not None:
                a = self.relu(eta * (constraint_ff + noise_emb))
            else:
                a = self.relu(eta * (constraint_ff - feedup - feedback + noise_emb))   
        else:
            # basis update             
            update = - feedup - a_inter - feedback + noise_emb            
            a = a_prev + eta * update + torch.sqrt(eta) * torch.randn_like(a_prev)*T
            if self.encoder is not None:
                # TODO: still need to implement partial injection
                if self.constraint_energy == "SC":
                    ff = self.encoder(x_c - self.decoder(a_prev))
                else:
                    ff = self.encoder(x_c)
                a = a + eta * ff
            a = self.relu(a)
        # Decode (reconstruct) from the latent representation.
        # print(self.M_intra_T,a.shape)
        upstream_grad = self.M_intra_T(a) if self.M_intra_T is not None else 0 
        decoded = self.decoder(a) if self.decoder is not None else None
        # print(decoded.shape)
        return a, upstream_grad, decoded



class neural_sheet(nn.Module):
    def __init__(self,in_channels,num_basis,kernel_size=3,
    stride=2,output_padding=1,whiten_dim=None,
    learning_horizontal=True,eta_base=0.1,
    jfb_no_grad_iters=None,jfb_with_grad_iters=None,
    jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
    channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
    init_lambda=0.1,noise_embedding=True,bias=True,
    relu_6=False,T=0.1,groups=1,h_groups=1,constraint_energy="SC",
    per_dim_threshold=True,positive_threshold=False,multiscale=False,intra=True,
    k_inter=None,n_hid_layers=-1,eta_emb=False,tied_transpose=True):
        super().__init__()
        # network topology: chain, ring, sheet?

        self.eta_base = eta_base
        self.jfb_no_grad_iters = (0, 6) if jfb_no_grad_iters is None else tuple(jfb_no_grad_iters)
        self.jfb_with_grad_iters = (1, 3) if jfb_with_grad_iters is None else tuple(jfb_with_grad_iters)
        self.jfb_reuse_solution_rate = jfb_reuse_solution_rate
        self.jfb_ddp_safe = jfb_ddp_safe
        self._last_a = None
        self.mixer_value=mixer_value
        self.T=T
        self.positive_threshold = positive_threshold
        self.multiscale = multiscale
        self.num_scales = len(num_basis)

        if multiscale:
            self.encoder = GaussianPyramidEncoder(levels = self.num_scales-1,concat_to_channels=False)
            self.decoder = self.encoder.decode
            self.injection_ls = [in_channels]*self.num_scales
        else:
            if len(num_basis) ==1:
                self.encoder = lambda x: [x]
                self.decoder = lambda x: x[0]
                self.injection_ls = [in_channels]
            else:
                self.encoder = lambda x: [x] + [None for _ in range(self.num_scales-1)]
                self.decoder = lambda x: x[0]
                self.injection_ls = [in_channels] + [None]*(self.num_scales-1)
            
        levels = []
        prev_channels  = None
        print("intra",intra)
        counter = 0
        self.injection_dic = {}
        for i, nb in enumerate(num_basis):
            in_channels = self.injection_ls[i]
            # print(in_channels)
            levels.append(
                neural_node(in_channels, prev_channels, nb, kernel_size=kernel_size,
                eta=eta_base,stride=stride,padding = kernel_size // 2 if i > 0 else 3, output_padding=output_padding,
                learning_horizontal=learning_horizontal, groups=groups, bias=bias,relu_6=relu_6, 
                h_groups=h_groups, constraint_energy = constraint_energy, k_inter = k_inter, tied_transpose=tied_transpose)
            )
            self.injection_dic[counter] = i
            counter+=1

            if n_hid_layers > 0:
                for _ in range(n_hid_layers):
                    levels.append(
                        neural_node(None, nb, nb, kernel_size=kernel_size,
                        eta=eta_base,stride=stride,padding = kernel_size // 2 if i > 0 else 3, output_padding=output_padding,
                        learning_horizontal=learning_horizontal, groups=groups, bias=bias,relu_6=relu_6, 
                        h_groups=h_groups, constraint_energy = constraint_energy, k_inter = k_inter, down=False, tied_transpose=tied_transpose)
                    )
                    counter+=1
            if intra:
                prev_channels = nb
        self.levels = nn.ModuleList(levels)

        self.n_levels=len(levels)
        self.noise_embedding = noise_embedding
        if noise_embedding:
            # --- Embedding maps (noise/label/augment -> emb_channels -> affine->c1) ---
            emb_channels   = num_basis[0] * channel_mult_emb
            noise_channels = num_basis[0] * channel_mult_noise

            self.map_noise   = PositionalEmbedding(num_channels=noise_channels, endpoint=True)
            self.map_layer0 = nn.Linear(noise_channels, emb_channels, bias=True)
            self.eta_affine = nn.Linear(emb_channels, 1, bias=True) if eta_emb else None
            # self.affines     = nn.ModuleList([nn.Linear(emb_channels, d if per_dim_threshold else 1, bias=True) for d in num_basis])
            self.affines =  nn.ModuleList([nn.Linear(emb_channels, node.num_basis if per_dim_threshold else 1, bias=True) for node in self.levels])
            for lin in self.affines:
                nn.init.zeros_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        else:
            self.lambda_bias = nn.ModuleList([torch.nn.Parameter(torch.zeros(1,1,1,1)) for _ in num_basis])
            
    def _reset_all_wnorm(self):
        if not getattr(self, '_wnorm', False):
            return
        # Recompute normalized conv weights inside the recurrent block
        for lvl in self.levels:
            if hasattr(lvl, 'reset_wnorm'):
                lvl.reset_wnorm()
                
    def _decompose_M_inter(self,n_components=1,steer_components=None):
        for lvl in self.levels:
            if hasattr(lvl, 'decompose_M_inter'):
                lvl.decompose_M_inter(n_components=n_components,steer_components=steer_components)
    

    def forward(self, x, a=None, upstream_grad=None, noise_labels=None,
     T=None,return_feature=False,infer_mode=False,n_iters=None,
     n_components =None,steer_components=None, neuron_steer = None,return_history=False):
        """If deq_mode=True: stochastic JFB; else: legacy unrolled."""
        B = x.shape[0]
        if T is None:
            T = self.T
        self._reset_all_wnorm()
        if n_components is not None:
            self._decompose_M_inter(n_components,steer_components)
        # --- Build embedding exactly in your order ---
        if noise_labels is None:
            noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

        if self.noise_embedding:

            emb = self.map_noise(noise_labels)                       # [B, Cn]
            emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
            emb = F.relu(self.map_layer0(emb))
            # noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            noise_emb_ls = [affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype) for affine in self.affines]
            if self.positive_threshold:
                # print("Using positive threshold")
                noise_emb_ls = [F.relu(emb) for emb in noise_emb_ls]
                # print(len(noise_emb))
            # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
            if self.eta_affine is not None:
                eta_emb = F.relu(self.eta_affine(emb)).to(x.dtype).unsqueeze(2).unsqueeze(3) # [B,1,1,1]
            else:
                eta_emb = None

            if neuron_steer is not None:
                for i,noise_emb in enumerate(noise_emb_ls):
                    _,_,h,w = neuron_steer["a_shape"][i]
                    noise_emb_ls[i] = noise_emb.repeat(1,1,h,w)
                    i,j,k = neuron_steer["neuron_idx"]
                    L = neuron_steer["neuron_level"]
                    noise_emb_ls[L][:,i,j,k] = noise_emb_ls[L][:,i,j,k]*neuron_steer["steer_value"][0] + neuron_steer["steer_value"][1]

                    

        else:
            noise_emb_ls = [self.lambda_bias for _ in range(self.n_levels)]

        x = self.encoder(x)

        # ----- JFB / DEQ mode -----
        if a is None:
            a = [None] * self.n_levels
        if upstream_grad is None:
            upstream_grad = [None] * self.n_levels
        decoded = [None] * self.n_levels

        # k ~ U{0..N}|dice>jfb_reuse_solution_rate
        k0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value =self.mixer_value) if self.jfb_reuse_solution_rate>random.random() else 0
        # n ~ U{0..N}
        n0 = sample_uniformly_with_long_tail(self.jfb_no_grad_iters[0], self.jfb_no_grad_iters[1],mixer_value=self.mixer_value)
        # m ~ U{1..M}
        m1 = random.randint(self.jfb_with_grad_iters[0], self.jfb_with_grad_iters[1])

        if infer_mode:
            n0 = n_iters
            m1 = 0
            k0 = 0
            
        # ---- no-grad phase ----
        if n0 > 0:
            with torch.no_grad():
                # either start with zero init, i.e. a, and upstream_grad are zero, or start with random init, 
                # i.e. a, and upstream_grad are infered from random image with same noise level
                
                if k0>0:
                    perm_idx = torch.randperm(B)
                    # print(perm_idx)
                    x_perm = [x_scale[perm_idx] for x_scale in x]
                    for _ in range(k0):
                        # self.forward_inter(x_perm, a, upstream_grad, noise_emb_ls, T=T)
                        self.forward_inter(x_perm, a, upstream_grad, decoded, noise_emb_ls, T=T,n_components=n_components,steer_components=steer_components,eta_emb=eta_emb)
                        
                history = []
                for _ in range(n0):
                    self.forward_inter(x, a, upstream_grad, decoded, noise_emb_ls, T=T,n_components=n_components,steer_components=steer_components,eta_emb=eta_emb)
                    if return_history:
                        a = [ai.detach() if ai is not None else None for ai in a]
                        upstream_grad = [ui.detach() if isinstance(ui,torch.Tensor) else 0 for ui in upstream_grad]
                        decoded = [decoded[i].detach().clone() for i in self.injection_dic.keys()]
                        features =  {"a":a,"decoded":decoded,"upstream_grad":upstream_grad,"denoised":self.decoder(decoded)}
                        history.append(features)

        # cut graph between phases
        a = [ai.detach() if ai is not None else None for ai in a]

        # ---- with-grad phase ----
        for _ in range(m1):
            self.forward_inter(x, a, upstream_grad, decoded, noise_emb_ls,n_components=n_components,steer_components=steer_components,eta_emb=eta_emb)

        decoded = [decoded[i] for i in self.injection_dic.keys()]
        # return decoded[0].sum(dim=1,keepdim=True)
        if return_feature:
            if return_history:
                return history
            else:
                a = [ai.detach().clone() if ai is not None else None for ai in a]
                upstream_grad = [ui.detach().clone() if isinstance(ui,torch.Tensor) else 0 for ui in upstream_grad]
                features =  {"a":a,"decoded":decoded,"upstream_grad":upstream_grad,"denoised":self.decoder(decoded)}
                return features
        else:
            return self.decoder(decoded)

    def forward_inter(self, x_in, a, upstream_grad, decoded, noise_emb_ls=None, T=0,n_components=None,steer_components=None,eta_emb=None):
        # bottom-up
        for i in range(len(self.levels)):
            # print(i)
            inp = None if i == 0 else a[i - 1]
            top_signal = upstream_grad[i + 1] if i < self.n_levels - 1 else None
            x_c = x_in[self.injection_dic[i]] if i in self.injection_dic else None
            a[i], upstream_grad[i], decoded[i] = self.levels[i](inp, a_prev=a[i], x_c = x_c, top_signal=top_signal, noise_emb = noise_emb_ls[i], T=T,n_components=n_components,steer_components=steer_components,eta_emb=eta_emb)

        # top-down
        for i in range(self.n_levels - 1, -1, -1):
            inp = None if i == 0 else a[i - 1]
            top_signal = upstream_grad[i + 1] if i < self.n_levels - 1 else None
            x_c = x_in[self.injection_dic[i]] if i in self.injection_dic else None
            a[i], upstream_grad[i], decoded[i] = self.levels[i](inp, a_prev=a[i], x_c = x_c, top_signal=top_signal, noise_emb = noise_emb_ls[i], T=T,n_components=n_components,steer_components=steer_components,eta_emb=eta_emb)
            


            # print([d.shape if d is not None else None for d in decoded])
    # def infer(self, x, a = None, decoded = None, noise_labels=None,n_iter = 1,return_feature=False,return_hist =False,T=None):
    #     if T is None:
    #         T = self.T
    #     self._reset_all_wnorm()
    #     B = x.shape[0]
    #     if a is None:
    #         a = [None] * self.n_levels
    #     if decoded is None:
    #         decoded = [None] * self.n_levels
        
    #     decoded_hist = {"a":[],"decoded":[]}
    #     # --- Build embedding exactly in your order ---
    #     if noise_labels is None:
    #         noise_labels = torch.zeros(B, device=x.device, dtype=x.dtype)

    #     if self.noise_embedding:

    #         emb = self.map_noise(noise_labels)                       # [B, Cn]
    #         emb = emb.reshape(B, 2, -1).flip(1).reshape(B, -1)       # swap sin/cos
    #         emb = F.relu(self.map_layer0(emb))
    #         noise_emb = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
    #         # noise_emb = emb.unsqueeze(2).unsqueeze(3).to(x.dtype)  # [B, c1, 1, 1]
    #     else:
    #         noise_emb = self.lambda_bias
            
    #     x = self.encoder(x)
        
    #     with torch.no_grad():
    #         for _ in range(n_iter):
    #             self.forward_inter(x, a, decoded, noise_emb,T=T)
    #             decoded_hist["a"].append([i.cpu().detach() for i in a])
    #             decoded_hist["decoded"].append([i.cpu().detach() for i in decoded])
    #     decoded_out = self.decoder(decoded[0])
    #     # decoded_out = decoded[0].sum(dim=1,keepdim=True)
            
    #     if return_feature:
    #         if return_hist:
    #             return a, decoded, decoded_hist
    #         else:
    #             return a, decoded
    #     else:
    #         return decoded_out

def UNetBlind64(**kwargs):
    return UNetBlindDenoise(image_size=64, **kwargs)

