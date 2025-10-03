import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random


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
            output_padding=self.output_padding
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
        torch.nn.init.dirac_(self.M.weight)
        
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
                 kernel_size=5, stride=2, output_padding=1, whiten_dim=None,
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


# ----------------------------------------
# Factory functions, mirroring the originals
# ----------------------------------------


def UNetBlind64(**kwargs):
    return UNetBlindDenoise(image_size=64, **kwargs)

