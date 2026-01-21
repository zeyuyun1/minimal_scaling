import argparse
import json
import os
import re
from datetime import datetime

import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import WandbLogger
from torchvision.datasets import ImageFolder
from pytorch_lightning.utilities import rank_zero_only
from torchvision.utils import make_grid
# import sparselearning
# from sparselearning.core import Masking, CosineDecay, LinearDecay

from model import RecurrentConvNLayer,UNetBlind64,RecurrentConvNLayer2,RecurrentConvNLayer3
from model import RecurrentConvNLayer_cc,OneLayerAE_MinWithNoise,RecurrentOneLayer,RecurrentOneLayer_stable,RecurrentOneLayer_reuse,RecurrentOneLayer_DEQ,RecurrentOneLayer_splitNet
from model import RecurrentOneLayer_diverse,RecurrentLayers_diverse,neural_sheet,RecurrentConvNLayer_simple

from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math

import torch
from torch import nn

def _endswith_lookup(state_dict, suffix):
    for k in state_dict.keys():
        if k.endswith(suffix):
            return k
    return None


@torch.no_grad()
def _zero_module_weights_(mod: nn.Module, zero_bias: bool = True):
    """Zero conv/linear weights (handles spectral_norm's weight_orig if present)."""
    if hasattr(mod, "weight_orig"):   # spectral_norm / parametrizations
        mod.weight_orig.zero_()
        if hasattr(mod, "weight_v"):  # power-iter vector; keep it normalized
            mod.weight_v.normal_().div_(mod.weight_v.norm() + 1e-9)
    if hasattr(mod, "weight") and mod.weight is not None:
        mod.weight.zero_()
    if zero_bias and hasattr(mod, "bias") and mod.bias is not None:
        mod.bias.zero_()

@torch.no_grad()
def erase_all_M_weights(obj, zero_bias: bool = True, freeze: bool = False, verbose: bool = True) -> int:
    """
    Zero out every recurrent 'M' matrix in the model.
    Works if you pass a LightningModule (with .model/.ema_model) or a plain nn.Module.

    Args:
        obj: LightningModule or nn.Module
        zero_bias: also zero M.bias if it exists
        freeze: set requires_grad=False for all parameters in M
        verbose: print how many were zeroed

    Returns:
        Number of 'M' modules zeroed.
    """
    # resolve root model(s)
    roots = []
    if isinstance(obj, nn.Module) and hasattr(obj, "model"):
        roots.append(obj.model)
        # also do EMA if present
        if hasattr(obj, "ema_model") and isinstance(obj.ema_model, nn.Module):
            roots.append(obj.ema_model)
    elif isinstance(obj, nn.Module):
        roots.append(obj)
    else:
        raise TypeError("erase_all_M_weights expects a LightningModule or nn.Module.")
    total = 0
    for root in roots:
        # Preferred: iterate levels if present (fast and explicit)
        if hasattr(root, "levels"):
            for i, lev in enumerate(root.levels):
                if hasattr(lev, "M") and isinstance(lev.M, nn.Module):
                    _zero_module_weights_(lev.M, zero_bias=zero_bias)
                    if freeze:
                        for p in lev.M.parameters():
                            p.requires_grad = False
                    total += 1
        else:
            # Fallback: scan named submodules for anything literally named 'M'
            for name, mod in root.named_modules():
                if name.split(".")[-1] == "M" and isinstance(mod, nn.Module):
                    _zero_module_weights_(mod, zero_bias=zero_bias)
                    if freeze:
                        for p in mod.parameters():
                            p.requires_grad = False
                    total += 1

    if verbose:
        print(f"[erase_all_M_weights] zeroed {total} recurrent M module(s).")
    return total



@torch.no_grad()
def load_ff_encoders_from_checkpoint(lit_module, ckpt_path, copy_outer_encoder=True, copy_outer_decoder=True, strict_shape=True):
    """
    Copy ONLY feed-forward encoder weights (and biases) into:
      - lit_module.model
      - lit_module.ema_model
    from a checkpoint file that may be a PL checkpoint or a raw state_dict.

    Args:
        lit_module: your LightningModule (instance of LitDenoiser)
        ckpt_path:  path to .ckpt or .pth
        copy_outer_encoder: also copy model.encoder if present (whitening)
        copy_outer_decoder: also copy model.decoder if present
        strict_shape: if True, only copy when shapes match exactly
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)  # PL or raw

    def _copy_into(target_model):
        root = getattr(target_model, "model", target_model)  # in case of nested
        # (a) copy outer whitening encoder if requested
        if copy_outer_encoder and hasattr(root, "encoder") and isinstance(root.encoder, nn.Module):
            for suf in [".encoder.weight", ".encoder.conv.weight"]:
                key = _endswith_lookup(sd, f"{suf}")
                if key is None:
                    # try common prefixes
                    key = _endswith_lookup(sd, f".model{suf}") or _endswith_lookup(sd, f".model.model{suf}")
                if key is not None:
                    src_w = sd[key]
                    if hasattr(root.encoder, "weight"):
                        if (not strict_shape) or (root.encoder.weight.shape == src_w.shape):
                            root.encoder.weight.copy_(src_w)
                    # bias
                    bkey = key.replace("weight", "bias")
                    if hasattr(root.encoder, "bias") and (root.encoder.bias is not None) and (bkey in sd):
                        src_b = sd[bkey]
                        if (not strict_shape) or (root.encoder.bias.shape == src_b.shape):
                            root.encoder.bias.copy_(src_b)
                    break  # found one version

        # (a2) copy outer decoder if requested
        if copy_outer_decoder and hasattr(root, "decoder") and isinstance(root.decoder, nn.Module):
            for suf in [".decoder.weight", ".decoder.conv.weight"]:
                key = _endswith_lookup(sd, f"{suf}")
                if key is None:
                    # try common prefixes
                    key = _endswith_lookup(sd, f".model{suf}") or _endswith_lookup(sd, f".model.model{suf}")
                if key is not None:
                    src_w = sd[key]
                    if hasattr(root.decoder, "weight"):
                        if (not strict_shape) or (root.decoder.weight.shape == src_w.shape):
                            root.decoder.weight.copy_(src_w)
                    # bias
                    bkey = key.replace("weight", "bias")
                    if hasattr(root.decoder, "bias") and (root.decoder.bias is not None) and (bkey in sd):
                        src_b = sd[bkey]
                        if (not strict_shape) or (root.decoder.bias.shape == src_b.shape):
                            root.decoder.bias.copy_(src_b)
                    break  # found one version

        # (b) per-level encoders of RecurrentConvUnit
        if not hasattr(root, "levels"):
            return
        for i, lev in enumerate(root.levels):
            if not hasattr(lev, "encoder") or not isinstance(lev.encoder, nn.Module):
                continue
            # look for any of these suffixes in the ckpt:
            suffixes = [
                f".levels.{i}.encoder.weight",           # most common
                f".model.levels.{i}.encoder.weight",     # PL prefix
                f".model.model.levels.{i}.encoder.weight",
            ]
            key = None
            for suf in suffixes:
                key = _endswith_lookup(sd, suf)
                if key: break
            # fallback: suffix-only search
            if key is None:
                key = _endswith_lookup(sd, f".levels.{i}.encoder.weight")
            if key is None:
                # nothing found; skip this level
                continue
            src_w = sd[key]
            if hasattr(lev.encoder, "weight"):
                if (not strict_shape) or (lev.encoder.weight.shape == src_w.shape):
                    print("loading {} layer".format(i), src_w.shape)
                    lev.encoder.weight.copy_(src_w)
            # bias if present
            bkey = key.replace("weight", "bias")
            if hasattr(lev.encoder, "bias") and (lev.encoder.bias is not None) and (bkey in sd):
                src_b = sd[bkey]
                if (not strict_shape) or (lev.encoder.bias.shape == src_b.shape):
                    lev.encoder.bias.copy_(src_b)

    # copy into main model
    _copy_into(lit_module)
    # copy into EMA model too
    if hasattr(lit_module, "ema_model"):
        _copy_into(lit_module.ema_model)

import math, torch, torch.nn as nn, torch.nn.functional as F

@torch.no_grad()
def renorm_conv_to_kaiming_(conv: nn.Conv2d):
    if not isinstance(conv, nn.Conv2d): return
    w = conv.weight
    fan_in = w.shape[1] * w.shape[2] * w.shape[3]
    target_std = math.sqrt(2.0 / max(1, fan_in))
    cur_std = w.std().clamp_min(1e-8)
    scale = (target_std / cur_std).item()
    w.mul_(scale)
    # leave bias as-is

@torch.no_grad()
def spectral_rescale_conv2d_(conv: nn.Conv2d, x_shape=(1,1,64,64), target_sn=1.0, iters=10):
    """One-time power iteration to estimate ||W||_2 and rescale to <= target_sn."""
    if not isinstance(conv, nn.Conv2d): return
    device = conv.weight.device
    u = torch.randn(x_shape, device=device)
    u = u / (u.norm() + 1e-9)
    for _ in range(iters):
        v = F.conv2d(u, conv.weight, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        v = v / (v.norm() + 1e-9)
        u = F.conv_transpose2d(v, conv.weight, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
        u = u / (u.norm() + 1e-9)
    # Rayleigh quotient ≈ spectral norm
    v = F.conv2d(u, conv.weight, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups)
    sn = v.norm().item()
    if sn > 1e-8 and sn > target_sn:
        conv.weight.mul_(target_sn / sn)

def renorm_ff_stack_(root, mode="kaiming"):
    # outer whitening (if any)
    if hasattr(root, "encoder") and isinstance(root.encoder, nn.Conv2d):
        if mode=="kaiming": renorm_conv_to_kaiming_(root.encoder)
        else: spectral_rescale_conv2d_(root.encoder, target_sn=1.0)
    # per level
    for lev in getattr(root, "levels", []):
        if hasattr(lev, "encoder") and isinstance(lev.encoder, nn.Conv2d):
            if mode=="kaiming": renorm_conv_to_kaiming_(lev.encoder)
            else: spectral_rescale_conv2d_(lev.encoder, target_sn=1.0)



def _to_wandb_image(chw: torch.Tensor):
    """Convert CHW [0,1] → something wandb.Image likes."""
    chw = chw.detach().cpu()
    C = chw.size(0)
    if C == 1:
        return wandb.Image(chw[0].numpy())              # H×W
    elif C in (3, 4):
        return wandb.Image(chw.permute(1, 2, 0).numpy())  # H×W×3/4
    else:
        # collapse funny channel counts (e.g., 16) to grayscale
        return wandb.Image(chw.mean(dim=0).numpy())

# Import from our new modules
from utils import warmup_fn, vis_patches
from loss import EDMLossNoCond, SimpleUniformNoiseLoss, EDMStyleXPredLoss, SimpleNoiseLoss
from data import SubsetWithTransform, ImageDataModule, MNISTDataModule


def add_noise(x, sigma, distribution: str = "uniform",
              P_mean: float = -1.2, P_std: float = 1.2):
    """
    x: (B,C,H,W)
    sigma:
      * if distribution == "uniform": tuple/list (lo, hi) or float
      * if distribution == "edm": tuple/list (noise_min, noise_max) or None
    P_mean/P_std: EDM log-normal parameters (must match training)
    """
    B, _, _, _ = x.shape
    device = x.device

    if distribution == "edm":
        # σ ~ LogNormal(P_mean, P_std), then clamp to noise_range if given
        rnd = torch.randn(B, 1, 1, 1, device=device)
        sigma_sample = (rnd * P_std + P_mean).exp()
        if isinstance(sigma, (tuple, list)) and len(sigma) == 2:
            lo, hi = sigma
            if lo is not None:
                sigma_sample = sigma_sample.clamp_min(lo)
            if hi is not None:
                sigma_sample = sigma_sample.clamp_max(hi)
        return x + torch.randn_like(x) * sigma_sample, sigma_sample.flatten()

    # default: uniform (your original behavior)
    if isinstance(sigma, (tuple, list)) and len(sigma) == 2:
        lo, hi = sigma
        sigma_sample = torch.rand(B, 1, 1, 1, device=device) * (hi - lo) + lo
        return x + torch.randn_like(x) * sigma_sample, sigma_sample.flatten()
    else:
        return x + torch.randn_like(x) * float(sigma), float(sigma)

class EMACheckpointCallback(Callback):
    """Custom callback to save EMA model checkpoints."""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def on_validation_end(self, trainer, pl_module):
        """Save EMA model checkpoint at the end of validation."""
        if trainer.is_global_zero:  # Only save on main process
            ema_checkpoint_path = os.path.join(self.save_dir, "ema_model.ckpt")
            torch.save(pl_module.ema_model.state_dict(), ema_checkpoint_path)
            print(f"Saved EMA model checkpoint to: {ema_checkpoint_path}")
    
    def on_train_end(self, trainer, pl_module):
        """Save final EMA model checkpoinz at the end of training."""
        if trainer.is_global_zero:  # Only save on main process
            ema_checkpoint_path = os.path.join(self.save_dir, "final_ema_model.ckpt")
            torch.save(pl_module.ema_model.state_dict(), ema_checkpoint_path)
            print(f"Saved final EMA model checkpoint to: {ema_checkpoint_path}")


class LitDenoiser(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        num_basis: list,
        eta_base: float,
        n_iters_inter: int,
        kernel_size: int,
        stride: int,
        n_iters_intra: int,
        lr: float = 1e-3,
        sigma=(0.1, 1.5),
        model_arch = "h_sparse",
        ema_halflife_kimg: float = 500.0,
        ema_rampup_ratio: float = 0.05,
        P_mean: float = -2.0,
        P_std: float = 0.5,
        edm_weighting: bool = False,
        eta_ls: list = None,
        jfb_no_grad_iters: tuple = (0, 6),
        jfb_with_grad_iters: tuple = (1, 3),
        jfb_reuse_solution: bool = False,
        jfb_ddp_safe: bool = True,
        whiten_dim: int = None,
        learning_horizontal: bool = True,
        stable: bool = False,
        jfb_reuse_solution_rate: float = 0,
        frequency_groups: int = None,
        init_lambda: float = 0.0,
        whiten_ks: int = 3,
        noise_embedding: bool = True,
        loss_type: str = "edm",
        bias: bool = True,
        relu_6: bool = False,
        T: float = 0.1,
        pyramid: bool = False,
        groups: int = 2,
        h_groups: int = 2,
        energy_function: str = "elastic",
        per_dim_threshold: bool = False,
        positive_threshold: bool = False,
        multiscale: bool = False,
        num_basis_per_scale: list = None,
        constraint_energy: str = "SC",
        intra: bool = True,
        k_inter: int = None,
        n_hid_layers: int = -1,
        eta_emb: bool = False,
        tied_transpose: bool = True,
        num_classes: int = 0,
        cond_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        if loss_type == "uniform":
            self.loss_obj = SimpleUniformNoiseLoss(noise_range=sigma)
        else:
            self.loss_obj = EDMLossNoCond(noise_range=sigma,sigma_data=0.25,P_mean=P_mean,P_std=P_std,edm_weighting=edm_weighting)
        # self.loss_obj = SimpleUniformNoiseLoss(noise_range=sigma)
        # self.loss_obj = EDMLossNoCond(noise_range=sigma,sigma_data=0.15)
        # self.loss_obj = EDMLossNoCond(noise_range=sigma,sigma_data=0.15,P_mean=P_mean,P_std=P_std)
        # self.loss_obj = EDMStyleXPredLoss(noise_range=sigma,sigma_data=0.25)
        # self.loss_obj = SimpleNoiseLoss(noise_range=sigma,sigma_data=0.25,P_mean=-2,P_std=0.5,weighting='edm')
        if model_arch == "unet":
            self.model = UNetBlind64(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                n_iters_inter=n_iters_inter,
                kernel_size=kernel_size,
                stride=stride,
                n_iters_intra=n_iters_intra
            )
        elif model_arch == "recur_new":
            # print(eta_ls)
            self.model = RecurrentConvNLayer3(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                # n_iters_inter=n_iters_inter,
                kernel_size=kernel_size,
                stride=stride,
                eta_ls=eta_ls,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                # n_iters_intra=n_iters_intra,
                # whiten_dim=16,
            )
        elif model_arch == "recur_cc":
            self.model = RecurrentConvNLayer_cc(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                n_iters_inter=n_iters_inter,
                kernel_size=kernel_size,
                stride=stride,
                n_iters_intra=n_iters_intra,
                # whiten_dim=16,
            )
        elif model_arch == "AE":
            self.model = OneLayerAE_MinWithNoise(
                in_channels=in_channels,
                num_basis=num_basis,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                noise_embedding=noise_embedding,
                bias=bias,
            )
        elif model_arch == "SC":
            if stable:
                self.model = RecurrentOneLayer_stable(
                    in_channels=in_channels,
                    num_basis=num_basis,
                    eta_base=eta_base,
                    kernel_size=kernel_size,
                    stride=stride,
                    whiten_dim=whiten_dim,
                    jfb_no_grad_iters=jfb_no_grad_iters,
                    jfb_with_grad_iters=jfb_with_grad_iters,
                    jfb_reuse_solution=jfb_reuse_solution,
                    jfb_ddp_safe=jfb_ddp_safe,
                    learning_horizontal=learning_horizontal,
                )
            else:
                self.model = RecurrentOneLayer(
                    in_channels=in_channels,
                    num_basis=num_basis,
                    eta_base=eta_base,
                    kernel_size=kernel_size,
                    stride=stride,
                    whiten_dim=whiten_dim,
                    jfb_no_grad_iters=jfb_no_grad_iters,
                    jfb_with_grad_iters=jfb_with_grad_iters,
                    jfb_reuse_solution=jfb_reuse_solution,
                    jfb_ddp_safe=jfb_ddp_safe,
                    learning_horizontal=learning_horizontal,
                )
        elif model_arch == "SC_reuse":
            self.model = RecurrentOneLayer_reuse(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
            )
        elif model_arch == "SC_diverse":
            self.model = RecurrentOneLayer_diverse(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                energy_function=energy_function,
                pyramid=pyramid,
                groups=groups,
                h_groups=h_groups,
                per_dim_threshold=per_dim_threshold,
                positive_threshold=positive_threshold,
                multiscale=multiscale,
                num_basis_per_scale=num_basis_per_scale,
            )
        elif model_arch == "SC_NLayer_diverse":
            self.model = RecurrentLayers_diverse(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                energy_function=energy_function,
                pyramid=pyramid,
                groups=groups,
                h_groups=h_groups,
                per_dim_threshold=per_dim_threshold,
                positive_threshold=positive_threshold,
                multiscale=multiscale,
                num_basis_per_scale=num_basis_per_scale,
            )
        elif model_arch == "SC_DEQ":
            self.model = RecurrentOneLayer_DEQ(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
            )
        elif model_arch == "SC_splitNet":
            self.model = RecurrentOneLayer_splitNet(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                pyramid=pyramid,
                groups=groups,
                h_groups=h_groups,
            )
        elif model_arch == "SC_simple":
            self.model = RecurrentConvNLayer_simple(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                num_classes=num_classes,
                cond_drop_prob=cond_drop_prob,
            )
        elif model_arch == "neural_sheet":
            self.model = neural_sheet(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                init_lambda=init_lambda,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                constraint_energy=constraint_energy,
                groups=groups,
                h_groups=h_groups,
                per_dim_threshold=per_dim_threshold,
                positive_threshold=positive_threshold,
                multiscale=multiscale,
                intra=intra,
                k_inter=k_inter,
                n_hid_layers=n_hid_layers,
                eta_emb=eta_emb,
                tied_transpose=tied_transpose,
            )

# class neural_sheet(nn.Module):
#     def __init__(self,in_channels,num_basis,kernel_size=3,
#     stride=2,output_padding=1,whiten_dim=None,
#     learning_horizontal=True,eta_base=0.1,
#     jfb_no_grad_iters=None,jfb_with_grad_iters=None,
#     jfb_reuse_solution_rate=0,jfb_ddp_safe=True,
#     channel_mult_emb=2,channel_mult_noise=1,jfb_reuse_solution=0,mixer_value=0.0, 
#     init_lambda=0.1,noise_embedding=True,bias=True,
#     relu_6=False,T=0.1,groups=1,h_groups=1,constraint_energy="SC",
#     per_dim_threshold=True,positive_threshold=False,multiscale=False):
        
        else:
            self.model = RecurrentConvNLayer(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                n_iters_inter=n_iters_inter,
                kernel_size=kernel_size,
                stride=stride,
                n_iters_intra=n_iters_intra,
                # whiten_dim=16,
            )
        print(type(self.model))
        # Initialize EMA model mirroring the selected architecture
        if model_arch == "unet":
            self.ema_model = UNetBlind64(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                n_iters_inter=n_iters_inter,
                kernel_size=kernel_size,
                stride=stride,
                n_iters_intra=n_iters_intra,
            )
        elif model_arch == "recur_new":
            self.ema_model = RecurrentConvNLayer3(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                eta_ls=eta_ls,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
            )
        elif model_arch == "AE":
            self.ema_model = OneLayerAE_MinWithNoise(
                in_channels=in_channels,
                num_basis=num_basis,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                noise_embedding=noise_embedding,
                bias=bias,
            )
        elif model_arch == "SC":
            if stable:
                self.ema_model = RecurrentOneLayer_stable(
                    in_channels=in_channels,
                    num_basis=num_basis,
                    eta_base=eta_base,
                    kernel_size=kernel_size,
                    stride=stride,
                    whiten_dim=whiten_dim,
                    jfb_no_grad_iters=jfb_no_grad_iters,
                    jfb_with_grad_iters=jfb_with_grad_iters,
                    jfb_reuse_solution=jfb_reuse_solution,
                    jfb_ddp_safe=jfb_ddp_safe,
                    learning_horizontal=learning_horizontal,
                )
            else:
                self.ema_model = RecurrentOneLayer(
                    in_channels=in_channels,
                    num_basis=num_basis,
                    eta_base=eta_base,
                    kernel_size=kernel_size,
                    stride=stride,
                    whiten_dim=whiten_dim,
                    jfb_no_grad_iters=jfb_no_grad_iters,
                    jfb_with_grad_iters=jfb_with_grad_iters,
                    jfb_reuse_solution=jfb_reuse_solution,
                    jfb_ddp_safe=jfb_ddp_safe,
                    learning_horizontal=learning_horizontal,
                )
        elif model_arch == "SC_reuse":
            self.ema_model = RecurrentOneLayer_reuse(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
            )
        elif model_arch == "SC_DEQ":
            self.ema_model = RecurrentOneLayer_DEQ(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
            )
        elif model_arch == "SC_splitNet":
            self.ema_model = RecurrentOneLayer_splitNet(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                pyramid=pyramid,
                groups=groups,
                h_groups=h_groups,
            )
        elif model_arch == "SC_diverse":
            self.ema_model = RecurrentOneLayer_diverse(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                pyramid=pyramid,
                groups=groups,
                h_groups=h_groups,
                energy_function=energy_function,
                per_dim_threshold=per_dim_threshold,
                positive_threshold=positive_threshold,
                multiscale=multiscale,
                num_basis_per_scale=num_basis_per_scale,
            )
        elif model_arch == "SC_NLayer_diverse":
            self.ema_model = RecurrentLayers_diverse(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                frequency_groups=frequency_groups,
                init_lambda=init_lambda,
                whiten_ks=whiten_ks,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                energy_function=energy_function,
                pyramid=pyramid,
                groups=groups,
                h_groups=h_groups,
                per_dim_threshold=per_dim_threshold,
                positive_threshold=positive_threshold,
                multiscale=multiscale,
                num_basis_per_scale=num_basis_per_scale,
            )
        elif model_arch == "SC_simple":
            self.ema_model = RecurrentConvNLayer_simple(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                num_classes=num_classes,
                cond_drop_prob=cond_drop_prob,
            )
        elif model_arch == "neural_sheet":
           self.ema_model = neural_sheet(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                kernel_size=kernel_size,
                stride=stride,
                whiten_dim=whiten_dim,
                jfb_no_grad_iters=jfb_no_grad_iters,
                jfb_with_grad_iters=jfb_with_grad_iters,
                jfb_reuse_solution=jfb_reuse_solution,
                jfb_ddp_safe=jfb_ddp_safe,
                learning_horizontal=learning_horizontal,
                jfb_reuse_solution_rate=jfb_reuse_solution_rate,
                init_lambda=init_lambda,
                noise_embedding=noise_embedding,
                bias=bias,
                relu_6=relu_6,
                T=T,
                constraint_energy=constraint_energy,
                groups=groups,
                h_groups=h_groups,
                per_dim_threshold=per_dim_threshold,
                positive_threshold=positive_threshold,
                multiscale=multiscale,
                intra=intra,
                k_inter=k_inter,
                n_hid_layers=n_hid_layers,
                eta_emb=eta_emb,
            )
        else:
            self.ema_model = RecurrentConvNLayer(
                in_channels=in_channels,
                num_basis=num_basis,
                eta_base=eta_base,
                n_iters_inter=n_iters_inter,
                kernel_size=kernel_size,
                stride=stride,
                n_iters_intra=n_iters_intra,
            )        
        print("EMA",type(self.ema_model))
        # Copy initial parameters from main model to EMA model
        self.ema_model.load_state_dict(self.model.state_dict())
        # Freeze EMA model parameters (they will be updated via EMA, not gradients)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def forward(self, x, noise_labels=None):
        return self.model(x, noise_labels)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        per_pix = self.loss_obj(self.model, x, class_labels=labels)
        loss = per_pix.mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)        
        return loss


    def optimizer_step(self, *args, **kwargs):
        # perform the actual optimizer step first
        super().optimizer_step(*args, **kwargs)
        # then update EMA exactly once per optimizer step
        self.update_ema()
    
    @torch.no_grad()
    def update_ema(self):
        gb = self._global_batch_size()
        cur_nimg = self._cur_nimg()

        ema_halflife_nimg = int(self.hparams.ema_halflife_kimg * 1000)
        if self.hparams.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, max(cur_nimg, 1) * self.hparams.ema_rampup_ratio)

        beta = 0.5 ** (gb / max(ema_halflife_nimg, 1))
        for p_ema, p_net in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, beta))


    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_in, noise_labels = add_noise(x, self.hparams.sigma)
        # x_in = add_noise(x, self.hparams.sigma,distribution="edm",P_mean=self.hparams.P_mean,P_std=self.hparams.P_std)
        
        # Use EMA model for validation (better performance)
        with torch.no_grad():
            x_hat = self.ema_model(x_in, noise_labels=noise_labels)
    
        # compute batch signal and noise power
        sig_pow   = torch.sum(x ** 2)
        noise_pow = torch.sum((x - x_hat) ** 2)
    
        # compute SNR in dB and log
        snr_db = 10 * torch.log10(sig_pow / noise_pow)
        self.log("val_snr_db", snr_db, on_step=False, on_epoch=True, sync_dist=True)

        if batch_idx == 0:                     # only first batch each epoch
            self._log_images_rank0(x, x_in, x_hat)   # will execute only on global rank 0
    
        return snr_db

    @rank_zero_only
    def _log_images_rank0(self, x, x_in, x_hat):
        """
        x, x_in, x_hat: (B, C, H, W), values can be in [-1,1] or arbitrary — we normalize in vis_patches.
        """
        n_vis = min(4, x.shape[0])

        # 3 panels: clean | noisy | denoised  (each is a grid over n_vis images)
        a = vis_patches(x[:n_vis].detach().cpu().flatten(2),     show=False, return_tensor=True)  # CHW [0,1]
        b = vis_patches(x_in[:n_vis].detach().cpu().flatten(2),  show=False, return_tensor=True)
        c = vis_patches(x_hat[:n_vis].detach().cpu().flatten(2), show=False, return_tensor=True)

        # stack as a batch and tile horizontally
        panel = make_grid(torch.stack([a, b, c], dim=0), nrow=3, padding=4)  # CHW
        # re-normalize just in case; safe even if already [0,1]
        panel = (panel - panel.min()) / (panel.max() - panel.min() + 1e-8)

        # Resolve your net for pulling weights
        net = getattr(self, "ema_model", None) or self.model
        root = getattr(net, "model", net)

        # ---- layer-1 filters ----
        # w2: [out_c, in_c, kh, kw]
        model_arch = self.hparams.model_arch
        with torch.no_grad():
            if model_arch == "AE":
                w2 = root.D1.weight.detach()
                if self.hparams.whiten_dim is not None:
                    w2 = root.dec0(w2).cpu()
            else:
                if self.hparams.multiscale and self.hparams.model_arch != "neural_sheet":
                    w2 = root.levels[0].decoders[1].conv.weight.detach()
                else:
                    w2 = root.levels[0].decoder.conv.weight.detach()
                if self.hparams.whiten_dim is not None:
                    if self.hparams.frequency_groups is not None or self.hparams.groups is not None:
                        pass
                    else:
                        w2 = root.decoder(w2).cpu()
        # with torch.no_grad():
        #     w2 = root.decoder(w2).cpu()
        # .cpu()

        out_c, in_c, kh, kw = w2.shape
        patches = w2.view(out_c, in_c, kh * kw)   # (B=out_c, C=in_c, P)
        # If in_c not in {1,3}, collapse to 1 channel so W&B can render it
        if in_c not in (1, 3):
            patches = patches.mean(dim=1, keepdim=True)  # (B,1,P)

        basis_img = vis_patches(
            patches, normalize=True, return_tensor=True,
            ncol=int(math.ceil(math.sqrt(out_c)))
        )  # CHW in [0,1]

        # ---- Log to W&B ----
        self.logger.experiment.log(
            {
                "filters/layer0": _to_wandb_image(basis_img),
                "validation_panel": _to_wandb_image(panel),
            },
            step=self.global_step,
        )



    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer =  optim.Adam(self.parameters(), lr=self.hparams.lr)
        # total_steps = self.trainer.estimated_stepping_batches
        # decay = CosineDecay(args.prune_rate, T_max=total_steps)
        # mask = Masking(optimizer, prune_rate_decay=decay,
        #             prune_rate=args.prune_rate, growth_mode=args.growth,
        #             prune_mode=args.prune, redistribution_mode=args.redistribution,
        #             prune_every_k_steps=100)
        # mask.add_module(
        #     model,
        #     density=0.05,                 # density ONLY for that tensor
        #     sparse_init='constant',
        #     debug_find=True,
        #     keep_param_names=("M_inter.weight_v",)  # or the exact printed name prefix
        # )
        scheduler = {
            'scheduler': LambdaLR(optimizer, lr_lambda=warmup_fn),
            'interval': 'step',
        }
        return [optimizer], [scheduler]

    def _world_size(self) -> int:
        # PL 2.x usually has .world_size; fall back to num_devices/1
        return int(getattr(self.trainer, "world_size",
                        getattr(self.trainer, "num_devices", 1)) or 1)

    def _accum(self) -> int:
        return int(getattr(self.trainer, "accumulate_grad_batches", 1))

    def _per_device_bs(self) -> int:
        # works with your DataModules
        return int(getattr(self.trainer.datamodule, "batch_size", 1))

    def _global_batch_size(self) -> int:
        return self._per_device_bs() * self._world_size() * self._accum()

    def _cur_nimg(self) -> int:
        # global_step increments per optimizer step in PL
        return int(self.global_step) * self._global_batch_size()
    


if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    parser = argparse.ArgumentParser(description="Train a recurrent conv denoiser with PyTorch Lightning.")
    parser.add_argument("--project_name", type=str, default="main",
                        help="Name of the project")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Name of the experiment. Results will be saved under pretrained_model/<exp_name>.")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Path to a JSON config file. Overrides other args.")
    parser.add_argument("--config_dir", type=str, default="configs",
                        help="Directory to save generated config files.")
    parser.add_argument("--data_dir", type=str, default="~/celeba",
                        help="Directory to save generated config files.")
    # General training args
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    # Model hyperparameters
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_basis", type=lambda s: [int(item) for item in s.split(',')], default="64,64",
                        help="Comma-separated list for number of basis per layer, e.g. '64,64'.")
    parser.add_argument("--eta_base", type=float, default=0.25)
    parser.add_argument("--n_iters_inter", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n_iters_intra", type=int, default=1)
    parser.add_argument("--sigma", type=lambda s: tuple(map(float, s.split(','))), default="0.5,0.5",
                        help="Noise sigma as single float or range 'min,max'.")
    parser.add_argument("--model_arch", type=str, default="h_sparse")
    parser.add_argument("--dataset", type=str, choices=["mnist","celeba"], default="celeba",
                        help="Dataset to train on: 'mnist' or 'celeba'.")
    parser.add_argument("--grayscale", action="store_true", default=False,
                        help="If set, convert images to grayscale (1 channel) in the dataloader.")
    parser.add_argument("--img_size", type=int, default=64,
                        help="Target image size for resizing/cropping (square). Ignored if --no_resize.")
    parser.add_argument("--no_resize", action="store_true", default=False,
                        help="Disable resize/center-crop in the dataloader transforms.")
    parser.add_argument("--resize_size", type=int, default=512,
                        help="Resize size for the dataloader transforms.")
    parser.add_argument("--random_crop", action="store_true", default=False,
                        help="Use RandomCrop(img_size) instead of CenterCrop. If --no_resize is set, only crop.")
    # EDM noise distribution parameters
    parser.add_argument("--P_mean", type=float, default=-1.2,
                        help="Mean parameter for EDM log-normal noise distribution (default: -2.0)")
    parser.add_argument("--P_std", type=float, default=1.2,
                        help="Std parameter for EDM log-normal noise distribution (default: 0.5)")
    parser.add_argument("--edm_weighting", action="store_true", default=False,
                        help="Use EDM weighting for the loss function (default: False)")
    # EMA hyperparameters
    parser.add_argument("--ema_halflife_kimg", type=float, default=500.0,
                        help="EMA half-life in thousands of images (default: 500.0)")
    parser.add_argument("--ema_rampup_ratio", type=float, default=0.05,
                        help="EMA ramp-up ratio (default: 0.05)")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Path to checkpoint file to load for initialization (e.g., 'pretrained_model/main/00001_experiment/denoiser.ckpt')")
    parser.add_argument("--eta_ls", type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help="Comma-separated list for step size for each layer, e.g. '0.1,0.1'.")
    # jfb_no_grad_iters
    parser.add_argument("--jfb_no_grad_iters", type=lambda s: tuple(map(int, s.split(','))), default="0,6",
                        help="Comma-separated list for number of no-grad iterations for each layer, e.g. '0,6'.")
    parser.add_argument("--jfb_with_grad_iters", type=lambda s: tuple(map(int, s.split(','))), default="1,3",
                        help="Comma-separated list for number of with-grad iterations for each layer, e.g. '1,3'.")
    parser.add_argument("--jfb_reuse_solution", action="store_true", default=False,
                        help="Use reuse solution for JFB (default: False)")
    parser.add_argument("--jfb_ddp_safe", action="store_true", default=False,
                        help="Use DDP-safe JFB (default: True)")
    parser.add_argument("--load_all_weights", action="store_true", default=False,
                        help="Load all weights from checkpoint (default: False)")
    parser.add_argument("--whiten_dim", type=int, default=None,
                        help="Whiten dimension for AE (default: None)")
    parser.add_argument("--no_learning_horizontal", action="store_true", default=False,
                        help="No learning horizontal for gram (default: False)")
    parser.add_argument("--zero_M_weights", action="store_true", default=False,
                        help="Zero M weights (default: False)")
    parser.add_argument("--stable", action="store_true", default=False,
                        help="Stable model (default: False)")
    parser.add_argument("--jfb_reuse_solution_rate", type=float, default=0,
                        help="JFB reuse solution rate (default: 0)")
    parser.add_argument("--frequency_groups", type=int, default=None,
                        help="Frequency groups (default: None)")
    parser.add_argument("--init_lambda", type=float, default=0.0,
                        help="Initial lambda for the model (default: 0.0)")
    parser.add_argument("--whiten_ks", type=int, default=3,
                        help="Whiten kernel size (default: 3)")
    parser.add_argument("--no_noise_embedding", action="store_true", default=False,
                        help="No noise embedding (default: False)")
    parser.add_argument("--loss_type", type=str, choices=["edm", "uniform"], default="edm",
                        help="Loss type (default: edm)")
    parser.add_argument("--no_bias", action="store_true", default=False,
                        help="No bias (default: False)")
    parser.add_argument("--relu_6", action="store_true", default=False,
                        help="Use ReLU6 (default: False)")
    parser.add_argument("--T", type=float, default=0.0,
                        help="Temperature for the model (default: 0.1)")
    parser.add_argument("--pyramid", action="store_true", default=False,
                        help="Use Gaussian pyramid (default: False)")
    parser.add_argument("--groups", type=int, default=2,
                        help="Groups (default: 2)")
    parser.add_argument("--h_groups", type=int, default=2,
                        help="Horizontal groups (default: 2)")
    parser.add_argument("--energy_function", type=str, choices=["elastic", "BM","SC","hybrid"], default="elastic",
                        help="Energy function (default: elastic)")
    parser.add_argument("--per_dim_threshold", action="store_true", default=False,
                        help="Per-dimension threshold (default: False)")
    parser.add_argument("--positive_threshold", action="store_true", default=False,
                        help="Positive threshold (default: False)")
    parser.add_argument("--multiscale", action="store_true", default=False,
                        help="Multiscale (default: False)")
    parser.add_argument("--num_basis_per_scale", type=lambda s: [int(item) for item in s.split(',')], default=None,
                        help="Comma-separated list for number of basis per scale, e.g. '64,64'.")
    parser.add_argument("--constraint_energy", type=str, choices=["SC", "BM",], default="SC",
                        help="Constraint energy (default: SC)")
    parser.add_argument("--intra", action="store_true", default=False,
                        help="Intra (default: True)")
    parser.add_argument("--k_inter", type=int, default=None,
                        help="k for inter layer weight norm (default: None)")
    parser.add_argument("--n_hid_layers", type=int, default=-1,
                        help="Number of hidden layers (default: -1)")
    parser.add_argument("--eta_emb", action="store_true", default=False,
                        help="Use eta embedding (default: False)")
    parser.add_argument("--num_classes", type=int, default=0,
                        help="Number of classes (default: 0)")
    parser.add_argument("--cond_drop_prob", type=float, default=0.0,
                        help="Conditional dropout probability (default: 0.0)")
    parser.add_argument(
        "--no_tied_transpose",
        action="store_false",
        default=True,
        help="Disable tied transpose (default: enabled)"
    )
    # sparselearning.core.add_sparse_args(parser)
    args = parser.parse_args()
    
    # Set up experiment directory with index to avoid collisions
    # Check if we're in a distributed environment using environment variables
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    project_dir = os.path.join("pretrained_model", args.project_name)
    os.makedirs(project_dir, exist_ok=True)
    
    # Only rank 0 creates the directory and determines the index
    if local_rank == 0:
        # Find existing experiment directories and get the next index
        prev_run_dirs = []
        if os.path.isdir(project_dir):
            prev_run_dirs = [x for x in os.listdir(project_dir) if os.path.isdir(os.path.join(project_dir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        
        # Create experiment directory with index
        exp_dir_name = f"{cur_run_id:05d}_{args.exp_name}"
        base_dir = os.path.join(project_dir, exp_dir_name)
        os.makedirs(base_dir, exist_ok=True)
        
        # Write the directory name to a file for other processes to read
        with open(os.path.join(project_dir, ".current_exp"), 'w') as f:
            f.write(exp_dir_name)
    else:
        # Non-rank-0 processes wait a bit and then read the directory name
        import time
        time.sleep(0.1)  # Give rank 0 time to create the directory
        
        try:
            with open(os.path.join(project_dir, ".current_exp"), 'r') as f:
                exp_dir_name = f.read().strip()
            base_dir = os.path.join(project_dir, exp_dir_name)
        except (OSError, IOError):
            # Fallback if file doesn't exist
            exp_dir_name = f"00000_{args.exp_name}"
            base_dir = os.path.join(project_dir, exp_dir_name)

    # Load or save config
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config_file}")
    else:
        config = vars(args)

    # Save config in experiment folder
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Saved configuration to {config_path}")
    
    wandb_logger = WandbLogger(
        project=args.project_name,
        name=args.exp_name,
        save_dir=base_dir,           # where to store wandb files
        log_model=True               # automatically upload checkpoints
    )
    
    # Prepare data and model
    # dm = MNISTDataModule(batch_size=args.batch_size)
    lr = args.lr*args.batch_size/64*args.gpus
    # print(args.eta_ls)
    model = LitDenoiser(
        in_channels=args.in_channels,
        num_basis=args.num_basis,
        eta_base=args.eta_base,
        n_iters_inter=args.n_iters_inter,
        kernel_size=args.kernel_size,
        stride=args.stride,
        lr=lr,
        sigma=args.sigma,
        n_iters_intra=args.n_iters_intra,
        ema_halflife_kimg=args.ema_halflife_kimg,
        ema_rampup_ratio=args.ema_rampup_ratio,
        model_arch = args.model_arch,
        P_mean=args.P_mean,
        P_std=args.P_std,
        edm_weighting = args.edm_weighting,
        eta_ls=args.eta_ls,
        jfb_no_grad_iters=args.jfb_no_grad_iters,
        jfb_with_grad_iters=args.jfb_with_grad_iters,
        jfb_reuse_solution=args.jfb_reuse_solution,
        jfb_ddp_safe=args.jfb_ddp_safe,
        whiten_dim=args.whiten_dim,
        learning_horizontal=not args.no_learning_horizontal,
        stable=args.stable,
        jfb_reuse_solution_rate=args.jfb_reuse_solution_rate,
        frequency_groups=args.frequency_groups,
        init_lambda=args.init_lambda,
        whiten_ks=args.whiten_ks,
        noise_embedding=not args.no_noise_embedding,
        loss_type=args.loss_type,
        bias=not args.no_bias,
        relu_6=args.relu_6,
        T=args.T,
        pyramid=args.pyramid,
        groups=args.groups,
        h_groups=args.h_groups,
        energy_function=args.energy_function,
        per_dim_threshold=args.per_dim_threshold,
        positive_threshold=args.positive_threshold,
        multiscale=args.multiscale,
        num_basis_per_scale=args.num_basis_per_scale,
        constraint_energy=args.constraint_energy,
        intra=args.intra,
        k_inter=args.k_inter,
        n_hid_layers=args.n_hid_layers,
        eta_emb=args.eta_emb,
        tied_transpose=not args.no_tied_transpose,
        num_classes=args.num_classes,
        cond_drop_prob=args.cond_drop_prob,
    )
    
    # Load checkpoint if specified
    if args.checkpoint_path:
        if args.load_all_weights:
            if not os.path.exists(args.checkpoint_path):
                raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
            
            print(f"Loading checkpoint from: {args.checkpoint_path}")
            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
            
            def fix_state_dict_keys(state_dict, prefix="model."):
                """Fix state dict keys by adding the model prefix if needed"""
                fixed_dict = {}
                for key, value in state_dict.items():
                    if key.startswith(prefix):
                        # Already has the correct prefix
                        fixed_dict[key] = value
                    else:
                        # Add the prefix
                        fixed_dict[prefix + key] = value
                return fixed_dict
            
            # Load the model state dict
            if 'state_dict' in checkpoint:
                # PyTorch Lightning checkpoint format
                state_dict = checkpoint['state_dict']
                # Try loading as-is first
                try:
                    model.load_state_dict(state_dict, strict=True)
                    print("Loaded model weights from PyTorch Lightning checkpoint")
                except RuntimeError as e:
                    if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                        print("Key mismatch detected, attempting to fix keys...")
                        # Try fixing keys by adding model prefix
                        fixed_state_dict = fix_state_dict_keys(state_dict, "model.")
                        model.load_state_dict(fixed_state_dict, strict=False)
                        print("Loaded model weights with key fixes applied")
                    else:
                        raise e
            else:
                # Direct state dict format
                try:
                    model.load_state_dict(checkpoint, strict=True)
                    print("Loaded model weights from state dict")
                except RuntimeError as e:
                    if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                        print("Key mismatch detected, attempting to fix keys...")
                        # Try fixing keys by adding model prefix
                        fixed_state_dict = fix_state_dict_keys(checkpoint, "model.")
                        model.load_state_dict(fixed_state_dict, strict=False)
                        print("Loaded model weights with key fixes applied")
                    else:
                        raise e
            
            # Also try to load EMA model if available
            if 'ema_model' in checkpoint:
                try:
                    model.ema_model.load_state_dict(checkpoint['ema_model'], strict=True)
                    print("Loaded EMA model weights from checkpoint")
                except RuntimeError as e:
                    if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
                        print("EMA key mismatch detected, attempting to fix keys...")
                        fixed_ema_dict = fix_state_dict_keys(checkpoint['ema_model'], "model.")
                        model.ema_model.load_state_dict(fixed_ema_dict, strict=False)
                        print("Loaded EMA model weights with key fixes applied")
                    else:
                        raise e
            else:
                # If no EMA weights in checkpoint, copy from main model
                model.ema_model.load_state_dict(model.model.state_dict())
                print("Initialized EMA model with main model weights")
            if args.zero_M_weights:
                erase_all_M_weights(model, zero_bias=True, freeze=False, verbose=True)
        else:
            print("Loading ONLY feed-forward encoders from: {args.checkpoint_path}")
            load_ff_encoders_from_checkpoint(
                model,
                args.checkpoint_path,
                copy_outer_encoder=True,
                copy_outer_decoder=True,
                strict_shape=True,
            )
            renorm_ff_stack_(model.model, mode="kaiming")
            renorm_ff_stack_(model.ema_model, mode="kaiming")
            print("Checkpoint loading completed successfully!")




    
    # Select DataModule based on dataset
    if args.dataset == "mnist":
        dm = MNISTDataModule(data_dir="./data", batch_size=args.batch_size)
    else:
        dm = ImageDataModule(
            data_dir=args.data_dir,
            # data_dir="/home/zeyu/data/celeba",
            batch_size=args.batch_size,
            num_workers=4,
            img_size=args.img_size,
            val_split=0.1,
            test_split=0.1,
            seed=42,
            grayscale=args.grayscale,
            no_resize=args.no_resize,
            # resize_size=args.resize_size,
            random_crop=args.random_crop,
        )


    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=base_dir,
        filename="denoiser",
        save_top_k=1,
        # monitor="val_loss",
        # mode="min",
        monitor="val_snr_db",
        mode="max",
        save_weights_only=True,
    )

    # EMA checkpoint callback
    ema_callback = EMACheckpointCallback(save_dir=base_dir)
    
    # Trainer setup
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = args.gpus if torch.cuda.is_available() else 1
    strategy = "ddp" if accelerator == "gpu" and args.gpus > 1 else 'auto'
    # strategy = 'auto'
    # import os
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        # strategy="ddp_find_unused_parameters_true",
        max_epochs=args.n_epochs,
        callbacks=[checkpoint_callback, ema_callback],
        logger=wandb_logger,
    )

    # Run training and testing
    trainer.fit(model, dm)
    trainer.test(model, dm)

    print(f"Best model weights saved at: {checkpoint_callback.best_model_path}")
