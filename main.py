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

from model import OneLayerAE_MinWithNoise
from torch.optim.lr_scheduler import LambdaLR
import wandb
import numpy as np
import math
# Import from our new modules
from utils import warmup_fn, vis_patches
from loss import EDMLossNoCond, SimpleUniformNoiseLoss
from data import ImageDataModule, MNISTDataModule



import math, torch, torch.nn as nn, torch.nn.functional as F

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
        kernel_size: int,
        stride: int,
        lr: float = 1e-3,
        sigma=(0.1, 1.5),
        model_arch = "h_sparse",
        ema_halflife_kimg: float = 500.0,
        ema_rampup_ratio: float = 0.05,
        P_mean: float = -2.0,
        P_std: float = 0.5,
        edm_weighting: bool = False,
        eta_ls: list = None,
        whiten_dim: int = None,
        noise_embedding: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_obj = SimpleUniformNoiseLoss(noise_range=sigma)
        # self.loss_obj = EDMLossNoCond(
        #     noise_range=sigma,
        #     sigma_data=0.25,
        #     P_mean=P_mean,
        #     P_std=P_std,
        #     edm_weighting=edm_weighting,
        # )

        # Single supported architecture: AE with noise embedding
        self.model = OneLayerAE_MinWithNoise(
            in_channels=in_channels,
            num_basis=num_basis,
            kernel_size=kernel_size,
            stride=stride,
            whiten_dim=whiten_dim,
            noise_embedding=noise_embedding,
            bias=bias,
        )
        self.ema_model = OneLayerAE_MinWithNoise(
            in_channels=in_channels,
            num_basis=num_basis,
            kernel_size=kernel_size,
            stride=stride,
            whiten_dim=whiten_dim,
            noise_embedding=noise_embedding,
            bias=bias,
        )
        # Copy initial parameters from main model to EMA model
        self.ema_model.load_state_dict(self.model.state_dict())
        # Freeze EMA model parameters (they will be updated via EMA, not gradients)
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def forward(self, x, noise_labels=None):
        return self.model(x, noise_labels)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        per_pix = self.loss_obj(self.model, x)
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
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
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
    parser.add_argument("--eta_ls", type=lambda s: [float(item) for item in s.split(',')], default=None,
                        help="Comma-separated list for step size for each layer, e.g. '0.1,0.1'.")
    parser.add_argument("--load_all_weights", action="store_true", default=False,
                        help="Load all weights from checkpoint (default: False)")
    parser.add_argument("--whiten_dim", type=int, default=None,
                        help="Whiten dimension for AE (default: None)")
    parser.add_argument("--zero_M_weights", action="store_true", default=False,
                        help="Zero M weights (default: False)")
    parser.add_argument("--no_noise_embedding", action="store_true", default=False,
                        help="No noise embedding (default: False)")
    parser.add_argument("--no_bias", action="store_true", default=False,
                        help="No bias (default: False)")
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
    
    model = LitDenoiser(
        in_channels=args.in_channels,
        num_basis=args.num_basis,
        kernel_size=args.kernel_size,
        stride=args.stride,
        lr=lr,
        sigma=args.sigma,
        ema_halflife_kimg=args.ema_halflife_kimg,
        ema_rampup_ratio=args.ema_rampup_ratio,
        model_arch = args.model_arch,
        P_mean=args.P_mean,
        P_std=args.P_std,
        edm_weighting = args.edm_weighting,
        whiten_dim=args.whiten_dim,
        noise_embedding=not args.no_noise_embedding,
        bias=not args.no_bias,
        eta_ls=args.eta_ls,
    )
    
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
        max_epochs=args.n_epochs,
        callbacks=[checkpoint_callback, ema_callback],
        logger=wandb_logger,
    )

    # Run training and testing
    trainer.fit(model, dm)
    trainer.test(model, dm)

    print(f"Best model weights saved at: {checkpoint_callback.best_model_path}")
