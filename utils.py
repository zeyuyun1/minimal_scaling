import numpy as np
import torch
from torchvision.utils import make_grid
import math


def warmup_fn(step):
    warmup_steps = 500
    return min((step + 1) / warmup_steps, 1.0)


# def vis_patches(patches, title="", figsize=None, colorbar=False,
#                       ncol=None, pad_value="min", show=True,
#                       return_tensor=False, vmin=None, vmax=None, fontsize=20,
#                       dpi=None, normalize = False):
#     """
#     Given patches of images in the dataset, create a grid and display it.

#     Parameters
#     ----------
#     patches : Tensor of (batch_size, pixels_per_patch) or 
#         (batch_size, channels, pixels_per_patch)

#     title : String; title of figure. Optional.
#     """
    
#     if normalize:
#         # print(patches.norm(dim=[-1,-2],keepdim=True).shape)
#         patches=patches-patches.mean(dim=(1,2),keepdim=True)
#         # patches=patches/patches.norm(dim=[-1,-2],keepdim=True)
#         p2p = patches.amax(dim=(1,2),keepdim=True)-patches.amin(dim=(1,2),keepdim=True).clamp(min=1e-8)
#         patches=(patches/p2p).clamp(min=-2,max=2)
    
#     if patches.dim() == 2:
#         channels = 1
#         patches.unsqueeze_(1)
#     else:
#         channels = patches.size(1)
#     batch_size = patches.size(0)
#     size = int(np.sqrt(patches.size(-1)))

#     img_grid = []
#     for i in range(batch_size):
#         img = torch.reshape(patches[i], (channels, size, size))
#         img_grid.append(img)

#     if pad_value != 0:
#         if pad_value == "min":
#             pad_value = torch.min(patches)
#         elif pad_value == "max":
#             pad_value = torch.max(patches)

#     if not ncol:
#         ncol = int(np.sqrt(batch_size))
#     out = make_grid(img_grid, padding=1, nrow=ncol, pad_value=pad_value)
#     # normalize between 0 and 1 for rgb
#     if channels == 3:
#         out = ((out - torch.min(out))/(torch.max(out) - torch.min(out)))
#         # .permute(1, 2, 0)
#     else:
#         out = out[0]

#     if return_tensor:
#         return out

def vis_patches(
    patches, title="", figsize=None, colorbar=False,
    ncol=None, pad_value="min", show=True,
    return_tensor=False, vmin=None, vmax=None, fontsize=20,
    dpi=None, normalize=False
):
    """
    patches: (B, P) or (B, C, P) where P = H*W (square)
    Returns: CHW grid in [0,1] when return_tensor=True
    """
    # Ensure (B, C, P)
    if patches.dim() == 2:
        patches = patches.unsqueeze(1)  # (B,1,P)
    B, C, P = patches.shape
    size = int(round(P ** 0.5))
    assert size * size == P, f"P={P} is not a perfect square"

    if normalize:
        # per-example zero-mean + scale by peak-to-peak, clamp to [-2,2]
        patches = patches - patches.mean(dim=(1, 2), keepdim=True)
        p2p = (patches.amax(dim=(1, 2), keepdim=True) - patches.amin(dim=(1, 2), keepdim=True)).clamp_min(1e-8)
        patches = (patches / p2p).clamp_(-2, 2)

    # make a list of CHW images
    imgs = [patches[i].reshape(C, size, size) for i in range(B)]

    # pad value
    if pad_value in ("min", "max"):
        pv = float(patches.min()) if pad_value == "min" else float(patches.max())
    else:
        pv = float(pad_value)

    # grid shape
    if ncol is None:
        ncol = int(math.ceil(math.sqrt(B)))

    grid = make_grid(imgs, padding=1, nrow=ncol, pad_value=pv)  # CHW

    # global min-max → [0,1] for ANY C (1/3/4/+)
    gmin, gmax = grid.amin(), grid.amax()
    grid = (grid - gmin) / (gmax - gmin + 1e-8)

    if return_tensor:
        return grid  # CHW, [0,1]

    return None
