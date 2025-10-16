import torch

class SimpleNoiseLoss:
    """
    Blind denoising loss with configurable σ sampling and σ-aware weighting.

    - sigma_dist: 'uniform' (default) or 'lognormal' (Song-style)
    - weighting:
        'none'     -> unweighted MSE
        'inv_var'  -> 1 / (σ^2 + eps)         (strongly downweights large σ)
        'edm'      -> (σ^2 + σ_data^2) / (σ^2 σ_data^2 + eps)
        'power'    -> (σ / σ_data)^(-weight_power)

    Returns per-pixel squared error (optionally weighted). Keep reduction outside
    to match your current training loop.
    """
    def __init__(self,
                 noise_range=(0.01, 1.0),
                 sigma_dist='lognormal',
                 # Lognormal params (Song): σ = exp(P_mean + P_std * N(0,1))
                 P_mean=-1.2,
                 P_std=1.2,
                 clamp_lognormal_to_range=True,
                 # Weighting options
                 weighting='edm',
                 sigma_data=0.25,
                 weight_power=2.0,
                 eps=1e-8):
        self.noise_min, self.noise_max = noise_range
        if self.noise_min is None or self.noise_max is None:
            raise ValueError("SimpleNoiseLoss requires finite (sigma_min, sigma_max).")
        if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

        self.sigma_dist = sigma_dist.lower()
        if self.sigma_dist not in ('uniform', 'lognormal'):
            raise ValueError("sigma_dist must be 'uniform' or 'lognormal'.")

        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.clamp_lognormal_to_range = bool(clamp_lognormal_to_range)

        self.weighting = weighting.lower()
        if self.weighting not in ('none', 'inv_var', 'edm', 'power'):
            raise ValueError("weighting must be 'none', 'inv_var', 'edm', or 'power'.")

        self.sigma_data = float(sigma_data)
        self.weight_power = float(weight_power)
        self.eps = float(eps)

    def _sample_sigma(self, B, device):
        if self.sigma_dist == 'uniform':
            if self.noise_max > self.noise_min:
                u = torch.rand(B, 1, 1, 1, device=device)
                sigma = self.noise_min + u * (self.noise_max - self.noise_min)
            else:
                sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)
        else:
            # Song-style lognormal sampling
            rnd = torch.randn(B, 1, 1, 1, device=device)
            sigma = (rnd * self.P_std + self.P_mean).exp()
            if self.clamp_lognormal_to_range:
                sigma = sigma.clamp(min=self.noise_min, max=self.noise_max)
        return sigma

    def _loss_weight(self, sigma):
        """Return a broadcastable loss weight tensor shaped (B,1,1,1)."""
        if self.weighting == 'none':
            return 1.0
        elif self.weighting == 'inv_var':
            # Classic inverse-variance weighting to counter σ^2 growth of MSE.
            return 1.0 / (sigma**2 + self.eps)
        elif self.weighting == 'edm':
            # Karras EDM-style: behaves like 1/σ^2 at large σ but is finite at tiny σ.
            sd2 = self.sigma_data ** 2
            return (sigma**2 + sd2) / (sigma**2 * sd2 + self.eps)
        elif self.weighting == 'power':
            # Generalized power law around a reference σ_data.
            # p=2 recovers ~1/σ^2 (up to a constant).
            # Note: add eps in denominator via clamp to avoid blowup at tiny σ.
            return (torch.clamp(sigma, min=self.eps) / self.sigma_data).pow(-self.weight_power)
        else:
            raise RuntimeError("Unreachable")

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        sigma = self._sample_sigma(B, device)          # (B,1,1,1)
        noise = torch.randn_like(images) * sigma       # additive Gaussian noise
        y_noisy = images + noise

        D_yn = net(y_noisy)                            # no conditioning

        per_pixel_se = (D_yn - images) ** 2            # (B,C,H,W)
        w = self._loss_weight(sigma)                   # (B,1,1,1), broadcasts
        return per_pixel_se * w


class SimpleUniformNoiseLoss:
    """
    Uniform-in-σ denoising loss with *no conditioning*.
    Calls D_yn = net(y + n) only. Returns per-pixel squared error.
    """
    def __init__(self, noise_range=(0.01, 1.0)):
        self.noise_min, self.noise_max = noise_range
        if self.noise_min is None or self.noise_max is None:
            raise ValueError("SimpleUniformNoiseLoss requires finite (sigma_min, sigma_max).")
        if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        # σ ~ Uniform[min, max] (fixed if min == max)
        if self.noise_max > self.noise_min:
            u = torch.rand(B, 1, 1, 1, device=device)
            sigma = self.noise_min + u * (self.noise_max - self.noise_min)
        else:
            sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)

        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n)                # <-- no sigma/labels/aug passed

        return (D_yn - y) ** 2


class EDMLossNoCond:
    """
    EDM loss with log-normal σ sampling + EDM weighting.
    Calls net(y+n, sigma) where 'net' is the EDMPrecond wrapper.
    """
    def __init__(self, P_mean: float = -1.2, P_std: float = 1.2,
                 sigma_data: float = 0.25, noise_range=(None, None), edm_weighting=False):
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.noise_min, self.noise_max = noise_range
        self.edm_weighting = edm_weighting

    def __call__(self, net, images: torch.Tensor):
        device = images.device
        B = images.shape[0]

        # σ ~ LogNormal(P_mean, P_std)
        rnd = torch.randn(B, 1, 1, 1, device=device)
        sigma = (rnd * self.P_std + self.P_mean).exp()
        if self.noise_min is not None:
            sigma = sigma.clamp_min(self.noise_min)
        if self.noise_max is not None:
            sigma = sigma.clamp_max(self.noise_max)

        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, noise_labels=sigma.flatten())  # EDM-preconditioned forward
        if self.edm_weighting:
        # EDM weighting
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            return weight * ((D_yn - y) ** 2)
        else:
            return (D_yn - y) ** 2


# class SimpleUniformNoiseLoss:
#     """
#     Uniform-in-σ denoising loss with *no conditioning*.
#     Calls D_yn = net(y + n) only. Returns per-pixel squared error.
#     """
#     def __init__(self, noise_range=(0.01, 1.0)):
#         self.noise_min, self.noise_max = noise_range
#         if self.noise_min is None or self.noise_max is None:
#             raise ValueError("SimpleUniformNoiseLoss requires finite (sigma_min, sigma_max).")
#         if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
#             raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

#     def __call__(self, net, images):
#         device = images.device
#         B = images.shape[0]

#         # σ ~ Uniform[min, max] (fixed if min == max)
#         if self.noise_max > self.noise_min:
#             u = torch.rand(B, 1, 1, 1, device=device)
#             sigma = self.noise_min + u * (self.noise_max - self.noise_min)
#         else:
#             sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)

#         y = images
#         n = torch.randn_like(y) * sigma
#         D_yn = net(y + n)                # <-- no sigma/labels/aug passed

#         return (D_yn - y) ** 2


# class EDMStyleXPredLoss:
#     """
#     EDM-style x-prediction loss for a net that takes ONLY an image tensor.
#     - Samples σ ~ Uniform[min,max]
#     - Normalizes input with c_in = 1/sqrt(σ^2 + σ_data^2) before calling net
#     - Unscales the net output back to image domain
#     - Applies EDM weighting: λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
#     Returns a per-pixel squared error map (same shape as images).
#     """
#     def __init__(self, noise_range=(0.01, 1.0), sigma_data=0.5, eps=1e-8):
#         sigma_min, sigma_max = noise_range
#         if sigma_min is None or sigma_max is None:
#             raise ValueError("Provide finite (sigma_min, sigma_max).")
#         if not (sigma_min > 0 and sigma_max >= sigma_min):
#             raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")
#         self.sigma_min   = float(sigma_min)
#         self.sigma_max   = float(sigma_max)
#         self.sigma_data  = float(sigma_data)
#         self.eps         = float(eps)

#     def __call__(self, net, images):
#         device = images.device
#         B = images.shape[0]

#         # --- sample σ ~ Uniform[min,max] ---
#         if self.sigma_max > self.sigma_min:
#             u = torch.rand(B, 1, 1, 1, device=device, dtype=images.dtype)
#             sigma = self.sigma_min + u * (self.sigma_max - self.sigma_min)
#         else:
#             sigma = torch.full((B, 1, 1, 1), self.sigma_min, device=device, dtype=images.dtype)

#         # --- make noisy input y + n ---
#         y = images
#         n = torch.randn_like(y) * sigma
#         y_noisy = y + n

#         # --- EDM input normalization ---
#         # c_in = 1 / sqrt(σ^2 + σ_data^2)
#         c_in = 1.0 / torch.sqrt(sigma * sigma + self.sigma_data * self.sigma_data)

#         # net sees only normalized inputs
#         y_in = c_in * y_noisy
#         y_hat_scaled = net(y_in)

#         # --- unscale prediction back to image domain ---
#         # x_hat = y_hat_scaled / c_in
#         x_hat = y_hat_scaled / (c_in + self.eps)

#         # --- EDM x-pred weighting ---
#         # λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
#         weight = (sigma * sigma + self.sigma_data * self.sigma_data) / (
#             (sigma * self.sigma_data + self.eps) ** 2
#         )

#         # per-pixel loss map
#         loss_map = weight * (x_hat - y) ** 2
#         return loss_map

class SimpleUniformNoiseLoss:
    """
    Uniform-in-σ denoising loss with *no conditioning*.
    Calls D_yn = net(y + n) only. Returns per-pixel squared error.
    """
    def __init__(self, noise_range=(0.01, 1.0)):
        self.noise_min, self.noise_max = noise_range
        if self.noise_min is None or self.noise_max is None:
            raise ValueError("SimpleUniformNoiseLoss requires finite (sigma_min, sigma_max).")
        if not (self.noise_min > 0 and self.noise_max >= self.noise_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        # σ ~ Uniform[min, max] (fixed if min == max)
        if self.noise_max > self.noise_min:
            u = torch.rand(B, 1, 1, 1, device=device)
            sigma = self.noise_min + u * (self.noise_max - self.noise_min)
        else:
            sigma = torch.full((B, 1, 1, 1), float(self.noise_min), device=device)

        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n)                # <-- no sigma/labels/aug passed

        return (D_yn - y) ** 2


class EDMStyleXPredLoss:
    """
    EDM-style x-prediction loss for a net that takes ONLY an image tensor.
    - Samples σ ~ Uniform[min,max]
    - Normalizes input with c_in = 1/sqrt(σ^2 + σ_data^2) before calling net
    - Unscales the net output back to image domain
    - Applies EDM weighting: λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
    Returns a per-pixel squared error map (same shape as images).
    """
    def __init__(self, noise_range=(0.01, 1.0), sigma_data=0.5, eps=1e-8):
        sigma_min, sigma_max = noise_range
        if sigma_min is None or sigma_max is None:
            raise ValueError("Provide finite (sigma_min, sigma_max).")
        if not (sigma_min > 0 and sigma_max >= sigma_min):
            raise ValueError(f"Invalid noise_range {noise_range}: need 0 < min <= max.")
        self.sigma_min   = float(sigma_min)
        self.sigma_max   = float(sigma_max)
        self.sigma_data  = float(sigma_data)
        self.eps         = float(eps)

    def __call__(self, net, images):
        device = images.device
        B = images.shape[0]

        # --- sample σ ~ Uniform[min,max] ---
        if self.sigma_max > self.sigma_min:
            u = torch.rand(B, 1, 1, 1, device=device, dtype=images.dtype)
            sigma = self.sigma_min + u * (self.sigma_max - self.sigma_min)
        else:
            sigma = torch.full((B, 1, 1, 1), self.sigma_min, device=device, dtype=images.dtype)

        # --- make noisy input y + n ---
        y = images
        n = torch.randn_like(y) * sigma
        y_noisy = y + n

        x_hat = net(y_noisy)

        # --- EDM x-pred weighting ---
        # λ(σ) = (σ^2 + σ_data^2) / (σ * σ_data)^2
        weight = (sigma * sigma + self.sigma_data * self.sigma_data) / (
            (sigma * self.sigma_data + self.eps) ** 2
        )

        # per-pixel loss map
        loss_map = weight * (x_hat - y) ** 2
        return loss_map
