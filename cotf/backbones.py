"""Sparse-field reconstruction estimators; sources are listed in README."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def fourier_coordinates(x, bands=4):
    frequency = 2 ** torch.arange(bands, device=x.device, dtype=x.dtype) * math.pi
    phase = x[..., None] * frequency
    return torch.cat([x, phase.sin().flatten(-2), phase.cos().flatten(-2)], -1)


class Senseiver(nn.Module):
    """Sensor-to-latent cross-attention, latent self-attention, coordinate decoding."""

    def __init__(self, c):
        super().__init__()
        w, context = c.get("width", 64), c.get("context_dimensions", 0)
        self.token = nn.Linear(19, w)  # 2 coordinates + 16 Fourier features + value
        self.context = nn.Linear(max(1, context), w)
        self.latents = nn.Parameter(torch.randn(c.get("latents", 16), w) / math.sqrt(w))
        self.cross = nn.MultiheadAttention(w, c.get("heads", 4), batch_first=True)
        self.latent_blocks = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                w,
                c.get("heads", 4),
                2 * w,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            c.get("latent_layers", 2),
            enable_nested_tensor=False,
        )
        self.query = nn.Linear(18, w)
        self.decode = nn.MultiheadAttention(w, c.get("heads", 4), batch_first=True)
        self.output = nn.Sequential(
            nn.LayerNorm(w), nn.Linear(w, w), nn.GELU(), nn.Linear(w, 1)
        )

    def forward(self, x, available, query, context):
        sensor = self.token(
            torch.cat([fourier_coordinates(x[..., :2]), x[..., 2:3]], -1)
        )
        c = context if context.shape[1] else context.new_zeros((len(x), 1))
        # An always-present context token makes the empty sensor set well defined.
        tokens = torch.cat([sensor, self.context(c)[:, None]], 1)
        mask = torch.cat(
            [
                available <= 0,
                torch.zeros((len(x), 1), dtype=torch.bool, device=x.device),
            ],
            1,
        )
        latent = self.latents[None].expand(len(x), -1, -1)
        latent = (
            latent
            + self.cross(
                latent, tokens, tokens, key_padding_mask=mask, need_weights=False
            )[0]
        )
        latent = self.latent_blocks(latent)
        q = self.query(fourier_coordinates(query))
        decoded = self.decode(q, latent, latent, need_weights=False)[0]
        return self.output(q + decoded).squeeze(-1)


class ImplicitField(nn.Module):
    """Permutation-invariant sensor pooling with a continuous coordinate MLP."""

    def __init__(self, c):
        super().__init__()
        w, context = c.get("width", 64), c.get("context_dimensions", 0)
        self.encoder = nn.Sequential(
            nn.Linear(4 + context, w), nn.SiLU(), nn.Linear(w, w)
        )
        self.decoder = nn.Sequential(
            nn.Linear(w + 2 + context, w), nn.SiLU(), nn.Linear(w, 1)
        )

    def forward(self, x, available, query, context):
        tokens = self.encoder(
            torch.cat(
                [x, available[..., None], context[:, None].expand(-1, x.shape[1], -1)],
                -1,
            )
        )
        pooled = (tokens * available[..., None]).sum(1) / available.sum(
            1, keepdim=True
        ).clamp_min(1)
        features = torch.cat(
            [
                query,
                pooled[:, None].expand(-1, query.shape[1], -1),
                context[:, None].expand(-1, query.shape[1], -1),
            ],
            -1,
        )
        return self.decoder(features).squeeze(-1)


class DeepONet(nn.Module):
    """Fixed-slot branch and coordinate trunk combined by an inner product."""

    def __init__(self, c):
        super().__init__()
        w = c.get("width", 64)
        self.sensor_count = c["input_shape"][0]
        self.branch = nn.Sequential(
            nn.Linear(4 * self.sensor_count + c.get("context_dimensions", 0), w),
            nn.GELU(),
            nn.Linear(w, w),
            nn.GELU(),
            nn.Linear(w, w),
        )
        self.trunk = nn.Sequential(nn.Linear(2, w), nn.GELU(), nn.Linear(w, w))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, x, available, query, context):
        if x.shape[1] != self.sensor_count:
            raise ValueError(
                "DeepONet requires the fitted fixed number of sensor slots"
            )
        values = torch.cat(
            [x * available[..., None], available[..., None]], -1
        ).flatten(1)
        return (
            torch.einsum(
                "bw,bqw->bq",
                self.branch(torch.cat([values, context], -1)),
                self.trunk(query),
            )
            + self.bias
        )


class SpectralConv(nn.Module):
    """Learned positive/negative low Fourier modes of a real 2D field."""

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        self.weight = nn.Parameter(
            torch.randn(2, width, width, modes, modes, dtype=torch.cfloat) / width
        )

    def forward(self, x):
        spectrum = torch.fft.rfft2(x)
        m = min(self.modes, x.shape[-2] // 2, spectrum.shape[-1])
        result = torch.zeros_like(spectrum)
        result[..., :m, :m] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[..., :m, :m], self.weight[0, ..., :m, :m]
        )
        result[..., -m:, :m] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[..., -m:, :m], self.weight[1, ..., :m, :m]
        )
        return torch.fft.irfft2(result, s=x.shape[-2:])


class RasterField(nn.Module):
    """Voronoi observation embedding followed by a U-Net or RecFNO operator."""

    def __init__(self, c, spectral=False):
        super().__init__()
        self.c = c
        w = c.get("width", 64)
        self.lift = nn.Conv2d(4 + c.get("context_dimensions", 0), w, 1)
        self.spectral = spectral
        if spectral:
            self.fourier = nn.ModuleList(
                [
                    SpectralConv(w, c.get("modes", 8))
                    for _ in range(c.get("fourier_layers", 4))
                ]
            )
            self.local = nn.ModuleList([nn.Conv2d(w, w, 1) for _ in self.fourier])
        else:
            self.down = nn.Sequential(
                nn.Conv2d(w, 2 * w, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(2 * w, 2 * w, 3, padding=1),
                nn.GELU(),
            )
            self.up = nn.Sequential(
                nn.Conv2d(3 * w, w, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(w, w, 3, padding=1),
                nn.GELU(),
            )
        self.output = nn.Conv2d(w, 1, 1)

    def forward(self, x, available, query, context):
        size = self.c.get("grid_size", 32)
        low = x.new_tensor(self.c.get("coordinate_min", [0.0, 0.0]))
        high = x.new_tensor(self.c.get("coordinate_max", [1.0, 1.0]))
        axis = torch.linspace(0, 1, size, device=x.device)
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        normalized = torch.stack([xx, yy], -1)
        grid = (
            (low + normalized * (high - low))
            .reshape(-1, 2)[None]
            .expand(len(x), -1, -1)
        )
        distance = torch.cdist(grid, x[..., :2]).masked_fill(
            available[:, None] <= 0, float("inf")
        )
        nearest = distance.argmin(-1)
        values = x[..., 2].gather(1, nearest) * (available.sum(1) > 0)[:, None]
        sensor_index = (
            ((x[..., :2] - low) / (high - low) * (size - 1))
            .round()
            .long()
            .clamp(0, size - 1)
        )
        flat_index = sensor_index[..., 1] * size + sensor_index[..., 0]
        mask = (
            x.new_zeros((len(x), size * size))
            .scatter_add(1, flat_index, available)
            .clamp_max(1)
        )
        inputs = [
            values.reshape(len(x), 1, size, size),
            mask.reshape(len(x), 1, size, size),
            normalized.permute(2, 0, 1)[None].expand(len(x), -1, -1, -1),
            context[..., None, None].expand(-1, -1, size, size),
        ]
        h = self.lift(torch.cat(inputs, 1))
        if self.spectral:
            for fourier, local in zip(self.fourier, self.local):
                h = F.gelu(fourier(h) + local(h))
        else:
            down = self.down(F.avg_pool2d(h, 2))
            up = F.interpolate(
                down, size=h.shape[-2:], mode="bilinear", align_corners=True
            )
            h = self.up(torch.cat([h, up], 1))
        field = self.output(h)
        sample = (query - low) / (high - low) * 2 - 1
        return (
            F.grid_sample(field, sample[:, None], mode="bilinear", align_corners=True)
            .squeeze(1)
            .squeeze(1)
        )


class Kriging(nn.Module):
    """Ordinary Gaussian kriging with train-fitted covariance hyperparameters."""

    def __init__(self, c):
        super().__init__()
        self.c = c
        self.log_length = nn.Parameter(
            torch.tensor(float(c.get("kriging_length", 0.3))).log()
        )
        self.log_noise = nn.Parameter(
            torch.tensor(float(c.get("kriging_noise", 0.01))).log()
        )

    def predict(self, x, available, query, context):
        means, variances = [], []
        amplitude = self.c.get("temperature_scale", 1.0) ** 2
        for values, mask, coordinates in zip(x, available > 0, query):
            observed = values[mask]
            if not len(observed):
                means.append(
                    coordinates.new_full(
                        (len(coordinates),), self.c.get("training_mean", 0.0)
                    )
                )
                variances.append(coordinates.new_full((len(coordinates),), amplitude))
                continue
            distance = torch.cdist(observed[:, :2], observed[:, :2]).square()
            kernel = amplitude * torch.exp(
                -distance / (2 * self.log_length.exp().square())
            )
            kernel = kernel + torch.eye(len(observed), device=x.device) * (
                self.log_noise.exp().square() + 1e-6
            )
            cross = amplitude * torch.exp(
                -torch.cdist(coordinates, observed[:, :2]).square()
                / (2 * self.log_length.exp().square())
            )
            ones = torch.ones(len(observed), device=x.device)
            inverse_one = torch.linalg.solve(kernel, ones)
            inverse_value = torch.linalg.solve(kernel, observed[:, 2])
            level = (ones @ inverse_value) / (ones @ inverse_one)
            mean = level + cross @ torch.linalg.solve(kernel, observed[:, 2] - level)
            inverse_cross = torch.linalg.solve(kernel, cross.T)
            variance = (
                amplitude
                - (cross * inverse_cross.T).sum(-1)
                + (1 - cross @ inverse_one).square() / (ones @ inverse_one)
            )
            means.append(mean)
            variances.append(variance.clamp_min(1e-8))
        return torch.stack(means), torch.stack(variances)

    def forward(self, x, available, query, context):
        return self.predict(x, available, query, context)[0]


def build(c):
    """Construct an estimator with a mechanism distinct from the other controls."""
    name = c.get("backbone", "senseiver")
    if name == "senseiver":
        return Senseiver(c)
    if name == "implicit":
        return ImplicitField(c)
    if name == "deeponet":
        return DeepONet(c)
    if name in {"recfno", "voronoi_cnn"}:
        return RasterField(c, spectral=name == "recfno")
    if name == "kriging":
        return Kriging(c)
    raise ValueError("Unknown reconstruction backbone: " + name)
