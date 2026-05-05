import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pos_enc(x: torch.Tensor, n_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
    parts = [x]
    for f in freqs:
        parts.append(torch.sin(math.pi * f * x))
        parts.append(torch.cos(math.pi * f * x))
    return torch.cat(parts, dim=-1)


class NeuralMaterialField(nn.Module):

    def __init__(self, hidden: int = 64, n_pos_freqs: int = 6):
        super().__init__()
        self.n_pos_freqs = n_pos_freqs
        pos_enc_dim = 3 * (1 + 2 * n_pos_freqs)

        self.net = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )
        # Init: roughness ≈ 0.5, metallic ≈ 0.12 (mostly dielectric prior)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, 0.0)
        self.net[-1].bias.data[1] = -2.0

    def forward(self, xyz: torch.Tensor):
        pos_enc = _pos_enc(xyz, self.n_pos_freqs)
        out = self.net(pos_enc)
        roughness = torch.sigmoid(out[:, 0:1]) * 0.9 + 0.05
        metallic  = torch.sigmoid(out[:, 1:2])
        return roughness, metallic


class PhysicsGuidedPositionalShader(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        n_pos_freqs: int = 6,
        n_dir_freqs: int = 4,
        n_layers: int = 3,
        specular_scale: float = 1.0,
        legacy_softplus: bool = False,
    ):
        super().__init__()
        self.n_pos_freqs = n_pos_freqs
        self.n_dir_freqs = n_dir_freqs
        self.specular_scale = specular_scale
        self.legacy_softplus = legacy_softplus

        pos_enc_dim = 3 * (1 + 2 * n_pos_freqs)
        dir_enc_dim = 3 * (1 + 2 * n_dir_freqs)
        in_dim = pos_enc_dim + dir_enc_dim + 1

        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.log_spec_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        xyz: torch.Tensor,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        refl_dir: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> torch.Tensor:
        pos_enc  = _pos_enc(xyz, self.n_pos_freqs)
        refl_enc = _pos_enc(refl_dir, self.n_dir_freqs)
        x = torch.cat([pos_enc, refl_enc, roughness], dim=-1)

        if self.legacy_softplus:
            L_spec = F.softplus(self.net(x)) * self.specular_scale
        else:
            L_spec = torch.clamp(torch.tanh(self.net(x)), min=0.0) * self.specular_scale * self.log_spec_scale.exp()

        F0 = 0.04 * (1.0 - metallic) + albedo.detach() * metallic
        fresnel = F0 + (1.0 - F0) * (1.0 - cos_theta).pow(5)

        return fresnel * L_spec

def _ide(refl_dir: torch.Tensor, roughness: torch.Tensor, n_freqs: int) -> torch.Tensor:
    freqs = 2.0 ** torch.arange(n_freqs, device=refl_dir.device, dtype=refl_dir.dtype)
    parts = [refl_dir]
    for k, f in enumerate(freqs, start=1):
        attn = torch.exp(-(f * f) * roughness * roughness)
        parts.append(attn * torch.sin(math.pi * f * refl_dir))
        parts.append(attn * torch.cos(math.pi * f * refl_dir))
    return torch.cat(parts, dim=-1)


class PGPSIde(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        n_pos_freqs: int = 6,
        n_dir_freqs: int = 4,
        n_layers: int = 3,
        specular_scale: float = 1.0,
    ):
        super().__init__()
        self.n_pos_freqs = n_pos_freqs
        self.n_dir_freqs = n_dir_freqs
        self.specular_scale = specular_scale

        pos_enc_dim = 3 * (1 + 2 * n_pos_freqs)
        dir_enc_dim = 3 * (1 + 2 * n_dir_freqs)
        in_dim = pos_enc_dim + dir_enc_dim + 1

        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_spec_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        xyz: torch.Tensor,
        albedo: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        refl_dir: torch.Tensor,
        cos_theta: torch.Tensor,
    ) -> torch.Tensor:
        pos_enc  = _pos_enc(xyz, self.n_pos_freqs)
        refl_enc = _ide(refl_dir, roughness, self.n_dir_freqs)
        x = torch.cat([pos_enc, refl_enc, roughness], dim=-1)

        L_spec = torch.clamp(torch.tanh(self.net(x)), min=0.0) \
                 * self.specular_scale * self.log_spec_scale.exp()

        F0 = 0.04 * (1.0 - metallic) + albedo.detach() * metallic
        fresnel = F0 + (1.0 - F0) * (1.0 - cos_theta).pow(5)
        return fresnel * L_spec


