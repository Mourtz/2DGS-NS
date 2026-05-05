import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeferredShader(nn.Module):

    def __init__(self, hidden: int = 128, n_freqs: int = 4, specular_scale: float = 0.8):
        super().__init__()
        self.n_freqs = n_freqs
        self.specular_scale = specular_scale

        dir_enc_dim = 3 * (1 + 2 * n_freqs)
        in_ch = 3 + 3 + 1 + 1 + 1 + dir_enc_dim  # 36

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def encode_dirs(dirs: torch.Tensor, n_freqs: int) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(n_freqs, device=dirs.device, dtype=dirs.dtype)
        encs = [dirs]
        for freq in freqs:
            encs.append(torch.sin(math.pi * freq * dirs))
            encs.append(torch.cos(math.pi * freq * dirs))
        return torch.cat(encs, dim=0)

    def forward(
        self,
        albedo: torch.Tensor,
        normal: torch.Tensor,
        alpha: torch.Tensor,
        roughness_map: torch.Tensor,
        metallic_map: torch.Tensor,
        refl_dir: torch.Tensor,
    ) -> torch.Tensor:
        """Returns [3, H, W] final clamped colour."""
        refl_enc = self.encode_dirs(refl_dir, self.n_freqs)
        x = torch.cat(
            [albedo, normal, alpha, roughness_map, metallic_map, refl_enc], dim=0
        ).unsqueeze(0)
        residual = torch.tanh(self.net(x).squeeze(0)) * self.specular_scale
        return torch.clamp(albedo + residual, 0.0, 1.0)


def compute_material_gbuffer(rasterizer, means3D, means2D, opacity, scales, rotations,
                              roughness, metallic, cov3D_precomp=None):
    N = roughness.shape[0]
    mat_colors = torch.cat([
        roughness,
        metallic,
        torch.zeros(N, 1, device=roughness.device),
    ], dim=1)

    mat_map, _, _ = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=None,
        colors_precomp=mat_colors,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return mat_map[0:1], mat_map[1:2]


@torch.no_grad()
def compute_reflection_direction(camera, normal: torch.Tensor) -> torch.Tensor:
    H, W = normal.shape[1], normal.shape[2]
    device, dtype = normal.device, normal.dtype

    y = torch.arange(H, device=device, dtype=dtype)
    x = torch.arange(W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing='ij')

    tanfovx = math.tan(camera.FoVx * 0.5)
    tanfovy = math.tan(camera.FoVy * 0.5)
    xndc = (xx + 0.5) / W * 2.0 - 1.0
    yndc = (yy + 0.5) / H * 2.0 - 1.0

    x_cam = xndc * tanfovx
    y_cam = -yndc * tanfovy
    z_cam = torch.ones_like(x_cam)
    ray_cam = F.normalize(torch.stack([x_cam, y_cam, z_cam], dim=0), dim=0)

    R_v2w = camera.world_view_transform[:3, :3].T.to(dtype=dtype)
    ray_world = F.normalize(
        (R_v2w @ ray_cam.view(3, -1)).view(3, H, W), dim=0
    )

    view_dir = -ray_world

    n = F.normalize(normal, dim=0)
    dot_vn = (view_dir * n).sum(dim=0, keepdim=True)
    refl_dir = F.normalize(2.0 * dot_vn * n - view_dir, dim=0)

    return refl_dir
