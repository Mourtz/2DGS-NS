import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClusteredMaterialField(nn.Module):

    def __init__(self, n_clusters: int = 12,
                 roughness_init=None, metallic_init=None,
                 freeze_materials: bool = False):
        super().__init__()
        self.n_clusters = n_clusters
        self._clustered = False

        def _to_logit(vals, n):
            if vals is not None:
                v = torch.tensor(vals, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
                return torch.log(v / (1.0 - v))
            return None

        r_logits = _to_logit(roughness_init, n_clusters)
        m_logits = _to_logit(metallic_init,  n_clusters)

        r_param = r_logits if r_logits is not None else torch.zeros(n_clusters)
        m_param = m_logits if m_logits is not None else torch.full((n_clusters,), -2.0)

        if freeze_materials:
            self.register_buffer('roughness_logits', r_param)
            self.register_buffer('metallic_logits',  m_param)
        else:
            self.roughness_logits = nn.Parameter(r_param)
            self.metallic_logits  = nn.Parameter(m_param)

        self.register_buffer('cluster_centers', torch.zeros(n_clusters, 3))

    @torch.no_grad()
    def cluster(self, xyz: torch.Tensor, albedo: torch.Tensor = None,
                pos_weight: float = 1.0, color_weight: float = 3.0):
        import numpy as np
        from sklearn.cluster import KMeans
        xyz_np = xyz.detach().float().cpu().numpy()

        if albedo is not None:
            pos_std = float(xyz_np.std(axis=0).mean()) + 1e-6
            features = np.concatenate([
                xyz_np / pos_std * pos_weight,
                albedo.detach().float().cpu().numpy() * color_weight,
            ], axis=1)
        else:
            features = xyz_np

        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(features)

        centers = np.stack([
            xyz_np[labels == k].mean(axis=0) if (labels == k).any()
            else xyz_np.mean(axis=0)
            for k in range(self.n_clusters)
        ])
        new_centers = torch.from_numpy(centers).float().to(xyz.device)

        if self._clustered:
            with torch.no_grad():
                old_dists = torch.cdist(xyz.detach(), self.cluster_centers)
                old_ids   = old_dists.argmin(dim=1)
                old_r = torch.sigmoid(self.roughness_logits[old_ids]) * 0.9 + 0.05
                old_m = torch.sigmoid(self.metallic_logits[old_ids])
                new_labels = torch.from_numpy(labels).long().to(xyz.device)
                for k in range(self.n_clusters):
                    mask = new_labels == k
                    if mask.any():
                        mean_r = old_r[mask].mean().clamp(0.06, 0.94)
                        mean_m = old_m[mask].mean().clamp(1e-4, 1 - 1e-4)
                        if not isinstance(self.roughness_logits, torch.nn.Parameter):
                            continue  # frozen — skip
                        r_logit = torch.log((mean_r - 0.05) / (0.95 - (mean_r - 0.05)))
                        m_logit = torch.log(mean_m / (1.0 - mean_m))
                        self.roughness_logits.data[k] = r_logit
                        self.metallic_logits.data[k]  = m_logit

        self.cluster_centers.copy_(new_centers)
        self._clustered = True

        r = torch.sigmoid(self.roughness_logits) * 0.9 + 0.05
        m = torch.sigmoid(self.metallic_logits)
        n_used = len(set(labels.tolist()))
        alb_info = f", color_weight={color_weight}" if albedo is not None else ""
        print(f"[ClusteredMaterialField] K={self.n_clusters} clusters "
              f"({'xyz+albedo' if albedo is not None else 'xyz only'}{alb_info}) | "
              f"active={n_used}/{self.n_clusters} | "
              f"roughness [{r.min():.2f},{r.max():.2f}] "
              f"metallic [{m.min():.2f},{m.max():.2f}]")

    def forward(self, xyz: torch.Tensor):
        """Returns roughness [N,1] and metallic [N,1]."""
        if not self._clustered:
            # If centers already loaded from a checkpoint, skip re-clustering
            if self.cluster_centers.abs().sum() > 0:
                self._clustered = True
            else:
                self.cluster(xyz)
        dists = torch.cdist(xyz.detach(), self.cluster_centers)
        ids   = dists.argmin(dim=1)
        roughness = torch.sigmoid(self.roughness_logits[ids]).unsqueeze(1) * 0.9 + 0.05
        metallic  = torch.sigmoid(self.metallic_logits[ids]).unsqueeze(1)
        return roughness, metallic


class ClusteredCookTorranceShader(nn.Module):

    def __init__(self, n_lights: int = 2, n_clusters: int = 12,
                 roughness_init=None, metallic_init=None,
                 freeze_materials: bool = False):
        super().__init__()
        self.cmf = ClusteredMaterialField(
            n_clusters=n_clusters,
            roughness_init=roughness_init,
            metallic_init=metallic_init,
            freeze_materials=freeze_materials,
        )
        self.lighting = DirectionalLightModel(n_lights=n_lights)

    @staticmethod
    def _ggx_ndf(NdotH, alpha2):
        denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0
        return alpha2 / (math.pi * denom * denom + 1e-7)

    @staticmethod
    def _smith_ggx(NdotV, NdotL, roughness):
        k   = (roughness + 1.0) ** 2 / 8.0
        g_v = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
        g_l = NdotL / (NdotL * (1.0 - k) + k + 1e-7)
        return g_v * g_l

    def forward(self, xyz, albedo, normal, view_dir):
        roughness, metallic = self.cmf(xyz)
        alpha2 = roughness.clamp(0.04, 1.0) ** 4

        F0  = 0.04 * (1.0 - metallic) + albedo.detach() * metallic
        NdotV = (normal * view_dir).sum(-1, keepdim=True).clamp(1e-4, 1.0)
        F_n = F0 + (1.0 - F0) * (1.0 - NdotV).pow(5)

        light_dirs, light_colors, ambient = self.lighting.get_lights()

        color = torch.zeros_like(albedo)
        for i in range(self.lighting.n_lights):
            l  = light_dirs[i].unsqueeze(0).expand_as(normal)
            lc = light_colors[i]
            NdotL = (normal * l).sum(-1, keepdim=True).clamp(0.0, 1.0)
            h     = F.normalize(view_dir + l, dim=-1)
            NdotH = (normal * h).sum(-1, keepdim=True).clamp(0.0, 1.0)
            VdotH = (view_dir * h).sum(-1, keepdim=True).clamp(0.0, 1.0)
            F_h = F0 + (1.0 - F0) * (1.0 - VdotH).pow(5)
            D   = self._ggx_ndf(NdotH, alpha2)
            G   = self._smith_ggx(NdotV, NdotL, roughness)
            specular = D * G * F_h / (4.0 * NdotV * NdotL + 1e-7)
            diffuse  = albedo / math.pi * (1.0 - metallic) * (1.0 - F_n)
            color = color + (diffuse + specular) * NdotL * lc

        color = color + ambient * (albedo * (1.0 - metallic) * (1.0 - F_n) + F_n)
        return color.clamp(0.0, 1.0)


class DirectionalLightModel(nn.Module):

    def __init__(self, n_lights: int = 2):
        super().__init__()
        self.n_lights = n_lights

        init_dirs = torch.zeros(n_lights, 3)
        init_dirs[0] = torch.tensor([ 0.5,  1.0,  0.3])   # key: top-right-forward
        if n_lights > 1:
            init_dirs[1] = torch.tensor([-0.3,  0.5, -0.2])  # fill: left-elevated
        self._dirs = nn.Parameter(init_dirs)

        init_log_i = torch.zeros(n_lights, 3)
        init_log_i[0] = torch.log(torch.tensor([1.1, 1.1, 1.1]))  # bright white key
        if n_lights > 1:
            init_log_i[1] = torch.log(torch.tensor([0.4, 0.4, 0.5]))  # cool fill
        self._log_intensities = nn.Parameter(init_log_i)

        self._log_ambient = nn.Parameter(torch.log(torch.tensor([0.2, 0.2, 0.2])))

    def get_lights(self):
        dirs   = F.normalize(self._dirs, dim=-1)
        colors = F.softplus(self._log_intensities)
        ambient = F.softplus(self._log_ambient)
        return dirs, colors, ambient


class CookTorranceShader(nn.Module):

    def __init__(self, n_lights: int = 2, nmf_hidden: int = 64, nmf_pos_freqs: int = 6):
        super().__init__()
        from scene.neural_brdf_field import NeuralMaterialField
        self.nmf     = NeuralMaterialField(hidden=nmf_hidden, n_pos_freqs=nmf_pos_freqs)
        self.lighting = DirectionalLightModel(n_lights=n_lights)

    @staticmethod
    def _ggx_ndf(NdotH: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        denom = NdotH * NdotH * (alpha2 - 1.0) + 1.0
        return alpha2 / (math.pi * denom * denom + 1e-7)

    @staticmethod
    def _smith_ggx(NdotV: torch.Tensor, NdotL: torch.Tensor,
                   roughness: torch.Tensor) -> torch.Tensor:
        k   = (roughness + 1.0) ** 2 / 8.0
        g_v = NdotV / (NdotV * (1.0 - k) + k + 1e-7)
        g_l = NdotL / (NdotL * (1.0 - k) + k + 1e-7)
        return g_v * g_l

    def forward(
        self,
        xyz:      torch.Tensor,
        albedo:   torch.Tensor,
        normal:   torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        roughness, metallic = self.nmf(xyz)
        alpha2 = roughness.clamp(0.04, 1.0) ** 4

        # Fresnel base reflectance (metallic workflow)
        F0 = 0.04 * (1.0 - metallic) + albedo.detach() * metallic

        NdotV = (normal * view_dir).sum(-1, keepdim=True).clamp(1e-4, 1.0)
        F_n = F0 + (1.0 - F0) * (1.0 - NdotV).pow(5)

        light_dirs, light_colors, ambient = self.lighting.get_lights()

        color = torch.zeros_like(albedo)
        for i in range(self.lighting.n_lights):
            l  = light_dirs[i].unsqueeze(0).expand_as(normal)
            lc = light_colors[i]

            NdotL = (normal * l).sum(-1, keepdim=True).clamp(0.0, 1.0)
            h     = F.normalize(view_dir + l, dim=-1)
            NdotH = (normal * h).sum(-1, keepdim=True).clamp(0.0, 1.0)
            VdotH = (view_dir * h).sum(-1, keepdim=True).clamp(0.0, 1.0)

            F_h = F0 + (1.0 - F0) * (1.0 - VdotH).pow(5)
            D   = self._ggx_ndf(NdotH, alpha2)
            G   = self._smith_ggx(NdotV, NdotL, roughness)

            specular = D * G * F_h / (4.0 * NdotV * NdotL + 1e-7)
            diffuse  = albedo / math.pi * (1.0 - metallic) * (1.0 - F_n)

            color = color + (diffuse + specular) * NdotL * lc

        color = color + ambient * (albedo * (1.0 - metallic) * (1.0 - F_n) + F_n)
        return color.clamp(0.0, 1.0)

class SphMipEnvironment(nn.Module):

    def __init__(self, n_levels: int = 8, h: int = 64, w: int = 128):
        super().__init__()
        self.n_levels = n_levels
        self.envmap = nn.Parameter(torch.zeros(1, 3, n_levels, h, w))
        self.irradiance_raw = nn.Parameter(torch.full((3,), -1.05))

    def query(self, refl_dir: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        phi   = torch.atan2(refl_dir[:, 2], refl_dir[:, 0])
        theta = torch.acos(refl_dir[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        u = phi   / math.pi
        v = theta / math.pi * 2.0 - 1.0
        d = roughness.squeeze(-1) * 2.0 - 1.0

        grid = torch.stack([u, v, d], dim=-1).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.envmap, grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )
        radiance = sampled.squeeze(0).squeeze(-1).squeeze(-1).T
        return F.softplus(radiance)

    @property
    def irradiance(self) -> torch.Tensor:
        return F.softplus(self.irradiance_raw)


class CookTorranceProbeShader(nn.Module):

    def __init__(
        self,
        nmf_hidden: int = 64,
        nmf_pos_freqs: int = 6,
        probe_resolution: int = 64,
        probe_update_interval: int = 500,
        probe_n_blur: int = 5,
    ):
        super().__init__()
        from scene.neural_brdf_field import NeuralMaterialField
        self.nmf = NeuralMaterialField(hidden=nmf_hidden, n_pos_freqs=nmf_pos_freqs)

        from scene.env_probe import EnvironmentProbe
        self.probe = EnvironmentProbe(
            resolution=probe_resolution,
            update_interval=probe_update_interval,
            n_blur_levels=probe_n_blur,
        )

    def forward(
        self,
        xyz:      torch.Tensor,
        albedo:   torch.Tensor,
        normal:   torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        roughness, metallic = self.nmf(xyz)

        F0     = 0.04 * (1.0 - metallic) + albedo.detach() * metallic
        NdotV  = (normal * view_dir).sum(-1, keepdim=True).clamp(1e-4, 1.0)
        fresnel = F0 + (1.0 - F0) * (1.0 - NdotV).pow(5)

        refl_dir = F.normalize(2.0 * NdotV * normal - view_dir, dim=-1)

        L_env    = self.probe.query(refl_dir, roughness)
        specular = L_env * fresnel

        irr     = self.probe.average_irradiance().to(xyz.device)
        diffuse = irr * albedo / math.pi * (1.0 - metallic) * (1.0 - fresnel)

        return (specular + diffuse).clamp(0.0, 1.0)


class SphMipFeatureEnv(nn.Module):

    def __init__(self, n_levels: int = 8, h: int = 8, w: int = 16, feat_dim: int = 8):
        super().__init__()
        self.feat_dim = feat_dim
        self.envmap = nn.Parameter(torch.randn(1, feat_dim, n_levels, h, w) * 0.01)

    def query(self, refl_dir: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        phi   = torch.atan2(refl_dir[:, 2], refl_dir[:, 0])
        theta = torch.acos(refl_dir[:, 1].clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        u = phi   / math.pi
        v = theta / math.pi * 2.0 - 1.0
        d = roughness.squeeze(-1) * 2.0 - 1.0

        grid = torch.stack([u, v, d], dim=-1).view(1, -1, 1, 1, 3)
        sampled = F.grid_sample(
            self.envmap, grid,
            mode='bilinear', padding_mode='border', align_corners=True,
        )
        return sampled.squeeze(0).squeeze(-1).squeeze(-1).T


class CookTorranceRefGSShader(nn.Module):

    def __init__(
        self,
        nmf_hidden: int = 64,
        nmf_pos_freqs: int = 6,
        feat_dim: int = 8,
        n_mip_levels: int = 8,
        env_h: int = 8,
        env_w: int = 16,
    ):
        super().__init__()
        from scene.neural_brdf_field import NeuralMaterialField
        self.nmf = NeuralMaterialField(hidden=nmf_hidden, n_pos_freqs=nmf_pos_freqs)
        self.env = SphMipFeatureEnv(n_levels=n_mip_levels, h=env_h, w=env_w, feat_dim=feat_dim)

        self.irradiance_raw = nn.Parameter(torch.full((3,), -1.05))

        in_dim = feat_dim + 3 * feat_dim
        hidden = max(32, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        xyz:      torch.Tensor,
        albedo:   torch.Tensor,
        normal:   torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        roughness, metallic = self.nmf(xyz)

        F0    = 0.04 * (1.0 - metallic) + albedo.detach() * metallic
        NdotV = (normal * view_dir).sum(-1, keepdim=True).clamp(1e-4, 1.0)
        fresnel = F0 + (1.0 - F0) * (1.0 - NdotV).pow(5)

        refl_dir = F.normalize(2.0 * NdotV * normal - view_dir, dim=-1)

        s = self.env.query(refl_dir, roughness)

        k = albedo.detach()

        outer = (k.unsqueeze(-1) * s.unsqueeze(-2)).reshape(k.shape[0], -1)
        mlp_in = torch.cat([s, outer], dim=-1)

        spec_env = F.relu(self.mlp(mlp_in))
        specular = spec_env * fresnel

        irr     = F.softplus(self.irradiance_raw).to(xyz.device)
        diffuse = irr * albedo / math.pi * (1.0 - metallic) * (1.0 - fresnel)

        return (specular + diffuse).clamp(0.0, 1.0)


class CookTorranceSphMipShader(nn.Module):

    def __init__(
        self,
        nmf_hidden: int = 64,
        nmf_pos_freqs: int = 6,
        n_mip_levels: int = 8,
        env_h: int = 64,
        env_w: int = 128,
    ):
        super().__init__()
        from scene.neural_brdf_field import NeuralMaterialField
        self.nmf = NeuralMaterialField(hidden=nmf_hidden, n_pos_freqs=nmf_pos_freqs)
        self.env = SphMipEnvironment(n_levels=n_mip_levels, h=env_h, w=env_w)

    def forward(
        self,
        xyz:      torch.Tensor,
        albedo:   torch.Tensor,
        normal:   torch.Tensor,
        view_dir: torch.Tensor,
    ) -> torch.Tensor:
        roughness, metallic = self.nmf(xyz)

        F0 = 0.04 * (1.0 - metallic) + albedo.detach() * metallic
        NdotV    = (normal * view_dir).sum(-1, keepdim=True).clamp(1e-4, 1.0)
        fresnel  = F0 + (1.0 - F0) * (1.0 - NdotV).pow(5)

        refl_dir = F.normalize(2.0 * NdotV * normal - view_dir, dim=-1)

        L_env    = self.env.query(refl_dir, roughness)
        specular = L_env * fresnel

        irr     = self.env.irradiance.to(xyz.device)
        diffuse = irr * albedo / math.pi * (1.0 - metallic) * (1.0 - fresnel)

        return (specular + diffuse).clamp(0.0, 1.0)


