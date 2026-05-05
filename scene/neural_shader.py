import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralShader(nn.Module):

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        n_freqs: int = 4,
        use_reflection: bool = True,
        specular_scale: float = 0.8,
    ):
        super().__init__()
        self.n_freqs = n_freqs
        self.use_reflection = use_reflection
        self.specular_scale = specular_scale

        # raw direction (3) + sin/cos per frequency (3 * 2 * n_freqs) = 27
        dir_enc_dim = 3 * (1 + 2 * n_freqs)
        n_dir_inputs = 2 if use_reflection else 1  # view dir + (optional) refl dir
        in_dim = feature_dim + dir_enc_dim * n_dir_inputs

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
        )

        # Zero-init output: specular contribution starts at zero
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def encode_dirs(self, dirs: torch.Tensor) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(self.n_freqs, device=dirs.device, dtype=dirs.dtype)
        encs = [dirs]
        for freq in freqs:
            encs.append(torch.sin(math.pi * freq * dirs))
            encs.append(torch.cos(math.pi * freq * dirs))
        return torch.cat(encs, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dirs_enc = self.encode_dirs(view_dirs)
        if self.use_reflection and normals is not None:
            # Reflection direction: r = 2*(v·n)*n - v
            dot_vn = (view_dirs * normals).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            refl_dirs = F.normalize(2.0 * dot_vn * normals - view_dirs, dim=-1)
            refl_enc = self.encode_dirs(refl_dirs)
            x = torch.cat([features, dirs_enc, refl_enc], dim=-1)
        else:
            x = torch.cat([features, dirs_enc], dim=-1)

        out = self.net(x)
        # Bounded residual: tanh keeps values in (-1, 1), scale controls range
        return torch.tanh(out) * self.specular_scale


class PositionalNeuralShader(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        n_pos_freqs: int = 6,
        n_dir_freqs: int = 4,
        use_reflection: bool = True,
        specular_scale: float = 0.8,
        n_layers: int = 3,
        dir_noise_std: float = 0.0,
        use_ide: bool = False,
        norm_xyz: bool = False,
    ):
        super().__init__()
        self.n_pos_freqs = n_pos_freqs
        self.n_dir_freqs = n_dir_freqs
        self.use_reflection = use_reflection
        self.specular_scale = specular_scale
        self.dir_noise_std = dir_noise_std  # noise on view/refl direction (training only)
        self.use_ide = use_ide
        self.norm_xyz = norm_xyz
        # Registered as buffers so they're included in state_dict and restored on load.
        self.register_buffer('scene_center', torch.zeros(3))
        self.register_buffer('scene_scale', torch.ones(1))

        pos_enc_dim = 3 * (1 + 2 * n_pos_freqs)
        dir_enc_dim = 3 * (1 + 2 * n_dir_freqs)
        n_dir = 2 if use_reflection else 1
        in_dim = pos_enc_dim + dir_enc_dim * n_dir

        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def encode(self, x: torch.Tensor, n_freqs: int) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
        parts = [x]
        for f in freqs:
            parts.append(torch.sin(math.pi * f * x))
            parts.append(torch.cos(math.pi * f * x))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor | None = None,
        roughness: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.training and self.dir_noise_std > 0.0:
            view_dirs = F.normalize(
                view_dirs + torch.randn_like(view_dirs) * self.dir_noise_std, dim=-1
            )

        if self.norm_xyz:
            xyz = (xyz - self.scene_center) / self.scene_scale

        pos_enc = self.encode(xyz, self.n_pos_freqs)
        dir_enc = self.encode(view_dirs, self.n_dir_freqs)

        if self.use_reflection and normals is not None:
            dot_vn = (view_dirs * normals).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            refl = F.normalize(2.0 * dot_vn * normals - view_dirs, dim=-1)
            if self.training and self.dir_noise_std > 0.0:
                refl = F.normalize(
                    refl + torch.randn_like(refl) * self.dir_noise_std, dim=-1
                )
            if self.use_ide and roughness is not None:
                from scene.neural_brdf_field import _ide
                refl_enc = _ide(refl, roughness, self.n_dir_freqs)
            else:
                refl_enc = self.encode(refl, self.n_dir_freqs)
            x = torch.cat([pos_enc, dir_enc, refl_enc], dim=-1)
        else:
            x = torch.cat([pos_enc, dir_enc], dim=-1)

        return torch.tanh(self.net(x)) * self.specular_scale


class SIRENPosShader(nn.Module):

    def __init__(
        self,
        hidden_dim: int = 128,
        n_pos_freqs: int = 6,
        n_dir_freqs: int = 4,
        use_reflection: bool = True,
        specular_scale: float = 0.8,
        n_layers: int = 4,
        omega0: float = 30.0,
        skip_connection: bool = True,
    ):
        super().__init__()
        self.n_pos_freqs = n_pos_freqs
        self.n_dir_freqs = n_dir_freqs
        self.use_reflection = use_reflection
        self.specular_scale = specular_scale
        self.omega0 = omega0
        self.skip_connection = skip_connection

        pos_enc_dim = 3 * (1 + 2 * n_pos_freqs)
        dir_enc_dim = 3 * (1 + 2 * n_dir_freqs)
        n_dir = 2 if use_reflection else 1
        self.in_dim = pos_enc_dim + dir_enc_dim * n_dir

        # Build layers manually (SIREN needs non-standard init per layer)
        self.layers = nn.ModuleList()
        prev = self.in_dim
        for i in range(n_layers - 1):
            lin = nn.Linear(prev, hidden_dim)
            # SIREN init
            if i == 0:
                bound = 1.0 / prev
            else:
                bound = math.sqrt(6.0 / prev)
            nn.init.uniform_(lin.weight, -bound, bound)
            nn.init.zeros_(lin.bias)
            self.layers.append(lin)
            # skip connection: after layer 1, concatenate original input
            if skip_connection and i == 1:
                prev = hidden_dim + self.in_dim
            else:
                prev = hidden_dim
        # Output layer (linear, zero-init)
        self.out_layer = nn.Linear(prev, 3)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def encode(self, x: torch.Tensor, n_freqs: int) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
        parts = [x]
        for f in freqs:
            parts.append(torch.sin(math.pi * f * x))
            parts.append(torch.cos(math.pi * f * x))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        xyz: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pos_enc = self.encode(xyz, self.n_pos_freqs)
        dir_enc = self.encode(view_dirs, self.n_dir_freqs)

        if self.use_reflection and normals is not None:
            dot_vn = (view_dirs * normals).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            refl = F.normalize(2.0 * dot_vn * normals - view_dirs, dim=-1)
            refl_enc = self.encode(refl, self.n_dir_freqs)
            inp = torch.cat([pos_enc, dir_enc, refl_enc], dim=-1)
        else:
            inp = torch.cat([pos_enc, dir_enc], dim=-1)

        x = inp
        for i, layer in enumerate(self.layers):
            if self.skip_connection and i == 2:
                x = torch.cat([x, inp], dim=-1)
            w = self.omega0 if i == 0 else 1.0
            x = torch.sin(w * layer(x))

        return torch.tanh(self.out_layer(x)) * self.specular_scale


class ClusteredBRDFShader(nn.Module):

    def __init__(
        self,
        n_clusters: int = 16,
        feature_dim: int = 8,
        hidden_dim: int = 64,
        n_dir_freqs: int = 4,
        use_reflection: bool = True,
        specular_scale: float = 0.8,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.feature_dim = feature_dim
        self.n_dir_freqs = n_dir_freqs
        self.use_reflection = use_reflection
        self.specular_scale = specular_scale

        # Learnable per-cluster material embeddings
        self.cluster_features = nn.Parameter(torch.randn(n_clusters, feature_dim) * 0.01)

        dir_enc_dim = 3 * (1 + 2 * n_dir_freqs)
        n_dir = 2 if use_reflection else 1
        in_dim = feature_dim + dir_enc_dim * n_dir

        layers = []
        prev = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.ReLU(inplace=True)]
            prev = hidden_dim
        layers.append(nn.Linear(prev, 3))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def encode(self, x: torch.Tensor, n_freqs: int) -> torch.Tensor:
        freqs = 2.0 ** torch.arange(n_freqs, device=x.device, dtype=x.dtype)
        parts = [x]
        for f in freqs:
            parts.append(torch.sin(math.pi * f * x))
            parts.append(torch.cos(math.pi * f * x))
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        cluster_ids: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mat_feats = self.cluster_features[cluster_ids]
        dir_enc = self.encode(view_dirs, self.n_dir_freqs)
        if self.use_reflection and normals is not None:
            dot_vn = (view_dirs * normals).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            refl = F.normalize(2.0 * dot_vn * normals - view_dirs, dim=-1)
            refl_enc = self.encode(refl, self.n_dir_freqs)
            x = torch.cat([mat_feats, dir_enc, refl_enc], dim=-1)
        else:
            x = torch.cat([mat_feats, dir_enc], dim=-1)
        return torch.tanh(self.net(x)) * self.specular_scale

    @torch.no_grad()
    def assign_clusters_kmeans(self, albedo: torch.Tensor, n_iters: int = 50) -> torch.Tensor:
        N, K = albedo.shape[0], self.n_clusters
        # Initialise centres as random subset of albedo values
        idx = torch.randperm(N, device=albedo.device)[:K]
        centres = albedo[idx].clone()
        for _ in range(n_iters):
            # Assign each point to nearest centre
            dists = torch.cdist(albedo, centres)
            ids = dists.argmin(dim=1)
            # Recompute centres
            for k in range(K):
                mask = ids == k
                if mask.any():
                    centres[k] = albedo[mask].mean(dim=0)
        return ids
