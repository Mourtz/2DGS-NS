import math
import numpy as np
import torch
import torch.nn.functional as F
from scene.cameras import MiniCam
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

_Y_UP  = np.array([0, 1, 0], dtype=np.float32)
_Z_UP  = np.array([0, 0,-1], dtype=np.float32)
_Z_DN  = np.array([0, 0, 1], dtype=np.float32)

def _face_R(forward_w, world_up):
    f = np.array(forward_w, dtype=np.float32)
    f = f / np.linalg.norm(f)
    u = np.array(world_up, dtype=np.float32)
    right = np.cross(u, f)
    right = right / np.linalg.norm(right)
    down = np.cross(f, right)
    down = down / np.linalg.norm(down)
    return np.column_stack([right, down, f])


_FACE_CONFIGS = [
    ([ 1,  0,  0],  _Y_UP,  '+X'),
    ([-1,  0,  0],  _Y_UP,  '-X'),
    ([ 0,  1,  0],  _Z_UP,  '+Y'),
    ([ 0, -1,  0],  _Z_DN,  '-Y'),
    ([ 0,  0,  1],  _Y_UP,  '+Z'),
    ([ 0,  0, -1],  _Y_UP,  '-Z'),
]

_FACE_AXES = []
for _fwd, _up, _lbl in _FACE_CONFIGS:
    _R = _face_R(_fwd, _up)
    _FACE_AXES.append({
        'right':   _R[:, 0],
        'down':    _R[:, 1],
        'forward': _R[:, 2],
        'label':   _lbl,
    })

class EnvironmentProbe:

    def __init__(
        self,
        resolution: int = 64,
        update_interval: int = 500,
        n_blur_levels: int = 5,
        center: tuple = (0.0, 0.0, 0.0),
    ):
        self.resolution = resolution
        self.update_interval = update_interval
        self.n_blur_levels = n_blur_levels
        self.center = np.array(center, dtype=np.float32)

        self.faces: torch.Tensor | None = None
        self.blurred: list[torch.Tensor] | None = None
        self._last_update = -1

        # Build MiniCam objects for each face (constant, built once)
        self._face_cams = [self._make_face_cam(i) for i in range(6)]

        # Precompute face axis tensors for GPU query
        self._right   = None
        self._down    = None
        self._forward = None

    def _make_face_cam(self, face_idx: int) -> MiniCam:
        """Build a MiniCam looking in face direction `face_idx`."""
        fwd, up, _ = _FACE_CONFIGS[face_idx]
        R = _face_R(fwd, up)
        T = np.zeros(3, dtype=np.float32)  # camera at world origin

        W2C = getWorld2View2(R, T)
        W2C_t = torch.tensor(W2C, dtype=torch.float32).T.cuda()  # transposed as expected

        fov = math.pi / 2.0
        proj = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=fov, fovY=fov
        ).transpose(0, 1).cuda()
        full_proj = W2C_t.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)

        return MiniCam(
            self.resolution, self.resolution,
            fov, fov, 0.01, 100.0,
            W2C_t, full_proj,
        )

    def should_update(self, iteration: int) -> bool:
        return self.faces is None or (iteration - self._last_update) >= self.update_interval

    @torch.no_grad()
    def update(self, gaussians, pipeline, background: torch.Tensor, iteration: int = 0):
        from gaussian_renderer import render

        faces = []
        for face_idx, cam in enumerate(self._face_cams):
            pkg = render(cam, gaussians, pipeline, background)
            face = pkg["render"].detach().clamp(0.0, 1.0)
            faces.append(face)

        self.faces = torch.stack(faces, dim=0)

        # Build roughness mip levels: repeated Gaussian-like blur
        blurred = [self.faces.clone()]
        for _ in range(self.n_blur_levels - 1):
            prev = blurred[-1]
            B, C, H, W = prev.shape
            # Average-pool blur (fast approximation of Gaussian)
            nxt = F.avg_pool2d(
                prev.reshape(B * C, 1, H, W),
                kernel_size=5, stride=1, padding=2,
            ).reshape(B, C, H, W)
            blurred.append(nxt)
        self.blurred = blurred

        # Precompute axis tensors on GPU
        rights   = np.stack([_FACE_AXES[i]['right']   for i in range(6)], axis=0)
        downs    = np.stack([_FACE_AXES[i]['down']    for i in range(6)], axis=0)
        forwards = np.stack([_FACE_AXES[i]['forward'] for i in range(6)], axis=0)
        dev = self.faces.device
        self._right   = torch.tensor(rights,   dtype=torch.float32, device=dev)
        self._down    = torch.tensor(downs,    dtype=torch.float32, device=dev)
        self._forward = torch.tensor(forwards, dtype=torch.float32, device=dev)

        self._last_update = iteration

    def query(self, refl_dir: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        if self.faces is None:
            return torch.zeros(refl_dir.shape[0], 3, device=refl_dir.device)

        device = refl_dir.device
        N = refl_dir.shape[0]
        right   = self._right.to(device)
        down    = self._down.to(device)
        forward = self._forward.to(device)

        dots = torch.matmul(refl_dir, forward.T)
        face_idx = dots.argmax(dim=-1)

        r_f = right[face_idx]
        d_f = down[face_idx]
        fw_f = forward[face_idx]

        fw_dot = (refl_dir * fw_f).sum(-1, keepdim=True).clamp(min=1e-6)
        u = (refl_dir * r_f).sum(-1, keepdim=True) / fw_dot
        v = (refl_dir * d_f).sum(-1, keepdim=True) / fw_dot
        grid = torch.cat([u, v], dim=-1).clamp(-1.0, 1.0)

        blur_idx = (roughness.squeeze(-1) * (self.n_blur_levels - 1)).long().clamp(0, self.n_blur_levels - 1)

        colors = torch.zeros(N, 3, device=device)
        for f in range(6):
            for b in range(self.n_blur_levels):
                mask = (face_idx == f) & (blur_idx == b)
                if not mask.any():
                    continue
                face_tex = self.blurred[b][f].to(device)
                g = grid[mask].unsqueeze(0).unsqueeze(0)
                sampled = F.grid_sample(
                    face_tex.unsqueeze(0), g,
                    mode='bilinear', padding_mode='border', align_corners=True,
                )
                colors[mask] = sampled.squeeze(0).squeeze(1).T

        return colors.clamp(0.0, 1.0)

    def average_irradiance(self) -> torch.Tensor:
        if self.faces is None:
            return torch.tensor([0.3, 0.3, 0.3])
        return self.faces.mean(dim=[0, 2, 3])
