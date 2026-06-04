#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
from utils.general_utils import get_device

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width
        H = viewpoint_cam.image_height
        device = viewpoint_cam.world_view_transform.device
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().to(device).T
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),
            cy = intrins[1,2].item(), 
            fx = intrins[0,0].item(), 
            fy = intrins[1,1].item()
        )

        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic
        camera.intrinsic = intrinsic
        camera_traj.append(camera)

    return camera_traj

class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        a class that extracts attributes a scene presented by 2DGS

        Usage example:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        device = get_device()
        background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        self.gaussians = gaussians
        self.render = partial(render, pipe=pipe, bg_color=background)
        self.clean()

    @torch.no_grad()
    def clean(self):
        self.depthmaps = []
        self.rgbmaps = []
        self.depth_normals = []
        self.viewpoint_stack = []

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        reconstruct radiance field given cameras
        """
        self.clean()
        self.viewpoint_stack = viewpoint_stack
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)
            rgb = render_pkg['render']
            depth = render_pkg['surf_depth']
            depth_normal = render_pkg['surf_normal']
            self.rgbmaps.append(rgb.cpu())
            self.depthmaps.append(depth.cpu())
            self.depth_normals.append(depth_normal.cpu())
        
        self.estimate_bounding_sphere()

    def estimate_bounding_sphere(self):
        """
        Estimate the bounding sphere given camera pose
        """
        from utils.render_utils import focus_point_fn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        center = (focus_point_fn(poses))
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().to(self.gaussians.device)
        print(f"The estimated bounding radius is {self.radius:.2f}")
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        Perform TSDF fusion given a fixed depth range, used in the paper.

        voxel_size: the voxel size of the volume
        sdf_trunc: truncation value
        depth_trunc: maximum depth range, should depended on the scene's scales
        mask_backgrond: whether to mask backgroud, only works when the dataset have masks

        return o3d.mesh
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        # Use tensor-based VoxelBlockGrid — ScalableTSDFVolume segfaults on open3d 0.18 + CachyOS.
        if torch.cuda.is_available():
            o3d_device = o3d.core.Device('CUDA:0')
        else:
            o3d_device = o3d.core.Device('CPU:0')
        trunc_multiplier = sdf_trunc / voxel_size

        if torch.cuda.is_available():
            _free_bytes = torch.cuda.mem_get_info()[0]
        else:
            _free_bytes = 16 * 1024 ** 3  # assume 16 GB free on CPU
        _bytes_per_block = 65 * 1024 # empirical bytes per block for 16^3 voxels
        
        # block_count consumes ~70 % of currently free VRAM, minus a 2 GB safety margin.
        block_count = max(10_000, int((_free_bytes - 2 * 1024 ** 3) * 0.70 / _bytes_per_block))
        print(f'block_count: {block_count}  ({block_count * _bytes_per_block / 1024**3:.1f} GB reserved on {o3d_device})')
        print(f'tip: pass --mesh_res 4096 (or higher) for finer voxels; current voxel_size={voxel_size:.5f} m')

        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=('tsdf', 'weight', 'color'),
            attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=float(voxel_size),
            block_resolution=16,
            block_count=block_count,
            device=o3d_device,
        )

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]
            depth = self.depthmaps[i].clone()
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # if we have mask provided, use it
            if mask_backgrond and (viewpoint_cam.gt_alpha_mask is not None):
                depth[(viewpoint_cam.gt_alpha_mask < 0.5)] = 0

            W = viewpoint_cam.image_width
            H = viewpoint_cam.image_height
            device = viewpoint_cam.world_view_transform.device
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, 0, 1]]).float().to(device).T
            intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
            intrinsic = o3d.core.Tensor(np.array([
                [intrins[0,0].item(), 0,                   intrins[0,2].item()],
                [0,                   intrins[1,1].item(), intrins[1,2].item()],
                [0,                   0,                   1                  ],
            ], dtype=np.float64))
            extrinsic = o3d.core.Tensor(
                np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy(), dtype=np.float64))

            # depth: [H, W] float32;  color: [H, W, 3] float32 ∈ [0,1]
            depth_o3d = o3d.t.geometry.Image(
                o3d.core.Tensor(depth.squeeze(0).contiguous().numpy(), device=o3d_device))
            color_o3d = o3d.t.geometry.Image(
                o3d.core.Tensor(np.clip(rgb.permute(1,2,0).contiguous().numpy(), 0.0, 1.0).astype(np.float32), device=o3d_device))

            frustum_coords = vbg.compute_unique_block_coordinates(
                depth_o3d, intrinsic, extrinsic, depth_scale=1.0, depth_max=depth_trunc)
            vbg.integrate(frustum_coords, depth_o3d, color_o3d,
                          intrinsic, extrinsic,
                          depth_scale=1.0, depth_max=depth_trunc,
                          trunc_voxel_multiplier=trunc_multiplier)

        mesh = vbg.extract_triangle_mesh()
        return mesh.to_legacy()

    @torch.no_grad()
    def extract_mesh_bounded_poisson(self, depth_trunc=None, mask_backgrond=True,
                                     poisson_depth=12, point_voxel_size=None):
        """
        Extract mesh via Screened Poisson Reconstruction.
        Uses rendered depth + surf_normal maps from all training views.
        Produces cleaner topology than TSDF: no voxelisation artifacts, handles
        thin structures and sharp edges better.

        poisson_depth: octree depth (9=fast/coarse, 11=good, 12=high, 13=very high)
        point_voxel_size: downsample voxel size (None = depth_trunc/2048)
        """
        if depth_trunc is None:
            depth_trunc = self.radius * 2.0
        if point_voxel_size is None:
            point_voxel_size = depth_trunc / 2048

        print("Fusing depth maps into oriented point cloud ...")
        print(f'depth_trunc: {depth_trunc}  point_voxel_size: {point_voxel_size:.5f}  poisson_depth: {poisson_depth}')

        all_points  = []
        all_normals = []
        all_colors  = []

        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="Back-projecting"):
            rgb          = self.rgbmaps[i]              # [3, H, W]
            depth        = self.depthmaps[i].clone()    # [1, H, W]
            surf_normal  = self.depth_normals[i]        # [3, H, W] world-space, alpha-weighted

            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            if mask_backgrond and (viewpoint_cam.gt_alpha_mask is not None):
                depth[(viewpoint_cam.gt_alpha_mask < 0.5)] = 0
            depth[depth > depth_trunc] = 0

            H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
            device = viewpoint_cam.world_view_transform.device
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W-1) / 2],
                [0, H / 2, 0, (H-1) / 2],
                [0, 0, 0, 1]]).float().to(device).T
            intrins = (viewpoint_cam.projection_matrix @ ndc2pix)[:3, :3].T
            fx, fy = intrins[0, 0].item(), intrins[1, 1].item()
            cx, cy = intrins[0, 2].item(), intrins[1, 2].item()

            depth_np = depth.squeeze(0).numpy()  # [H, W]
            # keep pixels where depth is valid and normal magnitude > 0.1
            # (surf_normal is alpha-weighted so magnitude ≈ render_alpha)
            nrm_np = surf_normal.permute(1, 2, 0).numpy()  # [H, W, 3]
            nrm_mag = np.linalg.norm(nrm_np, axis=2)       # [H, W]
            valid = (depth_np > 0) & (nrm_mag > 0.1)
            if not valid.any():
                continue

            v_idx, u_idx = np.where(valid)
            z = depth_np[valid]

            # back-project to camera space then to world space
            pts_cam = np.stack([
                (u_idx - cx) * z / fx,
                (v_idx - cy) * z / fy,
                z,
                np.ones_like(z),
            ], axis=1)  # [N, 4]
            c2w = np.linalg.inv(viewpoint_cam.world_view_transform.T.cpu().numpy())
            pts_world = (pts_cam @ c2w.T)[:, :3]  # [N, 3]

            # renormalise alpha-weighted surf_normal
            nrms = nrm_np[valid]                                         # [N, 3]
            nrms = nrms / (nrm_mag[valid, None] + 1e-8)

            clrs = np.clip(rgb.permute(1, 2, 0).numpy()[valid], 0.0, 1.0)

            all_points.append(pts_world.astype(np.float32))
            all_normals.append(nrms.astype(np.float32))
            all_colors.append(clrs.astype(np.float32))

        points  = np.concatenate(all_points,  axis=0)
        normals = np.concatenate(all_normals, axis=0)
        colors  = np.concatenate(all_colors,  axis=0)

        if torch.cuda.is_available():
            gpu = o3d.core.Device('CUDA:0')
        else:
            gpu = o3d.core.Device('CPU:0')
        pcd_t = o3d.t.geometry.PointCloud(gpu)
        pcd_t.point['positions'] = o3d.core.Tensor(points,  dtype=o3d.core.float32, device=gpu)
        pcd_t.point['normals']   = o3d.core.Tensor(normals, dtype=o3d.core.float32, device=gpu)
        pcd_t.point['colors']    = o3d.core.Tensor(colors,  dtype=o3d.core.float32, device=gpu)

        print(f"Downsampling {len(pcd_t.point['positions']):,} points (voxel_size={point_voxel_size:.5f}) ...")
        pcd_t = pcd_t.voxel_down_sample(point_voxel_size)
        # renormalise averaged normals on GPU
        nrms_t = pcd_t.point['normals']
        mag    = (nrms_t * nrms_t).sum(1, keepdim=True).sqrt()
        pcd_t.point['normals'] = nrms_t / (mag + 1e-8)
        print(f"{len(pcd_t.point['positions']):,} points after downsampling")

        # move to CPU only for the Poisson call (no tensor-API equivalent)
        pcd = pcd_t.to(o3d.core.Device('CPU:0')).to_legacy()

        print(f"Running Screened Poisson (depth={poisson_depth}) ...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth, width=0, scale=1.1, linear_fit=False)

        # trim low-density boundary artifacts (bottom 2 % by density)
        densities = np.asarray(densities)
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
        return o3d.mesh
        """
        def contract(x):
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
        
        def uncontract(y):
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
                compute per frame sdf
            """
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]
            pix_coords = (new_points[..., :2] / new_points[..., -1:])
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            sampled_depth = torch.nn.functional.grid_sample(depthmap[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
                Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
            """
            if inv_contraction is not None:
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # adaptive sdf_truncation
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)
            else:
                sdf_trunc = 5 * voxel_size

            tsdfs = torch.ones_like(samples[:,0]) * (-1)
            rgbs = torch.zeros((samples.shape[0], 3), device=samples.device)

            weights = torch.ones_like(samples[:,0])
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # volume integration
                sdf = sdf.flatten()
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]
                wp = w + 1
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        normalize = lambda x: (x - self.center) / self.radius
        unnormalize = lambda x: (x * self.radius) + self.center
        inv_contraction = lambda x: unnormalize(uncontract(x))

        N = resolution
        voxel_size = (self.radius * 2 / N)
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        from utils.mcube_utils import marching_cubes_with_contraction
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)
        R = min(R+0.01, 1.9)

        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,
            bounding_box_min=(-R, -R, -R),
            bounding_box_max=(R, R, R),
            level=0,
            resolution=N,
            inv_contraction=inv_contraction,
        )
        
        # coloring the mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        mesh = mesh.as_open3d
        print("texturing mesh ... ")
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().to(self.gaussians.device), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        render_path = os.path.join(path, "renders")
        gts_path = os.path.join(path, "gt")
        vis_path = os.path.join(path, "vis")
        os.makedirs(render_path, exist_ok=True)
        os.makedirs(vis_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))
