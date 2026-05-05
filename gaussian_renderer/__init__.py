#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
import math
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.point_utils import depth_to_normal
from utils.general_utils import build_rotation

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, sh_dropout = 0.0):
    """
    Render the scene. 

    Background tensor (bg_color) must be on GPU!
    """

    if sh_dropout > 0.0:
        features_rest_drop = F.dropout(pc._features_rest, p=sh_dropout, training=True)
        sh_features = torch.cat([pc._features_dc, features_rest_drop], dim=1)
    else:
        sh_features = pc.get_features

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        # currently don't support normal consistency loss if use precomputed covariance
        splat2world = pc.get_covariance(scaling_modifier)
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        world2pix =  viewpoint_camera.full_proj_transform @ ndc2pix
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9) # column major
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
    
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    pipe.convert_SHs_python = False
    shs = None
    colors_precomp = None
    if override_color is None:
        if pc.pgps_ide_shader is not None and pc.pgps_ide_active and pc.neural_material_field is not None:
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1)
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)

            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh1_degree = min(1, pc.active_sh_degree)
            albedo = torch.clamp(eval_sh(sh1_degree, shs_view, dir_pp_normalized) + 0.5, 0.0, 1.0)

            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            view_dir = -dir_pp_normalized
            surf_normals = surf_normals * torch.sign((view_dir * surf_normals).sum(-1, keepdim=True).clamp(min=1e-8))
            cos_theta = (view_dir * surf_normals).sum(-1, keepdim=True).clamp(0.0, 1.0)
            refl_dir = F.normalize(2.0 * cos_theta * surf_normals - view_dir, dim=-1)

            roughness, metallic = pc.neural_material_field(pc.get_xyz)
            specular = pc.pgps_ide_shader(pc.get_xyz, albedo, roughness, metallic, refl_dir, cos_theta)
            colors_precomp = torch.clamp(albedo + specular, 0.0, 1.0)

        elif pc.ct_shader is not None and pc.ct_active:
            C0 = 0.28209479177387814
            albedo = torch.clamp(pc._features_dc.squeeze(1) * C0 + 0.5, 0.0, 1.0)

            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_xyz.shape[0], 1)
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)
            view_dir = -dir_pp_normalized

            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            surf_normals = surf_normals * torch.sign((view_dir * surf_normals).sum(-1, keepdim=True).clamp(min=1e-8))

            colors_precomp = pc.ct_shader(pc.get_xyz, albedo, surf_normals, view_dir)

        elif pc.pgps_shader is not None and pc.pgps_active and pc.neural_material_field is not None:
            dir_pp = pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1)
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)

            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh1_degree = min(1, pc.active_sh_degree)
            albedo = torch.clamp(eval_sh(sh1_degree, shs_view, dir_pp_normalized) + 0.5, 0.0, 1.0)

            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            view_dir = -dir_pp_normalized
            surf_normals = surf_normals * torch.sign((view_dir * surf_normals).sum(-1, keepdim=True).clamp(min=1e-8))
            cos_theta = (view_dir * surf_normals).sum(-1, keepdim=True).clamp(0.0, 1.0)
            refl_dir = F.normalize(2.0 * cos_theta * surf_normals - view_dir, dim=-1)

            roughness, metallic = pc.neural_material_field(pc.get_xyz)
            specular = pc.pgps_shader(pc.get_xyz, albedo, roughness, metallic, refl_dir, cos_theta)
            colors_precomp = torch.clamp(albedo + specular, 0.0, 1.0)

        elif pc.clustered_shader is not None and pc.clustered_brdf_active and pc._cluster_ids.shape[0] == pc.get_xyz.shape[0]:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)
            albedo = torch.clamp(eval_sh(0, shs_view, dir_pp_normalized) + 0.5, 0.0, 1.0)
            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            residual = pc.clustered_shader(pc._cluster_ids, dir_pp_normalized, surf_normals)
            colors_precomp = torch.clamp(albedo + residual, 0.0, 1.0)

        elif pc.siren_shader is not None and pc.siren_active:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)
            albedo = eval_sh(0, shs_view, dir_pp_normalized) + 0.5
            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            residual = pc.siren_shader(pc.get_xyz, dir_pp_normalized, surf_normals)
            colors_precomp = torch.clamp(albedo + residual, 0.0, 1.0)

        elif pc.pos_shader is not None and pc.pos_neural_active:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)
            sh1_degree = min(1, pc.active_sh_degree)
            albedo = eval_sh(sh1_degree, shs_view, dir_pp_normalized) + 0.5
            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            residual = pc.pos_shader(pc.get_xyz, dir_pp_normalized, surf_normals)
            colors_precomp = torch.clamp(albedo + residual, 0.0, 1.0)
        elif pc.shader is not None and pc.neural_active and pc._neural_features.shape[0] > 0:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
            dir_pp_normalized = F.normalize(dir_pp, dim=-1)
            sh_color = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized) + 0.5
            R = build_rotation(pc.get_rotation)
            surf_normals = F.normalize(R[:, :, 2], dim=-1)
            specular = pc.shader(pc._neural_features, dir_pp_normalized, surf_normals)
            colors_precomp = torch.clamp(sh_color + specular, 0.0, 1.0)
        elif pipe.convert_SHs_python:
            shs_view = sh_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(sh_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = sh_features
    else:
        colors_precomp = override_color
    
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp
    )

    if getattr(pc, 'deferred_shader', None) is not None and getattr(pc, 'deferred_active', False):
        from scene.deferred_shader import compute_reflection_direction

        albedo_map = rendered_image

        N = means3D.shape[0]
        dev = means3D.device
        if pc._roughness.shape[0] == N:
            rough_ch = torch.sigmoid(pc._roughness.to(dev))
            metal_ch = torch.sigmoid(pc._metallic.to(dev))
        else:
            rough_ch = torch.full((N, 1), 0.5, device=dev)
            metal_ch = torch.full((N, 1), 0.0, device=dev)
        mat_colors = torch.cat([rough_ch, metal_ch, torch.zeros(N, 1, device=dev)], dim=1)

        mat_map, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D.detach(),
            shs=None,
            colors_precomp=mat_colors,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        roughness_map = mat_map[0:1]
        metallic_map  = mat_map[1:2]

        render_normal_ds = allmap[2:5]
        render_normal_ds = (
            render_normal_ds.permute(1, 2, 0)
            @ (viewpoint_camera.world_view_transform[:3, :3].T)
        ).permute(2, 0, 1)
        render_alpha_ds = allmap[1:2]

        refl_dir = compute_reflection_direction(viewpoint_camera, render_normal_ds)

        rendered_image = pc.deferred_shader(
            albedo_map, render_normal_ds, render_alpha_ds,
            roughness_map, metallic_map, refl_dir
        )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rets =  {"render": rendered_image,
            "viewspace_points": means2D,
            "visibility_filter" : radii > 0,
            "radii": radii,
    }


    # additional regularizations
    render_alpha = allmap[1:2]

    # get normal map
    # transform normal from view space to world space
    render_normal = allmap[2:5]
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # get median depth map
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # get expected depth map
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    
    # get depth distortion map
    render_dist = allmap[6:7]

    # psedo surface attributes
    # surf depth is either median or expected by setting depth_ratio to 1 or 0
    # for bounded scene, use median depth, i.e., depth_ratio = 1; 
    # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # remember to multiply with accum_alpha since render_normal is unnormalized.
    surf_normal = surf_normal * (render_alpha).detach()


    rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
    })

    return rets