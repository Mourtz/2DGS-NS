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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.deferred_shader import DeferredShader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    neural_feature_dim = dataset.neural_feature_dim if dataset.neural_shading else 0
    neural_hidden_dim = dataset.neural_hidden_dim if dataset.neural_shading else 64
    gaussians = GaussianModel(
        dataset.sh_degree,
        neural_feature_dim=neural_feature_dim,
        neural_hidden_dim=neural_hidden_dim,
        neural_use_reflection=dataset.neural_use_reflection if dataset.neural_shading else True,
        neural_specular_scale=dataset.neural_specular_scale if dataset.neural_shading else 0.8,
    )
    scene = Scene(dataset, gaussians)

    if getattr(dataset, "warmstart_ply", ""):
        print(f"[warmstart] Loading geometry from {dataset.warmstart_ply}")
        gaussians.load_ply_warmstart(dataset.warmstart_ply)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    pos_neural_optimizer = None
    if dataset.pos_neural_shading:
        from scene.neural_shader import PositionalNeuralShader
        gaussians.pos_shader = PositionalNeuralShader(
            hidden_dim=dataset.pos_neural_hidden_dim,
            n_layers=dataset.pos_neural_n_layers,
            n_pos_freqs=dataset.pos_neural_n_pos_freqs,
            n_dir_freqs=dataset.pos_neural_n_dir_freqs,
            dir_noise_std=dataset.pos_neural_dir_noise,
            norm_xyz=dataset.pos_neural_norm_xyz,
        ).cuda()
        if dataset.pos_neural_norm_xyz:
            with torch.no_grad():
                xyz = gaussians.get_xyz
                scene_min = xyz.min(dim=0).values
                scene_max = xyz.max(dim=0).values
                gaussians.pos_shader.scene_center.copy_((scene_min + scene_max) / 2.0)
                gaussians.pos_shader.scene_scale.copy_(
                    ((scene_max - scene_min) / 2.0).max().clamp(min=1e-3)
                )
            print(f"Positional Neural Shader: xyz normalisation enabled — "
                  f"center={gaussians.pos_shader.scene_center.tolist()}, "
                  f"scale={gaussians.pos_shader.scene_scale.item():.4f}")
        pos_neural_optimizer = torch.optim.Adam(gaussians.pos_shader.parameters(), lr=opt.shader_lr)
        n_params = sum(p.numel() for p in gaussians.pos_shader.parameters())
        print(f"Positional Neural Shader: {n_params:,} params, "
              f"n_pos_freqs={dataset.pos_neural_n_pos_freqs}, "
              f"n_dir_freqs={dataset.pos_neural_n_dir_freqs}, "
              f"hidden={dataset.pos_neural_hidden_dim}, layers={dataset.pos_neural_n_layers}")

    siren_optimizer = None
    if dataset.siren_shading:
        from scene.neural_shader import SIRENPosShader
        gaussians.siren_shader = SIRENPosShader(
            hidden_dim=dataset.pos_neural_hidden_dim,
            n_pos_freqs=dataset.pos_neural_n_pos_freqs,
            n_dir_freqs=dataset.pos_neural_n_dir_freqs,
            n_layers=dataset.pos_neural_n_layers,
            specular_scale=dataset.neural_specular_scale,
            omega0=dataset.siren_omega0,
            skip_connection=dataset.siren_skip,
        ).cuda()
        siren_optimizer = torch.optim.Adam(gaussians.siren_shader.parameters(), lr=opt.shader_lr)
        n_params = sum(p.numel() for p in gaussians.siren_shader.parameters())
        print(f"[Exp 35] SIRENPosShader: {n_params:,} params | omega0={dataset.siren_omega0}, skip={dataset.siren_skip}")

    clustered_brdf_optimizer = None
    if dataset.clustered_brdf_shading:
        from scene.neural_shader import ClusteredBRDFShader
        gaussians.clustered_shader = ClusteredBRDFShader(
            n_clusters=dataset.clustered_brdf_n_clusters,
            feature_dim=dataset.clustered_brdf_feature_dim,
            hidden_dim=dataset.clustered_brdf_hidden_dim,
            n_layers=dataset.clustered_brdf_n_layers,
        ).cuda()
        clustered_brdf_optimizer = torch.optim.Adam(gaussians.clustered_shader.parameters(), lr=opt.shader_lr)
        n_params = sum(p.numel() for p in gaussians.clustered_shader.parameters())
        print(f"Clustered BRDF Shader: {n_params} params ({dataset.clustered_brdf_n_clusters} clusters), "
              f"activates at iter {opt.neural_warmup_iter}")

    pgps_ide_optimizer = None
    if dataset.pgps_ide:
        from scene.neural_brdf_field import NeuralMaterialField, PGPSIde
        if gaussians.neural_material_field is None:
            gaussians.neural_material_field = NeuralMaterialField(
                hidden=dataset.nmf_hidden_dim,
                n_pos_freqs=dataset.nmf_pos_freqs,
            ).cuda()
        gaussians.pgps_ide_shader = PGPSIde(
            hidden_dim=dataset.pgps_hidden_dim,
            n_pos_freqs=dataset.nmf_pos_freqs,
            n_dir_freqs=4,
            n_layers=dataset.pgps_n_layers,
        ).cuda()
        pgps_ide_optimizer = torch.optim.Adam(
            list(gaussians.neural_material_field.parameters()) +
            list(gaussians.pgps_ide_shader.parameters()),
            lr=opt.nmf_lr,
        )
        n_nmf = sum(p.numel() for p in gaussians.neural_material_field.parameters())
        n_p28  = sum(p.numel() for p in gaussians.pgps_ide_shader.parameters())
        print(f"PGPSIde: NMF={n_nmf} params, L_spec+IDE={n_p28} params, activates at iter {opt.nmf_warmup_iter}")

    ct_optimizer = None
    if dataset.cook_torrance_shading:
        if dataset.ct_clustered:
            from scene.cook_torrance_shader import ClusteredCookTorranceShader
            def _parse_floats(s):
                return [float(x) for x in s.split(',')] if s.strip() else None
            r_init = _parse_floats(dataset.ct_roughness_init)
            m_init = _parse_floats(dataset.ct_metallic_init)
            gaussians.ct_shader = ClusteredCookTorranceShader(
                n_lights=dataset.ct_n_lights,
                n_clusters=dataset.ct_n_object_clusters,
                roughness_init=r_init,
                metallic_init=m_init,
                freeze_materials=dataset.ct_freeze_materials,
            ).cuda()
            n_ct    = sum(p.numel() for p in gaussians.ct_shader.parameters())
            n_light = sum(p.numel() for p in gaussians.ct_shader.lighting.parameters())
            frozen  = " [FROZEN]" if dataset.ct_freeze_materials else ""
            print(f"CookTorrance (Clustered-K{dataset.ct_n_object_clusters}{frozen}): "
                  f"{n_ct} params (materials={n_ct-n_light}{frozen}, lighting={n_light}), "
                  f"activates at iter {opt.nmf_warmup_iter}")
        elif dataset.ct_probe:
            from scene.cook_torrance_shader import CookTorranceProbeShader
            gaussians.ct_shader = CookTorranceProbeShader(
                nmf_hidden=dataset.ct_nmf_hidden,
                nmf_pos_freqs=dataset.ct_nmf_pos_freqs,
                probe_resolution=dataset.ct_probe_resolution,
                probe_update_interval=dataset.ct_probe_update_interval,
                probe_n_blur=dataset.ct_probe_n_blur,
            ).cuda()
            variant = f"Probe-{dataset.ct_probe_resolution}px-u{dataset.ct_probe_update_interval}"
            n_ct = sum(p.numel() for p in gaussians.ct_shader.parameters())
            print(f"CookTorrance ({variant}): {n_ct} params (NMF only, probe has no params), "
                  f"activates at iter {opt.nmf_warmup_iter}")
        elif dataset.ct_refgs:
            from scene.cook_torrance_shader import CookTorranceRefGSShader
            gaussians.ct_shader = CookTorranceRefGSShader(
                nmf_hidden=dataset.ct_nmf_hidden,
                nmf_pos_freqs=dataset.ct_nmf_pos_freqs,
                feat_dim=dataset.ct_refgs_feat_dim,
                n_mip_levels=dataset.ct_sph_mip_levels,
                env_h=dataset.ct_refgs_env_h,
                env_w=dataset.ct_refgs_env_w,
            ).cuda()
            variant = f"RefGS-C{dataset.ct_refgs_feat_dim}-{dataset.ct_sph_mip_levels}x{dataset.ct_refgs_env_h}x{dataset.ct_refgs_env_w}"
            n_ct  = sum(p.numel() for p in gaussians.ct_shader.parameters())
            n_env = sum(p.numel() for p in gaussians.ct_shader.env.parameters())
            print(f"CookTorrance ({variant}): {n_ct} params (NMF={n_ct-n_env}, env+mlp={n_env}), "
                  f"activates at iter {opt.nmf_warmup_iter}")
        elif dataset.ct_sph_mip:
            from scene.cook_torrance_shader import CookTorranceSphMipShader
            gaussians.ct_shader = CookTorranceSphMipShader(
                nmf_hidden=dataset.ct_nmf_hidden,
                nmf_pos_freqs=dataset.ct_nmf_pos_freqs,
                n_mip_levels=dataset.ct_sph_mip_levels,
                env_h=dataset.ct_sph_mip_h,
                env_w=dataset.ct_sph_mip_w,
            ).cuda()
            variant = f"SphMip-{dataset.ct_sph_mip_levels}x{dataset.ct_sph_mip_h}x{dataset.ct_sph_mip_w}"
            n_ct = sum(p.numel() for p in gaussians.ct_shader.parameters())
            n_env = sum(p.numel() for p in gaussians.ct_shader.env.parameters())
            print(f"CookTorrance ({variant}): {n_ct} params (NMF={n_ct-n_env}, env={n_env}), "
                  f"activates at iter {opt.nmf_warmup_iter}")
        else:
            from scene.cook_torrance_shader import CookTorranceShader
            gaussians.ct_shader = CookTorranceShader(
                n_lights=dataset.ct_n_lights,
                nmf_hidden=dataset.ct_nmf_hidden,
                nmf_pos_freqs=dataset.ct_nmf_pos_freqs,
            ).cuda()
            variant = f"{dataset.ct_n_lights}-light"
            n_ct = sum(p.numel() for p in gaussians.ct_shader.parameters())
            n_light = sum(p.numel() for p in gaussians.ct_shader.lighting.parameters())
            print(f"CookTorrance ({variant}): {n_ct} params (NMF={n_ct-n_light}, lighting={n_light}), "
                  f"activates at iter {opt.nmf_warmup_iter}")
        ct_optimizer = torch.optim.Adam(gaussians.ct_shader.parameters(), lr=opt.nmf_lr)

    pgps_optimizer = None
    if dataset.physics_guided_shader:
        from scene.neural_brdf_field import NeuralMaterialField, PhysicsGuidedPositionalShader
        if gaussians.neural_material_field is None:
            gaussians.neural_material_field = NeuralMaterialField(
                hidden=dataset.nmf_hidden_dim,
                n_pos_freqs=dataset.nmf_pos_freqs,
            ).cuda()
        gaussians.pgps_shader = PhysicsGuidedPositionalShader(
            hidden_dim=dataset.pgps_hidden_dim,
            n_pos_freqs=dataset.nmf_pos_freqs,
            n_dir_freqs=4,
            n_layers=dataset.pgps_n_layers,
        ).cuda()
        pgps_optimizer = torch.optim.Adam(
            list(gaussians.neural_material_field.parameters()) +
            list(gaussians.pgps_shader.parameters()),
            lr=opt.nmf_lr,
        )
        n_nmf = sum(p.numel() for p in gaussians.neural_material_field.parameters())
        n_pgps = sum(p.numel() for p in gaussians.pgps_shader.parameters())
        print(f"PGPS: NMF={n_nmf} params, L_spec MLP={n_pgps} params, activates at iter {opt.nmf_warmup_iter}")

    deferred_optimizer = None
    if dataset.deferred_shading:
        gaussians.deferred_shader = DeferredShader(
            hidden=opt.deferred_hidden,
            n_freqs=opt.deferred_n_freqs,
            specular_scale=opt.deferred_specular_scale,
        ).cuda()
        deferred_optimizer = torch.optim.Adam(gaussians.deferred_shader.parameters(), lr=opt.deferred_lr)
        n_params = sum(p.numel() for p in gaussians.deferred_shader.parameters())
        print(f"Deferred shader: {n_params} params, activates at iter {opt.deferred_warmup_iter}")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Activate neural shading after geometry has stabilised
        if gaussians.neural_feature_dim > 0 and not gaussians.neural_active and iteration >= opt.neural_warmup_iter:
            gaussians.neural_active = True
            print(f"\n[ITER {iteration}] Neural shading activated")

        # Activate positional neural shader
        if gaussians.pos_shader is not None and not gaussians.pos_neural_active and iteration >= opt.neural_warmup_iter:
            gaussians.pos_neural_active = True
            print(f"\n[ITER {iteration}] Positional neural shader activated")


        # Activate Exp 35: SIREN shader
        if gaussians.siren_shader is not None and not gaussians.siren_active and iteration >= opt.neural_warmup_iter:
            gaussians.siren_active = True
            print(f"\n[ITER {iteration}] SIREN shader activated")

        # Activate clustered BRDF: run K-means on DC albedo once at warmup
        if gaussians.clustered_shader is not None and not gaussians.clustered_brdf_active and iteration >= opt.neural_warmup_iter:
            with torch.no_grad():
                albedo = torch.clamp(gaussians._features_dc.squeeze(1) * 0.2820948 + 0.5, 0.0, 1.0)
                gaussians._cluster_ids = gaussians.clustered_shader.assign_clusters_kmeans(albedo)
            gaussians.clustered_brdf_active = True
            counts = [(gaussians._cluster_ids == k).sum().item() for k in range(dataset.clustered_brdf_n_clusters)]
            print(f"\n[ITER {iteration}] Clustered BRDF activated — cluster sizes: {counts}")

        # Activate deferred neural shader (Exp 19)
        if gaussians.deferred_shader is not None and not gaussians.deferred_active and iteration >= opt.deferred_warmup_iter:
            gaussians.deferred_active = True
            print(f"\n[ITER {iteration}] Deferred shader activated")

        # Activate Exp 42/45: Cook-Torrance shader
        if gaussians.ct_shader is not None and not gaussians.ct_active and iteration >= opt.nmf_warmup_iter:
            gaussians.ct_active = True
            print(f"\n[ITER {iteration}] Cook-Torrance shader activated")

        # Exp 48: albedo-guided re-clustering for ClusteredMaterialField.
        _recluster_iter = getattr(opt, 'ct_albedo_recluster_iter', 15000)
        if (_recluster_iter > 0
                and iteration == _recluster_iter
                and gaussians.ct_shader is not None
                and hasattr(gaussians.ct_shader, 'cmf')):
            C0 = 0.28209479177387814
            with torch.no_grad():
                albedo_dc = (gaussians._features_dc.squeeze(1) * C0 + 0.5).clamp(0.0, 1.0)
            gaussians.ct_shader.cmf._clustered = False  # allow re-cluster
            gaussians.ct_shader.cmf.cluster(
                gaussians.get_xyz.detach(), albedo=albedo_dc,
                color_weight=getattr(opt, 'ct_albedo_color_weight', 3.0))
            print(f"\n[ITER {iteration}] ClusteredMaterialField: albedo-guided re-cluster done")

        # Exp 45: update environment probe periodically
        if (gaussians.ct_shader is not None and gaussians.ct_active
                and hasattr(gaussians.ct_shader, 'probe')
                and gaussians.ct_shader.probe.should_update(iteration)):
            gaussians.ct_shader.probe.update(gaussians, pipe, background, iteration)
            print(f"\n[ITER {iteration}] Environment probe updated")

        # Activate Physics-Guided Positional Neural Shader (Exp 27)
        if gaussians.pgps_shader is not None and not gaussians.pgps_active and iteration >= opt.nmf_warmup_iter:
            gaussians.pgps_active = True
            print(f"\n[ITER {iteration}] Physics-Guided Positional Shader activated")

        # Activate Exp 28: PGPS + IDE
        if gaussians.pgps_ide_shader is not None and not gaussians.pgps_ide_active and iteration >= opt.nmf_warmup_iter:
            gaussians.pgps_ide_active = True
            print(f"\n[ITER {iteration}] PGPSIde (IDE) activated")

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, sh_dropout=opt.sh_dropout)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # neural feat L2 regularisation
        neural_reg_loss = torch.tensor(0.0, device="cuda")
        if gaussians.neural_active and gaussians.neural_feature_dim > 0:
            neural_reg_loss = opt.lambda_neural_reg * gaussians._neural_features.norm(dim=-1).mean()

        # loss
        total_loss = loss + dist_loss + normal_loss + neural_reg_loss

        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log


            if iteration % 10 == 0:
                if gaussians.clustered_brdf_active:
                    shader_status = "clustered"
                elif gaussians.pos_neural_active:
                    shader_status = "pos_neural"
                elif gaussians.ct_active:
                    shader_status = "CT"
                elif gaussians.neural_active:
                    shader_status = "neural"
                elif gaussians.deferred_active:
                    shader_status = "def"
                else:
                    shader_status = "off"
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "shader": shader_status,
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                if gaussians.deferred_shader is not None:
                    deferred_path = os.path.join(scene.model_path, "deferred_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {'hidden': 128, 'n_freqs': 4, 'specular_scale': 0.8},
                        'model': gaussians.deferred_shader.state_dict(),
                        'deferred_active': gaussians.deferred_active,
                    }, deferred_path)
                    print("[ITER {}] Saved deferred shader → {}".format(iteration, deferred_path))
                if gaussians.clustered_shader is not None:
                    path = os.path.join(scene.model_path, "clustered_brdf_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {
                            'n_clusters': dataset.clustered_brdf_n_clusters,
                            'feature_dim': dataset.clustered_brdf_feature_dim,
                            'hidden_dim': dataset.clustered_brdf_hidden_dim,
                            'n_layers': dataset.clustered_brdf_n_layers,
                        },
                        'model': gaussians.clustered_shader.state_dict(),
                        'cluster_ids': gaussians._cluster_ids.cpu(),
                    }, path)
                    print("[ITER {}] Saved clustered BRDF shader → {}".format(iteration, path))
                if gaussians.pos_shader is not None:
                    pos_path = os.path.join(scene.model_path, "pos_neural_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {
                            'hidden_dim': dataset.pos_neural_hidden_dim,
                            'n_layers': dataset.pos_neural_n_layers,
                            'n_pos_freqs': dataset.pos_neural_n_pos_freqs,
                            'n_dir_freqs': dataset.pos_neural_n_dir_freqs,
                            'norm_xyz': dataset.pos_neural_norm_xyz,
                        },
                        'model': gaussians.pos_shader.state_dict(),
                        'pos_neural_active': gaussians.pos_neural_active,
                    }, pos_path)
                    print("[ITER {}] Saved positional neural shader → {}".format(iteration, pos_path))
                # Exp 35: SIREN shader
                if gaussians.siren_shader is not None:
                    path = os.path.join(scene.model_path, "siren_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {
                            'hidden_dim': dataset.pos_neural_hidden_dim,
                            'n_pos_freqs': dataset.pos_neural_n_pos_freqs,
                            'n_dir_freqs': dataset.pos_neural_n_dir_freqs,
                            'n_layers': dataset.pos_neural_n_layers,
                            'omega0': dataset.siren_omega0,
                            'skip_connection': dataset.siren_skip,
                        },
                        'model': gaussians.siren_shader.state_dict(),
                        'siren_active': gaussians.siren_active,
                    }, path)
                    print("[ITER {}] Saved SIREN shader → {}".format(iteration, path))
                if gaussians.ct_shader is not None:
                    ct_path = os.path.join(scene.model_path, "ct_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {
                            'sph_mip': dataset.ct_sph_mip,
                            'refgs': dataset.ct_refgs,
                            'probe': dataset.ct_probe,
                            'clustered': dataset.ct_clustered,
                            'n_object_clusters': dataset.ct_n_object_clusters,
                            'freeze_materials': dataset.ct_freeze_materials,
                            'n_lights': dataset.ct_n_lights,
                            'nmf_hidden': dataset.ct_nmf_hidden,
                            'nmf_pos_freqs': dataset.ct_nmf_pos_freqs,
                            'sph_mip_levels': dataset.ct_sph_mip_levels,
                            'sph_mip_h': dataset.ct_sph_mip_h,
                            'sph_mip_w': dataset.ct_sph_mip_w,
                            'refgs_feat_dim': dataset.ct_refgs_feat_dim,
                            'refgs_env_h': dataset.ct_refgs_env_h,
                            'refgs_env_w': dataset.ct_refgs_env_w,
                        },
                        'ct_model': gaussians.ct_shader.state_dict(),
                        'ct_active': gaussians.ct_active,
                    }, ct_path)
                    print("[ITER {}] Saved CT shader → {}".format(iteration, ct_path))
                if gaussians.pgps_shader is not None:
                    pgps_path = os.path.join(scene.model_path, "pgps_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {
                            'hidden': dataset.nmf_hidden_dim,
                            'n_pos_freqs': dataset.nmf_pos_freqs,
                            'pgps_hidden_dim': dataset.pgps_hidden_dim,
                            'pgps_n_layers': dataset.pgps_n_layers,
                        },
                        'nmf_model': gaussians.neural_material_field.state_dict(),
                        'pgps_model': gaussians.pgps_shader.state_dict(),
                        'pgps_active': gaussians.pgps_active,
                    }, pgps_path)
                    print("[ITER {}] Saved PGPS → {}".format(iteration, pgps_path))
                # Exp 28: PGPS + IDE
                if gaussians.pgps_ide_shader is not None:
                    path = os.path.join(scene.model_path, "pgps_ide_state_{}.pth".format(iteration))
                    torch.save({
                        'config': {
                            'hidden': dataset.nmf_hidden_dim,
                            'n_pos_freqs': dataset.nmf_pos_freqs,
                            'pgps_hidden_dim': dataset.pgps_hidden_dim,
                            'pgps_n_layers': dataset.pgps_n_layers,
                        },
                        'nmf_model': gaussians.neural_material_field.state_dict(),
                        'pgps_ide_model': gaussians.pgps_ide_shader.state_dict(),
                    }, path)
                    print("[ITER {}] Saved PGPSIde → {}".format(iteration, path))


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                if gaussians.shader_optimizer is not None:
                    gaussians.shader_optimizer.step()
                    gaussians.shader_optimizer.zero_grad(set_to_none=True)
                if deferred_optimizer is not None and gaussians.deferred_active:
                    deferred_optimizer.step()
                    deferred_optimizer.zero_grad(set_to_none=True)
                if pos_neural_optimizer is not None and gaussians.pos_neural_active:
                    pos_neural_optimizer.step()
                    pos_neural_optimizer.zero_grad(set_to_none=True)
                if siren_optimizer is not None and gaussians.siren_active:
                    siren_optimizer.step()
                    siren_optimizer.zero_grad(set_to_none=True)
                if clustered_brdf_optimizer is not None and gaussians.clustered_brdf_active:
                    clustered_brdf_optimizer.step()
                    clustered_brdf_optimizer.zero_grad(set_to_none=True)
                if ct_optimizer is not None and gaussians.ct_active:
                    ct_optimizer.step()
                    ct_optimizer.zero_grad(set_to_none=True)
                if pgps_optimizer is not None and gaussians.pgps_active:
                    pgps_optimizer.step()
                    pgps_optimizer.zero_grad(set_to_none=True)
                if pgps_ide_optimizer is not None and gaussians.pgps_ide_active:
                    pgps_ide_optimizer.step()
                    pgps_ide_optimizer.zero_grad(set_to_none=True)

            if (iteration in saving_iterations) and gaussians.neural_feature_dim > 0:
                neural_path = scene.model_path + "/neural_state_{}.pth".format(iteration)
                gaussians.save_neural_state(neural_path)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        with torch.no_grad():        
            if network_gui.conn == None:
                network_gui.try_connect(dataset.render_items)
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # Send the data
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break
                except Exception as e:
                    # raise e
                    network_gui.conn = None

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, seed=args.seed)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
