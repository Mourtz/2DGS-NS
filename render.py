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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh
from utils.render_utils import generate_path, create_videos

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--skip_mesh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render_path", action="store_true")
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: voxel grid resolution for TSDF')
    parser.add_argument("--poisson", action="store_true", help='Mesh: use Screened Poisson instead of TSDF (better quality for bounded scenes)')
    parser.add_argument("--poisson_depth", default=12, type=int, help='Mesh: Poisson octree depth (11=good, 12=high, 13=very high)')
    parser.add_argument("--num_faces", default=1_000_000, type=int, help='Mesh: target face count after decimation (0 = skip)')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    pos_neural_path = os.path.join(args.model_path, "pos_neural_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(pos_neural_path):
        from scene.neural_shader import PositionalNeuralShader
        pn_state = torch.load(pos_neural_path, map_location='cuda', weights_only=True)
        cfg = pn_state['config']
        gaussians.pos_shader = PositionalNeuralShader(
            hidden_dim=cfg.get('hidden_dim', 128),
            n_pos_freqs=cfg.get('n_pos_freqs', 6),
            n_dir_freqs=cfg.get('n_dir_freqs', 4),
            n_layers=cfg.get('n_layers', 3),
            norm_xyz=cfg.get('norm_xyz', False),
        ).cuda().eval()
        gaussians.pos_shader.load_state_dict(pn_state['model'])
        gaussians.pos_neural_active = pn_state.get('pos_neural_active', True)
        print("Loaded Positional Neural Shader from {}".format(pos_neural_path))

    siren_path = os.path.join(args.model_path, "siren_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(siren_path):
        from scene.neural_shader import SIRENPosShader
        s_state = torch.load(siren_path, map_location='cuda', weights_only=True)
        cfg = s_state['config']
        gaussians.siren_shader = SIRENPosShader(
            hidden_dim=cfg.get('hidden_dim', 128),
            n_pos_freqs=cfg.get('n_pos_freqs', 6),
            n_dir_freqs=cfg.get('n_dir_freqs', 4),
            n_layers=cfg.get('n_layers', 3),
            omega0=cfg.get('omega0', 30.0),
            skip_connection=cfg.get('skip_connection', True),
        ).cuda().eval()
        gaussians.siren_shader.load_state_dict(s_state['model'])
        gaussians.siren_active = s_state.get('siren_active', True)
        print("Loaded SIREN shader from {}".format(siren_path))

    clustered_path = os.path.join(args.model_path, "clustered_brdf_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(clustered_path):
        from scene.neural_shader import ClusteredBRDFShader
        cb_state = torch.load(clustered_path, map_location='cuda', weights_only=True)
        cfg = cb_state['config']
        gaussians.clustered_shader = ClusteredBRDFShader(
            n_clusters=cfg.get('n_clusters', 64),
            feature_dim=cfg.get('feature_dim', 16),
            hidden_dim=cfg.get('hidden_dim', 128),
            n_layers=cfg.get('n_layers', 3),
        ).cuda().eval()
        gaussians.clustered_shader.load_state_dict(cb_state['model'])
        gaussians._cluster_ids = cb_state['cluster_ids'].cuda()
        gaussians.clustered_brdf_active = True
        print("Loaded clustered BRDF shader from {}".format(clustered_path))

    render_func = render
    ds_path = os.path.join(args.model_path, "deferred_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(ds_path):
        from scene.deferred_shader import DeferredShader
        ds_state = torch.load(ds_path, map_location='cuda')
        gaussians.deferred_shader = DeferredShader(**ds_state['config']).cuda().eval()
        gaussians.deferred_shader.load_state_dict(ds_state['model'])
        gaussians.deferred_active = ds_state.get('deferred_active', True)
        print("Loaded deferred shader from {}".format(ds_path))

    pgps_ide_path = os.path.join(args.model_path, "pgps_ide_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(pgps_ide_path):
        from scene.neural_brdf_field import NeuralMaterialField, PGPSIde
        ide_state = torch.load(pgps_ide_path, map_location='cuda', weights_only=True)
        cfg = ide_state['config']
        gaussians.neural_material_field = NeuralMaterialField(
            hidden=cfg.get('hidden', 64),
            n_pos_freqs=cfg.get('n_pos_freqs', 6),
        ).cuda().eval()
        gaussians.neural_material_field.load_state_dict(ide_state['nmf_model'])
        gaussians.pgps_ide_shader = PGPSIde(
            hidden_dim=cfg.get('pgps_hidden_dim', 128),
            n_pos_freqs=cfg.get('n_pos_freqs', 6),
            n_dir_freqs=cfg.get('n_dir_freqs', 4),
            n_layers=cfg.get('pgps_n_layers', 3),
        ).cuda().eval()
        gaussians.pgps_ide_shader.load_state_dict(ide_state['pgps_ide_model'])
        gaussians.pgps_ide_active = ide_state.get('pgps_ide_active', True)
        print("Loaded PGPS+IDE from {}".format(pgps_ide_path))

    pgps_path = os.path.join(args.model_path, "pgps_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(pgps_path):
        from scene.neural_brdf_field import NeuralMaterialField, PhysicsGuidedPositionalShader
        pgps_state = torch.load(pgps_path, map_location='cuda')
        cfg = pgps_state['config']
        gaussians.neural_material_field = NeuralMaterialField(
            hidden=cfg['hidden'], n_pos_freqs=cfg['n_pos_freqs']
        ).cuda().eval()
        gaussians.neural_material_field.load_state_dict(pgps_state['nmf_model'])
        legacy = 'log_spec_scale' not in pgps_state['pgps_model']
        gaussians.pgps_shader = PhysicsGuidedPositionalShader(
            hidden_dim=cfg['pgps_hidden_dim'], n_pos_freqs=cfg['n_pos_freqs'],
            n_dir_freqs=4, n_layers=cfg['pgps_n_layers'], legacy_softplus=legacy
        ).cuda().eval()
        gaussians.pgps_shader.load_state_dict(pgps_state['pgps_model'], strict=False)
        gaussians.pgps_active = pgps_state.get('pgps_active', True)
        print("Loaded PGPS from {}".format(pgps_path))

    ct_path = os.path.join(args.model_path, "ct_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(ct_path):
        ct_state = torch.load(ct_path, map_location='cuda')
        cfg = ct_state['config']
        if cfg.get('clustered', False):
            from scene.cook_torrance_shader import ClusteredCookTorranceShader
            gaussians.ct_shader = ClusteredCookTorranceShader(
                n_lights=cfg.get('n_lights', 2),
                n_clusters=cfg.get('n_object_clusters', 12),
                freeze_materials=cfg.get('freeze_materials', False),
            ).cuda().eval()
        elif cfg.get('refgs', False):
            from scene.cook_torrance_shader import CookTorranceRefGSShader
            gaussians.ct_shader = CookTorranceRefGSShader(
                nmf_hidden=cfg['nmf_hidden'], nmf_pos_freqs=cfg['nmf_pos_freqs'],
                feat_dim=cfg['refgs_feat_dim'], n_mip_levels=cfg['sph_mip_levels'],
                env_h=cfg['refgs_env_h'], env_w=cfg['refgs_env_w'],
            ).cuda().eval()
        elif cfg.get('sph_mip', False):
            from scene.cook_torrance_shader import CookTorranceSphMipShader
            gaussians.ct_shader = CookTorranceSphMipShader(
                nmf_hidden=cfg['nmf_hidden'], nmf_pos_freqs=cfg['nmf_pos_freqs'],
                n_mip_levels=cfg['sph_mip_levels'], env_h=cfg['sph_mip_h'], env_w=cfg['sph_mip_w'],
            ).cuda().eval()
        elif cfg.get('probe', False):
            from scene.cook_torrance_shader import CookTorranceProbeShader
            gaussians.ct_shader = CookTorranceProbeShader(
                nmf_hidden=cfg['nmf_hidden'], nmf_pos_freqs=cfg['nmf_pos_freqs'],
                probe_resolution=cfg.get('probe_resolution', 64),
                probe_update_interval=cfg.get('probe_update_interval', 500),
                probe_n_blur=cfg.get('probe_n_blur', 5),
            ).cuda().eval()
        else:
            from scene.cook_torrance_shader import CookTorranceShader
            gaussians.ct_shader = CookTorranceShader(
                n_lights=cfg.get('n_lights', 2),
                nmf_hidden=cfg['nmf_hidden'], nmf_pos_freqs=cfg['nmf_pos_freqs'],
            ).cuda().eval()
        gaussians.ct_shader.load_state_dict(ct_state['ct_model'])
        gaussians.ct_active = ct_state.get('ct_active', True)
        print("Loaded CT shader ({}) from {}".format(
            'clustered' if cfg.get('clustered') else 'nmf', ct_path))

    # ── Load per-Gaussian neural shading state (Exp 5 / old Exp 17) ──────────
    neural_path = os.path.join(args.model_path, "neural_state_{}.pth".format(scene.loaded_iter))
    if os.path.exists(neural_path):
        print("Loading neural shading state from {}".format(neural_path))
        gaussians.load_neural_state(neural_path)

    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render_func, pipe, bg_color=bg_color)

    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTrainCameras())
        gaussExtractor.export_image(train_dir)

    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)
        gaussExtractor.reconstruction(scene.getTestCameras())
        gaussExtractor.export_image(test_dir)

    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)
        n_fames = 240
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)
        gaussExtractor.reconstruction(cam_traj)
        gaussExtractor.export_image(traj_dir)
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir,
                    out_name='render_traj',
                    num_frames=n_fames)

    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())
        # extract the mesh and save
        if args.unbounded:
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        elif args.poisson:
            name = 'fuse_poisson.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            mesh = gaussExtractor.extract_mesh_bounded_poisson(
                depth_trunc=depth_trunc, poisson_depth=args.poisson_depth)
        else:
            name = 'fuse.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh, write_ascii=False)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        # decimate to target face count using quadric error minimisation
        if args.num_faces > 0 and len(mesh_post.triangles) > args.num_faces:
            print(f"Decimating {len(mesh_post.triangles):,} → {args.num_faces:,} faces ...")
            mesh_post = mesh_post.simplify_quadric_decimation(args.num_faces)
            mesh_post.remove_degenerate_triangles()
            mesh_post.remove_unreferenced_vertices()
            print(f"Decimated to {len(mesh_post.triangles):,} faces, {len(mesh_post.vertices):,} vertices")
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post, write_ascii=False)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
