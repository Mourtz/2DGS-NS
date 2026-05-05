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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.render_items = ['RGB', 'Alpha', 'Normal', 'Depth', 'Edge', 'Curvature']
        self.warmstart_ply = ""
        self.neural_shading = False
        self.neural_feature_dim = 32
        self.neural_hidden_dim = 64
        self.neural_use_reflection = True
        self.neural_specular_scale = 0.8
        self.pos_neural_shading = False
        self.pos_neural_hidden_dim = 128
        self.pos_neural_n_layers = 3
        self.pos_neural_n_pos_freqs = 6
        self.pos_neural_n_dir_freqs = 4
        self.pos_neural_dir_noise = 0.0
        self.pos_neural_norm_xyz = False
        self.clustered_brdf_shading = False
        self.clustered_brdf_n_clusters = 16
        self.clustered_brdf_feature_dim = 8
        self.clustered_brdf_hidden_dim = 64
        self.clustered_brdf_n_layers = 3
        self.deferred_shading = False
        self.deferred_hidden = 128
        self.deferred_n_freqs = 4
        self.deferred_specular_scale = 0.8
        self.neural_material_field = False
        self.nmf_hidden_dim = 64
        self.nmf_pos_freqs = 6
        self.physics_guided_shader = False
        self.pgps_hidden_dim = 128
        self.pgps_n_layers = 3
        self.pgps_ide = False
        self.cook_torrance_shading = False
        self.ct_n_lights = 2
        self.ct_nmf_hidden = 64
        self.ct_nmf_pos_freqs = 6
        self.ct_sph_mip = False
        self.ct_sph_mip_levels = 8
        self.ct_sph_mip_h = 64
        self.ct_sph_mip_w = 128
        self.ct_probe = False
        self.ct_probe_resolution = 64
        self.ct_probe_update_interval = 500
        self.ct_probe_n_blur = 5           
        self.ct_refgs = False
        self.ct_refgs_feat_dim = 8
        self.ct_refgs_env_h = 8            
        self.ct_refgs_env_w = 16           
        self.ct_clustered = False
        self.ct_n_object_clusters = 12
        self.ct_roughness_init = ""
        self.ct_metallic_init  = ""
        self.ct_freeze_materials = False
        self.siren_shading = False
        self.siren_omega0 = 30.0
        self.siren_skip = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 0.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_dist = 0.0
        self.lambda_normal = 0.05
        self.sh_dropout = 0.0
        self.opacity_cull = 0.05

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.neural_feature_lr = 0.0001
        self.shader_lr = 1e-4
        self.neural_warmup_iter = 20000
        self.lambda_neural_reg = 0.001
        self.deferred_lr = 0.001
        self.deferred_warmup_iter = 15000
        self.roughness_lr = 0.01
        self.metallic_lr = 0.01
        self.nmf_lr = 0.001
        self.nmf_warmup_iter = 3000
        self.ct_albedo_recluster_iter = 15000
        self.ct_albedo_color_weight = 3.0  # albedo vs position weight in joint K-Means
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
