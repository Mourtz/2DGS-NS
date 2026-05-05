# Reproducing the Thesis Runs

# Table 1 — Main results (Materials, 30k and 60k)

| Row                                 | Section |
|-------------------------------------|---------|
| SH3 baseline (3 seeds)              | 1.1     |
| GaussianShader                      | 1.2     |
| SH1 + Pos MLP (3 seeds)             | 1.3     |
| SH1 + PGPS                          | 1.4     |
| **SH1 + PGPS + IDE**                | 1.5     |
| **CT BRDF, 2 directional lights**   | 1.6     |
| **CT BRDF, SphMip 8×8×16**          | 1.7     |

## 1.1 — SH3 baseline (3 seeds, λ<sub>n</sub>=0.1)

```bash
for s in 1 2 3; do
  python train.py \
    -s data/nerf_synthetic/materials \
    -m output/seed_sweep/sh3_s$s \
    --white_background --eval \
    --sh_degree 3 --lambda_normal 0.1 \
    --iterations 60000 --position_lr_max_steps 60000 \
    --test_iterations 30000 60000 \
    --save_iterations 30000 60000 \
    --seed $s
done
```

## 1.2 — GaussianShader

Not in this repo — reproduced from the original authors' code
(<https://github.com/Asparagus15/GaussianShader>):

```bash
# from inside the GaussianShader repo
python train.py -s data/nerf_synthetic/materials -m output/gs_materials \
  --white_background --eval --brdf_dim 0 --iterations 30000
```

## 1.3 — SH1 + Pos MLP (3 seeds, hidden=256, 4 layers, λ<sub>n</sub>=0.1)

```bash
for s in 1 2 3; do
  python train.py \
    -s data/nerf_synthetic/materials \
    -m output/seed_sweep_posbig/posbig_s$s \
    --white_background --eval \
    --sh_degree 1 --pos_neural_shading \
    --pos_neural_hidden_dim 256 --pos_neural_n_layers 4 \
    --shader_lr 0.001 --lambda_normal 0.1 \
    --iterations 60000 --position_lr_max_steps 60000 \
    --test_iterations 30000 60000 \
    --save_iterations 30000 60000 \
    --seed $s
done
```

## 1.4 — SH1 + PGPS (Materials, 60k)

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t1_pgps \
  --white_background --eval \
  --sh_degree 1 --physics_guided_shader \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 1.5 — SH1 + PGPS + IDE (Materials, 60k)

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t1_pgps_ide \
  --white_background --eval \
  --sh_degree 1 --pgps_ide \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 1.6 — CT BRDF, 2 directional lights (Materials, 60k, 3 seeds)

All Cook-Torrance runs use `--sh_degree 0`: the CT shader reads only the SH0 (DC) coefficient,
so allocating SH-rest is wasted memory and produces dead gradients during the 0–3000 NMF warmup.

```bash
for s in 1 2 3; do
  python train.py \
    -s data/nerf_synthetic/materials \
    -m output/mat_ct2_sh0_seed$s \
    --white_background --eval \
    --sh_degree 0 \
    --cook_torrance_shading --ct_n_lights 2 \
    --iterations 60000 --position_lr_max_steps 60000 \
    --test_iterations 30000 60000 \
    --save_iterations 30000 60000 \
    --seed $s
done
```

## 1.7 — CT BRDF + SphMip 8×8×16 (Materials, 60k)

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/ct_sphmip_small_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_sph_mip \
  --ct_sph_mip_levels 8 --ct_sph_mip_h 8 --ct_sph_mip_w 16 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

---

# Table 2 — Ablation (Materials, 60k)

| Row                                 | Section |
|-------------------------------------|---------|
| Per-Gaussian feat=32                | 2.1     |
| Clustered K=64                      | 2.2     |
| Deferred MLP                        | 2.3     |
| SH0 + Pos MLP                       | 2.4     |
| SH2 + Pos MLP                       | 2.5     |
| SIREN 4L + skip (hidden=128)        | 2.6     |
| SIREN 256-hidden                    | 2.7     |

## 2.1 — Per-Gaussian neural features (feat=32)

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_per_gaussian_feat32 \
  --white_background --eval \
  --sh_degree 0 --neural_shading --neural_feature_dim 32 \
  --neural_warmup_iter 3000 --neural_feature_lr 0.001 --shader_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 2.2 — Clustered K=64

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_clustered_k64 \
  --white_background --eval \
  --sh_degree 0 --clustered_brdf_shading \
  --clustered_brdf_n_clusters 64 --clustered_brdf_feature_dim 16 \
  --clustered_brdf_hidden_dim 128 --clustered_brdf_n_layers 3 \
  --neural_warmup_iter 3000 --shader_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 2.3 — Deferred MLP (screen-space)

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_deferred \
  --white_background --eval \
  --sh_degree 0 --deferred_shading \
  --deferred_warmup_iter 5000 --deferred_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 2.4 — SH0 + Pos MLP

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_sh0_pos \
  --white_background --eval \
  --sh_degree 0 --pos_neural_shading \
  --pos_neural_hidden_dim 128 --pos_neural_n_layers 3 \
  --neural_warmup_iter 3000 --shader_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 2.5 — SH2 + Pos MLP

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_sh2_pos \
  --white_background --eval \
  --sh_degree 2 --pos_neural_shading \
  --pos_neural_hidden_dim 128 --pos_neural_n_layers 3 \
  --neural_warmup_iter 3000 --shader_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 2.6 — SIREN 4L + skip (hidden=128)

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_siren_4l \
  --white_background --eval \
  --sh_degree 1 --siren_shading --siren_skip --siren_omega0 30.0 \
  --pos_neural_hidden_dim 128 --pos_neural_n_layers 4 \
  --neural_warmup_iter 3000 --shader_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

## 2.7 — SIREN 256-hidden

```bash
python train.py \
  -s data/nerf_synthetic/materials \
  -m output/t2_siren_256 \
  --white_background --eval \
  --sh_degree 1 --siren_shading --siren_skip --siren_omega0 30.0 \
  --pos_neural_hidden_dim 256 --pos_neural_n_layers 4 \
  --neural_warmup_iter 3000 --shader_lr 0.001 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 \
  --save_iterations 30000 60000
```

---

# Table 3 — Cook-Torrance lighting ablation (Materials, 60k)

All share `--cook_torrance_shading`. Only the lighting flags differ.

## 3.1 — 2 directional lights

Same as 1.6 — single row reused across tables.

## 3.2 — 3 directional lights

```bash
python train.py -s data/nerf_synthetic/materials -m output/ct_3lights_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_n_lights 3 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 --save_iterations 30000 60000
```

## 3.3 — RefGS outer-product feature

```bash
python train.py -s data/nerf_synthetic/materials -m output/ct_refgs_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_refgs --ct_refgs_feat_dim 8 \
  --ct_sph_mip_levels 8 --ct_refgs_env_h 8 --ct_refgs_env_w 16 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 --save_iterations 30000 60000
```

## 3.4 — SphMip small (8×8×16)

Same as 1.7.

## 3.5 — Scene environment probe (0 lighting params)

```bash
python train.py -s data/nerf_synthetic/materials -m output/ct_probe_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_probe \
  --ct_probe_resolution 64 --ct_probe_update_interval 500 --ct_probe_n_blur 5 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 --save_iterations 30000 60000
```

## 3.6 — SphMip large (8×64×128, 196k params)

```bash
python train.py -s data/nerf_synthetic/materials -m output/ct_sphmip_large_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_sph_mip \
  --ct_sph_mip_levels 8 --ct_sph_mip_h 64 --ct_sph_mip_w 128 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 --save_iterations 30000 60000
```

## 3.7 — Clustered material, K=12, learnable

```bash
python train.py -s data/nerf_synthetic/materials -m output/ct_clustered_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_clustered \
  --ct_n_object_clusters 12 --ct_n_lights 2 \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 --save_iterations 30000 60000
```

## 3.8 — Clustered material, K=12, frozen metallic=1

```bash
python train.py -s data/nerf_synthetic/materials -m output/ct_clustered_frozen_sh0 \
  --white_background --eval \
  --sh_degree 0 \
  --cook_torrance_shading --ct_clustered \
  --ct_n_object_clusters 12 --ct_n_lights 2 \
  --ct_metallic_init "1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0" \
  --ct_freeze_materials \
  --iterations 60000 --position_lr_max_steps 60000 \
  --test_iterations 30000 60000 --save_iterations 30000 60000
```
