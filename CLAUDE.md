# Distillation of machine learned interatomic potentials into flow maps
This project aims at using the concept of flow maps (https://arxiv.org/pdf/2505.18825) to learn a function that can take an "initial" or "unrelaxed" crystal structures and then output the "(energy-)relaxed" structure.

# Code
The project is a deep learning project and hence uses PyTorch. The mamba environment "mlip" provides necessary code.

## Running
- Setup: `mamba activate mlip`, then `pip install -e .` (see `INSTALLATION.md`).
- Single GPU / CPU: `python main.py [--config cfg.json] [--lr 2e-4 ...]`
- Multi-GPU: `torchrun --nproc_per_node=N main.py ...` — `distributed_utils.setup_ddp` detects this via `WORLD_SIZE`.
- Config: CLI args are defined in `mlip_distillation/common/args_and_config.py`. A JSON config (optionally with `"include": "other.json"`) overrides defaults, and CLI flags override the config. Unknown keys in a config file raise an assert, so new options must be added to the parser.
- Formatting: pre-commit with black + isort (`--profile black`).
- There are no tests yet.

## Data
- Training reads the ASE DB given by `--data_path` (default `data/omat_init_and_final_relaxed`; `data/` is gitignored). It must have the form `data/[dataset name]` (relative, exactly one level below `data/`); `get_config` asserts this, and the dataset name is used in the run name.
- That DB is built from OMat with `scripts/filter_omat.py`, which keeps one `<base>_0` (initial) and the highest `<base>_N` (relaxed) entry per base sid.
- `AsePairDBDataset` (`dataset/pair_dataset.py`) pairs entries by sid: exactly one `_0` and one other `_N` per base sid, returned as `(initial, relaxed)` fairchem `AtomicData`. `pair_atomicdata_list_to_batch` collates them into two batches `(x0, x1)`.

## Code map
- `main.py`: entry point. Builds the dataset, model, DDP wrapper, optimizer and `LSDTrainer`.
- `models/flow_map.py`: `FlowMap` wraps a velocity model as `X_st = X_s + (t - s) * v(X_s, s, t)`. Positions are stepped in fractional coordinates (`step`): `f_st = f_s + (t - s) * dpos @ L_s^-1`, wrapped into [0, 1), then `pos_st = f_st @ L_st`. So `dpos` is the velocity relative to the cell (in Å), and the motion from the deforming cell comes from mapping back with `L_st`. `partial_t` computes the derivative via a JVP and returns the position part as `df_st/dt @ L_st`, in the same units as `dpos`. `compute_weight` is the optional learned loss weighting (`WeightsMLP`, enabled by `--learnable_weights_dim`).
- `models/equiformer_v3/`: Equiformer V3, copied from fairchem. Only `equiformer_v3_flowmap.py` (our variant: conditions on `s, t` and outputs `(dpos, dcell)`) may be edited. Treat the rest as vendored and read-only.
- `distillation/flow_map_trainer.py`: base trainer. `derotate` rigidly rotates each `x1` (cell and positions) so that `L0^-1 L1` is symmetric, i.e. a pure stretch (a choice of rotation; the energy is rotation invariant). `LSDTrainer` calls it on every batch. Builds the stochastic interpolant `I_t` between `x0` and `x1` and its time derivative. Positions follow the torus geodesic in fractional coordinates, `f_t = f0 + t * wrap(f1 - f0)` mapped back with `L_t`, and their target is `wrap(f1 - f0) @ L_t` (no `f_t @ dL_t/dt` term, matching `FlowMap`). The cell follows the pure-stretch lattice geodesic (`remove_rotation=True`). It also samples `0 <= s <= t <= 1`, and defines the weighted loss `mean(exp(-w) * ||r||^2 + w)`.
- `distillation/lsd_trainer.py`: Lagrangian self-distillation. A fraction `eta` (`--distillation_eta`) of each batch is used for the diagonal loss (s = t, velocity matches `dI_t/dt`). The rest is used for the LSD loss (`dX_st/dt` matches `v(X_st, t, t)` with the target detached).
- `common/periodic.py`: `wrap` (shortest displacement on the unit torus) and `cart_to_frac` / `frac_to_cart`, which run in fp32 with autocast disabled.
- `common/lattice_decomposition.py`: cell interpolation. Writes `L1 = L0 @ X` with `X = R @ S` (polar decomposition) and interpolates `R` and `S` along geodesics. `compute_Lt` / `compute_Lt_dot` share an `aux` dict. Cell rows are lattice vectors, and `det(L0)` and `det(L1)` must have the same sign.
- `distributed_utils.py`: DDP setup and cleanup, `is_main_process`, and `DDPWrapper`, which forwards unknown attributes (e.g. `.v`, `.partial_t`) to the wrapped module.

## Conventions and gotchas
- The forward/loss pass runs under bf16 autocast. Linear algebra (eigh, solve, det, inverse) must go through the `safe_*` helpers in `lattice_decomposition.py`, which upcast to fp32.
- In `FlowMap.v`, the model's raw cell output (a stress-like tensor from the stress head) is symmetrized and mapped to `dcell = cell @ sigma`, i.e. a pure strain with no rotation (row convention). This matches the interpolant's cell velocity exactly because `derotate` makes `L0^-1 L1` symmetric. This is the current choice, not a settled one: it is open for discussion, so don't change it silently.
- Under DDP, print and save only when `is_main_process()` is true. Unwrap `.module` before calling `state_dict()`.
- Checkpoints are written to `checkpoints/checkpoint_epoch={epoch}.pt`. The `checkpoints/` directory must already exist.

## Evaluation, status and roadmap
- Current focus is self-distillation: there is no pre-trained teacher. EquiformerV3 is the backbone of the velocity model and is trained from scratch on (initial, relaxed) pairs with the LSD objective.
<!-- TODO: how success is measured (e.g. RMSD to relaxed structure, energy), what is currently in progress, and planned next steps. -->
