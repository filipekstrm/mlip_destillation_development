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
- `models/equiformer_v3/`: Equiformer V3, copied from fairchem. Only `equiformer_v3_flow.py` (our variant `EquiformerV3Flow`: conditions on `s, t`, with `t` defaulting to `s`, and outputs `(dpos, dcell)`) may be edited. Treat the rest as vendored and read-only.
- `distillation/trainer.py`: `Trainer`, shared by all trainers (`distillation/flow_map_trainer.py` adds `FlowMapTrainer`, with `sample_s_t` and the weighted loss). `derotate` rigidly rotates each `x1` (cell and positions) so that `L0^-1 L1` is symmetric, i.e. a pure stretch (a choice of rotation; the energy is rotation invariant). `LSDTrainer` calls it on every batch. Builds the stochastic interpolant `I_t` between `x0` and `x1` and its time derivative. Positions follow the torus geodesic in fractional coordinates, `f_t = f0 + t * wrap(f1 - f0)` mapped back with `L_t`, and their target is `wrap(f1 - f0) @ L_t` (no `f_t @ dL_t/dt` term, matching `FlowMap`). The cell follows the pure-stretch lattice geodesic (`remove_rotation=True`). It also samples `0 <= s <= t <= 1`, and defines the weighted loss `mean(exp(-w) * ||r||^2 + w)`.
- `Trainer.train` owns the epoch loop (sampler epoch, `derotate`, optimizer step, `global_mean` epoch logging, checkpointing). Subclasses implement `training_step(model, x0, x1, device) -> (loss, log_dict)`, with the same log keys on every call and rank. Each subclass handles its own autocast.
- `distillation/flow_matching_trainer.py`: naive flow matching. Unweighted MSE of `v(I_t, t, t)` against `dI_t/dt`, using the same `FlowMap` model, so its checkpoints load into LSD.
- `distillation/lsd_trainer.py`: Lagrangian self-distillation. A fraction `eta` (`--distillation_eta`) of each batch is used for the diagonal loss (s = t, velocity matches `dI_t/dt`). The rest is used for the LSD loss (`dX_st/dt` matches `v(X_st, t, t)`). The target is computed under `no_grad` (the stop-gradient) and in eval mode, so with dropout (`--drop_path_rate`, `--attn_weights_drop`, both default 0) it is the full network.
- `common/periodic.py`: `wrap` (shortest displacement on the unit torus) and `cart_to_frac` / `frac_to_cart`, which run in fp32 with autocast disabled.
- `common/lattice_decomposition.py`: cell interpolation. Writes `L1 = L0 @ X` with `X = R @ S` (polar decomposition) and interpolates `R` and `S` along geodesics. `compute_Lt` / `compute_Lt_dot` share an `aux` dict. Cell rows are lattice vectors, and `det(L0)` and `det(L1)` must have the same sign.
- `common/loggers.py`: loggers registered by name with `@register_logger` (`none`/`noop`, `model_only`, `local_only`, `wandb`), chosen by `--logger`, which has no default: `get_config` asserts that it is set, so a run never silently goes unlogged. `get_logger(config)` returns the chosen logger on the main process and a `NoOpLogger` on other ranks. Except for the no-op logger, they create `checkpoints/<run_name>/`, and all but `model_only` also create `logs/<run_name>/`. `common/constants.py` holds the wandb project and checkpoint file names.
- `distributed_utils.py`: DDP setup and cleanup, `is_main_process`, and `DDPWrapper`, which forwards unknown attributes (e.g. `.v`, `.partial_t`) to the wrapped module.

## Conventions and gotchas
- The forward/loss pass runs under bf16 autocast. Linear algebra (eigh, solve, det, inverse) must go through the `safe_*` helpers in `lattice_decomposition.py`, which upcast to fp32.
- In `FlowMap.v`, the model's raw cell output (a stress-like tensor from the stress head) is symmetrized and mapped to `dcell = cell @ sigma`, i.e. a pure strain with no rotation (row convention). This matches the interpolant's cell velocity exactly because `derotate` makes `L0^-1 L1` symmetric. This is the current choice, not a settled one: it is open for discussion, so don't change it silently.
- Under DDP, print and save only when `is_main_process()` is true. Unwrap `.module` before calling `state_dict()`.
- Trainers get the logger as `self.logger`. Log and checkpoint through it, not with `torch.save` or ad hoc files. `Trainer.save_checkpoint` saves `{"model", "optimizer", "epoch"}` on the main process to `checkpoints/<run_name>/parameters_{epoch}.pt`.
- `--load <checkpoint>` resumes training: it loads model and optimizer state, continues from `epoch + 1`, and starts a new run directory, copying the old logs into it. The path must lie inside `checkpoints/<run_name>/`. `run_name` is a timestamp followed by the CLI flags (except `--logger`).

## Evaluation, status and roadmap
- Current focus is self-distillation: there is no pre-trained teacher. EquiformerV3 is the backbone of the velocity model and is trained from scratch on (initial, relaxed) pairs with the LSD objective.
<!-- TODO: how success is measured (e.g. RMSD to relaxed structure, energy), what is currently in progress, and planned next steps. -->
