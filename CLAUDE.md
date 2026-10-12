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
- `models/flow_map.py`: `FlowMap` wraps a velocity model as `X_st = X_s + (t - s) * v(X_s, s, t)`, acting on both `pos` and `cell`. `partial_t` computes `dX_st/dt` via a JVP. `compute_weight` is the optional learned loss weighting (`WeightsMLP`, enabled by `--learnable_weights_dim`).
- `models/equiformer_v3/`: Equiformer V3, copied from fairchem. Only `equiformer_v3_flowmap.py` (our variant: conditions on `s, t` and outputs `(dpos, dcell)`) may be edited. Treat the rest as vendored and read-only.
- `distillation/flow_map_trainer.py`: base trainer. Builds the stochastic interpolant `I_t` between `x0` and `x1` (positions interpolated linearly, cell via the lattice geodesic) and its time derivative, samples `0 <= s <= t <= 1`, and defines the weighted loss `mean(exp(-w) * ||r||^2 + w)`.
- `distillation/lsd_trainer.py`: Lagrangian self-distillation. A fraction `eta` (`--distillation_eta`) of each batch is used for the diagonal loss (s = t, velocity matches `dI_t/dt`). The rest is used for the LSD loss (`dX_st/dt` matches `v(X_st, t, t)` with the target detached).
- `common/lattice_decomposition.py`: cell interpolation. Writes `L1 = L0 @ X` with `X = R @ S` (polar decomposition) and interpolates `R` and `S` along geodesics. `compute_Lt` / `compute_Lt_dot` share an `aux` dict. Cell rows are lattice vectors, and `det(L0)` and `det(L1)` must have the same sign.
- `distributed_utils.py`: DDP setup and cleanup, `is_main_process`, and `DDPWrapper`, which forwards unknown attributes (e.g. `.v`, `.partial_t`) to the wrapped module.

## Conventions and gotchas
- The forward/loss pass runs under bf16 autocast. Linear algebra (eigh, solve, det, inverse) must go through the `safe_*` helpers in `lattice_decomposition.py`, which upcast to fp32.
- In `FlowMap.v`, the model's raw cell output (a stress-like tensor from the stress head) is symmetrized and mapped to `dcell = cell @ sigma`, i.e. a pure strain with no rotation (row convention). The interpolant's cell velocity is only exactly representable if `L0^-1 L1` is symmetric, i.e. x1 has no rigid rotation relative to x0. This is the current choice, not a settled one: it is open for discussion, so don't change it silently.
- Under DDP, print and save only when `is_main_process()` is true. Unwrap `.module` before calling `state_dict()`.
- Checkpoints are written to `checkpoints/checkpoint_epoch={epoch}.pt`. The `checkpoints/` directory must already exist.

## Evaluation, status and roadmap
- Current focus is self-distillation: there is no pre-trained teacher. EquiformerV3 is the backbone of the velocity model and is trained from scratch on (initial, relaxed) pairs with the LSD objective.
<!-- TODO: how success is measured (e.g. RMSD to relaxed structure, energy), what is currently in progress, and planned next steps. -->
