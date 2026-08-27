"""
Batched box interpolation via polar decomposition.

Given two batches of "boxes" L0, L1 of shape [batch, 3, 3] (rows = base
vectors), find X such that L1 = L0 @ X, decompose X = R @ S (rotation *
SPD scale/shear), and produce a continuously parametrized L(t) with
L(0) = L0, L(1) = L1, via geodesic interpolation of R (axis-angle /
Rodrigues) and S (eigenvalue exponentiation).

Requires det(L0) and det(L1) to have the same sign (no mirroring) --
fit_transform raises a ValueError otherwise. There is no continuous
rotation+scale path between a box and its mirror image (O(3) has two
disconnected components), so that case is intentionally unsupported here.

L(t) and its time derivative dL/dt are computed by two separate
functions (compute_Lt, compute_Lt_dot) that share intermediate results
via an "aux" dict, so you never pay for the derivative unless you ask
for it.
"""

import torch

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def hat(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix from vectors. v: [..., 3] -> [..., 3, 3]"""
    z = torch.zeros_like(v[..., 0])
    row0 = torch.stack([z, -v[..., 2], v[..., 1]], dim=-1)
    row1 = torch.stack([v[..., 2], z, -v[..., 0]], dim=-1)
    row2 = torch.stack([-v[..., 1], v[..., 0], z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _linalg_upcast_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp16/bf16 aren't supported by cuSOLVER-backed ops (eigh, solve, det on
    CUDA) in most torch versions, and are numerically unreliable for
    decompositions anyway (~8 bits of mantissa for bf16). Upcast to fp32
    for these ops; leave fp32/fp64 untouched."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def safe_eigh(A: torch.Tensor):
    """torch.linalg.eigh, upcasting fp16/bf16 to fp32 and casting back."""
    orig_dtype = A.dtype
    compute_dtype = _linalg_upcast_dtype(orig_dtype)
    if compute_dtype != orig_dtype:
        A = A.to(compute_dtype)
    eigval, eigvec = torch.linalg.eigh(A)
    return eigval.to(orig_dtype), eigvec.to(orig_dtype)


def safe_solve(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """torch.linalg.solve, upcasting fp16/bf16 to fp32 and casting back."""
    orig_dtype = A.dtype
    compute_dtype = _linalg_upcast_dtype(orig_dtype)
    if compute_dtype != orig_dtype:
        A, B = A.to(compute_dtype), B.to(compute_dtype)
    return torch.linalg.solve(A, B).to(orig_dtype)


def safe_det(A: torch.Tensor) -> torch.Tensor:
    """torch.linalg.det, upcasting fp16/bf16 to fp32 and casting back."""
    orig_dtype = A.dtype
    compute_dtype = _linalg_upcast_dtype(orig_dtype)
    if compute_dtype != orig_dtype:
        A = A.to(compute_dtype)
    return torch.linalg.det(A).to(orig_dtype)


def safe_inverse(A: torch.Tensor) -> torch.Tensor:
    """torch.linalg.inv, upcasting fp16/bf16 to fp32 and casting back."""
    orig_dtype = A.dtype
    compute_dtype = _linalg_upcast_dtype(orig_dtype)
    if compute_dtype != orig_dtype:
        A = A.to(compute_dtype)
    return torch.linalg.inv(A).to(orig_dtype)


def polar_decompose(X: torch.Tensor):
    """
    Batched polar decomposition X = R @ S.
    R: rotation (orthogonal, det=+1 if det(X) > 0 -- checked by caller)
    S: symmetric positive-definite (scale/shear)
    Returns R, eigvals (of S), eigvecs (of S) -- eigvecs/eigvals reused
    later so we don't have to re-decompose S.
    """
    XtX = X.transpose(-1, -2) @ X
    eigval, eigvec = safe_eigh(XtX)  # ascending, eigval >= 0
    eigval = eigval.clamp(min=1e-12)
    s = eigval.sqrt()  # eigenvalues of S itself

    S = eigvec @ torch.diag_embed(s) @ eigvec.transpose(-1, -2)
    S_inv = eigvec @ torch.diag_embed(1.0 / s) @ eigvec.transpose(-1, -2)
    R = X @ S_inv
    return R, s, eigvec


def rotation_to_axis_angle(R: torch.Tensor):
    """R: [..., 3, 3] -> axis [..., 3] (unit), theta [...]"""
    trace = R.diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_theta = ((trace - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos_theta)

    skew = R - R.transpose(-1, -2)
    axis = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], dim=-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)

    small = sin_theta.abs() < 1e-6
    axis = axis / (2 * sin_theta + 1e-12)
    axis = torch.where(small, torch.zeros_like(axis), axis)
    return axis, theta


def axis_angle_to_rotation(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula, batched. axis: [..., 3] unit, theta: [...]"""
    K = hat(axis)
    eye = torch.eye(3, device=axis.device, dtype=axis.dtype).expand(K.shape)
    sin_t = torch.sin(theta)[..., None, None]
    cos_t = torch.cos(theta)[..., None, None]
    return eye + sin_t * K + (1 - cos_t) * (K @ K)


def _as_time_tensor(t, like: torch.Tensor) -> torch.Tensor:
    """Broadcast a python float or 0-d tensor `t` to match `like`'s batch shape."""
    if not torch.is_tensor(t):
        t = torch.tensor(t, dtype=like.dtype, device=like.device)
    return t.expand_as(like) if t.dim() == 0 else t


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------


def fit_transform(L0: torch.Tensor, L1: torch.Tensor):
    """
    Solve L1 = L0 @ X for X, decompose into rotation + scale.
    Returns a dict of the pieces needed to evaluate L(t) later.

    Raises ValueError if det(L0) and det(L1) have opposite signs for
    any batch element (the box would be mirrored -- unsupported).
    """
    det_ratio = safe_det(L1) * safe_det(L0)
    if (det_ratio < 0).any():
        raise ValueError(
            "det(L0) and det(L1) have opposite signs for some batch elements: "
            "the box is mirrored, there's no continuous rotation+scale path."
        )

    X = safe_solve(L0, L1)  # X s.t. L0 @ X = L1
    R, s, V = polar_decompose(X)  # X = R @ S,  S = V diag(s) V^T
    axis, theta = rotation_to_axis_angle(R)

    return {"axis": axis, "theta": theta, "s": s, "V": V}


def compute_generators(params: dict):
    """
    Precompute the (constant, t-independent) generators:
      Omega = theta * hat(axis)          -- angular velocity matrix (skew)
      G_S   = V diag(log s) V^T          -- log-stretch-rate matrix (symmetric)
    R(t) = expm(t Omega),  S(t) = expm(t G_S).
    """
    axis, theta, s, V = params["axis"], params["theta"], params["s"], params["V"]
    Omega = theta[..., None, None] * hat(axis)
    G_S = V @ torch.diag_embed(torch.log(s)) @ V.transpose(-1, -2)
    return Omega, G_S


def compute_Lt_from_params(L0: torch.Tensor, params: dict, t):
    """
    Evaluate L(t) = L0 @ R(t) @ S(t), given params from fit_transform(L0, L1).

    Returns (L_t, aux), where aux bundles the intermediate quantities
    (R_t, S_t, and the constant generators Omega, G_S) needed by
    compute_Lt_dot -- so the derivative never has to recompute them.
    """
    axis, theta, s, V = params["axis"], params["theta"], params["s"], params["V"]
    Omega, G_S = compute_generators(params)

    t = _as_time_tensor(t, theta)

    R_t = axis_angle_to_rotation(axis, t * theta)
    s_t = s ** t.unsqueeze(-1)  # [..., 3]
    S_t = V @ torch.diag_embed(s_t) @ V.transpose(-1, -2)

    X_t = R_t @ S_t
    L_t = L0 @ X_t

    aux = {"R_t": R_t, "S_t": S_t, "Omega": Omega, "G_S": G_S}
    return L_t, aux


def compute_Lt(L0: torch.Tensor, L1: torch.Tensor, t):
    """
    Evaluate L(t) directly from the two boxes L0, L1 (runs fit_transform
    internally, then compute_Lt_from_params).

    Returns (L_t, aux) -- same as compute_Lt_from_params. Note: this
    recomputes fit_transform's polar decomposition on every call, so if
    you need L(t) at many different t (e.g. a trajectory), it's cheaper
    to call fit_transform(L0, L1) once yourself and reuse compute_Lt_from_params.
    """
    params = fit_transform(L0, L1)
    return compute_Lt_from_params(L0, params, t)


def compute_Lt_dot(L0: torch.Tensor, aux: dict) -> torch.Tensor:
    """
    Evaluate dL/dt given L0 and the aux dict produced by compute_Lt.

    dL/dt = L0 @ (Omega @ R(t) + R(t) @ G_S) @ S(t)
    (S(t) factored out of both product-rule terms via distributivity,
    saving one batched matmul vs. computing Rdot(t)@S(t) + R(t)@Sdot(t)
    directly.)
    """
    R_t, S_t, Omega, G_S = aux["R_t"], aux["S_t"], aux["Omega"], aux["G_S"]
    G_t = Omega @ R_t + R_t @ G_S  # combined generator, S(t) factored out
    Xdot_t = G_t @ S_t
    return L0 @ Xdot_t


# ---------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    batch = 4

    def random_box(batch, dtype=torch.float32):
        # random rotation * random positive-definite scale, applied to identity box
        Q, _ = torch.linalg.qr(torch.randn(batch, 3, 3, dtype=dtype))
        scale = torch.diag_embed(0.5 + torch.rand(batch, 3, dtype=dtype))
        return Q @ scale

    L0 = random_box(batch)
    L1 = random_box(batch)

    params = fit_transform(L0, L1)

    L_start, _ = compute_Lt_from_params(L0, params, 0.0)
    L_end, _ = compute_Lt_from_params(L0, params, 1.0)
    print("max |L(0) - L0| :", (L_start - L0).abs().max().item())
    print("max |L(1) - L1| :", (L_end - L1).abs().max().item())

    # full trajectory, e.g. for animation
    ts = torch.linspace(0, 1, steps=10)
    trajectory = torch.stack(
        [compute_Lt_from_params(L0, params, t)[0] for t in ts]
    )  # [10, batch, 3, 3]
    print("trajectory shape:", trajectory.shape)

    # --- verify analytic velocity against finite differences (float64) ---
    L0d, L1d = random_box(batch, dtype=torch.float64), random_box(
        batch, dtype=torch.float64
    )

    t0 = 0.37
    eps = 1e-6
    L_t, aux = compute_Lt(
        L0d, L1d, t0
    )  # one-shot: fit_transform + compute_Lt_from_params
    Ldot_analytic = compute_Lt_dot(L0d, aux)

    L_plus, _ = compute_Lt(L0d, L1d, t0 + eps)
    L_minus, _ = compute_Lt(L0d, L1d, t0 - eps)
    Ldot_fd = (L_plus - L_minus) / (2 * eps)
    print(
        "\nmax |Ldot_analytic - Ldot_finite_diff| (float64):",
        (Ldot_analytic - Ldot_fd).abs().max().item(),
    )

    # --- mirroring is rejected, not silently handled ---
    print("\n--- mirrored box (should raise) ---")
    L0m = random_box(batch, dtype=torch.float64)
    L1m = random_box(batch, dtype=torch.float64)
    L1m[:, 0, :] *= -1  # flip one row to force a sign mismatch in det
    try:
        fit_transform(L0m, L1m)
        print("ERROR: expected a ValueError but none was raised")
    except ValueError as e:
        print("Raised as expected:", e)

    # --- bfloat16 end-to-end check ---
    print("\n--- bfloat16 dtype check ---")
    L0_bf16 = random_box(batch, dtype=torch.float32).to(torch.bfloat16)
    L1_bf16 = random_box(batch, dtype=torch.float32).to(torch.bfloat16)
    params_bf16 = fit_transform(L0_bf16, L1_bf16)
    L_t_bf16, aux_bf16 = compute_Lt_from_params(L0_bf16, params_bf16, 0.5)
    Ldot_bf16 = compute_Lt_dot(L0_bf16, aux_bf16)
    print("dtype of L_t:", L_t_bf16.dtype, " dtype of Ldot_t:", Ldot_bf16.dtype)
    print("ran without error on bfloat16 input")

    L0_inv_bf16 = safe_inverse(L0_bf16)
    print(
        "safe_inverse dtype:",
        L0_inv_bf16.dtype,
        " round-trip max diff:",
        (L0_bf16.to(torch.float32) @ L0_inv_bf16.to(torch.float32) - torch.eye(3))
        .abs()
        .max()
        .item(),
    )
