# src/OCD_gpu.py
"""
GPU-accelerated version of your OCD solver using:
 - torch (for all linear algebra on GPU)
 - faiss (GPU) for radius / neighbor search
 - RAPIDS cuML DBSCAN for clustering on GPU

Requirements (must be installed in your environment):
 - torch (with CUDA)
 - faiss-gpu
 - cupy
 - cudf
 - cuml

Notes:
 - All tensors are float32 on the current CUDA device.
 - FAISS range_search returns variable-length neighbor lists; we convert
   those to torch.LongTensor lists on the same device.
"""

import torch
import faiss                 # faiss-gpu
import cudf
import cupy as cp
from cuml.cluster import DBSCAN as cuDBSCAN
from torch.utils.dlpack import to_dlpack, from_dlpack
import numpy as np
from typing import List, Tuple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ====== Helpers for conversions ======
def torch_to_cupy(t: torch.Tensor) -> cp.ndarray:
    """Convert CUDA torch tensor -> cupy ndarray via DLPack."""
    if not t.is_cuda:
        return cp.asarray(t.cpu().numpy())
    return cp.from_dlpack(to_dlpack(t))

def torch_to_cudf(t: torch.Tensor) -> cudf.DataFrame:
    """Convert CUDA torch tensor -> cudf.DataFrame (float32)."""
    cp_arr = torch_to_cupy(t.contiguous())
    # cupy -> cudf.DataFrame by columns
    cols = {}
    # cp_arr shape: (n, d)
    for j in range(cp_arr.shape[1]):
        cols[str(j)] = cp_arr[:, j]
    return cudf.DataFrame(cols)

def torch_to_numpy_cpu(t: torch.Tensor) -> np.ndarray:
    """Convert torch tensor -> numpy on CPU (float32)."""
    return t.detach().cpu().numpy()

# ====== FAISS GPU range search (returns list of torch.LongTensor indices per query) ======
class FaissGpuRangeSearcher:
    def __init__(self, dim: int, gpu_id: int = 0):
        self.dim = dim
        self.gpu_id = gpu_id
        # We'll create a GpuResources as needed when building an index
        self.res = faiss.StandardGpuResources()
        self.index = None

    def build_index(self, data: np.ndarray):
        """
        Build a GPU IndexFlatL2 (exact) for data.
        data: numpy array float32, shape (N, dim)
        """
        assert data.dtype == np.float32
        cpu_index = faiss.IndexFlatL2(self.dim)
        self.index = faiss.index_cpu_to_gpu(self.res, self.gpu_id, cpu_index)
        self.index.add(data)

    def range_search(self, queries: np.ndarray, radius: float) -> List[torch.LongTensor]:
        """
        Performs range_search on GPU FAISS.
        queries: numpy array float32 shape (Q, dim)
        returns a list of length Q of torch.LongTensor (indices) on DEVICE
        """
        assert self.index is not None, "Call build_index first"
        # faiss returns lims, distances, indices
        lims, D, I = self.index.range_search(queries, radius)
        # lims is length Q+1
        Q = queries.shape[0]
        out = []
        for q in range(Q):
            start = lims[q]
            end = lims[q + 1]
            if end > start:
                inds = I[start:end].astype(np.int64)
                out.append(torch.as_tensor(inds, device=DEVICE, dtype=torch.long))
            else:
                out.append(torch.empty((0,), device=DEVICE, dtype=torch.long))
        return out

# ====== rangesearch wrapper (torch tensor in/out) ======
def rangesearch_faiss(x: torch.Tensor, e: float, searcher: FaissGpuRangeSearcher = None):
    """
    Return neighbor lists for every row of x within radius e.
    x: torch tensor (N, d) on DEVICE, float32
    e: radius (float)
    searcher: optional FaissGpuRangeSearcher object. If None, build a temporary one.
    returns: list of torch.LongTensor (indices) of length N, each on DEVICE
    """
    assert x.dtype == DTYPE
    N, d = x.shape
    arr = torch_to_numpy_cpu(x.contiguous())
    if searcher is None:
        searcher = FaissGpuRangeSearcher(dim=d)
        searcher.build_index(arr)
    else:
        # If index exists, but different size, rebuild
        if (searcher.index is None) or (searcher.index.ntotal != arr.shape[0]):
            searcher.build_index(arr)
    # queries = arr (we query each point against whole dataset)
    out = searcher.range_search(arr, e)
    return out

# ====== compute_cond (piecewise constant) - GPU using index_add ======
def compute_cond(idx: List[torch.LongTensor], y: torch.Tensor) -> torch.Tensor:
    """
    Piecewise constant conditional expectation.
    idx: list of indices (LongTensor) for each group, each on DEVICE
    y: (Nparticles, dim), torch tensor on DEVICE, float32
    returns: (len(idx), dim) means per group
    """
    device = y.device
    n_groups = len(idx)
    d = y.shape[1]
    # flatten and create group id mapping
    if n_groups == 0:
        return torch.empty((0, d), device=device, dtype=DTYPE)
    lens = torch.tensor([i.numel() for i in idx], device=device, dtype=DTYPE)  # float for division later
    # handle empty groups properly
    # create flat_idx and repeats
    flat_indices = torch.cat([i for i in idx]) if sum([i.numel() for i in idx]) > 0 else torch.empty((0,), dtype=torch.long, device=device)
    if flat_indices.numel() == 0:
        return torch.zeros((n_groups, d), device=device, dtype=DTYPE)
    repeats = torch.cat([torch.full((i.numel(),), j, dtype=torch.long, device=device) for j, i in enumerate(idx)])
    # sum per group
    sums = torch.zeros((n_groups, d), device=device, dtype=DTYPE)
    sums.index_add_(0, repeats, y[flat_indices])
    lens_int = torch.tensor([i.numel() for i in idx], device=device, dtype=DTYPE).unsqueeze(1)
    means = sums / lens_int
    return means

# ====== compute_cond_vectorized_no_loop (piecewise linear approximation) ======
def compute_cond_vectorized_no_loop(X_m: torch.Tensor, Y_m: torch.Tensor, Idx_m: List[torch.LongTensor], NparticleThreshold: int = 4) -> torch.Tensor:
    """
    Vectorized piecewise linear conditional expectation on GPU.
    X_m: (N, d)
    Y_m: (Np, d)  Note: in your original code X_m and Y_m have same coordinate dims
    Idx_m: list of neighbor indices for each X point (list of LongTensor)
    returns y_Cond_x_m: (N, d)
    """
    device = X_m.device
    N = len(Idx_m)
    d = X_m.shape[1]
    if N == 0:
        return torch.empty((0, d), device=device, dtype=DTYPE)

    # Build flat index and group ids
    lens = torch.tensor([i.numel() for i in Idx_m], device=device, dtype=torch.long)
    total = int(lens.sum().item())
    if total == 0:
        # return zeros or means? follow original: Y_m_means computed per-group -> but groups empty -> zeros
        return torch.zeros((N, d), device=device, dtype=DTYPE)

    flat_idx = torch.cat(Idx_m)  # indices into Y_m
    repeats = torch.repeat_interleave(torch.arange(N, device=device, dtype=torch.long), lens)

    # Compute Y and X means per group
    # Use index_add to sum and divide
    Y_sums = torch.zeros((N, d), device=device, dtype=DTYPE)
    Y_sums.index_add_(0, repeats, Y_m[flat_idx])
    X_sums = torch.zeros((N, d), device=device, dtype=DTYPE)
    X_sums.index_add_(0, repeats, X_m[flat_idx])

    lens_f = lens.to(dtype=DTYPE).unsqueeze(1)
    Y_means = Y_sums / lens_f
    X_means = X_sums / lens_f

    y_Cond_x_m = Y_means.clone()

    # valid indices: groups with enough particles
    valid_mask = (lens > NparticleThreshold)
    valid_idx = torch.nonzero(valid_mask).flatten()
    if valid_idx.numel() == 0:
        return y_Cond_x_m

    # center X and Y for the groups (only compute for valid groups)
    X_centered_all = X_m[flat_idx] - X_means[repeats]
    Y_centered_all = Y_m[flat_idx] - Y_means[repeats]

    # We'll compute J_idx and sigma_x per valid group
    # For memory reasons, compute per-group (loop over valid groups). Each group uses local matmul
    for g in valid_idx.tolist():
        # indices in flat arrays where repeats == g
        mask = (repeats == g)
        if mask.sum().item() == 0:
            continue
        Xc = X_centered_all[mask]  # (ng, d)
        Yc = Y_centered_all[mask]  # (ng, d)
        ng = Xc.shape[0]
        # compute covariances (d x d)
        J = (Xc.t() @ Yc) / float(ng)          # d x d
        Sigma = (Xc.t() @ Xc) / float(ng)      # d x d
        # pseudo-inverse
        Sigma_inv = torch.linalg.pinv(Sigma)
        # difference X_m for group g: X entries are the "X locations" associated with group (here groups correspond to X points in original algorithm),
        # but in original code X_m[valid_idx] - X_m_means[valid_idx] was used; here group g corresponds to one X location => use X_m[g] - X_means[g]
        # Note: In original code they used X_m[valid_idx] - X_m_means[valid_idx] (shape (#valid, d))
        # So compute:
        Xdiff = (X_m[g] - X_means[g]).unsqueeze(0)  # (1, d)
        # compute linear correction: (J @ Sigma_inv) @ Xdiff.T -> shape (d,1) -> add to Y_means[g]
        A = J @ Sigma_inv  # d x d
        # einsum equivalent:
        corr = (A @ Xdiff.T).squeeze(1)  # (d,)
        y_Cond_x_m[g] = Y_means[g] + corr

    return y_Cond_x_m

# ====== find_nclusters using cuML DBSCAN on GPU ======
def find_nclusters_gpu(x: torch.Tensor, y: torch.Tensor, eps: float) -> int:
    """
    x: (N, dim) torch tensor on DEVICE
    y: (Np, dim) torch tensor on DEVICE
    eps: float
    returns: number of clusters (excluding noise label -1)
    """
    # concatenate along features (axis 1) as in original code
    joint = torch.cat([x, y], dim=1)  # (N, 2*dim) if y same dim, original used np.concatenate([...], axis=1)
    # convert to cudf for cuML DBSCAN
    gdf = torch_to_cudf(joint)
    # use cuML DBSCAN
    # note: cuML DBSCAN api similar to sklearn
    cdb = cuDBSCAN(eps=float(eps), min_samples=1)
    labels = cdb.fit_predict(gdf)  # returns cudf.Series
    # convert to cupy then numpy
    labels_cp = labels.to_cupy()
    labels_np = cp.asnumpy(labels_cp)
    unique = np.unique(labels_np)
    nclusters = len(set(unique.tolist()) - {-1}) if -1 in unique else len(unique)
    return int(nclusters)

# ====== find_opt_eps2 (search for eps) ======
def find_opt_eps2_gpu(X0: torch.Tensor, Y0: torch.Tensor, log_eps_range=[-3, 1], nepss=10, perc=0.95):
    """
    Same as original find_opt_eps2 but using GPU find_nclusters.
    """
    N = X0.shape[0]
    epss = np.logspace(log_eps_range[0], log_eps_range[1], nepss)
    nclusters = np.zeros(nepss, dtype=np.int64)
    eps0 = epss[-1]
    for i in range(nepss):
        eps = float(epss[i])
        nclusters[i] = find_nclusters_gpu(X0, Y0, eps)
        if nclusters[i] < perc * N:
            eps0 = epss[i]
            break
    return eps0

# ====== OCD integrators (Euler & RK4) - all heavy ops on GPU ======
def ocd_map_gpu(X00: torch.Tensor, Y00: torch.Tensor, dt: float = 0.01, Nt: int = 1000, sigma: float = 0.1,
                epsX: float = None, epsY: float = None, tol: float = 1e-14, minNt: int = 100, NparticleThreshold: int = 10,
                use_faiss_searcher: FaissGpuRangeSearcher = None):
    """
    GPU version of ocd_map (Euler) - returns X, Y, dists, err_m2X, err_m2Y
    Input tensors must be float32 on DEVICE.
    """
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma

    X = X00.clone().to(DEVICE).to(DTYPE)
    Y = Y00.clone().to(DEVICE).to(DTYPE)
    m2X = torch.mean(X00, dim=0).to(DEVICE).to(DTYPE)
    m2Y = torch.mean(Y00, dim=0).to(DEVICE).to(DTYPE)

    np_particles, d = Y.shape

    y_Cond_x = torch.zeros_like(X, device=DEVICE, dtype=DTYPE)
    x_Cond_y = torch.zeros_like(Y, device=DEVICE, dtype=DTYPE)

    err_m2X = []
    err_m2Y = []
    dists = []
    dists.append(torch.mean(torch.sum((X - Y) ** 2, dim=1)).item())

    rx = float(epsX)
    ry = float(epsY)

    # create FAISS searchers for X and Y
    fx_searcher = FaissGpuRangeSearcher(dim=d)
    fy_searcher = FaissGpuRangeSearcher(dim=d)

    for it in range(Nt):
        if it % 1000 == 0 and it > 0:
            print("it:", it, "dist:", dists[it])

        err_m2X.append(torch.abs(m2X - torch.mean(X, dim=0)).cpu().numpy())
        err_m2Y.append(torch.abs(m2Y - torch.mean(Y, dim=0)).cpu().numpy())

        X0 = X.clone()
        Y0 = Y.clone()

        # Euler update
        X = X0 + (Y0 - y_Cond_x) * dt
        Y = Y0 + (X0 - x_Cond_y) * dt

        # rangesearch using FAISS (need numpy float32 on CPU for building index; FAISS will transfer)
        Idx = rangesearch_faiss(X, rx, searcher=fx_searcher)
        Idy = rangesearch_faiss(Y, ry, searcher=fy_searcher)

        # compute conditionals (all on GPU torch)
        y_Cond_x = compute_cond_vectorized_no_loop(X, Y, Idx, NparticleThreshold)
        x_Cond_y = compute_cond_vectorized_no_loop(Y, X, Idy, NparticleThreshold)

        dists.append(torch.mean(torch.sum((X - Y) ** 2, dim=1)).item())
        if it > minNt:
            if abs(dists[it + 1] - dists[it]) < tol:
                break

    return X, Y, dists, err_m2X, err_m2Y


def ocd_map_RK4_gpu(X00: torch.Tensor, Y00: torch.Tensor, dt: float = 0.01, Nt: int = 1000, sigma: float = 0.1,
                    epsX: float = None, epsY: float = None, tol: float = 1e-14, minNt: int = 100, NparticleThreshold: int = 10):
    """
    GPU version of RK4 integrator. All heavy ops on GPU.
    """
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma

    X = X00.clone().to(DEVICE).to(DTYPE)
    Y = Y00.clone().to(DEVICE).to(DTYPE)
    m2X = torch.mean(X00, dim=0).to(DEVICE).to(DTYPE)
    m2Y = torch.mean(Y00, dim=0).to(DEVICE).to(DTYPE)

    np_particles, d = Y.shape

    y_Cond_x = torch.zeros_like(X, device=DEVICE, dtype=DTYPE)
    x_Cond_y = torch.zeros_like(Y, device=DEVICE, dtype=DTYPE)

    err_m2X = []
    err_m2Y = []
    dists = []
    dists.append(torch.mean(torch.sum((X - Y) ** 2, dim=1)).item())

    rx = float(epsX)
    ry = float(epsY)

    # FAISS searchers
    fx_searcher = FaissGpuRangeSearcher(dim=d)
    fy_searcher = FaissGpuRangeSearcher(dim=d)

    for it in range(Nt):
        if it % 1000 == 0 and it > 0:
            print("it:", it, "dist:", dists[it])

        err_m2X.append(torch.abs(m2X - torch.mean(X, dim=0)).cpu().numpy())
        err_m2Y.append(torch.abs(m2Y - torch.mean(Y, dim=0)).cpu().numpy())

        X0 = X.clone()
        Y0 = Y.clone()

        k1_X = (Y0 - y_Cond_x) * dt
        k1_Y = (X0 - x_Cond_y) * dt
        X_mid = X0 + 0.5 * k1_X
        Y_mid = Y0 + 0.5 * k1_Y

        Idx_mid = rangesearch_faiss(X_mid, rx, searcher=fx_searcher)
        Idy_mid = rangesearch_faiss(Y_mid, ry, searcher=fy_searcher)
        y_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, Y_mid, Idx_mid, NparticleThreshold)
        x_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, X_mid, Idy_mid, NparticleThreshold)
        k2_X = (Y_mid - y_Cond_x_mid) * dt
        k2_Y = (X_mid - x_Cond_y_mid) * dt

        X_mid = X0 + 0.5 * k2_X
        Y_mid = Y0 + 0.5 * k2_Y
        Idx_mid = rangesearch_faiss(X_mid, rx, searcher=fx_searcher)
        Idy_mid = rangesearch_faiss(Y_mid, ry, searcher=fy_searcher)
        y_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, Y_mid, Idx_mid, NparticleThreshold)
        x_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, X_mid, Idy_mid, NparticleThreshold)
        k3_X = (Y_mid - y_Cond_x_mid) * dt
        k3_Y = (X_mid - x_Cond_y_mid) * dt

        X_end = X0 + k3_X
        Y_end = Y0 + k3_Y
        Idx_end = rangesearch_faiss(X_end, rx, searcher=fx_searcher)
        Idy_end = rangesearch_faiss(Y_end, ry, searcher=fy_searcher)
        y_Cond_x_end = compute_cond_vectorized_no_loop(X_end, Y_end, Idx_end, NparticleThreshold)
        x_Cond_y_end = compute_cond_vectorized_no_loop(Y_end, X_end, Idy_end, NparticleThreshold)
        k4_X = (Y_end - y_Cond_x_end) * dt
        k4_Y = (X_end - x_Cond_y_end) * dt

        X = X0 + (k1_X + 2 * k2_X + 2 * k3_X + k4_X) / 6.0
        Y = Y0 + (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y) / 6.0

        Idx = rangesearch_faiss(X, rx, searcher=fx_searcher)
        Idy = rangesearch_faiss(Y, ry, searcher=fy_searcher)
        y_Cond_x = compute_cond_vectorized_no_loop(X, Y, Idx, NparticleThreshold)
        x_Cond_y = compute_cond_vectorized_no_loop(Y, X, Idy, NparticleThreshold)

        dists.append(torch.mean(torch.sum((X - Y) ** 2, dim=1)).item())
        if it > minNt:
            if abs(dists[it + 1] - dists[it]) < tol:
                break

    return X, Y, dists, err_m2X, err_m2Y
