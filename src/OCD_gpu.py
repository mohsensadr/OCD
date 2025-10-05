import torch
import cupy as cp
from cuml.cluster import DBSCAN
import faiss
import faiss.contrib.torch_utils  # enables torch<->faiss GPU tensors

# =====================================================
# === FAISS GPU index wrapper =========================
# =====================================================

class FaissGpuIndex:
    """Reusable FAISS GPU index for repeated neighbor queries."""
    def __init__(self, dim, k=2048):
        self.dim = dim
        self.k = min(k, 2048)
        self.res = faiss.StandardGpuResources()
        self.index = faiss.GpuIndexFlatL2(self.res, dim)
        self._n = 0

    def reset_and_add(self, X):
        if not X.is_contiguous():
            X = X.contiguous()
        try:
            self.index.reset()
        except Exception:
            self.index = faiss.GpuIndexFlatL2(self.res, self.dim)
        self.index.add(X)
        self._n = X.shape[0]

    def search(self, X, k=None):
        if k is None:
            k = self.k
        k = min(k, self._n)
        D, I = self.index.search(X, k)
        return D, I

# =====================================================
# === Vectorized conditional expectation ============
# =====================================================

def compute_cond_gpu(X, Y, I, D, radius, NparticleThreshold=4, eps_reg=1e-6, device="cuda"):
    """
    Compute conditional expectation vectorized on GPU.
    """
    N, d = X.shape
    r2 = radius ** 2

    mask = D <= r2
    lens = mask.sum(dim=1)
    valid = lens > 0
    max_len = lens.max().item()

    I_masked = torch.where(mask, I, torch.full_like(I, -1))
    pad_idx = I_masked.clone()
    pad_idx[pad_idx < 0] = 0
    Xn = X[pad_idx]
    Yn = Y[pad_idx]
    mask3 = mask.unsqueeze(2)
    Xn = Xn * mask3
    Yn = Yn * mask3

    lens_f = lens.clamp(min=1).float().unsqueeze(1)
    X_mean = Xn.sum(dim=1) / lens_f
    Y_mean = Yn.sum(dim=1) / lens_f
    y_Cond = Y_mean.clone()

    valid_big = lens > NparticleThreshold
    if valid_big.any():
        vb = valid_big.nonzero(as_tuple=False).squeeze(1)
        Xc = Xn[vb] - X_mean[vb].unsqueeze(1)
        Yc = Yn[vb] - Y_mean[vb].unsqueeze(1)
        maskb = mask[vb].unsqueeze(2)
        Xc_mask = Xc * maskb
        Yc_mask = Yc * maskb
        J = torch.matmul(Xc_mask.transpose(1, 2), Yc_mask) / lens_f[vb].unsqueeze(2)
        Sigma = torch.matmul(Xc_mask.transpose(1, 2), Xc_mask) / lens_f[vb].unsqueeze(2)
        Iden = torch.eye(d, device=device).unsqueeze(0).expand(J.shape)
        Sigma_reg = Sigma + eps_reg * Iden
        Sigma_inv = torch.linalg.solve(Sigma_reg, Iden)
        X_diff = (X[vb] - X_mean[vb]).unsqueeze(2)
        correction = torch.bmm(J @ Sigma_inv, X_diff).squeeze(2)
        y_Cond[vb] = Y_mean[vb] + correction

    return y_Cond

# =====================================================
# === GPU DBSCAN clustering ==========================
# =====================================================

def find_nclusters_cuml(X, Y, eps, min_samples=1):
    """
    Count clusters using cuML DBSCAN on GPU.
    X, Y: torch.Tensor on GPU
    eps: radius
    """
    joint = torch.cat([X, Y], dim=1).contiguous()
    joint_cp = cp.asarray(joint)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(joint_cp)
    n_clusters = int(labels.max().item() + 1)
    return n_clusters

# =====================================================
# === GPU optimal epsilon finder =====================
# =====================================================

def find_opt_eps2_gpu(X0, Y0, log_eps_range=[-3,1], nepss=10, perc=0.95, min_samples=1):
    device = X0.device if isinstance(X0, torch.Tensor) else "cuda"
    X0 = X0.to(device) if torch.is_tensor(X0) else torch.tensor(X0, device=device, dtype=torch.float32)
    Y0 = Y0.to(device) if torch.is_tensor(Y0) else torch.tensor(Y0, device=device, dtype=torch.float32)

    N = X0.shape[0]
    epss = torch.logspace(log_eps_range[0], log_eps_range[1], nepss, device=device)
    eps0 = epss[-1].item()

    for eps in epss:
        eps_val = eps.item()
        n_clusters = find_nclusters_cuml(X0, Y0, eps_val, min_samples)
        if n_clusters < perc * N:
            eps0 = eps_val
            break
    return eps0

# =====================================================
# === OCD map with Euler method ======================
# =====================================================

def ocd_map(X00, Y00, dt=0.01, Nt=1000, sigma=0.1,
            epsX=None, epsY=None, tol=1e-14,
            minNt=100, NparticleThreshold=10, k=2048, device="cuda"):

    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma

    X = torch.tensor(X00, dtype=torch.float32, device=device)
    Y = torch.tensor(Y00, dtype=torch.float32, device=device)
    m2X, m2Y = X.mean(dim=0), Y.mean(dim=0)
    y_Cond_x = torch.zeros_like(X)
    x_Cond_y = torch.zeros_like(Y)

    dists = [torch.sum((X - Y) ** 2, dim=1).mean().item()]
    err_m2X, err_m2Y = [], []
    rx, ry = epsX, epsY

    indexX = FaissGpuIndex(X.shape[1], k)
    indexY = FaissGpuIndex(Y.shape[1], k)

    for it in range(Nt):
        err_m2X.append((m2X - X.mean(dim=0)).abs().cpu().numpy())
        err_m2Y.append((m2Y - Y.mean(dim=0)).abs().cpu().numpy())

        X0, Y0 = X.clone(), Y.clone()
        X = X0 + (Y0 - y_Cond_x) * dt
        Y = Y0 + (X0 - x_Cond_y) * dt

        indexX.reset_and_add(X)
        indexY.reset_and_add(Y)
        DX, IX = indexX.search(X)
        DY, IY = indexY.search(Y)
        y_Cond_x = compute_cond_gpu(X, Y, IX, DX, rx, NparticleThreshold)
        x_Cond_y = compute_cond_gpu(Y, X, IY, DY, ry, NparticleThreshold)

        dist = torch.sum((X - Y) ** 2, dim=1).mean().item()
        dists.append(dist)

        if it > minNt and abs(dists[-1] - dists[-2]) < tol:
            break

    return X.cpu().numpy(), Y.cpu().numpy(), dists, err_m2X, err_m2Y

# =====================================================
# === OCD Map with RK4 integrator =====================
# =====================================================

def ocd_map_gpu_RK4(X00, Y00, dt=0.01, Nt=1000, sigma=0.1,
                         epsX=None, epsY=None, tol=1e-14,
                         minNt=100, NparticleThreshold=10, k=2048, device="cuda"):
    if epsX is None: epsX = sigma
    if epsY is None: epsY = sigma

    X = torch.tensor(X00, dtype=torch.float32, device=device)
    Y = torch.tensor(Y00, dtype=torch.float32, device=device)

    y_Cond_x = torch.zeros_like(X)
    x_Cond_y = torch.zeros_like(Y)
    rx, ry = epsX, epsY

    indexX = FaissGpuIndex(X.shape[1], k)
    indexY = FaissGpuIndex(Y.shape[1], k)

    dists = []
    for it in range(Nt):
        X0, Y0 = X, Y

        # ---- Stage 1 ----
        k1_X = (Y0 - y_Cond_x) * dt
        k1_Y = (X0 - x_Cond_y) * dt

        # ---- Stage 2 ----
        X_mid = X0 + 0.5 * k1_X
        Y_mid = Y0 + 0.5 * k1_Y
        indexX.reset_and_add(X_mid)
        indexY.reset_and_add(Y_mid)
        DX, IX = indexX.search(X_mid)
        DY, IY = indexY.search(Y_mid)
        y_Cond_mid = compute_cond_gpu(X_mid, Y_mid, IX, DX, rx, NparticleThreshold)
        x_Cond_mid = compute_cond_gpu(Y_mid, X_mid, IY, DY, ry, NparticleThreshold)
        k2_X = (Y_mid - y_Cond_mid) * dt
        k2_Y = (X_mid - x_Cond_mid) * dt

        # ---- Stage 3 ---- (reuse same mid)
        X_mid = X0 + 0.5 * k2_X
        Y_mid = Y0 + 0.5 * k2_Y
        k3_X, k3_Y = k2_X, k2_Y  # reuse mid-stage estimate

        # ---- Stage 4 ----
        X_end = X0 + k3_X
        Y_end = Y0 + k3_Y
        indexX.reset_and_add(X_end)
        indexY.reset_and_add(Y_end)
        DX, IX = indexX.search(X_end)
        DY, IY = indexY.search(Y_end)
        y_Cond_end = compute_cond_gpu(X_end, Y_end, IX, DX, rx, NparticleThreshold)
        x_Cond_end = compute_cond_gpu(Y_end, X_end, IY, DY, ry, NparticleThreshold)
        k4_X = (Y_end - y_Cond_end) * dt
        k4_Y = (X_end - x_Cond_end) * dt

        # ---- Update ----
        X = X0 + (k1_X + 2*k2_X + 2*k3_X + k4_X) / 6
        Y = Y0 + (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y) / 6

        # ---- One final cond ----
        indexX.reset_and_add(X)
        indexY.reset_and_add(Y)
        DX, IX = indexX.search(X)
        DY, IY = indexY.search(Y)
        y_Cond_x = compute_cond_gpu(X, Y, IX, DX, rx, NparticleThreshold)
        x_Cond_y = compute_cond_gpu(Y, X, IY, DY, ry, NparticleThreshold)

        dists.append(torch.sum((X - Y) ** 2, dim=1).mean().item())

        if it > minNt and abs(dists[-1] - dists[-2]) < tol:
            break

    return X.cpu().numpy(), Y.cpu().numpy(), dists


'''
def ocd_map_gpu_RK4(X00, Y00, dt=0.01, Nt=1000, sigma=0.1,
                    epsX=None, epsY=None, tol=1e-14,
                    minNt=100, NparticleThreshold=10, k=2048, device="cuda"):
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma

    X = torch.tensor(X00, dtype=torch.float32, device=device)
    Y = torch.tensor(Y00, dtype=torch.float32, device=device)

    m2X = X.mean(dim=0)
    m2Y = Y.mean(dim=0)
    y_Cond_x = torch.zeros_like(X)
    x_Cond_y = torch.zeros_like(Y)

    dists = [torch.sum((X - Y) ** 2, dim=1).mean().item()]
    err_m2X, err_m2Y = [], []

    rx, ry = epsX, epsY

    indexX = FaissGpuIndex(X.shape[1], k)
    indexY = FaissGpuIndex(Y.shape[1], k)

    for it in range(Nt):
        err_m2X.append((m2X - X.mean(dim=0)).abs().cpu().numpy())
        err_m2Y.append((m2Y - Y.mean(dim=0)).abs().cpu().numpy())

        X0, Y0 = X.clone(), Y.clone()

        # --- RK4 step ---
        k1_X = (Y0 - y_Cond_x) * dt
        k1_Y = (X0 - x_Cond_y) * dt

        X_mid = X0 + 0.5 * k1_X
        Y_mid = Y0 + 0.5 * k1_Y
        indexX.reset_and_add(X_mid)
        indexY.reset_and_add(Y_mid)
        DX, IX = indexX.search(X_mid)
        DY, IY = indexY.search(Y_mid)
        y_Cond_mid = compute_cond_gpu(X_mid, Y_mid, IX, DX, rx, NparticleThreshold)
        x_Cond_mid = compute_cond_gpu(Y_mid, X_mid, IY, DY, ry, NparticleThreshold)
        k2_X = (Y_mid - y_Cond_mid) * dt
        k2_Y = (X_mid - x_Cond_mid) * dt

        X_mid = X0 + 0.5 * k2_X
        Y_mid = Y0 + 0.5 * k2_Y
        indexX.reset_and_add(X_mid)
        indexY.reset_and_add(Y_mid)
        DX, IX = indexX.search(X_mid)
        DY, IY = indexY.search(Y_mid)
        y_Cond_mid = compute_cond_gpu(X_mid, Y_mid, IX, DX, rx, NparticleThreshold)
        x_Cond_mid = compute_cond_gpu(Y_mid, X_mid, IY, DY, ry, NparticleThreshold)
        k3_X = (Y_mid - y_Cond_mid) * dt
        k3_Y = (X_mid - x_Cond_mid) * dt

        X_end = X0 + k3_X
        Y_end = Y0 + k3_Y
        indexX.reset_and_add(X_end)
        indexY.reset_and_add(Y_end)
        DX, IX = indexX.search(X_end)
        DY, IY = indexY.search(Y_end)
        y_Cond_end = compute_cond_gpu(X_end, Y_end, IX, DX, rx, NparticleThreshold)
        x_Cond_end = compute_cond_gpu(Y_end, X_end, IY, DY, ry, NparticleThreshold)
        k4_X = (Y_end - y_Cond_end) * dt
        k4_Y = (X_end - x_Cond_end) * dt

        X = X0 + (k1_X + 2*k2_X + 2*k3_X + k4_X) / 6
        Y = Y0 + (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y) / 6

        indexX.reset_and_add(X)
        indexY.reset_and_add(Y)
        DX, IX = indexX.search(X)
        DY, IY = indexY.search(Y)
        y_Cond_x = compute_cond_gpu(X, Y, IX, DX, rx, NparticleThreshold)
        x_Cond_y = compute_cond_gpu(Y, X, IY, DY, ry, NparticleThreshold)

        dist = torch.sum((X - Y) ** 2, dim=1).mean().item()
        dists.append(dist)

        if it > minNt and abs(dists[-1] - dists[-2]) < tol:
            break

    return X.cpu().numpy(), Y.cpu().numpy(), dists, err_m2X, err_m2Y
'''

