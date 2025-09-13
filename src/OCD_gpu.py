# src/OCD_gpu_faiss.py

import torch
import faiss
from sklearn.cluster import DBSCAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rangesearch_faiss(X, radius, k=None):
    """
    FAISS GPU radius search workaround using kNN + filtering.
    X: torch tensor (N,d)
    radius: float
    k: int, max neighbors to query (if None, use N)
    """
    X_np = X.detach().cpu().numpy().astype('float32')
    n, d = X_np.shape
    if k is None:
        k = n  # search all points

    # FAISS GPU index
    res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatL2(res, d)
    index.add(X_np)

    D, I = index.search(X_np, k)  # kNN search
    neighbors = []
    radius2 = radius ** 2
    for i in range(n):
        mask = D[i] <= radius2
        neighbors.append(I[i][mask].tolist())
    return neighbors


def compute_cond_vectorized_no_loop(X_m, Y_m, Idx_m, NparticleThreshold=4):
    lens = torch.tensor([len(i) for i in Idx_m], device=device)
    flat_idx = torch.cat([torch.tensor(i, device=device) for i in Idx_m])
    repeats = torch.repeat_interleave(torch.arange(len(Idx_m), device=device), lens)

    X_means = torch.zeros((len(Idx_m), X_m.shape[1]), device=device)
    Y_means = torch.zeros((len(Idx_m), Y_m.shape[1]), device=device)
    start = 0
    for i, l in enumerate(lens):
        idxs = flat_idx[start:start + l]
        X_means[i] = X_m[idxs].mean(dim=0)
        Y_means[i] = Y_m[idxs].mean(dim=0)
        start += l

    y_Cond_x_m = Y_means.clone()
    valid_idx = lens > NparticleThreshold

    if valid_idx.any():
        X_centered = X_m[flat_idx] - X_means[repeats]
        Y_centered = Y_m[flat_idx] - Y_means[repeats]

        d = X_m.shape[1]
        J = torch.zeros((len(Idx_m), d, d), device=device)
        Sigma = torch.zeros((len(Idx_m), d, d), device=device)
        start = 0
        for i, l in enumerate(lens):
            idxs = slice(start, start + l)
            Xi = X_centered[idxs].unsqueeze(2)
            Yi = Y_centered[idxs].unsqueeze(1)
            J[i] = (Xi @ Yi).mean(dim=0)
            Sigma[i] = (Xi @ Xi.transpose(1, 2)).mean(dim=0)
            start += l

        for i, valid in enumerate(valid_idx):
            if valid:
                sigma_inv = torch.pinverse(Sigma[i])
                X_diff = X_m[i] - X_means[i]
                y_Cond_x_m[i] = Y_means[i] + (J[i] @ sigma_inv @ X_diff)

    return y_Cond_x_m


def find_nclusters(X, Y, eps):
    joint = torch.cat([X, Y], dim=1).cpu().numpy()
    clustering = DBSCAN(eps=eps, min_samples=1).fit(joint)
    labels = clustering.labels_
    return len(set(labels)) - (1 if -1 in labels else 0)

def ocd_map(X00, Y00, dt=0.01, Nt=1000, sigma=0.1,
                      epsX=None, epsY=None, tol=1e-14,
                      minNt=100, NparticleThreshold=10, k=None):
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
    err_m2X = []
    err_m2Y = []

    rx = epsX
    ry = epsY

    for it in range(Nt):
        err_m2X.append((m2X - X.mean(dim=0)).abs().cpu().numpy())
        err_m2Y.append((m2Y - Y.mean(dim=0)).abs().cpu().numpy())

        X0 = X.clone()
        Y0 = Y.clone()

        X = X0 + (Y0 - y_Cond_x) * dt
        Y = Y0 + (X0 - x_Cond_y) * dt

        Idx = rangesearch_faiss(X, rx, k=k)
        Idy = rangesearch_faiss(Y, ry, k=k)

        y_Cond_x = compute_cond_vectorized_no_loop(X, Y, Idx, NparticleThreshold)
        x_Cond_y = compute_cond_vectorized_no_loop(Y, X, Idy, NparticleThreshold)

        dists.append(torch.sum((X - Y) ** 2, dim=1).mean().item())
        if it > minNt and abs(dists[-1] - dists[-2]) < tol:
            break

    return X.cpu().numpy(), Y.cpu().numpy(), dists, err_m2X, err_m2Y

def ocd_map_RK4(X00, Y00, dt=0.01, Nt=1000, sigma=0.1,
                          epsX=None, epsY=None, tol=1e-14,
                          minNt=100, NparticleThreshold=10, k=None):
    """
    OCD map using RK4 integration (GPU + FAISS)
    """
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
    err_m2X = []
    err_m2Y = []

    rx = epsX
    ry = epsY

    for it in range(Nt):
        err_m2X.append((m2X - X.mean(dim=0)).abs().cpu().numpy())
        err_m2Y.append((m2Y - Y.mean(dim=0)).abs().cpu().numpy())

        X0 = X.clone()
        Y0 = Y.clone()

        # RK4 steps
        k1_X = (Y0 - y_Cond_x) * dt
        k1_Y = (X0 - x_Cond_y) * dt

        X_mid = X0 + 0.5 * k1_X
        Y_mid = Y0 + 0.5 * k1_Y
        Idx_mid = rangesearch_faiss(X_mid, rx, k=k)
        Idy_mid = rangesearch_faiss(Y_mid, ry, k=k)
        y_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, Y_mid, Idx_mid, NparticleThreshold)
        x_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, X_mid, Idy_mid, NparticleThreshold)
        k2_X = (Y_mid - y_Cond_x_mid) * dt
        k2_Y = (X_mid - x_Cond_y_mid) * dt

        X_mid = X0 + 0.5 * k2_X
        Y_mid = Y0 + 0.5 * k2_Y
        Idx_mid = rangesearch_faiss(X_mid, rx, k=k)
        Idy_mid = rangesearch_faiss(Y_mid, ry, k=k)
        y_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, Y_mid, Idx_mid, NparticleThreshold)
        x_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, X_mid, Idy_mid, NparticleThreshold)
        k3_X = (Y_mid - y_Cond_x_mid) * dt
        k3_Y = (X_mid - x_Cond_y_mid) * dt

        X_end = X0 + k3_X
        Y_end = Y0 + k3_Y
        Idx_end = rangesearch_faiss(X_end, rx, k=k)
        Idy_end = rangesearch_faiss(Y_end, ry, k=k)
        y_Cond_x_end = compute_cond_vectorized_no_loop(X_end, Y_end, Idx_end, NparticleThreshold)
        x_Cond_y_end = compute_cond_vectorized_no_loop(Y_end, X_end, Idy_end, NparticleThreshold)
        k4_X = (Y_end - y_Cond_x_end) * dt
        k4_Y = (X_end - x_Cond_y_end) * dt

        X = X0 + (k1_X + 2 * k2_X + 2 * k3_X + k4_X) / 6
        Y = Y0 + (k1_Y + 2 * k2_Y + 2 * k3_Y + k4_Y) / 6

        Idx = rangesearch_faiss(X, rx, k=k)
        Idy = rangesearch_faiss(Y, ry, k=k)
        y_Cond_x = compute_cond_vectorized_no_loop(X, Y, Idx, NparticleThreshold)
        x_Cond_y = compute_cond_vectorized_no_loop(Y, X, Idy, NparticleThreshold)

        dists.append(torch.sum((X - Y) ** 2, dim=1).mean().item())
        if it > minNt and abs(dists[-1] - dists[-2]) < tol:
            break

    return X.cpu().numpy(), Y.cpu().numpy(), dists, err_m2X, err_m2Y
