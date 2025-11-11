import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance as wd
import time
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.neighbors import BallTree
from sklearn.cluster import DBSCAN

def rangesearch(x, e):
    ## This function is used to find particle ids within a radius e
    tree = BallTree(x)
    ind = tree.query_radius(x, r=e)
    return ind

def compute_cond(idx, y):
    ## This function computes piecewise constant approximation to the conditional expectation  
    # Flatten idx and create an offset array
    flat_idx = np.concatenate(idx)
    lens = [len(i) for i in idx]

    # Use np.add.reduceat to sum the elements
    sums = np.add.reduceat(y[flat_idx], np.r_[0, np.cumsum(lens[:-1])])

    avg = sums/np.array(lens)[:,None]
    
    return avg

def compute_cond_vectorized_no_loop(X_m, Y_m, Idx_m, NparticleThreshold = 4):
    ## This function computes piecewise linear approximation to the conditional expectation

    # Flatten idx_m and create an offset array
    flat_idx = np.concatenate(Idx_m)
    lens = np.array([len(i) for i in Idx_m])

    # Repeating indices for broadcasting
    repeats = np.repeat(np.arange(len(Idx_m)), lens)
    
    # Calculate the means for each group
    X_m_means = np.add.reduceat(X_m[flat_idx], np.r_[0, np.cumsum(lens[:-1])]) / lens[:, None]
    Y_m_means = np.add.reduceat(Y_m[flat_idx], np.r_[0, np.cumsum(lens[:-1])]) / lens[:, None]
    
    # Initialize the result array with Y_m_means
    y_Cond_x_m = Y_m_means.copy()

    # Handle cases with more than one neighbor
    valid_idx = lens > NparticleThreshold

    if np.any(valid_idx):
        # Centering X_m and Y_m for valid indices
        X_m_centered = X_m[flat_idx] - X_m_means[repeats]
        Y_m_centered = Y_m[flat_idx] - Y_m_means[repeats]
        
        # Compute J_idx and sigma_x for valid indices
        J_idx = np.add.reduceat(X_m_centered[:, :, None] * Y_m_centered[:, None, :], np.r_[0, np.cumsum(lens[:-1])]) / lens[:, None, None]
        sigma_x = np.add.reduceat(X_m_centered[:, :, None] * X_m_centered[:, None, :], np.r_[0, np.cumsum(lens[:-1])]) / lens[:, None, None]
        
        # Inverse of sigma_x (using pseudo-inverse to handle singular matrices)
        sigma_x_inv = np.linalg.pinv(sigma_x[valid_idx])

        # Compute X_m difference for valid indices
        X_m_diff = X_m[valid_idx] - X_m_means[valid_idx]
        
        # Compute the conditional expectations y_Cond_x for valid indices
        y_Cond_x_m[valid_idx] = Y_m_means[valid_idx] + np.einsum('ijk,ik->ij', J_idx[valid_idx] @ sigma_x_inv, X_m_diff)

    return y_Cond_x_m


def find_nclusters(x, y, eps):
    ## This function finds the number of clusters using DBSCAN
    # inputs: x: (N,dim),
    #         y: (Np,dim),
    #         eps: bandwidth
    # output: number of clusters
    joint = np.concatenate([x,y],axis=1)
    
    clustering = DBSCAN(eps=1*eps, min_samples=1).fit(joint)

    labels = clustering.labels_

    return len(set(labels)) - (1 if -1 in labels else 0)

    
def find_opt_eps2(X0, Y0, log_eps_range=[-3,1], nepss = 10, perc=0.95):
    ## This function finds the maximum bandwidth epsilon where N_cluster/Np>perc
    # inputs: X0: (N,dim),
    #         Y0: (Np,dim),
    #         log_eps_range: the range to consider for eps in logarithmic scale
    #         nepss: number of grid points
    #         prec: the criteria for finding maximum bandwidth where N_cluster/Np>perc
    # outputs: eps0: the estimated optimal bandwidth
    
    N = X0.shape[0]
    epss = np.logspace(log_eps_range[0],log_eps_range[1],nepss)
    nclusters = np.zeros(nepss)
    for i in range(nepss):
        eps = epss[i]
        X = X0.copy()
        Y = Y0.copy()
        nclusters[i] = find_nclusters(X, Y, eps)
        
        if nclusters[i] < perc*N:
            eps0 = epss[i]
            break
    return eps0
    
def ocd_map(X00, Y00, dt=0.01, Nt=1000, sigma=0.1, epsX=None, epsY=None, tol=1e-14, minNt = 100, NparticleThreshold=10):
    ## This function finds the map between X and Y by solving OCD dynamics using Euler method
    # inputs: X: (N,dim),
    #         Y: (Np,dim),
    #         dt: time step size,
    #         Nt: number of steps
    #         sigma: the kernel bandwidth for computing conditional expectation
    #         tol: convergence tolerance
    #         minNt: minimum number of iterations
    #         NparticleThreshold: number of particles as the threshold to switch between 
    #                             piecewse constant and linear approximation of conditional expectation
    # outputs: X (N,dim),
    #          Y (Np,dim)
    #          dists: history of W2 distance
    #          err_m2X, err_m2Y: history of the error in 2nd order moments for X and Y
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma
        
    X = X00.copy()
    Y = Y00.copy()
    m2X = np.mean(X00, axis=0)
    m2Y = np.mean(Y00, axis=0)

    np_particles, d = Y.shape

    y_Cond_x = np.zeros_like(X)   # Initialize Y conditional X
    x_Cond_y = np.zeros_like(Y)   # Initialize X conditional Y

    err_m2X = []
    err_m2Y = []
    dists = []
    dists.append(np.mean(np.sum((X - Y) ** 2, axis=1)))
    
    rx = epsX
    ry = epsY
        
    for it in range(Nt):
        if it % 1000 == 0 and it>0:
            print("it: ", it, " dist: ", dists[it])
        err_m2X.append(abs(m2X-np.mean(X, axis=0)))
        err_m2Y.append(abs(m2Y-np.mean(Y, axis=0)))
        
        X0 = X.copy()
        Y0 = Y.copy()

        # Update X and Y
        X = X0 + (Y0 - y_Cond_x) * dt
        Y = Y0 + (X0 - x_Cond_y) * dt

        # Finding neighbors of X in eps
        Idx = rangesearch(X, rx)
        Idy = rangesearch(Y, ry)
        
        y_Cond_x = compute_cond_vectorized_no_loop(X, Y, Idx, NparticleThreshold)
        x_Cond_y = compute_cond_vectorized_no_loop(Y, X, Idy, NparticleThreshold)
        
        ## keep a history of W2 distance
        dists.append(np.mean(np.sum((X - Y) ** 2, axis=1)))
        if it>minNt:
            if abs(dists[it+1]-dists[it]) < tol:
                break
    return X, Y, dists, err_m2X, err_m2Y

def ocd_map_lp(X00, Y00, p=2, dt=0.01, Nt=1000, sigma=0.1, epsX=None, epsY=None, tol=1e-14, minNt = 100, NparticleThreshold=10):
    ## This function finds the map between X and Y by solving OCD dynamics using Euler method
    # inputs: X: (N,dim),
    #         Y: (Np,dim),
    #         p: the lp order of Wasserstein distance c=|x-y|^p
    #         dt: time step size,
    #         Nt: number of steps
    #         sigma: the kernel bandwidth for computing conditional expectation
    #         tol: convergence tolerance
    #         minNt: minimum number of iterations
    #         NparticleThreshold: number of particles as the threshold to switch between 
    #                             piecewse constant and linear approximation of conditional expectation
    # outputs: X (N,dim),
    #          Y (Np,dim)
    #          dists: history of Wp distance
    #          err_m2X, err_m2Y: history of the error in 2nd order moments for X and Y
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma
        
    X = X00.copy()
    Y = Y00.copy()
    m2X = np.mean(X00, axis=0)
    m2Y = np.mean(Y00, axis=0)

    np_particles, d = Y.shape

    gradcx_Cond_x = np.zeros_like(X)   # Initialize Y conditional X
    gradcy_Cond_y = np.zeros_like(Y)   # Initialize X conditional Y

    err_m2X = []
    err_m2Y = []
    dists = []
    dists.append(np.mean(np.sum((X - Y) ** p, axis=1)))
    
    rx = epsX
    ry = epsY
        
    for it in range(Nt):
        if it % 1000 == 0 and it>0:
            print("it: ", it, " dist: ", dists[it])
        err_m2X.append(abs(m2X-np.mean(X, axis=0)))
        err_m2Y.append(abs(m2Y-np.mean(Y, axis=0)))
        
        X0 = X.copy()
        Y0 = Y.copy()

        # Update X and Y
        X = X0 + (-p*(X0-Y0)**(p-1) + gradcx_Cond_x) * dt
        Y = Y0 + (-p*(Y0-X0)**(p-1) + gradcy_Cond_y) * dt

        # Finding neighbors of X in eps
        Idx = rangesearch(X, rx)
        Idy = rangesearch(Y, ry)
        
        gradcx_Cond_x = compute_cond_vectorized_no_loop(X, p*(X-Y)**(p-1), Idx, NparticleThreshold)
        gradcy_Cond_y = compute_cond_vectorized_no_loop(Y, p*(Y-X)**(p-1), Idy, NparticleThreshold)
        
        ## keep a history of Wp distance
        dists.append(np.mean(np.sum((X - Y) ** p, axis=1)))
        if it>minNt:
            if abs(dists[it+1]-dists[it]) < tol:
                break
    return X, Y, dists, err_m2X, err_m2Y

def ocd_map_RK4(X00, Y00, dt=0.01, Nt=1000, sigma=0.1, epsX=None, epsY=None, tol=1e-14, minNt = 100, NparticleThreshold=10):
    ## This function finds the map between X and Y by solving OCD dynamics using RK4
    # inputs: X: (N,dim),
    #         Y: (Np,dim),
    #         dt: time step size,
    #         Nt: number of steps
    #         sigma: the kernel bandwidth for computing conditional expectation
    #         tol: convergence tolerance
    #         minNt: minimum number of iterations
    #         NparticleThreshold: number of particles as the threshold to switch between 
    #                             piecewse constant and linear approximation of conditional expectation
    # outputs: X (N,dim),
    #          Y (Np,dim)
    #          dists: history of W2 distance
    #          err_m2X, err_m2Y: history of the error in 2nd order moments for X and Y
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma
        
    X = X00.copy()
    Y = Y00.copy()
    m2X = np.mean(X00, axis=0)
    m2Y = np.mean(Y00, axis=0)

    np_particles, d = Y.shape

    y_Cond_x = np.zeros_like(X)   # Initialize Y conditional X
    x_Cond_y = np.zeros_like(Y)   # Initialize X conditional Y

    err_m2X = []
    err_m2Y = []
    dists = []
    dists.append(np.mean(np.sum((X - Y) ** 2, axis=1)))
    
    rx = epsX
    ry = epsY
        
    for it in range(Nt):
        if it % 1000 == 0 and it>0:
            print("it: ", it, " dist: ", dists[it])
        err_m2X.append(abs(m2X-np.mean(X, axis=0)))
        err_m2Y.append(abs(m2Y-np.mean(Y, axis=0)))
        
        X0 = X.copy()
        Y0 = Y.copy()
        
        k1_X = (Y0 - y_Cond_x) * dt
        k1_Y = (X0 - x_Cond_y) * dt
        X_mid = X0 + 0.5 * k1_X
        Y_mid = Y0 + 0.5 * k1_Y
        Idx_mid = rangesearch(X_mid, rx)
        Idy_mid = rangesearch(Y_mid, ry)
        y_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, Y_mid, Idx_mid, NparticleThreshold)
        x_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, X_mid, Idy_mid, NparticleThreshold)
        k2_X = (Y_mid - y_Cond_x_mid) * dt 
        k2_Y = (X_mid - x_Cond_y_mid) * dt
        X_mid = X0 + 0.5 * k2_X
        Y_mid = Y0 + 0.5 * k2_Y
        Idx_mid = rangesearch(X_mid, rx)
        Idy_mid = rangesearch(Y_mid, ry)
        y_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, Y_mid, Idx_mid, NparticleThreshold)
        x_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, X_mid, Idy_mid, NparticleThreshold)
        k3_X = (Y_mid - y_Cond_x_mid) * dt
        k3_Y = (X_mid - x_Cond_y_mid) * dt
        X_end = X0 + k3_X
        Y_end = Y0 + k3_Y
        Idx_end = rangesearch(X_end, rx)
        Idy_end = rangesearch(Y_end, ry)
        y_Cond_x_end = compute_cond_vectorized_no_loop(X_end, Y_end, Idx_end, NparticleThreshold)
        x_Cond_y_end = compute_cond_vectorized_no_loop(Y_end, X_end, Idy_end, NparticleThreshold)
        k4_X = (Y_end - y_Cond_x_end) * dt
        k4_Y = (X_end - x_Cond_y_end) * dt
        X = X0 + (k1_X + 2*k2_X + 2*k3_X + k4_X) / 6
        Y = Y0 + (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y) / 6
        
        Idx = rangesearch(X, rx)
        Idy = rangesearch(Y, ry)
        y_Cond_x = compute_cond_vectorized_no_loop(X, Y, Idx, NparticleThreshold) 
        x_Cond_y = compute_cond_vectorized_no_loop(Y, X, Idy, NparticleThreshold)
        
        ## keep a history of W2 distance
        dists.append(np.mean(np.sum((X - Y) ** 2, axis=1)))
        if it>minNt:
            if abs(dists[it+1]-dists[it]) < tol:
                break
    return X, Y, dists, err_m2X, err_m2Y

def ocd_map_RK4_lp(X00, Y00, p=2, dt=0.01, Nt=1000, sigma=0.1, epsX=None, epsY=None, tol=1e-14, minNt = 100, NparticleThreshold=10):
    ## This function finds the map between X and Y by solving OCD dynamics using RK4
    # inputs: X: (N,dim),
    #         Y: (Np,dim),
    #         p: the lp order of Wasserstein distance c=|x-y|^p
    #         dt: time step size,
    #         Nt: number of steps
    #         sigma: the kernel bandwidth for computing conditional expectation
    #         tol: convergence tolerance
    #         minNt: minimum number of iterations
    #         NparticleThreshold: number of particles as the threshold to switch between 
    #                             piecewse constant and linear approximation of conditional expectation
    # outputs: X (N,dim),
    #          Y (Np,dim)
    #          dists: history of Wp distance
    #          err_m2X, err_m2Y: history of the error in 2nd order moments for X and Y
    if epsX is None:
        epsX = sigma
    if epsY is None:
        epsY = sigma
        
    X = X00.copy()
    Y = Y00.copy()
    m2X = np.mean(X00, axis=0)
    m2Y = np.mean(Y00, axis=0)

    np_particles, d = Y.shape

    gradcx_Cond_x = np.zeros_like(X)   # Initialize Y conditional X
    gradcy_Cond_y = np.zeros_like(Y)   # Initialize X conditional Y

    err_m2X = []
    err_m2Y = []
    dists = []
    dists.append(np.mean(np.sum((X - Y) ** p, axis=1)))
    
    rx = epsX
    ry = epsY
        
    for it in range(Nt):
        if it % 1000 == 0 and it>0:
            print("it: ", it, " dist: ", dists[it])
        err_m2X.append(abs(m2X-np.mean(X, axis=0)))
        err_m2Y.append(abs(m2Y-np.mean(Y, axis=0)))
        
        X0 = X.copy()
        Y0 = Y.copy()

        k1_X = (-p*(X0-Y0)**(p-1) + gradcx_Cond_x) * dt
        k1_Y = (-p*(Y0-X0)**(p-1) + gradcy_Cond_y) * dt
        X_mid = X0 + 0.5 * k1_X
        Y_mid = Y0 + 0.5 * k1_Y
        Idx_mid = rangesearch(X_mid, rx)
        Idy_mid = rangesearch(Y_mid, ry)
        gradcx_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, p*(X_mid-Y_mid)**(p-1), Idx_mid, NparticleThreshold)
        gradcy_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, p*(Y_mid-X_mid)**(p-1), Idy_mid, NparticleThreshold)
        k2_X = (-p*(X_mid-Y_mid)**(p-1) + gradcx_Cond_x_mid) * dt
        k2_Y = (-p*(Y_mid-X_mid)**(p-1) + gradcy_Cond_y_mid) * dt
        X_mid = X0 + 0.5 * k2_X
        Y_mid = Y0 + 0.5 * k2_Y
        Idx_mid = rangesearch(X_mid, rx)
        Idy_mid = rangesearch(Y_mid, ry)
        gradcx_Cond_x_mid = compute_cond_vectorized_no_loop(X_mid, p*(X_mid-Y_mid)**(p-1), Idx_mid, NparticleThreshold)
        gradcy_Cond_y_mid = compute_cond_vectorized_no_loop(Y_mid, p*(Y_mid-X_mid)**(p-1), Idy_mid, NparticleThreshold)
        k3_X = (-p*(X_mid-Y_mid)**(p-1) + gradcx_Cond_x_mid) * dt
        k3_Y = (-p*(Y_mid-X_mid)**(p-1) + gradcy_Cond_y_mid) * dt
        X_end = X0 + k3_X
        Y_end = Y0 + k3_Y
        Idx_end = rangesearch(X_end, rx)
        Idy_end = rangesearch(Y_end, ry)
        gradcx_Cond_x_end = compute_cond_vectorized_no_loop(X_end, p*(X_end-Y_end)**(p-1), Idx_end, NparticleThreshold)
        gradcy_Cond_y_end = compute_cond_vectorized_no_loop(Y_end, p*(Y_end-X_end)**(p-1), Idy_end, NparticleThreshold)
        k4_X = (-p*(X_end-Y_end)**(p-1) + gradcx_Cond_x_end) * dt
        k4_Y = (-p*(Y_end-X_end)**(p-1) + gradcy_Cond_y_end) * dt
        X = X0 + (k1_X + 2*k2_X + 2*k3_X + k4_X) / 6.0
        Y = Y0 + (k1_Y + 2*k2_Y + 2*k3_Y + k4_Y) / 6.0
        
        Idx = rangesearch(X, rx)
        Idy = rangesearch(Y, ry)
        gradcx_Cond_x = compute_cond_vectorized_no_loop(X, p*(X-Y)**(p-1), Idx, NparticleThreshold) 
        gradcy_Cond_y = compute_cond_vectorized_no_loop(Y, p*(Y-X)**(p-1), Idy, NparticleThreshold)

        ## keep a history of Wp distance
        dists.append(np.mean(np.sum((X - Y) ** p, axis=1)))
        if it>minNt:
            if abs(dists[it+1]-dists[it]) < tol:
                break
    return X, Y, dists, err_m2X, err_m2Y