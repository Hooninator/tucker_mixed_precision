import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import torch
import argparse
import pandas as pd

from dataclasses import dataclass
from functools import reduce

from scipy.io import loadmat

import time

##################################################
#
#               UTILITY FUNCTIONS
#
##################################################


def get_tensorname(path):
    return path.split("/")[-1].split(".")[0]


class Config:

    def __init__(self):
        return  # Everything blank for now

    def set_from_args(self, args):
        self.lra_u = args.lra_u
        self.ttmc_u = args.ttmc_u
        if args.mode == "cuda":
            self.lra_fn = gpu_lra_fns[args.lra]
            self.qr_fn = gpu_qr_fns[args.qrd]
            self.init_fn = gpu_lra_fns[args.init]
        else:
            self.lra_fn = lra_fns[args.lra]
            self.qr_fn = qr_fns[args.qrd]
            self.init_fn = lra_fns[args.init]
        self.ranks = args.ranks
        self.tensorname = get_tensorname(args.tensorpath)
        self.mode = args.mode

    def print(self):
        print("~~~~~~~~~~ CONFIGURATION ~~~~~~~~~~")
        print(f"    Tensor: {self.tensorname}")
        print(f"    Ranks: {self.ranks}")
        print(f"    Low-Rank Approximation Precision: {self.lra_u}")
        print(f"    TTMc Precision: {self.ttmc_u}")
        print(f"    Factor Matrix Init: {self.init_fn.__name__}")
        print(f"    Factor Matrix Update: {self.lra_fn.__name__}")
        print(f"    QR Decomposition: {self.qr_fn.__name__}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


config = Config()

precisions = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64
}


def generate_normal_100():
    return torch.randn((100, 100, 100), dtype=torch.float64)


def generate_small_test():
    data = [[[1, 2, 1], [2, 1, 2], [7, 2, 2]],
            [[1, 2, 1], [3, 1, 4], [2, 5, 2]],
            [[1, 2, 1], [2, 1, 4], [2, 2, 2]]]
    return torch.tensor(data, dtype=torch.float64)


def generate_evil():
    u = "fp16"
    upper = torch.finfo(precisions[u]).max * 1e5
    lower = torch.finfo(precisions[u]).tiny * 1e-5
    size = (50, 50, 50, 50)
    T = torch.randn(size) * upper + lower
    S = torch.randint(0, 2, (size))
    T = torch.mul(T, S)  # Zero out some random entries
    S = S.to(dtype=torch.float64).uniform_(
        lower, lower * 2)  # uniform random noise
    return T + S


tensors = {
    "IL2": tl.datasets.load_IL2data().tensor,
    "covid19_serology": tl.datasets.load_covid19_serology().tensor,
    "indian_pines": tl.datasets.load_indian_pines().tensor,
    "kinetic": tl.datasets.load_kinetic().tensor,
    "normal_100": generate_normal_100(),
    "small_test": generate_small_test(),
    "evil": generate_evil()
}


def read_tensor(args):
    if '/' in args.tensorpath:
        X = loadmat(args.tensorpath, mat_dtype=True)['X']
        X_tensor = torch.tensor(
            X[0][0][0], dtype=torch.float64)
        return X_tensor
    else:
        return torch.tensor(tensors[args.tensorpath], dtype=torch.float64)


def compute_error(X, G, U_list):
    T = reconstruct(G, U_list)
    return (torch.linalg.norm(X - T) / torch.linalg.norm(X)).item()


def compare_norm(X1, X2):
    assert X1.shape == X2.shape
    return (torch.norm(X1) - torch.norm(X2)) / torch.norm(X1).item()


def gpu_compute_error(d_X_fp64, d_G, d_U_list):

    assert d_X_fp64.dtype == torch.float64

    # Compute this in fp64
    d_G = to_u(d_G, 'fp64')
    d_U_list = m_to_u(d_U_list, 'fp64')
    d_T = ttmc_faster(d_G, d_U_list, False)

    d_U_list = m_to_u(d_U_list, config.ttmc_u)
    return (torch.linalg.norm(d_X_fp64 - d_T) / torch.linalg.norm(d_X_fp64)).item()


def unfold(X, mode):
    if config.mode == "cuda":
        return gpu_unfold(X, mode)
    else:
        return cpu_unfold(X, mode)


def cpu_unfold(X, mode):
    X_k = tl.base.unfold(X.numpy(), mode)
    return torch.tensor(X_k)


def gpu_unfold(d_X, mode):
    # Slow
    h_X = d_X.to(device='cpu')
    return cpu_unfold(h_X, mode).to(device='cuda')


def gpu_fold(d_X, mode, dims):
    h_X = d_X.to(device='cpu')
    h_X = torch.tensor(tl.fold(h_X.numpy(), mode, dims))
    return h_X.to(device='cuda')


def make_gaussian(m, n, device='cpu', precision='fp64'):
    return torch.randn(m, n, dtype=precisions[precision], device=device)


def to_u(X, u):
    return X.to(dtype=precisions[u])


def m_to_u(tensors, u):
    return [to_u(t, u) for t in tensors]


def to_u_gpu(X, u):
    return X.to(device='cuda', dtype=precisions[u])


def m_to_u_gpu(tensors, precision):
    result = []
    for t in tensors:
        result.append(to_u_gpu(t, precision))
    return (*result,)


def to_u_cpu(X, u):
    return X.to(device='cpu', dtype=precisions[u])


def m_to_u_cpu(tensors, precision):
    result = []
    for t in tensors:
        result.append(to_u_cpu(t, precision))
    return (*result,)


def alloc_gpu_tensor(dims, u):
    return torch.zeros(dims, dtype=precisions[u], device='cuda')


def converged(e1, e2, tol):
    return abs(e1 - e2) < tol


def form_core(X, U_list):
    Y = tl.tenalg.multi_mode_dot(X.numpy(), U_list, transpose=True)
    return torch.tensor(Y)


def gpu_form_core(d_X, d_U_list):
    d_G = ttmc_faster(d_X, d_U_list, True)
    return d_G


def reconstruct(G, U_list):
    Y = tl.tenalg.multi_mode_dot(G.numpy(), U_list)
    return torch.tensor(Y)


def nrm_scale(A, B, dim, inv):
    norms = torch.norm(B, dim=dim)
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1
    if inv:
        norms = torch.reciprocal(norms)
    if dim == 0:
        A_scaled = A / norms[None, :]  # Scale each column
    elif dim == 1:
        A_scaled = A / norms[:, None]  # Scale each row
    else:
        raise Exception("invalid dimension")
    return A_scaled


def compute_scaling_row(d_X):
    norms = torch.norm(d_X, dim=1)
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1
    return norms


# TODO: row and column scaling -- iterative row and column scaling
# can write a least squares problem to get the best possible diagonal entries
# scales rows and columns iteratively -- one/two times
# Evertything in GPU --- scaling may not be needed
# Persistenyl store input tensor and factors in fp16 -- dont' convert a bunch
# and also apply some noramlzjation thing at the start to the tensor once
# keep accumulating lambada scaling factor -- this can be in high precision -- then scale back once at the very end
def scaled_ugemm(A, B, precision):
    norms = torch.norm(A, dim=1)
    for i in range(len(norms)):
        if norms[i] == 0:
            norms[i] = 1
    A_scaled = A / norms[:, None]
    d_A, d_B = m_to_u_gpu([A_scaled, B], precision)
    d_C_scaled = d_A @ d_B
    C_scaled = to_u_cpu(d_C_scaled, "fp64")
    inv_norms = torch.reciprocal(norms)
    C = C_scaled / inv_norms[:, None]
    return C


def ugemm(A, B, precision):
    d_A, d_B = m_to_u_gpu([A, B], precision)
    d_C = d_A @ d_B
    C = to_u_cpu(d_C, "fp64")
    return C


##################################################
#
#               QR Decomposition
#
##################################################

def hh_qr(X):
    Q, _ = torch.linalg.qr(X, mode='reduced')
    return Q


def chol_qr(X):
    X_T = X.T
    d_X_T, d_X = m_to_u_gpu([X_T, X], "fp64")
    d_B = torch.matmul(d_X_T, d_X)
    B = to_u_cpu(d_B, "fp64")
    L = torch.linalg.cholesky(B)
    Q = torch.linalg.solve_triangular(L, X_T, upper=False).T
    return Q


def gpu_chol_qr(d_X):
    d_B = d_X.T @ d_X
    L = torch.linalg.cholesky(B)
    Q = torch.linalg.solve_triangular(L, X_T, upper=False).T
    return Q


qr_fns = {
    "hh_qr": hh_qr,
    "chol_qr": chol_qr
}


gpu_qr_fns = {
    "hh_qr": hh_qr,
    "chol_qr": gpu_chol_qr
}


def qr(X):
    qr_fn = config.qr_fn
    return qr_fn(X)

##################################################
#
#               FACTOR MATRIX INIT/UPDATE
#
##################################################


def rrf(X, r):
    l = 5
    nrm = torch.linalg.norm(X)
    S = make_gaussian(X.shape[1], r+l)
    Y = scaled_ugemm(X, S, config.lra_u)
    return qr(Y)


def gpu_rrf(d_X, r):
    l = 5
    # TODO: Only do this once?
    S = make_gaussian(d_X.shape[1], r+l, 'cuda', config.ttmc_u)
    Y = d_X @ S
    Y = to_u(Y, config.lra_u)
    return qr(Y)


def svd(X, r):
    U, _, _ = torch.linalg.svd(to_u(X, config.lra_u))
    U_r = to_u(U[:, :r], config.ttmc_u)
    return U_r


def rand_svd(X, r):
    Q = rrf(X, r)
    B_T = scaled_ugemm(X.T, Q, config.lra_u).T
    U_tilde, _, _ = torch.linalg.svd(B_T)  # O(R^{2N})
    return scaled_ugemm(Q, U_tilde[:, :r], config.lra_u)  # Q @ U_tilde[:, :r]


def gpu_rand_svd(d_X, r):
    d_X = to_u(d_X, config.ttmc_u)
    Q = gpu_rrf(d_X, r)
    Q = to_u(Q, config.ttmc_u)
    B = Q.T @ d_X
    B = to_u(B, config.lra_u)
    d_U_tilde, _, _, = torch.linalg.svd(B)
    d_U_tilde = to_u(d_U_tilde, config.ttmc_u)
    return Q @ d_U_tilde[:, :r]


lra_fns = {
    "rrf": rrf,
    "svd": svd,
    "rand_svd": rand_svd,
}


gpu_lra_fns = {
    "rrf": gpu_rrf,
    "svd": svd,
    "rand_svd": gpu_rand_svd,
}


def init_factors(X, ranks):
    order = X.ndim
    assert order == len(ranks)
    factors = []
    for k in range(order):
        X_k = unfold(X, k)
        U_k = config.init_fn(X_k, ranks[k])
        factors.append(U_k)
    return factors


def update_factor(Y, k, rank):
    Y_k = unfold(Y, k)
    U_k = config.lra_fn(Y_k, rank)
    return U_k

##################################################
#
#               SCALING MATRICES
#
##################################################


def init_scaling_matrices(d_X):
    d_D_list = []
    N = d_X.ndim
    dim_lst = np.arange(0, N)
    for n in range(N):
        # norm of each n-slice
        I = d_X.shape[n]
        d_D = torch.sum(d_X, dim=([d for d in dim_lst if d != n]))
        d_D_list.append(d_D)
    return d_D_list


def init_scaling_matrix_sum(d_X):
    d_D = torch.sum(d_X, dim=([d for d in range(d_X.dim()) if d != 0]))
    return d_D


def init_scaling_matrix_inf(d_X):
    # Slow
    d_D = torch.stack([torch.max(torch.select(d_X, 0, i))
                      for i in range(d_X.shape[0])])
    return torch.tensor(d_D, device='cuda', dtype=precisions[config.lra_u])


def init_scaling_matrix_single(d_X):
    val = torch.max(d_X)
    return torch.eye(d_X.shape[0], dtype=torch.float64) * val


# Solve an instance of the orthogonal Procrustes problem
# min_Q \|QX - E\|_F, where E is a matrix of all ones and Q is orthogonal
# This is solved by computing M = EX^T, which is equivalent to summing all the columns of X and replicating it n times
# then taking the SVD of M, M = UDV^T, and computing UV^T
def init_scaling_matrix_ortho(d_X):
    m = torch.sum(d_X, dim=([d for d in range(d_X.dim()) if d != 0]))
    M = torch.stack([m for _ in range(d_X.shape[0])])
    U, D, V_T = torch.linalg.svd(M)
    return U @ V_T


def apply_scaling_matrices(d_X, d_D_list, inv):
    N = d_X.ndim
    assert N == len(d_D_list)
    shape = [1] * N
    for n in range(N):
        # Scale n-slices
        shape[n] = -1
        if inv:
            d_X = d_X / d_D_list[n].view(*shape)
        else:
            d_X = d_X * d_D_list[n].view(*shape)
        shape[n] = 1
    return d_X


def apply_scaling_matrix(d_X, d_D, inv):
    if inv:
        d_X = d_X / d_D.view([1 if d != 0 else -1 for d in range(d_X.dim())])
    else:
        d_X = d_X * d_D.view([1 if d != 0 else -1 for d in range(d_X.dim())])
    return d_X


##################################################
#
#               TTMC FUNCTIONS
#
##################################################


def ttmc(X, matrices, transpose, exclude=[]):
    n = len(matrices)
    dims = list(X.shape)
    Y = X
    for i in range(n):
        if i in exclude:
            continue
        Y = torch.tensor(tl.unfold(Y.numpy(), i))
        U = matrices[i]
        if transpose:
            Y = scaled_ugemm(Y.T, U, config.ttmc_u).T
            dims[i] = U.shape[1]
        else:
            Y = scaled_ugemm(U, Y, config.ttmc_u)
            dims[i] = U.shape[0]
        Y = torch.tensor(tl.fold(Y.numpy(), i, dims))
    return Y


def ttmc_fast(d_X, matrices, transpose, exclude=[]):
    n = len(matrices)
    dims = list(d_X.shape)
    d_Y = d_X
    for i in range(n):
        if i in exclude:
            continue
        d_Y = gpu_unfold(d_Y, i)
        d_U = matrices[i]
        if transpose:
            d_Y = d_U.T @ d_Y
            dims[i] = d_U.shape[1]
        else:
            d_Y = d_U @ d_Y
            dims[i] = d_U.shape[0]
        d_Y = gpu_fold(d_Y, i, dims)
    return d_Y


def ttmc_faster(d_X, matrices, transpose, exclude=[]):

    n = len(matrices)
    dims = list(d_X.shape)
    d_Y = d_X

    for i in range(n):

        if i in exclude:
            continue

        d_U = matrices[i]

        if i == 0:
            d_Y = d_Y.reshape(dims[i], -1)
            if transpose:
                d_Y = d_U.T @ d_Y
                dims[i] = d_U.shape[1]
            else:
                d_Y = d_U @ d_Y
                dims[i] = d_U.shape[0]
            d_Y = d_Y.view(*dims)
            continue

        nk = d_Y.shape[i]

        M_k = reduce(lambda a, b: a*b, [d_Y.shape[d] for d in range(i)])

        if i == n-1:
            P_k = 1
        else:
            P_k = reduce(lambda a, b: a*b, [d_Y.shape[d]
                         for d in range(i+1, n)])

        d_Y = d_Y.reshape(M_k, nk, P_k)

        if transpose:
            d_Y_out = torch.zeros((M_k, d_U.shape[1], P_k))
        else:
            d_Y_out = torch.zeros((M_k, d_U.shape[0], P_k))
        d_Y_out = d_Y_out.to(device='cuda', dtype=d_Y.dtype)

        if transpose:
            for l in range(P_k):
                d_Y_out[:, :, l] = d_Y[:, :, l] @ d_U
            dims[i] = d_U.shape[1]
        else:
            for l in range(P_k):
                d_Y_out[:, :, l] = d_Y[:, :, l] @ d_U.T
            dims[i] = d_U.shape[0]
        d_Y = d_Y_out.reshape(*dims)

    return d_Y

##################################################
#
#               MAIN HOOI FUNCTION
#
##################################################


def hooi(X, ranks, maxiters):

    # Initialize factor matrices
    U_list = init_factors(X, ranks)
    G = form_core(X, U_list)
    err_curr = compute_error(X, G, U_list)
    err_prev = torch.linalg.norm(X)
    errors = [err_curr]

    # Main loop
    iter = 0
    N = X.ndim
    while iter < maxiters:

        for n in range(N):
            Y = ttmc(X, U_list, True, [n])
            U_n = update_factor(Y, n, ranks[n])
            U_list[n] = U_n

        G = form_core(X, U_list)
        err_prev = err_curr
        err_curr = compute_error(X, G, U_list)

        iter += 1
        errors.append(err_curr)

    print(f"Final error: {err_curr}")
    return errors


def hooi_fast(X, ranks, maxiters):

    # Move tensor to GPU
    # Keep things in fp64 for now, since we need to do scaling
    d_X = to_u_gpu(X, 'fp64')
    d_X_fp64 = to_u(d_X, 'fp64')  # For error computation

    # Compute scaling matrix
    d_D = init_scaling_matrix_inf(d_X)

    # Init factor matrices
    d_U_list = init_factors(d_X, ranks)

    # Convert to ttmc precision
    d_X = apply_scaling_matrix(d_X, d_D, True)
    d_X = to_u(d_X, config.ttmc_u)

    # Compute initial error
    d_U_list[0] = apply_scaling_matrix(d_U_list[0], d_D, False)
    d_U_list[0] = to_u(d_U_list[0], config.ttmc_u)
    d_G = gpu_form_core(d_X, d_U_list)
    d_U_list[0] = apply_scaling_matrix(d_U_list[0], d_D, True)
    d_U_list[0] = to_u(d_U_list[0], config.ttmc_u)
    err_curr = gpu_compute_error(d_X_fp64, d_G, d_U_list)
    print(f"Initial error: {err_curr}")
    err_prev = torch.linalg.norm(d_X).to(device='cpu')
    errors = [err_curr]

    # Main loop
    iter = 0
    N = d_X.ndim
    times = [0]
    while iter < maxiters:

        stime = time.time()

        for n in range(N):

            d_Y_n = ttmc_faster(d_X, d_U_list, True, [n])

            if n == 0:
                d_Y_n = apply_scaling_matrix(d_Y_n, d_D, False)

            d_U_n = update_factor(d_Y_n, n, ranks[n])

            if n == 0:
                d_U_n = apply_scaling_matrix(d_U_n, d_D, False)
                d_U_n = to_u(d_U_n, config.ttmc_u)

            d_U_list[n] = d_U_n

        etime = time.time()
        times.append((etime - stime))

        err_prev = err_curr
        d_G = gpu_form_core(d_X, d_U_list)
        d_U_list[0] = apply_scaling_matrix(d_U_list[0], d_D, True)
        d_U_list[0] = to_u(d_U_list[0], config.ttmc_u)
        err_curr = gpu_compute_error(d_X_fp64, d_G, d_U_list)

        iter += 1
        errors.append(err_curr)

    print(f"Final error: {err_curr}")
    return errors, times


def hooi_scaled(X, ranks, maxiters):
    return


##################################################
#
#               DRIVER AND STATS
#
##################################################

def rankstr(ranks):
    return 'x'.join([str(s) for s in ranks])


def get_filename(args):
    return f"{get_tensorname(args.tensorpath)}-lra:{args.lra}-qr:{args.qrd}-init:{args.init}-ttmcu:{args.ttmc_u}-lrau:{args.lra_u}-{rankstr(args.ranks)}.csv"


datadir = "./results/results_v2/"


def write_stats(args, error_mat):
    df = pd.DataFrame(error_mat)
    df.to_csv(f"{datadir}error_{get_filename(args)}")


def write_times(args, times_mat):
    df = pd.DataFrame(times_mat)
    df.to_csv(f"{datadir}timing_{get_filename(args)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorpath", type=str)
    parser.add_argument("--maxiters", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--ntrials", type=int, default=100)
    parser.add_argument("--lra", type=str)
    parser.add_argument("--qrd", type=str)
    parser.add_argument("--init", type=str)
    parser.add_argument("--ttmc_u", type=str)
    parser.add_argument("--lra_u", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--ranks", type=int, nargs='+')
    args = parser.parse_args()

    X = read_tensor(args)
    ranks = args.ranks

    config.set_from_args(args)
    config.print()

    # Pytorch setup
    torch.backends.cuda.matmul.allow_tf32 = True

    # Reference error
    G_ref, U_list_ref = tl.decomposition.tucker(
        X.numpy(), ranks, n_iter_max=args.maxiters, tol=args.tol)
    G_ref = torch.tensor(G_ref)
    for i in range(len(U_list_ref)):
        U_list_ref[i] = torch.tensor(U_list_ref[i])
    ref_error = compute_error(X, G_ref, U_list_ref)
    print(f"Reference Error: {ref_error}")

    error_mat = np.zeros(shape=(args.ntrials, args.maxiters + 1))
    times_mat = np.zeros(shape=(args.ntrials, args.maxiters + 1))
    for i in range(args.ntrials):
        errors, times = hooi_fast(X, ranks, args.maxiters)
        error_mat[i] = errors
        times_mat[i] = times
    print(f"Final times: {times_mat[args.ntrials-1]}")

    write_stats(args, error_mat)
    write_times(args, times_mat)
