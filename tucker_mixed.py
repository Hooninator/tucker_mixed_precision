
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
import torch
import argparse

from dataclasses import dataclass

from scipy.io import loadmat


##################################################
#
#               UTILITY FUNCTIONS
#
##################################################

class Config:

    def __init__(self):
        return

    def set_from_args(self, args):
        return


precisions = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64
}

tensors = {
    "IL2": tl.datasets.load_IL2data,
    "covid19_serology": tl.datasets.load_covid19_serology,
    "indian_pines": tl.datasets.load_indian_pines,
    "kinetic": tl.datasets.load_kinetic
}


def read_tensor(args):
    if '/' in args.tensorpath:
        X = loadmat(args.tensorpath, mat_dtype=True)['X']
        X_tensor = torch.tensor(
            X[0][0][0], dtype=torch.float64, device=args.device)
        return X_tensor
    else:
        return torch.tensor(tensors[args.tensorpath]().tensor, dtype=torch.float64)


def compute_error(X, G, U_list):
    T = reconstruct(G, U_list)
    return (torch.linalg.norm(X - T) / torch.linalg.norm(X)).item()


def unfold(X, mode):
    X_k = tl.base.unfold(X.numpy(), mode)
    return torch.tensor(X_k)


def make_gaussian(m, n):
    return torch.randn(m, n, dtype=torch.float64)


def to_u_gpu(X, u):
    return X.to(device='cuda', dtype=precisions[u])


def m_to_u_gpu(tensors, precisions):
    result = []
    for t, p in zip(tensors, precisions):
        result.append(to_u_gpu(t, p))
    return (*result,)


def to_u_cpu(X, u):
    return X.to(device='cpu', dtype=precisions[u])


def m_to_u_cpu(tensors, precisions):
    result = []
    for t, p in zip(tensors, precisions):
        result.append(to_u_cpu(t, p))
    return (*result,)


def converged(e1, e2, tol):
    return abs(e1 - e2) < tol

def form_core(X, U_list):
    Y = tl.tenalg.multi_mode_dot(X.numpy(), U_list, transpose=True)
    return torch.tensor(Y)


def reconstruct(G, U_list):
    Y = tl.tenalg.multi_mode_dot(G.numpy(), U_list)
    return torch.tensor(Y)


##################################################
#
#               QR Decomposition
#
##################################################

def hh_qr(X):
    Q, _ = torch.linalg.qr(X, mode='reduced')
    return Q


def chol_qr(X):
    d_X, d_X_T = m_to_u_gpu([X, X.T], ["fp64", "fp64"])
    d_B = torch.matmul(d_X_T, d_X)
    B = to_u_cpu(d_B, "fp64")
    L = torch.linalg.cholesky(B)
    Q = torch.linalg.solve_triangular(L, X.T, upper=False).T
    return Q


qr_ops = {
    "hh_qr": hh_qr,
    "chol_qr": chol_qr
}


def qr(args, X):
    qr_fn = qr_ops[args.qrd]
    return qr_fn(X)

##################################################
#
#               FACTOR MATRIX INIT/UPDATE
#
##################################################


def rrf(args, X, r):
    nrm = torch.linalg.norm(X)
    S = make_gaussian(X.shape[1], r)
    d_S, d_X = m_to_u_gpu([S,X], ["fp64", "fp64"])
    Y = torch.matmul(d_X, d_S)
    Y = to_u_cpu(Y, "fp64")
    return qr(args, Y)


def svd(args, X, r):
    U, _, _ = torch.linalg.svd(X)
    return U[:, :r]


def rand_svd(args, X, r):
    Q = rrf(args, X, r)
    B = Q.T @ X
    U_tilde, _, _ = torch.linalg.svd(B)
    return Q @ U_tilde[:, :r]


lra_ops = {
    "rrf": rrf,
    "svd": svd,
    "rand_svd": rand_svd,
}


def init_factors(args, X, ranks):
    order = X.ndim
    assert order == len(ranks)
    factors = []
    for k in range(order):
        X_k = unfold(X, k)
        U_k = lra_ops[args.init](args, X_k, ranks[k])
        factors.append(U_k)
    return factors


def update_factor(args, Y, k, rank):
    Y_k = unfold(Y, k)
    lra_fn = lra_ops[args.lra]
    U_k = lra_fn(args, Y_k, rank)
    return U_k

##################################################
#
#               TTMC FUNCTION
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
        d_Y, d_U = m_to_u_gpu([Y, U], ["fp64", "fp64"])
        if transpose:
            d_Y = d_U.T @ d_Y
            dims[i] = U.shape[1]
        else:
            d_Y = d_U @ d_Y
            dims[i] = U.shape[0]
        Y = to_u_cpu(d_Y, "fp64")
        Y = torch.tensor(tl.fold(Y.numpy(), i, dims))
    return Y

##################################################
#
#               MAIN HOOI FUNCTION
#
##################################################


def hooi(args, X, ranks):

    # Initialize factor matrices with RRF
    U_list = init_factors(args, X, ranks)
    G = form_core(X, U_list)
    err_curr = compute_error(X, G, U_list)
    err_prev = torch.linalg.norm(X)
    errors = [err_curr]

    # Main loop
    maxiters = args.maxiters
    iter = 0
    N = X.ndim
    while iter < maxiters and not converged(err_curr, err_prev, args.tol):

        for n in range(N):
            Y = ttmc(X, U_list, True, [n])
            U_n = update_factor(args, Y, n, ranks[n])
            U_list[n] = U_n

        G = form_core(X, U_list)
        err_prev = err_curr
        err_curr = compute_error(X, G, U_list)

        iter += 1
        errors.append(err_curr)

    print(err_curr)
    return errors

##################################################
#
#               DRIVER AND STATS
#
##################################################


def write_stats(args, error_mat):
    return


def print_args(args):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorpath", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--maxiters", type=int)
    parser.add_argument("--tol", type=float)
    parser.add_argument("--ntrials", type=int, default=100)
    parser.add_argument("--lra", type=str)
    parser.add_argument("--qrd", type=str)
    parser.add_argument("--init", type=str)
    parser.add_argument("--init_u", type=str)
    parser.add_argument("--ttmc_u", type=str)
    parser.add_argument("--lra_u", type=str)
    parser.add_argument("--ranks", type=int, nargs='+')
    args = parser.parse_args()
    print_args(args)

    X = read_tensor(args)
    print(X.shape)
    ranks = args.ranks

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
    for i in range(args.ntrials):
        errors = hooi(args, X, ranks)
        error_mat[i] = errors

    write_stats(args, error_mat)
