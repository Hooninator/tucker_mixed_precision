
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
        return # Everything blank for now

    def set_from_args(self, args):
        self.lra_u = args.lra_u
        self.ttmc_u = args.ttmc_u
        self.lra_fn = lra_fns[args.lra]
        self.qr_fn = qr_fns[args.qrd]
        self.init_fn = lra_fns[args.init]

    def print(self):
        print("~~~~~~~~~~ CONFIGURATION ~~~~~~~~~~")
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


def converged(e1, e2, tol):
    return abs(e1 - e2) < tol


def form_core(X, U_list):
    Y = tl.tenalg.multi_mode_dot(X.numpy(), U_list, transpose=True)
    return torch.tensor(Y)


def reconstruct(G, U_list):
    Y = tl.tenalg.multi_mode_dot(G.numpy(), U_list)
    return torch.tensor(Y)

# Take row or columnwise norm of B, then scale appropriate dimension of A
def nrm_scale(A, B, dim, inv):
    norms = torch.norm(B, dim=dim)
    if inv:
        norms = torch.reciprocal(norms)
    if dim==0:
        A_scaled = A / norms[None, :] # Scale each column
    elif dim==1:
        A_scaled = A / norms[:, None] # Scale each row
    else:
        raise Exception("invalid dimension")
    return A_scaled


def scaled_ugemm(A, B, precision):
    norms = torch.norm(A, dim=1)
    for i in range(len(norms)):
        if norms[i]==0:
            norms[i] = 1
    A_scaled = A / norms[:, None]
    d_A, d_B = m_to_u_gpu([A_scaled, B], precision)
    d_C_scaled = d_A @ d_B
    C_scaled = to_u_cpu(d_C_scaled, "fp64")
    inv_norms = torch.reciprocal(norms)
    C = C_scaled / inv_norms[:, None]
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


qr_fns = {
    "hh_qr": hh_qr,
    "chol_qr": chol_qr
}


def qr(args, X):
    qr_fn = config.qr_fn
    return qr_fn(X)

##################################################
#
#               FACTOR MATRIX INIT/UPDATE
#
##################################################


def rrf(X, r):
    oversampling = 5
    nrm = torch.linalg.norm(X)
    S = make_gaussian(X.shape[1], r+oversampling)
    Y = scaled_ugemm(X, S, config.lra_u)
    return qr(args, Y)


def svd(X, r):
    U, _, _ = torch.linalg.svd(X)
    return U[:, :r]


def rand_svd(X, r):
    Q = rrf(X, r) 
    B_T = scaled_ugemm(X.T, Q, config.lra_u).T
    U_tilde, _, _ = torch.linalg.svd(B_T) # O(R^{2N})
    return scaled_ugemm(Q, U_tilde[:,:r], config.lra_u)#Q @ U_tilde[:, :r]


lra_fns = {
    "rrf": rrf,
    "svd": svd,
    "rand_svd": rand_svd,
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
        if transpose:
            Y = scaled_ugemm(Y.T, U, config.ttmc_u).T
            dims[i] = U.shape[1]
        else:
            Y = scaled_ugemm(U, Y, config.ttmc_u)
            dims[i] = U.shape[0]
        Y = torch.tensor(tl.fold(Y.numpy(), i, dims))
    return Y

##################################################
#
#               MAIN HOOI FUNCTION
#
##################################################


def hooi(args, X, ranks):

    # Initialize factor matrices with RRF
    U_list = init_factors(X, ranks)
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
            U_n = update_factor(Y, n, ranks[n])
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
    parser.add_argument("--ttmc_u", type=str)
    parser.add_argument("--lra_u", type=str)
    parser.add_argument("--ranks", type=int, nargs='+')
    args = parser.parse_args()
    config.set_from_args(args)
    config.print()

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
