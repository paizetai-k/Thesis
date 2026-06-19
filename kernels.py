"""
GP kernel, model, and training loop for NBA shot prediction.

Kernel: ScaleKernel(Matérn-5/2, ARD-2) on (x_norm, y_norm) — independent length-scales ls_x, ls_y.
Input shape: (n, 2).
"""
import warnings
import numpy as np
import torch
import gpytorch


# ---------------------------------------------------------------------------
# GP model
# ---------------------------------------------------------------------------

class KernelGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel, learn_inducing=True):
        var_dist  = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        var_strat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, var_dist, learn_inducing_locations=learn_inducing
        )
        super().__init__(var_strat)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ---------------------------------------------------------------------------
# Kernel building & hyperparameter logging
# ---------------------------------------------------------------------------

def build_kernel(variant="standard"):
    if variant != "standard":
        raise ValueError(f"Unknown variant: {variant!r}")
    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2))


def hp_str(model, variant="standard"):
    """Short hyperparameter summary for training-progress logging."""
    k = model.covar_module
    ls = k.base_kernel.lengthscale[0]  # shape (2,) with ARD
    return (f"ls_x={ls[0].item():.3f}  ls_y={ls[1].item():.3f}  "
            f"os={k.outputscale.item():.3f}")


# ---------------------------------------------------------------------------
# Inducing point initialisation
# ---------------------------------------------------------------------------

def init_inducing(train_x, n_inducing):
    """K-means init; replaces any NaN centroids (empty clusters) with random points."""
    from scipy.cluster.vq import kmeans2
    rng  = np.random.default_rng(42)
    data = train_x.numpy().astype(float)
    init = data[rng.choice(len(data), n_inducing, replace=False)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centres, _ = kmeans2(data, init, iter=20, minit="matrix")
    nan_rows = np.isnan(centres).any(axis=1)
    if nan_rows.any():
        centres[nan_rows] = data[rng.choice(len(data), int(nan_rows.sum()), replace=False)]
    return torch.tensor(centres, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_gp(train_x, train_y, kernel, learn_inducing=True,
             n_iter=500, lr=0.01, n_inducing=200, print_every=100, log_fn=None):
    """
    Train a sparse variational GP classifier (BernoulliLikelihood + ELBO).
    log_fn(model) -> str  is called every print_every iterations for progress output.
    Returns (model, likelihood).
    """
    inducing_pts = init_inducing(train_x, n_inducing)
    likelihood   = gpytorch.likelihoods.BernoulliLikelihood()
    model        = KernelGPModel(inducing_pts, kernel, learn_inducing=learn_inducing)

    model.train(); likelihood.train()
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_iter)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_x.size(0))

    for i in range(n_iter):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if print_every and (i + 1) % print_every == 0:
            suffix = log_fn(model) if log_fn else ""
            print(f"    iter {i+1:3d}/{n_iter}  loss={loss.item():.4f}  {suffix}")

    return model, likelihood
