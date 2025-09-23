import logging
from typing import Any

import numpy as np
import pymc as pm
from pymc import math as pm_math

logger = logging.getLogger("ppcx")

RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


def build_marginalized_mixture_model(
    data: np.ndarray,
    prior_probs: np.ndarray,
    sectors: dict[str, Any],
    mu_params: dict[str, Any] | None = None,
    sigma_params: dict[str, Any] | None = None,
    feature_weights: np.ndarray | list[float] | None = None,
):
    """
    Build a marginalized mixture PyMC model (no discrete z).
    Returns a PyMC model object (not sampled).
    """
    n_data = data.shape[0]
    n_features = data.shape[1]
    k = len(sectors)

    model = pm.Model(
        coords={"obs": range(n_data), "cluster": range(k), "feature": range(n_features)}
    )

    # Parse mu and sigma parameters
    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}

    # If feature weights provided, scale data accordingly
    if feature_weights is not None:
        if len(feature_weights) != n_features:
            raise ValueError(
                f"Feature weights length {len(feature_weights)} does not match number of features {n_features}."
            )
        feature_weights = np.array(feature_weights)
        data = data * feature_weights[np.newaxis, :]

    with model:
        obs_data = pm.Data("obs_data", data, dims=("obs", "feature"))
        prior_w = pm.Data(
            "prior_w", prior_probs.reshape(n_data, k), dims=("obs", "cluster")
        )

        mu = pm.Normal(
            "mu", mu_params["mu"], mu_params["sigma"], dims=("cluster", "feature")
        )
        sigma = pm.HalfNormal(
            "sigma", sigma_params["sigma"], dims=("cluster", "feature")
        )

        # Log weights with small constant to avoid log(0)
        log_w = pm.Deterministic(
            "log_w", pm_math.log(prior_w + 1e-12), dims=("obs", "cluster")
        )

        # Per-cluster log-likelihood
        x_centered = (obs_data[:, None, :] - mu[None, :, :]) / sigma[None, :, :]
        logp_feat = -0.5 * (
            pm_math.log(2 * np.pi) + 2 * pm_math.log(sigma[None, :, :]) + x_centered**2
        )
        logp_clusters = logp_feat.sum(axis=2)  # (obs, cluster)

        # Mixture log likelihood (marginalized over clusters)
        log_mix = pm.logsumexp(logp_clusters + log_w, axis=1)  # (obs,)

        # Total logp as Potential
        pm.Potential("mixture_logp", log_mix.sum())

    logger.info("Marginalized mixture model (un-sampled) created.")
    return model


# # Simple, not marginalized model (z discrete) --> slower sampling, but direct cluster assignments

# n_features = X_scaled.shape[1]
# n_data = X_scaled.shape[0]
# k = len(sectors)  # number of clusters = number of sectors
#
# with pm.Model(
#     coords={"cluster": range(k), "feature": range(n_features), "obs": range(ndata)}
# ) as simple_model:
#     # Cluster means
#     μ = pm.Normal("μ", mu=0, sigma=1, dims=("cluster", "feature"))

#     # Cluster standard deviations (diagonal covariance)
#     σ = pm.HalfNormal("σ", sigma=0.5, dims=("cluster", "feature"))

#     # Cluster assignments with spatial priors
#     z = pm.Categorical("z", p=prior_probs, dims="obs")

#     # Likelihood: each point comes from its assigned cluster
#     observations = pm.Normal(
#         "x_obs", mu=μ[z], sigma=σ[z], observed=X_scaled, dims=("obs", "feature")
#     )

# with simple_model:
#     prior_samples = pm.sample_prior_predictive(100)

# fig, ax = plt.subplots(figsize=(8, 4))
# az.plot_dist(
#     X_scaled,
#     kind="hist",
#     color="C1",
#     hist_kwargs={"alpha": 0.6},
#     label="observed",
# )
# az.plot_dist(
#     prior_samples.prior_predictive["x_obs"],
#     kind="hist",
#     hist_kwargs={"alpha": 0.6},
#     label="simulated",
# )
# plt.xticks(rotation=45);
