import logging
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from scipy.stats import norm as scipy_norm

logger = logging.getLogger("ppcx")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


def assign_spatial_priors(df, sectors, prior_strength=0.8):
    """Assign spatial prior probabilities based on polygon sectors."""
    ndata = len(df)
    k = len(sectors)
    prior_probs = np.ones((ndata, k)) / k  # default uniform

    for idx, sector in enumerate(sectors):
        mask = sector.contains_points(df["x"].values, df["y"].values)
        # Strong prior for points inside polygon
        prior_probs[mask] = (1 - prior_strength) / (
            k - 1
        )  # small prob for other clusters
        prior_probs[mask, idx] = prior_strength  # high prob for this cluster
        logger.info(f"Sector {idx}: {mask.sum()} points with strong prior")

    return prior_probs


def plot_spatial_priors(
    df,
    prior_probs,
    img=None,
    cmap="Reds",
    point_size=1,
    alpha=0.7,
    figsize_per_panel=(4, 4),
):
    """
    Plot spatial prior probability maps for each cluster.

    Parameters
    - df: DataFrame with columns 'x' and 'y' for point coordinates
    - prior_probs: array-like shape (n_points, k) with prior probability for each cluster
    - img: optional background image (2D/3D array) to show under the scatter
    - cmap: colormap for priors (default 'Reds')
    - point_size: scatter point size
    - alpha: scatter alpha
    - figsize_per_panel: tuple (width, height) per panel in inches

    Returns
    - fig, axes: matplotlib figure and axes array
    """

    prior_probs = np.asarray(prior_probs)
    n_points, k = prior_probs.shape

    nrows = int(np.ceil(np.sqrt(k)))
    ncols = int(np.ceil(k / nrows))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
    )
    # Normalize axes to flat list for easy indexing
    if isinstance(axes, np.ndarray):
        axes_flat = axes.flatten()
    else:
        axes_flat = [axes]

    for cluster in range(k):
        ax = axes_flat[cluster]
        if img is not None:
            ax.imshow(img, alpha=0.3)
        scatter = ax.scatter(
            df["x"],
            df["y"],
            c=prior_probs[:, cluster],
            cmap=cmap,
            s=point_size,
            alpha=alpha,
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Prior for Cluster {cluster}")
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.colorbar(scatter, ax=ax)

    # Hide any unused axes
    total_panels = nrows * ncols
    if total_panels > k:
        for ia in range(k, total_panels):
            axes_flat[ia].axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def compute_posterior_assignments(
    idata,
    X_scaled,
    prior_probs,
    *,
    n_posterior_samples=None,
    use_posterior_mean=False,
    random_seed=RANDOM_SEED,
):
    """
    Compute assignment probabilities, hard labels and uncertainty for a marginalized model.

    Parameters
    - idata: arviz InferenceData from marginalized model (must contain "mu" and "sigma")
    - X_scaled: (n_points, n_features) array used for inference (can be new single-day data)
    - prior_probs: (n_points, k) spatial priors for the same X_scaled rows
    - n_posterior_samples: if int, randomly subsample this many posterior draws for Monte Carlo
      (default: use all draws). Use to speed-up computation on large datasets.
    - use_posterior_mean: if True, compute responsibilities using posterior mean mu/sigma (cheap).
    - random_seed: RNG seed for subsampling posterior draws.

    Returns
    - posterior_probs: (n_points, k) averaged responsibilities
    - cluster_pred: (n_points,) hard labels = argmax_k posterior_probs
    - uncertainty: (n_points,) entropy of posterior_probs
    """

    # Extract posterior mu and sigma
    mu_samples = idata.posterior["mu"].values  # (chains, draws, k, n_features)
    sigma_samples = idata.posterior["sigma"].values

    # collapse chain/draw dims
    S_full = mu_samples.shape[0] * mu_samples.shape[1]
    k = mu_samples.shape[2]
    n_features = mu_samples.shape[3]
    n_points = X_scaled.shape[0]

    mu_flat = mu_samples.reshape(-1, k, n_features)  # (S_full, k, d)
    sigma_flat = sigma_samples.reshape(-1, k, n_features)

    if use_posterior_mean:
        # Fast deterministic assignment using posterior mean parameters
        mu_mean = mu_flat.mean(axis=0)  # (k, d)
        sigma_mean = sigma_flat.mean(axis=0)
        # compute log-likelihood for each cluster (vectorized)
        # shape -> (n_points, k)
        log_lik = np.stack(
            [
                scipy_norm.logpdf(X_scaled, loc=mu_mean[kk], scale=sigma_mean[kk]).sum(
                    axis=1
                )
                for kk in range(k)
            ],
            axis=1,
        )
        log_prior = np.log(prior_probs + 1e-12)  # (n_points, k)
        log_resp = log_prior + log_lik
        # normalize
        a = log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp - a)
        resp /= resp.sum(axis=1, keepdims=True)
        posterior_probs = resp
    else:
        # Monte Carlo average over posterior draws (may be expensive)
        rng = np.random.default_rng(random_seed)
        if n_posterior_samples is None or n_posterior_samples >= mu_flat.shape[0]:
            sel_idx = np.arange(mu_flat.shape[0])
        else:
            sel_idx = rng.choice(
                mu_flat.shape[0], size=n_posterior_samples, replace=False
            )

        S = sel_idx.shape[0]
        # accumulate responsibilities per draw
        resp_acc = np.zeros((S, n_points, k), dtype=float)
        for si, s in enumerate(sel_idx):
            # vectorized over points and features:
            # for each component kk compute logpdf across features and sum
            for kk in range(k):
                lp = scipy_norm.logpdf(
                    X_scaled, loc=mu_flat[s, kk], scale=sigma_flat[s, kk]
                )  # (n_points, d)
                log_lik = lp.sum(axis=1)  # (n_points,)
                log_prior = np.log(prior_probs[:, kk] + 1e-12)  # (n_points,)
                resp_acc[si, :, kk] = log_prior + log_lik
            # stabilize & normalize for this draw
            a = resp_acc[si].max(axis=1, keepdims=True)
            resp_acc[si] = np.exp(resp_acc[si] - a)
            resp_acc[si] /= resp_acc[si].sum(axis=1, keepdims=True)

        # average over selected draws
        posterior_probs = resp_acc.mean(axis=0)  # (n_points, k)

    # hard assignment, uncertainty (entropy)
    cluster_pred = posterior_probs.argmax(axis=1)
    uncertainty = -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)

    return posterior_probs, cluster_pred, uncertainty


def plot_1d_velocity_clustering(
    df_features,
    img,
    *,
    idata,
    cluster_pred,
    posterior_probs,
    scaler=None,
) -> tuple[Any, np.ndarray]:
    """
    Plot 1D velocity clustering results for marginalized model.
    - cluster_pred: array of cluster assignments (from responsibilities, not z samples)
    - posterior_probs: (n_points, k) assignment probabilities for each point

    Parameters
    - df_features: DataFrame with columns 'x', 'y', 'u', 'v', 'V'
    - img: optional background image (2D/3D array) to show under the scatter
    - idata: arviz InferenceData from marginalized model (must contain "mu" and "sigma")
    - cluster_pred: (n_points,) hard labels = argmax_k posterior_probs
    - posterior_probs: (n_points, k) averaged responsibilities
    - scaler: optional StandardScaler used to scale the velocity feature (for inverse transform)

    Returns
    - fig, uncertainty: matplotlib figure and (n_points,) entropy of posterior_probs

    """

    # Compute uncertainty (entropy) for each point
    uncertainty = -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)
    max_probs = posterior_probs[np.arange(len(cluster_pred)), cluster_pred]

    # Get model parameters
    mu_posterior = idata.posterior["mu"].mean(dim=["chain", "draw"]).values.flatten()
    sigma_posterior = (
        idata.posterior["sigma"].mean(dim=["chain", "draw"]).values.flatten()
    )

    # Distinct colors
    unique_labels = np.unique(cluster_pred)
    colors = [
        "#E31A1C",
        "#1F78B4",
        "#33A02C",
        "#FF7F00",
        "#6A3D9A",
        "#B15928",
        "#A6CEE3",
        "#B2DF8A",
        "#FB9A99",
        "#FDBF6F",
    ][: len(unique_labels)]
    color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Velocity field with quiver plot
    ax0 = axes[0, 0]
    ax0.set_title("Velocity Vector Field", fontsize=14, pad=10)
    if img is not None:
        ax0.imshow(img, alpha=0.5, cmap="gray")
    magnitudes = df_features["V"].to_numpy()
    vmin = 0.0
    vmax = np.max(magnitudes)
    norm = Normalize(vmin=vmin, vmax=vmax)
    q = ax0.quiver(
        df_features["x"].to_numpy(),
        df_features["y"].to_numpy(),
        df_features["u"].to_numpy(),
        df_features["v"].to_numpy(),
        magnitudes,
        scale=None,
        scale_units="xy",
        angles="xy",
        cmap="viridis",
        norm=norm,
        width=0.008,
        headwidth=2.5,
        alpha=1.0,
    )
    cbar = fig.colorbar(q, ax=ax0, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label("Velocity Magnitude", rotation=270, labelpad=15)
    ax0.set_aspect("equal")
    ax0.set_xticks([])
    ax0.set_yticks([])
    ax0.grid(False)

    # Plot 2: Spatial clusters
    ax1 = axes[0, 1]
    ax1.set_title("Velocity-Based Spatial Clustering", fontsize=14, pad=10)
    if img is not None:
        ax1.imshow(img, alpha=0.3, cmap="gray")
    for label in unique_labels:
        mask = cluster_pred == label
        if np.any(mask):
            ax1.scatter(
                df_features.loc[mask, "x"],
                df_features.loc[mask, "y"],
                c=color_map[label],
                s=8,
                alpha=0.8,
                label=f"Cluster {label}",
                edgecolors="none",
            )
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax1.set_aspect("equal")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot 3: Assignment uncertainty (entropy)
    ax2 = axes[1, 0]
    ax2.set_title("Assignment Uncertainty (Entropy)", fontsize=14, pad=10)
    if img is not None:
        ax2.imshow(img, alpha=0.3, cmap="gray")
    scatter = ax2.scatter(
        df_features["x"],
        df_features["y"],
        c=uncertainty,
        cmap="plasma",
        s=8,
        alpha=0.8,
        vmin=0,
        vmax=uncertainty.max(),
    )
    ax2.set_aspect("equal")
    ax2.set_xticks([])
    ax2.set_yticks([])
    cbar2 = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar2.set_label("Entropy", rotation=270, labelpad=20)

    # Plot 4: Velocity distribution by cluster
    ax3 = axes[1, 1]
    ax3.set_title("Velocity Distribution by Cluster", fontsize=14, pad=10)
    velocity = df_features["V"].values
    for label in unique_labels:
        mask = cluster_pred == label
        if np.any(mask):
            ax3.hist(
                velocity[mask],
                bins=35,
                alpha=0.7,
                density=True,
                color=color_map[label],
                label=f"Cluster {label}",
                edgecolor="white",
                linewidth=0.5,
            )
    # Overlay model distributions
    v_range = np.linspace(velocity.min(), velocity.max(), 200)
    for label in unique_labels:
        if scaler is not None:
            mu_orig = scaler.inverse_transform([[mu_posterior[label]]])[0, 0]
            sigma_orig = sigma_posterior[label] * scaler.scale_[0]
        else:
            mu_orig = mu_posterior[label]
            sigma_orig = sigma_posterior[label]
        model_dist = scipy_norm.pdf(v_range, mu_orig, sigma_orig)
        ax3.plot(
            v_range,
            model_dist,
            "--",
            color=color_map[label],
            linewidth=2.5,
            alpha=0.9,
            label=f"Model {label}",
        )
    ax3.set_xlabel("Velocity Magnitude", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, framealpha=0.9)

    # Statistics box
    fig.subplots_adjust(right=0.98)
    stats_text = "CLUSTER STATISTICS\n" + "=" * 40 + "\n"
    for label in unique_labels:
        mask = cluster_pred == label
        count = mask.sum()
        if count == 0:
            continue
        v_mean = velocity[mask].mean()
        v_std = velocity[mask].std()
        v_median = np.median(velocity[mask])
        nmad = np.median(np.abs(velocity[mask] - v_median)) * 1.4826
        x_mean = df_features.loc[mask, "x"].mean()
        y_mean = df_features.loc[mask, "y"].mean()
        x_std = df_features.loc[mask, "x"].std()
        y_std = df_features.loc[mask, "y"].std()
        avg_uncertainty = uncertainty[mask].mean()
        avg_prob = max_probs[mask].mean()
        stats_text += f"CLUSTER {label} (pts: {count})\n"
        stats_text += f"├─ Center: ({x_mean:.1f}, {y_mean:.1f})\n"
        stats_text += f"├─ Spread: (σx={x_std:.1f}, σy={y_std:.1f})\n"
        stats_text += f"├─ Velocity: {v_mean:.4f} ± {v_std:.4f}\n"
        stats_text += f"├─ Median/NMAD: {v_median:.4f}/{nmad:.4f}\n"
        stats_text += (
            f"├─ Model μ/σ: {mu_posterior[label]:.4f}/{sigma_posterior[label]:.4f}\n"
        )
        stats_text += f"├─ Avg Uncertainty: {avg_uncertainty:.4f}\n"
        stats_text += f"└─ Avg Assignment Prob: {avg_prob:.4f}\n\n"
    # Place statistics box in empty space
    fig.text(
        0.72,
        0.05,
        stats_text,
        fontsize=8,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout(rect=[0, 0, 0.7, 1])
    return fig, uncertainty
