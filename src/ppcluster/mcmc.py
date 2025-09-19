import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scipy.stats import norm as scipy_norm

logger = logging.getLogger("ppcx")
RANDOM_SEED = 8927
rng = np.random.default_rng(RANDOM_SEED)


""" PRIOR """


def assign_spatial_priors(
    df: pd.DataFrame,
    polygons: dict[str, Any],
    prior_probs: dict[str, list[float]],
) -> np.ndarray:
    """
    Assign spatial prior probabilities to each point based on sector polygons and a prior probability dictionary.

    - polygons: dict mapping sector name -> Polygon object (must have .contains_points(x, y))
    - prior_probs: dict mapping sector name -> probability vector (list of floats, length = n_sectors)
    - Points not inside any sector get uniform probability.

    Returns:
        prior_probs_arr: (ndata, n_sectors) array
    """
    sector_names = list(polygons.keys())
    n_sectors = len(sector_names)
    ndata = len(df)

    # Validate prior_probs dictionary
    if set(prior_probs.keys()) != set(sector_names):
        raise ValueError(
            f"assign_spatial_priors: sector names in polygons and prior_probs do not match.\n"
            f"polygons: {sector_names}\nprior_probs: {list(prior_probs.keys())}"
        )
    for name, vec in prior_probs.items():
        arr = np.asarray(vec, dtype=float)
        if arr.ndim != 1 or arr.size != n_sectors:
            raise ValueError(
                f"assign_spatial_priors: prior vector for '{name}' must be 1D and length {n_sectors}."
            )
        if np.any(arr < 0) or arr.sum() <= 0:
            raise ValueError(
                f"assign_spatial_priors: probabilities for '{name}' must be non-negative and sum > 0."
            )

    # Default uniform prior
    uniform = np.ones(n_sectors, dtype=float) / n_sectors
    prior_probs_arr = np.tile(uniform, (ndata, 1))

    # Extract x,y coordinates of the points
    x = df["x"].values
    y = df["y"].values

    # For each sector, assign its prior to points inside
    for name, polygon in polygons.items():
        mask = polygon.contains_points(x, y)
        prior_probs_arr[mask, :] = np.asarray(prior_probs[name], dtype=float) / np.sum(
            prior_probs[name]
        )

    return prior_probs_arr


def assign_spatial_priors_legacy(
    df: pd.DataFrame,
    sectors: Sequence[Any],
    prior_strength: float = 0.8,
) -> np.ndarray:
    """Assign spatial prior probabilities based on polygon sectors.

    Args:
        df: DataFrame with at least columns ``x`` and ``y`` containing point coordinates.
        sectors: Sequence of region objects exposing ``contains_points(x, y) -> np.ndarray[bool]``.
            Typically matplotlib Path-like or custom objects with the same API.
        prior_strength: Probability mass assigned to the preferred cluster when a point is
            inside a sector (in ``[0, 1]``). The remaining mass is spread uniformly over
            the other clusters.

    Returns:
        np.ndarray: Array of shape ``(n_points, k)`` with prior probabilities per observation
        and cluster.
    """
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
    df: pd.DataFrame,
    prior_probs: np.ndarray | Image.Image,
    img: np.ndarray | None = None,
    cmap: str = "Reds",
    point_size: float = 1,
    alpha: float = 0.7,
    figsize_per_panel: tuple[float, float] = (4, 4),
) -> tuple[Figure, np.ndarray | Axes]:
    """Plot spatial prior probability maps for each cluster.

    Args:
        df: DataFrame with columns ``x`` and ``y`` for point coordinates.
        prior_probs: Array of shape ``(n_points, k)`` with prior probabilities per cluster.
        img: Optional background image array to show under the scatter.
        cmap: Colormap name for priors (default ``'Reds'``).
        point_size: Scatter point size.
        alpha: Scatter alpha.
        figsize_per_panel: Tuple ``(width, height)`` per panel in inches.

    Returns:
        Tuple[Figure, Union[np.ndarray, Axes]]: Matplotlib figure and axes grid.
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
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

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


""" POSTERIOR AND STATISTICS """


def compute_posterior_assignments(
    idata: Any,
    X_scaled: np.ndarray,
    prior_probs: np.ndarray,
    *,
    n_posterior_samples: int | None = None,
    use_posterior_mean: bool = False,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute posterior responsibilities, hard labels, and uncertainty.

    Args:
        idata: ArviZ ``InferenceData`` from marginalized model (must contain variables
            ``mu`` and ``sigma`` in ``idata.posterior`` with dims ``(chain, draw, k, d)``).
        X_scaled: Array of shape ``(n_points, n_features)`` used for inference.
        prior_probs: Array of shape ``(n_points, k)`` with spatial priors for each point.
        n_posterior_samples: If provided, randomly subsample this many posterior draws
            to approximate responsibilities; otherwise use all draws.
        use_posterior_mean: If True, compute responsibilities using posterior mean
            of ``mu`` and ``sigma`` (faster, deterministic).
        random_seed: RNG seed used when subsampling posterior draws.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - posterior_probs: ``(n_points, k)`` averaged responsibilities.
            - cluster_pred: ``(n_points,)`` hard labels (argmax over clusters).
            - uncertainty: ``(n_points,)`` entropy of responsibilities per point.
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


def compute_cluster_statistics(
    df_features: pd.DataFrame,
    cluster_pred: np.ndarray,
    posterior_probs: np.ndarray,
    *,
    idata: Any | None = None,
    scaler: Any | None = None,
    feature_index: int = 0,
) -> dict[int, dict[str, float]]:
    """
    Compute per-cluster statistics independently of plotting.

    Args:
        df_features: DataFrame with columns ``x``, ``y``, and ``V`` (velocity magnitude).
        cluster_pred: Array ``(n_points,)`` with hard labels per point.
        posterior_probs: Array ``(n_points, k)`` with responsibilities.
        idata: Optional ArviZ ``InferenceData`` to compute model posterior means for
            ``mu`` and ``sigma`` (reported as ``model_mu``/``model_sigma`` for the
            selected feature).
        scaler: Optional scaler to inverse-transform the selected feature (applied to
            model parameters if provided).
        feature_index: Feature index to report ``μ/σ`` for (default ``0``).

    Returns:
        Dict[int, Dict[str, float]]: Mapping cluster_id to statistics, including keys:
            ``count``, ``x_mean``, ``y_mean``, ``x_std``, ``y_std``,
            ``velocity_mean``, ``velocity_std``, ``velocity_median``, ``velocity_nmad``,
            ``avg_entropy``, ``avg_assignment_prob``, ``model_mu``, ``model_sigma``.
    """
    # Per-point entropy_pt and max posterior prob
    entropy_pt = -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)
    max_prob_pt = posterior_probs[np.arange(len(cluster_pred)), cluster_pred]

    # Optional model μ/σ from posterior means
    model_mu = None
    model_sigma = None
    if idata is not None and "mu" in idata.posterior and "sigma" in idata.posterior:
        mu_arr = idata.posterior["mu"].mean(dim=["chain", "draw"]).values
        sigma_arr = idata.posterior["sigma"].mean(dim=["chain", "draw"]).values
        # Ensure shapes (k, n_features)
        if mu_arr.ndim == 1:
            mu_arr = mu_arr[:, None]
            sigma_arr = sigma_arr[:, None]
        # Select requested feature
        model_mu = mu_arr[:, feature_index].copy()
        model_sigma = sigma_arr[:, feature_index].copy()
        # Optionally inverse-transform
        if scaler is not None:
            # Inverse-transform μ: expects shape (n_samples, n_features)
            mu_sel = mu_arr[:, [feature_index]]
            model_mu = scaler.inverse_transform(mu_sel)[:, 0]
            # Scale σ with feature scale (Robust/Standard scalers expose scale_)
            scale_vec = getattr(scaler, "scale_", None)
            if scale_vec is not None:
                model_sigma = model_sigma * scale_vec[feature_index]

    velocity = df_features["V"].to_numpy()
    stats: dict[int, dict[str, float]] = {}
    for label in np.unique(cluster_pred):
        mask = cluster_pred == label
        count = int(mask.sum())
        if count == 0:
            continue

        v_vals = velocity[mask]
        v_mean = float(v_vals.mean())
        v_std = float(v_vals.std())
        v_median = float(np.median(v_vals))
        nmad = float(np.median(np.abs(v_vals - v_median)) * 1.4826)

        x_vals = np.asarray(df_features.loc[mask, "x"])  # Series -> ndarray
        y_vals = np.asarray(df_features.loc[mask, "y"])  # Series -> ndarray
        x_mean = float(x_vals.mean())
        y_mean = float(y_vals.mean())
        x_std = float(x_vals.std())
        y_std = float(y_vals.std())
        avg_entropy = float(entropy_pt[mask].mean())
        avg_prob = float(max_prob_pt[mask].mean())

        entry = {
            "count": count,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_std": x_std,
            "y_std": y_std,
            "velocity_mean": v_mean,
            "velocity_std": v_std,
            "velocity_median": v_median,
            "velocity_nmad": nmad,
            "avg_entropy": avg_entropy,
            "avg_assignment_prob": avg_prob,
            "model_mu": None,
            "model_sigma": None,
        }
        if model_mu is not None and model_sigma is not None:
            # Assumes label indexes components directly
            entry["model_mu"] = float(model_mu[label])
            entry["model_sigma"] = float(model_sigma[label])

        stats[int(label)] = entry

    return stats


def plot_1d_velocity_clustering(
    df_features: pd.DataFrame,
    img: Image.Image | np.ndarray | None,
    *,
    idata: Any,
    cluster_pred: np.ndarray,
    posterior_probs: np.ndarray,
    scaler: Any | None = None,
) -> tuple[Figure, np.ndarray, dict[int, dict[str, float]]]:
    """Plot 1D velocity clustering results for marginalized model.

    Args:
        df_features: DataFrame with columns ``x``, ``y``, ``u``, ``v``, and ``V``.
        img: Optional background image array.
        idata: ArviZ ``InferenceData`` containing posterior draws for ``mu`` and ``sigma``.
        cluster_pred: Array of hard cluster assignments per point.
        posterior_probs: Array of responsibilities per point and cluster.
        scaler: Optional scaler to inverse-transform model parameters for overlay.

    Returns:
        Tuple[Figure, np.ndarray, Dict[int, Dict[str, float]]]:
            Matplotlib figure, uncertainty (entropy) per point, and the statistics
            dictionary returned by ``compute_cluster_statistics``.
    """

    # Per-point uncertainty (entropy) for scatter plot and return value
    entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)
    # Max probs can be useful for debugging/annotation; compute if needed
    # max_probs = posterior_probs[np.arange(len(cluster_pred)), cluster_pred]

    # Get model parameters for overlay (posterior means)
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
        c=entropy,
        cmap="plasma",
        s=8,
        alpha=0.8,
        vmin=0,
        vmax=entropy.max(),
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
    # Ensure numpy array for range computation
    velocity_arr = np.asarray(velocity)
    v_range = np.linspace(float(np.min(velocity_arr)), float(np.max(velocity_arr)), 200)
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

    # Render statistics box
    stats = compute_cluster_statistics(
        df_features,
        cluster_pred,
        posterior_probs,
        idata=idata,
        scaler=scaler,
        feature_index=0,
    )
    fig.subplots_adjust(right=0.98)
    stats_text = "CLUSTER STATISTICS\n" + "=" * 40 + "\n"
    for label in sorted(stats.keys()):
        s = stats[label]
        stats_text += f"CLUSTER {label} (pts: {s['count']})\n"
        stats_text += f"├─ Center: ({s['x_mean']:.1f}, {s['y_mean']:.1f})\n"
        stats_text += f"├─ Spread: (σx={s['x_std']:.1f}, σy={s['y_std']:.1f})\n"
        stats_text += (
            f"├─ Velocity: {s['velocity_mean']:.4f} ± {s['velocity_std']:.4f}\n"
        )
        stats_text += (
            f"├─ Median/NMAD: {s['velocity_median']:.4f}/{s['velocity_nmad']:.4f}\n"
        )
        if s["model_mu"] is not None and s["model_sigma"] is not None:
            stats_text += f"├─ Model μ/σ: {s['model_mu']:.4f}/{s['model_sigma']:.4f}\n"
        stats_text += f"├─ Avg Entropy: {s['avg_entropy']:.4f}\n"
        stats_text += f"└─ Avg Assignment Prob: {s['avg_assignment_prob']:.4f}\n\n"

    fig.text(
        0.72,
        0.05,
        stats_text,
        fontsize=8,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )

    plt.tight_layout(rect=(0, 0, 0.7, 1))
    return fig, entropy, stats
