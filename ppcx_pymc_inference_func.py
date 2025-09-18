import logging

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

logger = logging.getLogger("ppcx")


def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
):
    if grid_spacing is None:
        # Estimate grid spacing from minimum distances
        points = np.column_stack((x, y))
        tree = cKDTree(points)
        # Query the nearest neighbor point (k=2)
        distances, _ = tree.query(points, k=2)
        # distances[:, 1] is distance to nearest neighbor
        grid_spacing = np.median(distances[:, 1])
        if any(np.isnan(distances[:, 1])):
            logger.warning("Some points have no neighbors within the search radius.")
        if any(distances[:, 1] != grid_spacing):
            logger.warning(
                "Points are not on a regular grid. Interpolation may be needed."
            )
        logger.info(f"Estimated grid spacing: {grid_spacing:.2f}")

    # Create regular grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)

    # Initialize with NaN for no label
    label_grid = np.full(X.shape, np.nan)

    if labels is not None:
        # Map the labels to the grid
        for i, (xi, yi) in enumerate(zip(x, y, strict=False)):
            ix = np.argmin(np.abs(x_grid - xi))
            iy = np.argmin(np.abs(y_grid - yi))
            label_grid[iy, ix] = labels[i]

    return X, Y, label_grid


def remove_small_grid_components(label_grid, min_size=5, connectivity=8):
    """
    Remove/merge small connected components in a 2D label grid.
    Small components are reassigned to the most common neighbor label (if any),
    otherwise set to NaN.
    Args:
        label_grid: 2D numpy array with labels (can contain np.nan for empty cells)
        min_size: minimum size (in grid cells) to keep a component
        connectivity: 4 or 8 (neighbors)
    Returns:
        cleaned_grid: 2D array with small components merged/removed
    """
    cleaned = label_grid.copy().astype(float)
    structure = (
        np.ones((3, 3), dtype=bool)
        if connectivity == 8
        else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    )

    # iterate over each distinct label value (ignore NaN)
    unique_labels = np.unique(label_grid[~np.isnan(label_grid)])
    for lab in unique_labels:
        mask = label_grid == lab
        if not np.any(mask):
            continue
        components, ncomp = ndimage.label(mask, structure=structure)
        for comp_id in range(1, ncomp + 1):
            comp_mask = components == comp_id
            comp_size = comp_mask.sum()
            if comp_size < min_size:
                # dilate component to get neighbor cells
                nb_mask = ndimage.binary_dilation(
                    comp_mask, structure=np.ones((3, 3))
                ) & (~comp_mask)
                neighbor_labels = cleaned[nb_mask]
                # exclude NaNs and the same label
                neighbor_labels = neighbor_labels[~np.isnan(neighbor_labels)]
                neighbor_labels = neighbor_labels[neighbor_labels != lab]
                if neighbor_labels.size > 0:
                    # pick most common neighbor label
                    new_label = np.bincount(neighbor_labels.astype(int)).argmax()
                    cleaned[comp_mask] = new_label
                else:
                    cleaned[comp_mask] = np.nan
    return cleaned


def plot_cluster_labels_on_image(
    df_features,
    img,
    cluster_pred,
    ax=None,
    title="Velocity-Based Spatial Clustering",
    markersize=8,
):
    """Plot cluster labels over optional background image."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(title, fontsize=14, pad=10)
    if img is not None:
        ax.imshow(img, alpha=0.3, cmap="gray")

    unique_labels = np.unique(cluster_pred)
    # reserve a color for noise / NaN mapped to -1
    noise_label = -1
    labels_no_noise = [lab for lab in sorted(unique_labels) if lab != noise_label]

    # build a discrete colormap sized to the number of non-noise labels
    n = max(1, len(labels_no_noise))
    cmap = plt.cm.get_cmap("tab20", n)
    colors_arr = [cmap(i) for i in range(n)]

    color_map = {}
    for i, lab in enumerate(labels_no_noise):
        color_map[lab] = colors_arr[i]
    # assign noise color
    if noise_label in unique_labels:
        color_map[noise_label] = "#7f7f7f"

    for label in sorted(unique_labels):
        mask = cluster_pred == label
        if np.any(mask):
            ax.scatter(
                df_features.loc[mask, "x"],
                df_features.loc[mask, "y"],
                c=[color_map[label]],
                s=markersize,
                alpha=0.8,
                label=f"Cluster {label}",
                edgecolors="none",
            )

    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def close_small_holes(
    label_grid,
    max_hole_size=10,
    connectivity=8,
    require_single_neighbor=True,
):
    """
    Close small NaN-holes in a 2D label grid.

    Rules:
      - Only close holes whose size (number of NaN cells) <= max_hole_size.
      - Only close if the dilated border of the hole contains no NaNs
        (i.e. data all around the hole).
      - If require_single_neighbor is True the border must contain a single unique
        label; otherwise the most common border label is used.

    Args:
      label_grid: 2D ndarray with labels (NaN for empty cells).
      max_hole_size: maximum hole area (in grid cells) to fill.
      connectivity: 4 or 8 connectivity for labeling/dilation.
      require_single_neighbor: if True, require single neighbor label on border.

    Returns:
      new_grid: copy of label_grid with selected holes filled.
    """
    new_grid = label_grid.copy().astype(float)
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    # mask of holes (NaNs)
    hole_mask_all = np.isnan(label_grid)
    if not np.any(hole_mask_all):
        logger.debug("close_small_holes: no holes found (no NaNs in grid).")
        return new_grid

    # label each hole component
    comp_labels, ncomp = ndimage.label(hole_mask_all, structure=structure)
    filled_count = 0
    skipped_count = 0
    for comp_id in range(1, ncomp + 1):
        comp_mask = comp_labels == comp_id
        comp_size = int(comp_mask.sum())
        reason = None
        if comp_size > max_hole_size:
            reason = f"size>{max_hole_size}"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        # dilate to get border cells (use same connectivity structure)
        dilated = ndimage.binary_dilation(comp_mask, structure=structure)
        border_mask = dilated & (~comp_mask)

        # border values
        border_vals = new_grid[border_mask]
        if border_vals.size == 0:
            reason = "no border cells"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        # if any border cell is NaN -> not fully surrounded by data
        if np.any(np.isnan(border_vals)):
            reason = "border_has_nans"
            skipped_count += 1
            logger.debug(
                f"hole {comp_id}: size={comp_size} skipped ({reason}) border_nan_fraction={np.isnan(border_vals).mean():.3f}"
            )
            continue

        # get unique neighbor labels and counts
        unique_neighbors, counts = np.unique(border_vals, return_counts=True)
        if unique_neighbors.size == 0:
            reason = "no_neighbors"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        if require_single_neighbor:
            # require border to be all same label
            if unique_neighbors.size == 1:
                fill_label = unique_neighbors[0]
            else:
                reason = f"multiple_neighbors({unique_neighbors.tolist()})"
                skipped_count += 1
                logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
                continue
        else:
            # pick most common neighbor label
            fill_label = unique_neighbors[np.argmax(counts)]

        # sanity: ensure fill_label is finite
        if not np.isfinite(fill_label):
            reason = "fill_label_not_finite"
            skipped_count += 1
            logger.debug(f"hole {comp_id}: size={comp_size} skipped ({reason})")
            continue

        # coerce label to int-like if appropriate (avoid filling with floats like 1.0)
        if np.allclose(fill_label, np.round(fill_label)):
            fill_label = float(int(np.round(fill_label)))
        else:
            fill_label = float(fill_label)

        # fill hole with chosen label
        new_grid[comp_mask] = fill_label
        filled_count += 1
        logger.debug(
            f"hole {comp_id}: size={comp_size} filled with label={fill_label} (require_single_neighbor={require_single_neighbor})"
        )

    logger.info(
        f"close_small_holes: filled={filled_count} skipped={skipped_count} total={ncomp}"
    )
    return new_grid


def split_disconnected_components(label_grid, connectivity=8, start_label=0):
    """
    Split disconnected components of each label into unique labels.

    Args:
      label_grid: 2D array with labels (NaN = empty).
      connectivity: 4 or 8 connectivity for ndimage.label.
      start_label: integer to start new labels from.
      debug: if True logs summary.

    Returns:
      new_grid: 2D array with disconnected pieces assigned unique integer labels.
      mapping: dict original_label -> list of new labels assigned for its pieces.
    """
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    new_grid = np.full_like(label_grid, np.nan, dtype=float)
    mapping = {}
    next_label = int(start_label)

    unique_labels = np.unique(label_grid[~np.isnan(label_grid)])
    for lab in unique_labels:
        lab = float(lab)
        mask = label_grid == lab
        if not np.any(mask):
            continue
        comp, ncomp = ndimage.label(mask, structure=structure)
        mapping[int(lab)] = []
        for cid in range(1, ncomp + 1):
            comp_mask = comp == cid
            new_grid[comp_mask] = float(next_label)
            mapping[int(lab)].append(int(next_label))
            next_label += 1

    total_new = next_label - int(start_label)
    logger.debug(
        f"split_disconnected_components: original_labels={unique_labels.size} new_pieces={total_new} mapping={mapping}",
    )
    return new_grid, mapping


def map_grid_to_points(X, Y, label_grid, df, x_col="x", y_col="y", nan_fill=-1):
    """
    Map values from a 2D grid (label_grid) back to the original points in df.
    - X, Y: meshgrid returned by create_2d_grid (X.shape == label_grid.shape).
    - label_grid: 2D array with labels (np.nan = empty).
    - df: DataFrame with point coordinates.
    - x_col, y_col: column names in df for coordinates.
    - nan_fill: integer value to use when grid cell is NaN (default -1).
    Returns:
      numpy integer array of length len(df) with mapped labels.
    """
    x_grid = X[0, :]
    y_grid = Y[:, 0]
    pts = df[[x_col, y_col]].to_numpy()
    out = np.full(pts.shape[0], nan_fill, dtype=int)
    for i, (xi, yi) in enumerate(pts):
        ix = np.argmin(np.abs(x_grid - xi))
        iy = np.argmin(np.abs(y_grid - yi))
        val = label_grid[iy, ix]
        out[i] = nan_fill if np.isnan(val) else int(val)
    return out


def compute_cluster_statistics_simple(
    df_features: pd.DataFrame,
    cluster_pred: np.ndarray,
    posterior_probs: np.ndarray | None = None,
) -> dict[int, dict[str, float]]:
    """
    Simpler per-cluster statistics that works after grid cleaning / relabeling.

    - Does NOT rely on the original Bayesian model (no idata/scaler).
    - posterior_probs is optional; if provided we compute per-point entropy and
      use posterior_probs.max(axis=1) as a generic "assignment confidence".
      Note: posterior_probs may refer to original model components and therefore
      may not index new labels produced by post-processing. We only use the
      per-point max-prob and entropy (aggregated per cleaned cluster).

    Returns:
        Mapping cluster_id -> stats dict with keys:
          count, x_mean, y_mean, x_std, y_std,
          velocity_mean, velocity_std, velocity_median, velocity_nmad,
          avg_entropy (or None), avg_assignment_prob (or None)
    """
    cluster_pred = np.asarray(cluster_pred)
    velocity = np.asarray(df_features["V"])
    x_vals_all = np.asarray(df_features["x"])
    y_vals_all = np.asarray(df_features["y"])

    # Optional per-point metrics if posterior_probs available
    if posterior_probs is not None:
        pp = np.asarray(posterior_probs)
        # entropy per point (stable)
        entropy_pt = -np.sum(pp * np.log(pp + 1e-12), axis=1)
        # per-point max assignment probability (use max over components)
        max_prob_pt = pp.max(axis=1)
    else:
        entropy_pt = None
        max_prob_pt = None

    stats: dict[int, dict[str, float]] = {}
    for lab in np.unique(cluster_pred):
        mask = cluster_pred == lab
        count = int(mask.sum())
        if count == 0:
            continue

        v_vals = velocity[mask]
        v_mean = float(np.mean(v_vals))
        v_std = float(np.std(v_vals))
        v_median = float(np.median(v_vals))
        v_nmad = float(np.median(np.abs(v_vals - v_median)) * 1.4826)

        x_vals = x_vals_all[mask]
        y_vals = y_vals_all[mask]
        x_mean = float(np.mean(x_vals))
        y_mean = float(np.mean(y_vals))
        x_std = float(np.std(x_vals))
        y_std = float(np.std(y_vals))

        avg_entropy = float(entropy_pt[mask].mean()) if entropy_pt is not None else None
        avg_assign = (
            float(max_prob_pt[mask].mean()) if max_prob_pt is not None else None
        )

        stats[int(lab)] = {
            "count": count,
            "x_mean": x_mean,
            "y_mean": y_mean,
            "x_std": x_std,
            "y_std": y_std,
            "velocity_mean": v_mean,
            "velocity_std": v_std,
            "velocity_median": v_median,
            "velocity_nmad": v_nmad,
            "avg_entropy": avg_entropy,
            "avg_assignment_prob": avg_assign,
        }

    return stats


def plot_1d_velocity_clustering_simple(
    df_features: pd.DataFrame,
    img: Image.Image | np.ndarray | None,
    *,
    cluster_pred: np.ndarray,
    posterior_probs: np.ndarray,
) -> tuple[Figure, np.ndarray, dict[int, dict[str, float]]]:
    """Plot 1D velocity clustering results for marginalized model.

    Args:
        df_features: DataFrame with columns ``x``, ``y``, ``u``, ``v``, and ``V``.
        img: Optional background image array.
        cluster_pred: Array of hard cluster assignments per point.
        posterior_probs: Array of responsibilities per point and cluster.

    Returns:
        Tuple[Figure, np.ndarray, Dict[int, Dict[str, float]]]:
            Matplotlib figure, uncertainty (entropy) per point, and the statistics
            dictionary returned by ``compute_cluster_statistics``.
    """

    # Per-point uncertainty (entropy) for scatter plot and return value
    entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-12), axis=1)
    # Max probs can be useful for debugging/annotation; compute if needed
    # max_probs = posterior_probs[np.arange(len(cluster_pred)), cluster_pred]

    unique_labels = np.unique(cluster_pred)
    noise_label = -1
    labels_no_noise = [lab for lab in sorted(unique_labels) if lab != noise_label]
    n = max(1, len(labels_no_noise))
    cmap = plt.cm.get_cmap("tab20", n)
    colors_arr = [cmap(i) for i in range(n)]
    color_map = {}
    for i, lab in enumerate(labels_no_noise):
        color_map[lab] = colors_arr[i]
    if noise_label in unique_labels:
        color_map[noise_label] = "#7f7f7f"

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
                c=[color_map[label]],
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
    ax3.set_xlabel("Velocity Magnitude", fontsize=12)
    ax3.set_ylabel("Density", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10, framealpha=0.9)

    # Render statistics box
    stats = compute_cluster_statistics_simple(
        df_features,
        cluster_pred,
        posterior_probs,
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
