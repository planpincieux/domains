import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from scipy import ndimage
from scipy.spatial import cKDTree

from ppcluster.cvat import _parse_polygon_points, read_mask_element_from_cvat

logger = logging.getLogger("ppcx")
RANDOM_SEED = 8927


def apply_dic_filters(
    df: pd.DataFrame,
    filter_outliers: bool | None = True,
    tails_percentile: float | None = 0.01,
    min_velocity: float | None = 0.0,
    apply_2d_median: bool | None = False,
    median_window_size: int | None = 5,
    median_threshold_factor: float | None = 3.0,
    apply_2d_gaussian: bool | None = False,
    gaussian_sigma: float | None = 1.0,
) -> pd.DataFrame:
    """
    Apply all DIC data filters in sequence.

    Args:
        df: Raw DIC DataFrame
        filter_outliers: Whether to apply percentile-based outlier filtering
        tails_percentile: Percentile for tail filtering
        min_velocity: Minimum velocity threshold
        apply_2d_median: Whether to apply 2D median filter
        median_window_size: Window size for median filter
        median_threshold_factor: Threshold factor for median filter

    Returns:
        Filtered DataFrame
    """

    logger.info(f"Starting DIC filtering pipeline with {len(df)} points")
    df_filtered = df.copy()

    # 1. Apply percentile-based outlier filtering
    if filter_outliers and tails_percentile is not None and tails_percentile > 0:
        df_filtered = filter_outliers_by_percentile(
            df_filtered, tails_percentile=tails_percentile
        )

    # 2. Apply minimum velocity filtering
    if min_velocity is not None and min_velocity >= 0:
        df_filtered = filter_by_min_velocity(df_filtered, min_velocity=min_velocity)

    # 3. Apply 2D median filter
    if apply_2d_median:
        df_filtered = apply_2d_median_filter(
            df_filtered,
            window_size=median_window_size,
            threshold_factor=median_threshold_factor,
        )

    # 4. Apply 2D Gaussian smoothing to velocity magnitude
    if apply_2d_gaussian:
        df_filtered = apply_2d_gaussian_filter(df_filtered, sigma=gaussian_sigma)

    logger.info(
        f"DIC filtering pipeline completed: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} total)"
    )

    return df_filtered


def filter_dataframe_by_masks(
    xml_source: str | Path,
    df: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    exclude_labels: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Filter dataframe using polygonal <mask> annotations from a CVAT XML.

    - If no masks are present in the CVAT file, the original dataframe is returned unchanged.
    - If masks are present, points are kept if they fall in the union of mask polygons.
    - exclude_labels can be used to ignore masks by label.
    """
    raise NotImplementedError("This function not yet tested/verified")

    exclude = set(exclude_labels or ())
    masks = read_mask_element_from_cvat(
        xml_source, image_name=None, exclude_labels=exclude
    )
    if not masks:
        logger.info(
            "No CVAT masks found at %s — skipping CVAT mask filtering", xml_source
        )
        return df

    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()
    pts_xy = np.column_stack((x_vals, y_vals))
    include_mask = np.zeros(len(df), dtype=bool)

    for mask_info in masks:
        label = mask_info.get("label", "")
        pts_str = mask_info.get("points", None)
        if not pts_str:
            logger.debug("Mask '%s' has no 'points' attribute; skipping", label)
            continue
        try:
            verts = _parse_polygon_points(pts_str)  # (N,2)
            if verts.shape[0] < 3:
                logger.debug("Mask '%s' has fewer than 3 vertices; skipping", label)
                continue
            # ensure closed polygon for MplPath.contains_points behaviour
            if not np.allclose(verts[0], verts[-1]):
                verts = np.vstack([verts, verts[0]])
            path = MplPath(verts)
            hit = path.contains_points(pts_xy)
            include_mask |= hit
            logger.info("Mask '%s': %d points inside", label, int(hit.sum()))
        except Exception as exc:
            logger.warning("Failed to parse/apply mask '%s': %s", label, exc)
            continue

    df_filtered = df.loc[include_mask].reset_index(drop=True)
    logger.info("Data shape after CVAT mask filtering: %s", df_filtered.shape)
    return df_filtered


def filter_outliers_by_percentile(
    df: pd.DataFrame, tails_percentile: float = 0.01, velocity_column: str = "V"
) -> pd.DataFrame:
    """
    Filter out extreme tails based on the specified percentile.

    Args:
        df: DataFrame with DIC data
        tails_percentile: Percentile for tail filtering (e.g., 0.01 removes bottom 1% and top 1%)
        velocity_column: Column name for velocity magnitude

    Returns:
        Filtered DataFrame
    """
    prob_threshold = (tails_percentile, 1 - tails_percentile)
    velocity_percentiles = df[velocity_column].quantile(prob_threshold).values

    df_filtered = df[
        (df[velocity_column] >= velocity_percentiles[0])
        & (df[velocity_column] <= velocity_percentiles[1])
    ].reset_index(drop=True)

    logger.info(
        f"Percentile filtering: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} outliers)"
    )

    return df_filtered


def filter_by_min_velocity(
    df: pd.DataFrame, min_velocity: float, velocity_column: str = "V"
) -> pd.DataFrame:
    """
    Filter out low velocity vectors if specified.

    Args:
        df: DataFrame with DIC data
        min_velocity: Minimum velocity threshold
        velocity_column: Column name for velocity magnitude

    Returns:
        Filtered DataFrame
    """
    if min_velocity < 0:
        logger.info("Minimum velocity filtering disabled")
        return df

    df_filtered = df[df[velocity_column] >= min_velocity].reset_index(drop=True)

    logger.info(
        f"Min velocity filtering: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} points below {min_velocity})"
    )

    return df_filtered


def create_2d_grid(df: pd.DataFrame, grid_spacing: float | None = None) -> tuple:
    """
    Create a 2D grid from scattered DIC points.

    Args:
        df: DataFrame with columns 'x', 'y', 'u', 'v', 'V'
        grid_spacing: Spacing between grid points. If None, estimated from data

    Returns:
        tuple: (x_grid, y_grid, u_grid, v_grid, v_mag_grid, valid_mask)
    """
    if grid_spacing is None:
        # Estimate grid spacing from minimum distances
        points = df[["x", "y"]].values
        tree = cKDTree(points)
        distances, _ = tree.query(
            points, k=2
        )  # k=2 to get distance to nearest neighbor
        grid_spacing = np.median(
            distances[:, 1]
        )  # distances[:, 1] is distance to nearest neighbor
        logger.info(f"Estimated grid spacing: {grid_spacing:.2f}")

    # Create regular grid
    x_min, x_max = df["x"].min(), df["x"].max()
    y_min, y_max = df["y"].min(), df["y"].max()

    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    # Create meshgrid
    X, Y = np.meshgrid(x_grid, y_grid)

    # Initialize grids
    u_grid = np.full_like(X, np.nan)
    v_grid = np.full_like(Y, np.nan)
    v_mag_grid = np.full_like(X, np.nan)

    # Map points to grid
    for _, row in df.iterrows():
        i = np.argmin(np.abs(y_grid - row["y"]))
        j = np.argmin(np.abs(x_grid - row["x"]))

        u_grid[i, j] = row["u"]
        v_grid[i, j] = row["v"]
        v_mag_grid[i, j] = row["V"]

    # Create valid mask
    valid_mask = ~np.isnan(v_mag_grid)

    logger.info(f"Created 2D grid: {X.shape}, {np.sum(valid_mask)} valid points")

    return X, Y, u_grid, v_grid, v_mag_grid, valid_mask


def apply_2d_median_filter(
    df: pd.DataFrame,
    window_size: int | None = 5,
    threshold_factor: float | None = 3.0,
) -> pd.DataFrame:
    """
    Apply 2D median filter to remove outliers based on local neighborhood.
    Now evaluates u, v and V components. A point is removed if any component
    is an outlier relative to its local median +/- threshold_factor * MAD.
    """
    logger.info(
        f"Applying 2D median filter (u,v,V): window_size={window_size}, threshold_factor={threshold_factor}"
    )

    # Create 2D grid from scattered points
    X, Y, u_grid, v_grid, v_mag_grid, valid_mask = create_2d_grid(df)

    # Prepare work arrays with NaNs where invalid
    u_work = u_grid.copy().astype(float)
    v_work = v_grid.copy().astype(float)
    V_work = v_mag_grid.copy().astype(float)

    u_work[~valid_mask] = np.nan
    v_work[~valid_mask] = np.nan
    V_work[~valid_mask] = np.nan

    size = int(window_size) if window_size is not None else 5

    # Helper to compute nanmedian with generic_filter
    nanmedian = lambda arr: float(np.nanmedian(arr))

    u_median = ndimage.generic_filter(
        u_work, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    v_median = ndimage.generic_filter(
        v_work, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    V_median = ndimage.generic_filter(
        V_work, function=nanmedian, size=size, mode="constant", cval=np.nan
    )

    # compute local MAD (median of absolute deviations)
    u_abs_dev = np.abs(u_work - u_median)
    v_abs_dev = np.abs(v_work - v_median)
    V_abs_dev = np.abs(V_work - V_median)

    u_mad = ndimage.generic_filter(
        u_abs_dev, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    v_mad = ndimage.generic_filter(
        v_abs_dev, function=nanmedian, size=size, mode="constant", cval=np.nan
    )
    V_mad = ndimage.generic_filter(
        V_abs_dev, function=nanmedian, size=size, mode="constant", cval=np.nan
    )

    # thresholding (avoid zero MAD by adding tiny eps)
    eps = 1e-12
    outlier_u = (
        np.abs(u_work - u_median) > (threshold_factor * (u_mad + eps))
    ) & valid_mask
    outlier_v = (
        np.abs(v_work - v_median) > (threshold_factor * (v_mad + eps))
    ) & valid_mask
    outlier_V = (
        np.abs(V_work - V_median) > (threshold_factor * (V_mad + eps))
    ) & valid_mask

    # combined outlier: any component flagged
    outlier_mask = outlier_u | outlier_v | outlier_V

    n_outliers = int(np.sum(outlier_mask))
    logger.info(f"Detected {n_outliers} outliers (u/v/V combined) in 2D median filter")

    # Build list of valid indices to keep (not an outlier)
    keep_indices = []
    for idx, row in df.iterrows():
        i = np.argmin(np.abs(Y[:, 0] - row["y"]))
        j = np.argmin(np.abs(X[0, :] - row["x"]))
        if not outlier_mask[i, j]:
            keep_indices.append(idx)

    df_filtered = df.iloc[keep_indices].reset_index(drop=True)
    logger.info(
        f"2D median filtering (u,v,V): {len(df)} -> {len(df_filtered)} points (removed {len(df) - len(df_filtered)})"
    )
    return df_filtered


def apply_2d_gaussian_filter(
    df: pd.DataFrame, sigma: float | None = 1.0
) -> pd.DataFrame:
    """
    Smooth u, v and velocity magnitude with a 2D Gaussian filter.

    - Uses mask-normalized gaussian filtering to respect missing cells.
    - Maps smoothed grid values back to original points.
    """
    sigma = sigma or 1.0
    logger.info(f"Applying 2D Gaussian filter (u,v,V): sigma={sigma}")

    # Build grid from points
    X, Y, u_grid, v_grid, v_mag_grid, valid_mask = create_2d_grid(df)

    if not np.any(valid_mask):
        logger.warning(
            "apply_2d_gaussian_filter: no valid grid cells found, returning original df"
        )
        return df

    # Prepare arrays for convolution: replace NaN with 0 and use validity mask
    u_work = np.where(valid_mask, u_grid, 0.0)
    v_work = np.where(valid_mask, v_grid, 0.0)
    V_work = np.where(valid_mask, v_mag_grid, 0.0)
    mask_work = valid_mask.astype(float)

    # Smooth numerator and denominator for each component
    num_u = ndimage.gaussian_filter(u_work, sigma=sigma, mode="constant", cval=0.0)
    num_v = ndimage.gaussian_filter(v_work, sigma=sigma, mode="constant", cval=0.0)
    num_V = ndimage.gaussian_filter(V_work, sigma=sigma, mode="constant", cval=0.0)
    den = ndimage.gaussian_filter(mask_work, sigma=sigma, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        u_smoothed = num_u / den
        v_smoothed = num_v / den
        V_smoothed = num_V / den

    u_smoothed[den == 0] = np.nan
    v_smoothed[den == 0] = np.nan
    V_smoothed[den == 0] = np.nan

    # Map smoothed grid back to points
    df_out = df.copy()
    sm_u = []
    sm_v = []
    sm_V = []
    x_grid = X[0, :]
    y_grid = Y[:, 0]
    for _, row in df_out.iterrows():
        i = np.argmin(np.abs(y_grid - row["y"]))
        j = np.argmin(np.abs(x_grid - row["x"]))
        sm_u.append(u_smoothed[i, j])
        sm_v.append(v_smoothed[i, j])
        sm_V.append(V_smoothed[i, j])

    df_out["u"] = sm_u
    df_out["v"] = sm_v
    df_out["V"] = sm_V

    n_none = int(
        np.isnan(df_out["V"]).sum()
        + np.isnan(df_out["u"]).sum()
        + np.isnan(df_out["v"]).sum()
    )
    logger.info(
        f"apply_2d_gaussian_filter: smoothed values assigned, NaN mapped points (total components)={n_none}"
    )
    return df_out


# === SPATIAL SUBSAMPLING ===
def spatial_subsample(df, n_subsample=5, method="regular"):
    """
    Subsample DIC data spatially to reduce computational load.

    Parameters:
    - n_subsample: Take every nth point (for regular) or fraction (for random)
    - method: 'regular', 'random', or 'stratified'
    """
    if method == "regular":
        # Take every nth point in spatial order
        df_sorted = df.sort_values(["x", "y"]).reset_index(drop=True)
        subsample_idx = np.arange(0, len(df_sorted), n_subsample)
        df_sub = df_sorted.iloc[subsample_idx].copy()

    elif method == "random":
        # Random sampling (n_subsample as fraction)
        n_samples = (
            int(len(df) * n_subsample)
            if n_subsample < 1
            else int(len(df) / n_subsample)
        )
        df_sub = df.sample(n=min(n_samples, len(df)), random_state=RANDOM_SEED).copy()

    else:
        raise ValueError(f"Unknown subsampling method: {method}")

    print(
        f"Subsampled from {len(df)} to {len(df_sub)} points ({len(df_sub) / len(df) * 100:.1f}%)"
    )
    return df_sub
