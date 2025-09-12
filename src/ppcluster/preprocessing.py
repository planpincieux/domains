import logging

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ppcx")
RANDOM_SEED = 8927


def preproc_features(df):
    """
    Create additional features for clustering glacier displacement data
    """
    df_features = df.copy()

    # Direction angle (in radians and degrees)
    df_features["angle_rad"] = np.arctan2(df["v"], df["u"])
    df_features["angle_deg"] = np.degrees(df_features["angle_rad"])

    # Convert negative angles to positive (0-360 degrees)
    df_features["angle_deg"] = (df_features["angle_deg"] + 360) % 360

    # Directional components (unit vectors)
    df_features["u_unit"] = df["u"] / (df["V"] + 1e-10)  # Avoid division by zero
    df_features["v_unit"] = df["v"] / (df["V"] + 1e-10)

    # Square Magnitude of displacement
    df_features["V_squared"] = df["V"] ** 2

    # Log magnitude for better clustering of different scales
    df_features["log_magnitude"] = np.log1p(df["V"])

    # Magnitude categories
    magnitude_percentiles = np.percentile(df["V"], [25, 50, 75])
    df_features["magnitude_category"] = pd.cut(
        df["V"],
        bins=[0] + list(magnitude_percentiles) + [np.inf],
        labels=["low", "medium", "high", "very_high"],
    )

    return df_features


def normalize_data(
    df,
    spatial_weight=0.3,
    velocity_weight=0.7,
    spatial_features_names=None,
    velocity_features_names=None,
):
    """
    Custom clustering approach that considers both spatial proximity and velocity similarity

    Parameters:
    - spatial_weight: weight for spatial coordinates (x, y)
    - velocity_weight: weight for velocity features (u, v, magnitude, direction)
    """

    if spatial_features_names is None:
        spatial_features_names = ["x", "y"]
    if velocity_features_names is None:
        velocity_features_names = ["u", "v", "V", "angle_deg"]

    # Prepare features for clustering
    spatial_features = df[spatial_features_names].values
    velocity_features = df[velocity_features_names].values

    # Normalize features separately
    spatial_scaler = StandardScaler()
    velocity_scaler = StandardScaler()

    spatial_normalized = spatial_scaler.fit_transform(spatial_features)
    velocity_normalized = velocity_scaler.fit_transform(velocity_features)

    # Create a DataFrame with normalized features
    normalized_df = pd.DataFrame(
        np.hstack(
            [
                spatial_normalized * spatial_weight,
                velocity_normalized * velocity_weight,
            ]
        ),
        columns=spatial_features_names + velocity_features_names,
    )

    return normalized_df, spatial_scaler, velocity_scaler


def apply_dic_filters(
    df: pd.DataFrame,
    filter_outliers: bool | None = True,
    tails_percentile: float | None = 0.01,
    min_velocity: float | None = 0.0,
    apply_2d_median: bool | None = False,
    median_window_size: int | None = 5,
    median_threshold_factor: float | None = 3.0,
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
    logger.info(
        f"DIC filtering pipeline completed: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} total)"
    )

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


def create_2d_grid(df: pd.DataFrame, grid_spacing: float = None) -> tuple:
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
    velocity_column: str = "V",
) -> pd.DataFrame:
    """
    Apply 2D median filter to remove outliers based on local neighborhood.

    Args:
        df: DataFrame with DIC data containing 'x', 'y', 'u', 'v', 'V' columns
        window_size: Size of the median filter window (odd number recommended)
        threshold_factor: Factor for outlier detection (n * median_deviation)
        velocity_column: Column name for velocity magnitude

    Returns:
        Filtered DataFrame
    """
    logger.info(
        f"Applying 2D median filter: window_size={window_size}, threshold_factor={threshold_factor}"
    )

    # Create 2D grid from scattered points
    X, Y, u_grid, v_grid, v_mag_grid, valid_mask = create_2d_grid(df)

    # Apply median filter only to valid points
    v_mag_filtered = np.full_like(v_mag_grid, np.nan)

    # Create a copy for filtering
    v_mag_work = v_mag_grid.copy()
    v_mag_work[~valid_mask] = np.nan

    # Apply median filter
    v_mag_median = ndimage.median_filter(v_mag_work, size=window_size)

    # Calculate local median absolute deviation (MAD)
    # MAD = median(|x_i - median(x)|)
    mad_grid = np.abs(v_mag_work - v_mag_median)
    mad_median = ndimage.median_filter(mad_grid, size=window_size)

    # Create outlier mask
    outlier_threshold = threshold_factor * mad_median
    outlier_mask = np.abs(v_mag_work - v_mag_median) > outlier_threshold

    # Count outliers
    n_outliers = np.sum(outlier_mask & valid_mask)
    logger.info(f"Detected {n_outliers} outliers in 2D median filter")

    # Create list of valid point indices to keep
    keep_indices = []
    for idx, row in df.iterrows():
        # Find grid position
        i = np.argmin(np.abs(Y[:, 0] - row["y"]))
        j = np.argmin(np.abs(X[0, :] - row["x"]))

        # Check if this point should be kept
        if not outlier_mask[i, j]:
            keep_indices.append(idx)

    # Filter DataFrame
    df_filtered = df.iloc[keep_indices].reset_index(drop=True)
    logger.info(
        f"2D median filtering: {len(df)} -> {len(df_filtered)} points "
        f"(removed {len(df) - len(df_filtered)} outliers)"
    )

    return df_filtered


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

    elif method == "stratified":
        # Stratified sampling based on spatial grid
        x_bins = np.linspace(df["x"].min(), df["x"].max(), int(np.sqrt(len(df) / 100)))
        y_bins = np.linspace(df["y"].min(), df["y"].max(), int(np.sqrt(len(df) / 100)))

        df_temp = df.copy()
        df_temp["x_bin"] = pd.cut(df_temp["x"], x_bins, labels=False)
        df_temp["y_bin"] = pd.cut(df_temp["y"], y_bins, labels=False)

        # Sample from each spatial bin
        df_sub = (
            df_temp.groupby(["x_bin", "y_bin"])
            .apply(
                lambda x: x.sample(min(n_subsample, len(x)), random_state=RANDOM_SEED)
            )
            .reset_index(drop=True)
        )
        df_sub = df_sub.drop(["x_bin", "y_bin"], axis=1)

    print(
        f"Subsampled from {len(df)} to {len(df_sub)} points ({len(df_sub) / len(df) * 100:.1f}%)"
    )
    return df_sub
