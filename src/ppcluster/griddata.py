import logging
from typing import Literal, overload

import numpy as np
import pandas as pd

logger = logging.getLogger("ppcx")


def _infer_grid_spacing(x_vals: np.ndarray, y_vals: np.ndarray) -> float:
    """Infer a reasonable grid spacing from scattered points."""
    x_unique = np.unique(x_vals)
    y_unique = np.unique(y_vals)

    def _min_positive_diff(arr: np.ndarray) -> float:
        if arr.size < 2:
            return 1.0
        diffs = np.diff(arr)
        diffs = diffs[diffs > 0]
        return float(diffs.min()) if diffs.size else 1.0

    spacing = float(
        np.mean([_min_positive_diff(x_unique), _min_positive_diff(y_unique)])
    )
    return spacing if spacing > 0 else 1.0


@overload
def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
    *,
    return_axes: Literal[False] = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@overload
def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
    *,
    return_axes: Literal[True],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


def create_2d_grid(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray | None = None,
    grid_spacing: float | None = None,
    *,
    return_axes: bool = False,
) -> (
    tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
):
    if grid_spacing is None:
        grid_spacing = _infer_grid_spacing(np.asarray(x, float), np.asarray(y, float))
        logger.info(f"Estimated grid spacing: {grid_spacing:.2f}")

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_grid = np.arange(x_min, x_max + grid_spacing, grid_spacing)
    y_grid = np.arange(y_min, y_max + grid_spacing, grid_spacing)

    X, Y = np.meshgrid(x_grid, y_grid)
    label_grid = np.full(X.shape, np.nan)

    if labels is not None:
        labels_arr = np.asarray(labels)
        for xi, yi, lab in zip(x, y, labels_arr, strict=False):
            ix = np.argmin(np.abs(x_grid - xi))
            iy = np.argmin(np.abs(y_grid - yi))
            label_grid[iy, ix] = lab

    if return_axes:
        return X, Y, label_grid, x_grid, y_grid
    return X, Y, label_grid


def create_2d_grid_from_df(
    df: pd.DataFrame, grid_spacing: float | None = None
) -> tuple:
    """
    Create a 2D grid from scattered DIC points.

    Args:
        df: DataFrame with columns 'x', 'y', 'u', 'v', 'V'
        grid_spacing: Spacing between grid points. If None, estimated from data

    Returns:
        tuple: (x_grid, y_grid, u_grid, v_grid, v_mag_grid, valid_mask)
    """
    X, Y, _, x_grid, y_grid = create_2d_grid(
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        labels=None,
        grid_spacing=grid_spacing,
        return_axes=True,
    )

    u_grid = np.full(X.shape, np.nan, dtype=float)
    v_grid = np.full(X.shape, np.nan, dtype=float)
    v_mag_grid = np.full(X.shape, np.nan, dtype=float)

    for xi, yi, ui, vi, vmag in df[["x", "y", "u", "v", "V"]].itertuples(
        index=False, name=None
    ):
        ix = int(np.argmin(np.abs(x_grid - xi)))
        iy = int(np.argmin(np.abs(y_grid - yi)))
        u_grid[iy, ix] = ui
        v_grid[iy, ix] = vi
        v_mag_grid[iy, ix] = vmag

    valid_mask = ~np.isnan(v_mag_grid)
    logger.info(f"Created 2D grid: {X.shape}, {int(np.sum(valid_mask))} valid points")

    return X, Y, u_grid, v_grid, v_mag_grid, valid_mask


def map_grid_to_points(
    X,
    Y,
    label_grid,
    x_points,
    y_points,
    keep_nan=True,
    nan_fill=-1,
):
    """
    Map values from a 2D grid (label_grid) back to query points.

    Parameters:
    -----------
    X, Y : numpy.ndarray
        Meshgrid arrays returned by create_2d_grid (X.shape == label_grid.shape)
    label_grid : numpy.ndarray
        2D array with labels (np.nan = empty cells)
    x_points, y_points : numpy.ndarray
        1D arrays of query point coordinates
    keep_nan : bool, default=True
        If True, keep NaN points and fill with nan_fill value
        If False, filter out NaN points and return filtered coordinates
    nan_fill : int, default=-1
        Value to use when grid cell is NaN (only used if keep_nan=True)

    Returns:
    --------
        tuple : (labels, x, y)
            - label: Array of mapped labels (excluding NaN points if keep_nan=False)
            - x: Array of x coordinates (excluding NaN points if keep_nan=False)
            - y: Array of y coordinates (excluding NaN points if keep_nan=False)
    """
    # Ensure inputs are numpy arrays
    x_points = np.asarray(x_points)
    y_points = np.asarray(y_points)

    if len(x_points) != len(y_points):
        raise ValueError("x_points and y_points must have the same length")

    # Extract grid coordinates
    x_grid = X[0, :]
    y_grid = Y[:, 0]

    # Initialize output array
    n_points = len(x_points)
    labels = np.full(n_points, nan_fill, dtype=int)

    # Map each point to nearest grid cell
    for i, (xi, yi) in enumerate(zip(x_points, y_points, strict=False)):
        # Find nearest grid indices
        ix = np.argmin(np.abs(x_grid - xi))
        iy = np.argmin(np.abs(y_grid - yi))

        # Get grid value
        val = label_grid[iy, ix]
        labels[i] = nan_fill if np.isnan(val) else int(val)

    if not keep_nan:
        # Filter out NaN points
        valid_mask = labels != nan_fill
        labels = labels[valid_mask]
        x_points = x_points[valid_mask]
        y_points = y_points[valid_mask]

    return labels, x_points, y_points
