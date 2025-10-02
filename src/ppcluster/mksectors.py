import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from string import ascii_uppercase
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint, MultiPolygon, Polygon
from shapely.ops import unary_union

try:
    from skimage.draw import polygon as sk_polygon
except ImportError:
    sk_polygon = None

logger = logging.getLogger("ppcx")


Numeric = float | int
ImageShape = tuple[int, int] | tuple[int, int, int]
ArrayLike = Sequence[Numeric] | np.ndarray
LabelArray = Sequence[str] | np.ndarray


class PolygonDict(dict[str, np.ndarray]):
    def __init__(
        self,
        *args: Any,
        geometries: Mapping[str, Polygon] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.geometries: dict[str, Polygon] = dict(geometries or {})


def _chaikin(coords: np.ndarray, iterations: int) -> np.ndarray:
    pts = np.asarray(coords, dtype=float)
    if pts.shape[0] < 3 or iterations <= 0:
        return pts
    for _ in range(iterations):
        closed = np.vstack([pts, pts[0]])
        q0 = 0.75 * closed[:-1] + 0.25 * closed[1:]
        q1 = 0.25 * closed[:-1] + 0.75 * closed[1:]
        pts = np.empty((q0.shape[0] + q1.shape[0], 2), dtype=float)
        pts[0::2] = q0
        pts[1::2] = q1
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def _largest_polygon(geom: Polygon | MultiPolygon | None) -> Polygon | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    if geom.geom_type != "Polygon" or geom.area <= 0:
        return None
    return geom


def _base_polygon(
    points: np.ndarray, mode: Literal["boundary", "convex"]
) -> Polygon | None:
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return None
    hull = MultiPoint(pts).convex_hull
    return hull if hull.geom_type == "Polygon" else None


def compute_mk_sector_polygons(
    x: ArrayLike,
    y: ArrayLike,
    mk_label_str: LabelArray,
    *,
    smooth_iters: int = 2,
    prevent_overlap: bool = True,
    containment_strategy: Literal["difference", "keep", "none"] = "difference",
    polygon_mode: Literal["boundary", "convex"] = "boundary",
) -> PolygonDict:
    """Build smoothed sector polygons with optional containment/overlap handling."""
    strategy = containment_strategy.lower()
    if strategy not in {"difference", "keep", "none"}:
        raise ValueError(
            "containment_strategy must be 'difference', 'keep', or 'none'."
        )
    mode = polygon_mode.lower()
    if mode not in {"boundary", "convex"}:
        raise ValueError("polygon_mode must be 'boundary' or 'convex'.")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    labels = np.asarray(mk_label_str, dtype=object)

    unique_labels = [
        str(lab) for lab in np.unique(labels) if isinstance(lab, str) and lab
    ]

    polygons = PolygonDict()
    geometries: dict[str, Polygon] = {}

    for lab in unique_labels:
        mask = labels == lab
        pts = np.column_stack((x_arr[mask], y_arr[mask]))
        base_geom = _base_polygon(pts, mode)
        if base_geom is None or base_geom.area <= 0:
            continue

        coords = np.asarray(base_geom.exterior.coords)[:-1]
        coords = _chaikin(coords, max(smooth_iters, 0))

        poly_shape = _largest_polygon(Polygon(coords).buffer(0))
        if poly_shape is None:
            continue

        containing: list[str] = []
        contained: list[str] = []
        if strategy in {"difference", "keep"}:
            for prev_lab, prev_shape in geometries.items():
                if prev_shape.contains(poly_shape):
                    containing.append(prev_lab)
                if poly_shape.contains(prev_shape):
                    contained.append(prev_lab)

        if strategy == "difference":
            for prev_lab in containing:
                updated = _largest_polygon(geometries[prev_lab].difference(poly_shape))
                if updated is None:
                    geometries.pop(prev_lab, None)
                    polygons.pop(prev_lab, None)
                else:
                    geometries[prev_lab] = updated
                    polygons[prev_lab] = np.asarray(updated.exterior.coords)[:-1]
            if contained:
                subtract_union = unary_union([geometries[name] for name in contained])
                poly_shape = _largest_polygon(poly_shape.difference(subtract_union))
                if poly_shape is None:
                    continue

        if prevent_overlap and geometries:
            excluded: set[str] = set()
            if strategy == "keep":
                excluded.update(containing)
                excluded.update(contained)
            overlap_geoms = [
                shape for name, shape in geometries.items() if name not in excluded
            ]
            if overlap_geoms:
                poly_shape = _largest_polygon(
                    poly_shape.difference(unary_union(overlap_geoms))
                )
                if poly_shape is None:
                    continue

        polygons[lab] = np.asarray(poly_shape.exterior.coords)[:-1]
        geometries[lab] = poly_shape

    polygons.geometries = geometries
    return polygons


def auto_assign_mk_sectors(
    x: ArrayLike,
    y: ArrayLike,
    kin_cluster: ArrayLike,
    ordered_clusters_ids: Sequence[int],
    *,
    overlap_threshold: float = 0.6,
    convex_kwargs: Mapping[str, Any] | None = None,
    major_polygon_kwargs: Mapping[str, Any] | None = None,
    minor_polygon_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Automatically assign major/minor MK sectors based on polygon overlap."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    clusters = np.asarray(kin_cluster, dtype=int)
    ordered_clusters = [int(c) for c in ordered_clusters_ids]

    cluster_labels_str = clusters.astype(str)
    convex_opts: dict[str, Any] = dict(
        smooth_iters=0,
        prevent_overlap=False,
        containment_strategy="none",
        polygon_mode="convex",
    )
    if convex_kwargs:
        convex_opts.update(convex_kwargs)

    raw_polygons = compute_mk_sector_polygons(
        x_arr,
        y_arr,
        cluster_labels_str,
        **convex_opts,
    )

    cluster_shapes: dict[int, Polygon] = {}
    for lab_str, shape in raw_polygons.geometries.items():
        try:
            cluster_id = int(float(lab_str))
        except ValueError:
            continue
        if shape.is_empty or shape.area <= 0:
            continue
        cluster_shapes[cluster_id] = shape

    if not cluster_shapes:
        raise RuntimeError(
            "Unable to build polygons for any cluster; aborting assignment."
        )

    major_candidates = set(cluster_shapes.keys())
    minor_parent: dict[int, tuple[int, float]] = {}

    for child_id, child_poly in cluster_shapes.items():
        if child_poly.area <= 0:
            continue
        best_parent: int | None = None
        best_overlap = 0.0
        for parent_id, parent_poly in cluster_shapes.items():
            if parent_id == child_id or parent_poly.area <= child_poly.area:
                continue
            if not parent_poly.intersects(child_poly):
                continue
            overlap_ratio = parent_poly.intersection(child_poly).area / child_poly.area
            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_parent = parent_id
        if best_parent is not None and best_overlap >= overlap_threshold:
            minor_parent[child_id] = (best_parent, best_overlap)
            major_candidates.discard(child_id)

    ordered_major_clusters = [
        cid for cid in ordered_clusters if cid in major_candidates
    ]
    if not ordered_major_clusters:
        ordered_major_clusters = [
            cid for cid in ordered_clusters if cid in cluster_shapes
        ]
        major_candidates = set(ordered_major_clusters)
        minor_parent.clear()

    mk_label_str = np.full_like(clusters, "", dtype=object)
    mk_label_id = -1 * np.ones_like(clusters, dtype=int)

    label_to_index: dict[str, int] = {}
    major_label_map: dict[int, str] = {}
    cluster_to_label: dict[int, str] = {}
    minor_labels_map: dict[int, str] = {}
    minor_counts: defaultdict[str, int] = defaultdict(int)

    def assign_label(cluster_id: int, label: str) -> None:
        idx = label_to_index.setdefault(label, len(label_to_index))
        mask = clusters == cluster_id
        mk_label_str[mask] = label
        mk_label_id[mask] = idx
        cluster_to_label[cluster_id] = label

    for idx, cluster_id in enumerate(ordered_major_clusters):
        letter = ascii_uppercase[idx] if idx < len(ascii_uppercase) else f"S{idx}"
        major_label_map[cluster_id] = letter
        assign_label(cluster_id, letter)

    for cluster_id in ordered_clusters:
        if cluster_id not in minor_parent:
            continue
        parent_id, overlap_ratio = minor_parent[cluster_id]
        parent_label = major_label_map.get(parent_id)
        if parent_label is None:
            logger.warning(
                "Cluster %d expected as minor but parent %d unavailable; promoting to major.",
                cluster_id,
                parent_id,
            )
            letter = (
                ascii_uppercase[len(major_label_map)]
                if len(major_label_map) < len(ascii_uppercase)
                else f"S{len(major_label_map)}"
            )
            major_label_map[cluster_id] = letter
            assign_label(cluster_id, letter)
            major_candidates.add(cluster_id)
            continue
        minor_counts[parent_label] += 1
        label = f"{parent_label}{minor_counts[parent_label]}"
        assign_label(cluster_id, label)
        minor_labels_map[cluster_id] = label
        logger.info(
            "Cluster %d assigned to %s (minor of %s, overlap %.2f)",
            cluster_id,
            label,
            parent_label,
            overlap_ratio,
        )

    for cluster_id, label in major_label_map.items():
        logger.info("Cluster %d assigned to %s (major)", cluster_id, label)

    major_mask = np.isin(clusters, list(major_label_map.keys()))
    major_opts: dict[str, Any] = dict(
        smooth_iters=4,
        prevent_overlap=True,
        containment_strategy="difference",
    )
    if major_polygon_kwargs:
        major_opts.update(major_polygon_kwargs)
    polygons_major_dict: dict[str, np.ndarray] = {}
    if major_mask.any():
        polygons_major = compute_mk_sector_polygons(
            x_arr[major_mask],
            y_arr[major_mask],
            mk_label_str[major_mask],
            **major_opts,
        )
        polygons_major_dict = dict(polygons_major)

    minor_clusters = [cid for cid in ordered_clusters if cid in minor_parent]
    polygons_minor_dict: dict[str, np.ndarray] = {}
    if minor_clusters:
        minor_mask = np.isin(clusters, minor_clusters)
        minor_opts: dict[str, Any] = dict(
            smooth_iters=2,
            prevent_overlap=False,
            containment_strategy="none",
            polygon_mode="convex",
        )
        if minor_polygon_kwargs:
            minor_opts.update(minor_polygon_kwargs)
        polygons_minor = compute_mk_sector_polygons(
            x_arr[minor_mask],
            y_arr[minor_mask],
            mk_label_str[minor_mask],
            **minor_opts,
        )
        polygons_minor_dict = dict(polygons_minor)

    return {
        "mk_label_str": mk_label_str,
        "mk_label_id": mk_label_id,
        "major_clusters": ordered_major_clusters,
        "major_label_map": major_label_map,
        "minor_parent": minor_parent,
        "label_to_index": label_to_index,
        "cluster_to_label": cluster_to_label,
        "minor_labels_map": minor_labels_map,
        "raw_polygons": dict(raw_polygons),
        "raw_geometries": raw_polygons.geometries,
        "polygons_major": polygons_major_dict,
        "polygons_minor": polygons_minor_dict,
    }


def draw_polygon(
    ax_draw: plt.Axes,
    poly_coords: np.ndarray | None,
    label_text: str,
    color_hex: str,
    *,
    fill_alpha: float = 0.1,
    zorder: int = 1,
) -> None:
    if poly_coords is None or len(poly_coords) < 3:
        return
    ax_draw.fill(
        poly_coords[:, 0],
        poly_coords[:, 1],
        color=color_hex,
        alpha=fill_alpha,
        lw=0,
        zorder=zorder,
    )
    ax_draw.plot(
        np.r_[poly_coords[:, 0], poly_coords[0, 0]],
        np.r_[poly_coords[:, 1], poly_coords[0, 1]],
        color=color_hex,
        lw=2,
        zorder=zorder + 1,
    )
    cx, cy = np.mean(poly_coords, axis=0)
    ax_draw.text(
        cx,
        cy,
        label_text,
        color="k",
        fontsize=10,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        zorder=zorder + 2,
    )


def compute_mk_sector_stats(
    polygons: Mapping[str, np.ndarray],
    mk_label_str: LabelArray,
    *,
    x: ArrayLike,
    y: ArrayLike,
    v: ArrayLike | None = None,
    img_shape: ImageShape | None = None,
    rasterize: bool = False,
) -> pd.DataFrame:
    """Compute descriptive statistics for MK sectors."""
    labels = np.asarray(mk_label_str, dtype=object)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    v_arr = np.asarray(v, dtype=float) if v is not None else None
    stats_rows: list[dict[str, Any]] = []
    geometries: Mapping[str, Polygon] = getattr(polygons, "geometries", {})

    for lab, coords in polygons.items():
        if coords is None:
            continue
        poly_coords = np.asarray(coords, dtype=float)
        if poly_coords.ndim != 2 or poly_coords.shape[0] < 3:
            continue

        geometry = geometries.get(lab)
        if geometry is None:
            geometry = _largest_polygon(Polygon(poly_coords).buffer(0))
        else:
            geometry = _largest_polygon(geometry)

        if geometry is None or geometry.area <= 0:
            continue

        mask = labels == lab
        n_points = int(np.count_nonzero(mask))
        area = float(geometry.area)
        perimeter = float(geometry.length)
        centroid_x, centroid_y = (float(coord) for coord in geometry.centroid.coords[0])
        compactness = (
            float((4.0 * np.pi * area) / (perimeter**2 + 1e-12))
            if perimeter > 0
            else np.nan
        )
        point_density = float(n_points / area) if area > 0 else np.nan

        row: dict[str, Any] = {
            "label": lab,
            "n_points": n_points,
            "area_px2": area,
            "perimeter_px": perimeter,
            "compactness": compactness,
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "pixel_count": np.nan,
            "point_density_pts_per_px2": point_density,
            "v_mean": np.nan,
            "v_std": np.nan,
            "v_median": np.nan,
            "v_min": np.nan,
            "v_max": np.nan,
        }

        if v_arr is not None and n_points > 0:
            v_sel = v_arr[mask]
            row.update(
                v_mean=float(np.mean(v_sel)),
                v_std=float(np.std(v_sel)),
                v_median=float(np.median(v_sel)),
                v_min=float(np.min(v_sel)),
                v_max=float(np.max(v_sel)),
            )

        if rasterize and sk_polygon is not None and img_shape is not None:
            height, width = img_shape[:2]
            rr, cc = sk_polygon(
                poly_coords[:, 1], poly_coords[:, 0], shape=(height, width)
            )
            row["pixel_count"] = int(rr.size)

        stats_rows.append(row)

    columns = [
        "label",
        "n_points",
        "area_px2",
        "perimeter_px",
        "compactness",
        "centroid_x",
        "centroid_y",
        "pixel_count",
        "point_density_pts_per_px2",
        "v_mean",
        "v_std",
        "v_median",
        "v_min",
        "v_max",
    ]
    df = pd.DataFrame(stats_rows, columns=columns)
    return df.sort_values("label").reset_index(drop=True) if not df.empty else df
