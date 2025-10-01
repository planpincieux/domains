from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.path import Path as MplPath

logger = logging.getLogger("ppcx")


@dataclass
class Polygon:
    """Light wrapper exposing contains_points(x, y)"""

    name: str
    path: MplPath

    def contains_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pts = np.column_stack((x, y))
        return self.path.contains_points(pts)

    def bounds(self) -> tuple[float, float, float, float]:
        verts = np.asarray(self.path.vertices)
        xmin, ymin = verts.min(axis=0)
        xmax, ymax = verts.max(axis=0)
        return float(xmin), float(ymin), float(xmax), float(ymax)

    def plot(self, ax=None, close_polygon: bool = True, **plot_kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        verts = np.asarray(self.path.vertices)
        if close_polygon and not np.allclose(verts[0], verts[-1]):
            verts = np.vstack([verts, verts[0]])
        ax.plot(verts[:, 0], verts[:, 1], **plot_kwargs)
        return ax

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "vertices": np.array(self.path.vertices).tolist()}


def read_cvat_xml(xml_source: str | Path) -> ET.Element:
    """
    Read CVAT XML from a file path or from a raw XML string and return the ElementTree root.
    """
    text = None
    try:
        if hasattr(xml_source, "__fspath__") or Path(xml_source).exists():
            with open(xml_source, encoding="utf-8") as fh:
                text = fh.read()
        else:
            text = str(xml_source)
    except Exception:
        text = str(xml_source)
    return ET.fromstring(text)


def load_cvat_annotations(xml_source: str | Path) -> dict:
    """
    General CVAT reader that extracts common annotation types per image.

    Returns:
      { image_name: [ (ann_type, label, data, z_order), ... ], ... }

    ann_type is one of: 'polygon', 'polyline', 'points', 'box', 'mask', 'other'.
    - For polygon/polyline/points: data is an (N,2) ndarray from the 'points' attribute.
    - For box: data is dict {'xtl', 'ytl', 'xbr', 'ybr'} (float).
    - For mask: data is the dict produced by parse_mask_element.
    - For unknown types: data is a dict of attributes.
    """

    def _parse_polygon_points(points_str: str) -> np.ndarray:
        """Parse CVAT polygon points string "x1,y1;x2,y2;..." -> (N,2) float ndarray."""
        pts = []
        for p in points_str.split(";"):
            p = p.strip()
            if not p:
                continue
            x_str, y_str = p.split(",")
            pts.append((float(x_str), float(y_str)))
        return np.asarray(pts, dtype=float)

    root = read_cvat_xml(xml_source)
    out: dict = {}
    for img in root.findall(".//image"):
        name = img.get("name")
        items = []
        for element in img:
            tag = element.tag.lower()
            z = int(element.get("z_order", "0") or 0)
            label = element.get("label", "") or ""
            if tag in ("polygon", "polyline", "points"):
                pts_str = element.get("points", "") or ""
                if pts_str:
                    pts_arr = _parse_polygon_points(pts_str)
                else:
                    pts_arr = np.empty((0, 2), dtype=float)
                items.append(("polygon", label, pts_arr, z))
            elif tag == "box":
                try:
                    box = {
                        "xtl": float(element.get("xtl", 0.0)),
                        "ytl": float(element.get("ytl", 0.0)),
                        "xbr": float(element.get("xbr", 0.0)),
                        "ybr": float(element.get("ybr", 0.0)),
                    }
                except Exception:
                    box = {k: element.get(k) for k in ("xtl", "ytl", "xbr", "ybr")}
                items.append(("box", label, box, z))
            elif tag == "mask":
                mask_info = parse_mask_element(element)
                items.append(("mask", mask_info.get("label", label), mask_info, z))
            else:
                # fallback: keep attributes (and text if present)
                data = dict(element.items())
                text = (
                    element.text.strip()
                    if element.text and element.text.strip()
                    else None
                )
                if text is not None:
                    data["_text"] = text
                items.append(("other", label, data, z))
        out[name] = items
    return out


def read_polygons_from_cvat(
    xml_source: str | Path,
    image_name: str | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> dict[str, Polygon]:
    """
    Parse polygons from a CVAT export (using load_cvat_annotations) and return a list of Polygon objects.

    - If image_name is provided, only polygons for that image are returned.
    - exclude_labels: optional sequence of label names to ignore.
    - polygons are sorted by z_order to preserve annotation stacking order.
    """
    exclude = set(exclude_labels or ())
    parsed = load_cvat_annotations(xml_source)
    if not parsed:
        return {}

    if image_name is None:
        try:
            image_name = next(iter(parsed.keys()))
        except StopIteration:
            return {}

    items = parsed.get(image_name, [])
    # keep only polygon annotations: items are tuples (ann_type, label, data, z)
    poly_items = [it for it in items if it[0] == "polygon"]
    # sort by z-order (4th element)
    items_sorted = sorted(poly_items, key=lambda it: int(it[3]) if len(it) > 3 else 0)

    polygons: dict[str, Polygon] = {}
    for _ann_type, label, pts_arr, _z in items_sorted:
        if label in exclude:
            logger.debug("Skipping excluded label: %s", label)
            continue
        if pts_arr is None or getattr(pts_arr, "size", 0) == 0:
            continue
        verts = np.vstack([pts_arr, pts_arr[0]])
        codes = (
            [MplPath.MOVETO]
            + [MplPath.LINETO] * (len(pts_arr) - 1)
            + [MplPath.CLOSEPOLY]
        )
        path = MplPath(verts, codes)
        polygons[label or "unnamed"] = Polygon(name=label or "unnamed", path=path)

    return polygons


def filter_dataframe_by_polygons(
    df: pd.DataFrame,
    polygons: dict[str, Polygon] | Polygon | None,
    x_col: str = "x",
    y_col: str = "y",
    polygon_names: Iterable[str] | None = None,
    invert: bool = False,
    return_mask: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """
    Filter a DIC dataframe keeping only points inside (or outside if invert=True)
    one or more Polygon objects produced by read_spatial_priors_from_cvat().

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing point coordinates.
    polygons : dict[str, Polygon] | Polygon | None
        Polygons dictionary (name -> Polygon) or a single Polygon instance.
        If None the dataframe is returned unchanged (or inverted if invert=True).
    x_col, y_col : str
        Column names in df with x and y coordinates.
    polygon_names : iterable of str, optional
        If `polygons` is a dict, restrict to the given polygon keys (union).
        If None use all polygons in the dict.
    invert : bool
        If True return points outside the selected polygons.
    return_mask : bool
        If True return tuple (filtered_df, boolean_mask) where mask is aligned with df.

    Returns
    -------
    pandas.DataFrame
        Filtered dataframe (and mask if return_mask=True).
    """
    if polygons is None:
        mask = np.ones(len(df), dtype=bool)
        if invert:
            mask = ~mask
        out_df = df[mask]
        return (out_df, mask) if return_mask else out_df

    # extract coordinate arrays (handle possible missing columns)
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(
            f"Coordinates columns '{x_col}' and/or '{y_col}' not found in dataframe"
        )

    x_arr = df[x_col].to_numpy()
    y_arr = df[y_col].to_numpy()

    # build combined mask
    combined_mask = np.zeros_like(x_arr, dtype=bool)

    if isinstance(polygons, Polygon):
        combined_mask = polygons.contains_points(x_arr, y_arr)
    elif isinstance(polygons, dict):
        keys = list(polygons.keys())
        if polygon_names is not None:
            # ensure provided names exist
            sel = [k for k in polygon_names]
            missing = [k for k in sel if k not in polygons]
            if missing:
                raise KeyError(f"Requested polygon_names not found: {missing}")
            keys = sel
        for k in keys:
            poly = polygons[k]
            combined_mask |= poly.contains_points(x_arr, y_arr)
    else:
        raise TypeError("polygons must be a dict[str, Polygon], a Polygon, or None")

    if invert:
        combined_mask = ~combined_mask

    filtered_df = df[combined_mask]

    return (filtered_df, combined_mask) if return_mask else filtered_df


## -- Mask -- ##


def parse_mask_element(mask_el: ET.Element) -> dict:
    """
    Parse a single <mask> element from CVAT XML into a dictionary.
    Backwards-compatible helper used by load_cvat_annotations.
    """
    label = mask_el.get("label", "") or ""
    z = int(mask_el.get("z_order", "0") or 0)
    points = mask_el.get("points", None)
    data = None
    if mask_el.text:
        txt = mask_el.text.strip()
        if txt:
            data = txt
    attrs = dict(mask_el.items())
    return {"label": label, "z": z, "points": points, "data": data, "attrs": attrs}


def read_mask_element_from_cvat(
    xml_source: str | Path,
    image_name: str | None = None,
    exclude_labels: Sequence[str] | None = None,
) -> list[dict]:
    """
    Parse <mask> annotations from a CVAT export and return a list of mask info dicts.

    - If image_name is provided, only masks for that image are returned.
    - exclude_labels: optional sequence of label names to ignore.
    - masks are sorted by z_order to preserve annotation stacking order.

    Each returned dict contains at least: 'label', 'z', 'points', 'data', 'attrs'.
    """
    exclude = set(exclude_labels or ())
    parsed = load_cvat_annotations(xml_source)
    if not parsed:
        return []

    if image_name is None:
        try:
            image_name = next(iter(parsed.keys()))
        except StopIteration:
            return []

    items = parsed.get(image_name, [])
    # keep only mask annotations: items are tuples (ann_type, label, data, z)
    mask_items = [it for it in items if it[0] == "mask"]
    # sort by z-order (4th element)
    items_sorted = sorted(mask_items, key=lambda it: int(it[3]) if len(it) > 3 else 0)

    masks: list[dict] = []
    for _ann_type, label, mask_info, _z in items_sorted:
        if label in exclude:
            logger.debug("Skipping excluded mask label: %s", label)
            continue
        # mask_info is expected to be a dict produced by parse_mask_element
        if isinstance(mask_info, dict):
            info = dict(mask_info)  # copy to avoid mutating originals
        else:
            # fallback: preserve raw data
            info = {
                "label": label or "",
                "data": mask_info,
                "attrs": {},
                "points": None,
                "z": 0,
            }

        # ensure canonical keys
        info["label"] = info.get("label", label or "")
        info["z"] = int(info.get("z", 0) or 0)
        info.setdefault("points", None)
        info.setdefault("data", None)
        info.setdefault("attrs", {})

        masks.append(info)

    return masks
