# %% # ===  IMPORTS  === #
import argparse
from datetime import datetime
from pathlib import Path

import arviz as az
import joblib
import numpy as np
import pandas as pd
import pymc as pm
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

from ppcluster import logger, mcmc
from ppcluster.cvat import (
    filter_dataframe_by_polygons,
    read_polygons_from_cvat,
)
from ppcluster.griddata import create_2d_grid, map_grid_to_points
from ppcluster.mcmc.postproc import (
    aggregate_multiscale_clustering,
    remove_small_grid_components,
    split_disconnected_components,
)
from ppcluster.mksectors import (
    auto_assign_mk_sectors,
    compute_mk_sector_stats,
    draw_polygon,
)
from ppcluster.preprocessing import (
    apply_2d_gaussian_filter,
    apply_dic_filters,
    preprocess_velocity_features,
    spatial_subsample,
)
from ppcluster.utils.config import ConfigManager
from ppcluster.utils.database import (
    fetch_dic_analysis_ids,
    get_dic_analysis_by_ids,
    get_image,
    get_multi_dic_data,
)

INTERACTIVE = False  # set to True when running in an interactive environment

if not INTERACTIVE:
    plt.switch_backend("Agg")


def run_mcmc_clustering(
    df_input,
    prior_probs,
    sectors,
    output_dir,
    base_name,
    img=None,
    variables_names=None,
    transform_velocity="none",
    transform_params=None,
    mu_params=None,
    sigma_params=None,
    feature_weights=None,
    sample_args=None,
    mrf_regularization: bool = False,
    mrf_kwargs: dict | None = None,
    second_pass: str = "full",  # "skip" | "short" | "full"
    second_pass_sample_args: dict | None = None,
    random_seed=8927,
):
    """
    Run MCMC-based clustering on velocity data with flexible velocity transformations.

    Parameters:
    -----------
    df_input : pandas.DataFrame
        Input dataframe with 'x', 'y', 'V' columns
    transform_velocity : str, default="none"
        Type of velocity transformation: "power", "exponential", "threshold", "sigmoid", or "none"
    transform_params : dict, optional
        Parameters for velocity transformation (see preprocess_velocity_features for details)
    """

    # --- helper: build initvals from idata posterior means (warm-start) ---
    def _initvals_from_idata(idata_in, n_chains):
        mu_mean = idata_in.posterior["mu"].mean(dim=["chain", "draw"]).values
        sigma_mean = idata_in.posterior["sigma"].mean(dim=["chain", "draw"]).values
        # Ensure shapes match the model dims; return a list of per-chain dicts
        init = {"mu": mu_mean, "sigma": sigma_mean}
        return [init for _ in range(n_chains)]

    logger.info(f"Running MCMC clustering for {base_name}...")

    # Default parameters if not provided
    if mu_params is None:
        mu_params = {"mu": 0, "sigma": 1}
    if sigma_params is None:
        sigma_params = {"sigma": 1}
    if sample_args is None:
        sample_args = dict(
            target_accept=0.95,
            draws=2000,
            tune=1000,
            chains=4,
            cores=4,
            random_seed=random_seed,
        )
    if variables_names is None:
        variables_names = ["V"]

    if "V" not in df_input.columns:
        raise ValueError("Input dataframe must contain 'V' column for velocities.")

    # Preprocess velocity features to enhance high velocities
    velocities, transform_info = preprocess_velocity_features(
        velocities=df_input["V"].to_numpy(),
        velocity_transform=transform_velocity,
        velocity_params=transform_params,
    )

    # Extract data array for clustering
    if len(variables_names) > 1:
        # Concatenate other features to velocities
        additional_vars = variables_names.copy()
        if "V" in additional_vars:
            additional_vars.remove("V")
        additional_data = df_input[additional_vars].to_numpy()
        data_array = np.column_stack((velocities, additional_data))
    else:
        # Use only velocities
        data_array = velocities.reshape(-1, 1)

    # Scale data for model input
    scaler = StandardScaler()
    scaler.fit(data_array)
    joblib.dump(scaler, output_dir / f"{base_name}_scaler.joblib")
    data_array_scaled = scaler.transform(data_array)

    # Build model
    logger.info(f"Running MCMC clustering for {base_name}...")
    model = mcmc.build_marginalized_mixture_model(
        data_array_scaled,
        prior_probs,
        sectors,
        mu_params=mu_params,
        sigma_params=sigma_params,
        feature_weights=feature_weights,
    )

    # Sample model (1st pass)
    idata, convergence_flag = mcmc.sample_model(
        model, output_dir, base_name, **sample_args
    )
    if not convergence_flag:
        idata_summary = az.summary(idata, var_names=["mu", "sigma"])
        logger.info(f"MCMC did not converge. Summary:\n{idata_summary}")

    # --- MRF regularization of priors and optional re-sample ---
    prior_used = prior_probs
    if mrf_regularization:
        x_pos = df_input["x"].to_numpy()
        y_pos = df_input["y"].to_numpy()
        mkw = dict(n_neighbors=8, length_scale=50, beta=2.0, n_iter=5)
        if mrf_kwargs:
            mkw.update(mrf_kwargs)
        prior_mrf, q_mrf = mcmc.mrf_regularization(
            data_array_scaled, idata, prior_probs, x_pos, y_pos, **mkw
        )
        prior_used = prior_mrf

        # visualize refined priors
        try:
            fig, _ = mcmc.plot_spatial_priors(df_input, prior_mrf, img=img)
            fig.savefig(
                output_dir / f"{base_name}_mrf_priors.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        except Exception as exc:
            logger.warning(f"Could not plot MRF priors: {exc}")

    # Decide second pass strategy
    if mrf_regularization and second_pass.lower() == "skip":
        # Fastest: don't re-sample. Use q_mrf as final posterior_probs and argmax as labels.
        posterior_probs = q_mrf
        cluster_pred = np.argmax(posterior_probs, axis=1)
        uncertainty = 1.0 - posterior_probs.max(axis=1)
        # keep idata from 1st pass for plots/params
    else:
        # Re-sample with refined priors (short or full)
        if mrf_regularization:
            with model:
                pm.set_data({"prior_w": prior_used})

        # Allow short second pass and warm start
        sp2_args = dict(**sample_args)
        if second_pass.lower() == "short":
            # much fewer draws/tune; fewer chains can also help
            sp2_args.update(dict(draws=600, tune=400, chains=2, cores=2))
            if second_pass_sample_args:
                sp2_args.update(second_pass_sample_args)
        elif second_pass_sample_args:
            sp2_args.update(second_pass_sample_args)

        # Warm-start from previous posterior means
        initvals = _initvals_from_idata(idata, sp2_args.get("chains", 2))

        with model:
            # pass initvals through sample_model if it supports, else call pm.sample directly
            try:
                idata, convergence_flag = mcmc.sample_model(
                    model,
                    output_dir,
                    base_name + ("_mrf" if mrf_regularization else ""),
                    initvals=initvals,
                    **sp2_args,
                )
            except TypeError:
                # fallback if your wrapper doesn't accept initvals
                idata = pm.sample(**sp2_args)
                convergence_flag = True

        # Compute posterior-based assignments
        posterior_probs, cluster_pred, uncertainty = mcmc.compute_posterior_assignments(
            idata, n_posterior_samples=200
        )

    # Generate plots
    fig = mcmc.plot_velocity_clustering(
        df_features=df_input,
        img=img,
        idata=idata,
        cluster_pred=cluster_pred,
        posterior_probs=posterior_probs,
        scaler=scaler,
    )
    fig.savefig(
        output_dir / f"{base_name}_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Trace plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    az.plot_trace(
        idata, var_names=["mu", "sigma"], axes=axes, compact=True, legend=True
    )
    fig.savefig(output_dir / f"{base_name}_trace_plots.png", dpi=150)
    plt.close(fig)

    # Forest plots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    az.plot_forest(idata, var_names=["mu", "sigma"], combined=True, ess=True, ax=axes)
    fig.savefig(output_dir / f"{base_name}_forest_plot.png", dpi=150)
    plt.close(fig)

    # Collect and save metadata
    metadata = mcmc.collect_run_metadata(
        idata=idata,
        convergence_flag=convergence_flag,
        data_array_scaled=data_array_scaled,
        variables_names=variables_names,
        sectors=sectors,
        prior_probs=prior_probs,
        sample_args=sample_args,
        frame=locals(),
    )
    mcmc.save_run_metadata(output_dir, base_name, metadata)

    # Return results dictionary
    result = {
        "metadata": metadata,
        "idata": idata,
        "scaler": scaler,
        "convergence_flag": convergence_flag,
        "posterior_probs": posterior_probs,
        "cluster_pred": cluster_pred,
        "uncertainty": uncertainty,
    }

    plt.close("all")
    return result


def main(reference_date: str | None = None):
    """
    Run the pipeline. When called interactively you can pass:
      - reference_date: "YYYY-MM-DD" to override config.data.reference_date
      - overrides: dict of dot.notation keys -> values to override config entries
    """  # %% # ===  LOAD CONFIGURATION  === #

    # %% # ===  CONFIG MANAGER  === #
    config = ConfigManager()

    # Apply CLI overrides to
    if reference_date:
        config.set("data.reference_date", reference_date)

    # Retrieve reference date
    reference_date = config.get("data.reference_date", None)
    if reference_date is None:
        raise ValueError("reference_date must be specified either via CLI or config.")

    db_engine = create_engine(config.db_url)
    random_seed = config.get("random_seed", 8927)

    # %% # ===  DATA LOADING AND PREPROCESSING  === #
    data_config = config.get("data", {})

    # Output base directory (output will be saved in a subfolder with camera name and date range)
    output_base_dir = Path(data_config.get("output_dir", "output"))

    # Data selection parameters
    camera_name = data_config.get("camera_name", "PPCX_Tele")
    days_before_to_include = data_config.get("days_before_to_include", 0)
    days_after_to_include = data_config.get("days_after_to_include", 0)
    dt_min = data_config.get("dt_min", 72)
    dt_max = data_config.get("dt_max", 96)
    reference_start_date = datetime.strptime(reference_date, "%Y-%m-%d") - pd.Timedelta(
        days=days_before_to_include
    )
    reference_end_date = datetime.strptime(reference_date, "%Y-%m-%d") + pd.Timedelta(
        days=days_after_to_include
    )
    variables_names = data_config.get("variables_names", ["V"])

    # Read roi and spatial priors
    roi_path = Path(data_config.get("roi_path", "data/roi.xml"))
    sector_prior_file = Path(
        data_config.get("sector_prior_file", "data/priors_4_sectors.xml")
    )
    roi = read_polygons_from_cvat(roi_path, image_name=None)
    sectors = read_polygons_from_cvat(sector_prior_file, image_name=None)

    # Check that at least the reference date or an interval of dates is provided
    if not (reference_date or (reference_start_date and reference_end_date)):
        raise ValueError(
            "Either reference_date or both reference_start_date and reference_end_date must be provided."
        )

    # Fetch DIC ids
    dic_ids = fetch_dic_analysis_ids(
        db_engine,
        camera_name=camera_name,
        reference_date=reference_date,
        reference_date_start=reference_start_date,
        reference_date_end=reference_end_date,
        dt_hours_min=dt_min,
        dt_hours_max=dt_max,
    )
    if len(dic_ids) < 1:
        raise ValueError("No DIC analyses found for the given criteria")

    # Get DIC analysis metadata
    dic_analyses = get_dic_analysis_by_ids(db_engine=db_engine, dic_ids=dic_ids)
    logger.info("Fetched DIC analysis:")
    for _, row in dic_analyses.iterrows():
        print(
            f"DIC ID: {row['dic_id']}, date: {row['reference_date']}, dt (hrs): {row['dt_hours']}, Master: {row['master_timestamp']}, Slave: {row['slave_timestamp']}"
        )
    print("Summary of selected DIC analyses:")
    print(dic_analyses.describe())

    # Output paths
    date_start = dic_analyses.iloc[0]["master_timestamp"].strftime("%Y-%m-%d")
    date_end = dic_analyses.iloc[0]["slave_timestamp"].strftime("%Y-%m-%d")
    output_dir = output_base_dir / f"{camera_name}_{date_end}_mcmc_multiscale"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{date_start}_{date_end}"

    # Get master image
    master_image_id = dic_analyses["master_image_id"].iloc[0]
    img = get_image(image_id=master_image_id, config=config)

    # Fetch DIC data
    out = get_multi_dic_data(dic_ids, stack_results=False, config=config)
    logger.info(f"Found stack of {len(out)} DIC dataframes.")

    # Apply filter for each df in the dictionary and then stack them
    preprocessing_config = config.get("preprocessing", {})
    subsample_factor = preprocessing_config.get("subsample_factor", 1)
    subsample_method = preprocessing_config.get("subsample_method", "random")
    filter_kwargs = preprocessing_config.get("filter_kwargs", {})

    processed = []
    for src_id, df_src in out.items():
        try:
            # Filter only points inside the spatial priors sectors
            df_src = filter_dataframe_by_polygons(df_src, polygons=roi)

            # Apply other DIC filters if any
            df_src = apply_dic_filters(df_src, **filter_kwargs)

            # Append processed dataframe to the list
            processed.append(df_src)
        except Exception as exc:
            logger.warning("Filtering failed for %s: %s", src_id, exc)
    if not processed:
        raise RuntimeError("No dataframes left after filtering.")

    # Stack all processed dataframes
    df = pd.concat(processed, ignore_index=True)
    logger.info("Data shape after filtering and stacking: %s", df.shape)

    # Apply subsampling
    if subsample_factor > 1:
        df_subsampled = spatial_subsample(
            df, n_subsample=subsample_factor, method=subsample_method
        )
        df = df_subsampled
        logger.info(f"Data shape after subsampling: {df.shape}")

    # %% # ===  SPATIAL PRIORS AND INITIAL VISUALIZATIONS  === #

    # Assign spatial priors
    prior_config = config.get("priors", {})
    prior_probability = prior_config.get("probability", None)
    if not prior_probability:
        # Default: uniform priors across sectors
        n_sectors = len(sectors)
        uniform_prob = 1.0 / n_sectors
        prior_probability = {name: [uniform_prob] * n_sectors for name in sectors}
    fade_method = prior_config.get("fade_method", "constant")
    fade_method_options = prior_config.get("fade_options", {}).get(fade_method, {})
    prior_probs_array = mcmc.assign_spatial_priors(
        x=df["x"].to_numpy(),
        y=df["y"].to_numpy(),
        polygons=sectors,
        prior_probs=prior_probability,
        fade_method=fade_method,
        fade_options=fade_method_options,
    )

    fig, axes = mcmc.plot_spatial_priors(df, prior_probs_array, img=img)
    fig.savefig(
        output_dir / f"{base_name}_spatial_priors.jpg", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot velocity field
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_title("Velocity Field", fontsize=14, pad=10)
    ax.imshow(img, alpha=0.5, cmap="gray")
    magnitudes = df["V"].to_numpy()
    vmin = 0.0
    vmax = np.max(magnitudes)
    norm = Normalize(vmin=vmin, vmax=vmax)
    q = ax.quiver(
        df["x"].to_numpy(),
        df["y"].to_numpy(),
        df["u"].to_numpy(),
        df["v"].to_numpy(),
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
    cbar = fig.colorbar(q, ax=ax, shrink=0.8, aspect=20, pad=0.02)
    cbar.set_label("Velocity Magnitude", rotation=270, labelpad=15)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    fig.savefig(
        output_dir / f"{base_name}_velocity_field.jpg",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # %% # ===  RUN MCMC CLUSTERING  === #

    # MCMC parameters
    mcmc_config = config.get("mcmc", {})
    sample_options = mcmc_config.get("sample_options", {})
    sample_args = {
        "draws": sample_options.get("draws", 2000),
        "tune": sample_options.get("tune", 1000),
        "chains": sample_options.get("chains", 4),
        "cores": sample_options.get("cores", 4),
        "target_accept": sample_options.get("target_accept", 0.9),
        "random_seed": random_seed,
    }
    model_options = mcmc_config.get("model_options", {})
    mu_params = model_options.get("mu_params", {"mu": 0, "sigma": 1})
    sigma_params = model_options.get("sigma_params", {"sigma": 1})

    # Velocity transformation parameters
    velocity_transform = mcmc_config.get(
        "velocity_transform", None
    )  # also: "power", "exponential", "sigmoid"
    transform_params = mcmc_config.get(
        "transform_params", {}
    )  # also {"midpoint_percentile": 70, "steepness": 2.0},)

    # MRF regularization parameters
    mrf_regularization = mcmc_config.get("mrf_regularization", True)
    mrf_kwargs = mcmc_config.get("mrf_kwargs", {})
    second_pass = mcmc_config.get("second_pass", "short")
    second_pass_sample_args = mcmc_config.get("second_pass_sample_args", {})

    # Multiscale parameters
    multiscale_config = config.get("multiscale", {})
    sigma_values = multiscale_config.get("sigma_values", [2])

    # Aggregation parameters
    aggregation_config = multiscale_config.get("aggregation", {})
    similarity_threshold = aggregation_config.get("similarity_threshold", 0.7)
    overall_threshold = aggregation_config.get("overall_threshold", 0.8)

    # Loop through smoothing scales
    results = []
    for sigma in sigma_values:
        logger.info(f"Processing with Gaussian smoothing sigma={sigma}...")

        # Create scale-specific base name
        scale_base_name = f"{date_start}_{date_end}_sigma{sigma}"

        # Apply Gaussian smoothing if needed (skipped for sigma=0)
        df_run = apply_2d_gaussian_filter(df, sigma=sigma)

        # For larger sigma, tighten priors
        if sigma > 2:
            mu_params = {"mu": 0, "sigma": 0.5}
            sigma_params = {"sigma": 0.5}

        # Run MCMC clustering with the smoothed data
        result = run_mcmc_clustering(
            df_input=df_run,
            prior_probs=prior_probs_array,
            sectors=sectors,
            output_dir=output_dir,
            base_name=scale_base_name,
            img=img,
            variables_names=variables_names,
            sample_args=sample_args,
            transform_velocity=velocity_transform,
            transform_params=transform_params,
            mu_params=mu_params,
            sigma_params=sigma_params,
            random_seed=random_seed,
            mrf_regularization=mrf_regularization,
            mrf_kwargs=mrf_kwargs,
            second_pass=second_pass,
            second_pass_sample_args=second_pass_sample_args,
        )

        # Add scale information to result
        result["sigma"] = sigma

        # Append to results list
        results.append(result)

    # %% == =  AGGREGATE MULTI-SCALE RESULTS  (if multiscale approach)=== #

    # Multiscale parameters (grouped)
    multiscale_config = config.get("multiscale", {})
    aggregation_config = multiscale_config.get("aggregation", {})
    similarity_threshold = aggregation_config.get("similarity_threshold", 0.7)
    overall_threshold = aggregation_config.get("overall_threshold", 0.8)

    if len(sigma_values) > 1:
        aggregated_results = aggregate_multiscale_clustering(
            results,
            similarity_threshold=similarity_threshold,
            overall_threshold=overall_threshold,
            fig_path=output_dir
            / f"{reference_start_date}_{reference_end_date}_similarity_heatmap.jpg",
        )

        # Unpack aggregated results
        cluster_pred = aggregated_results["combined_cluster_pred"]
        posterior_probs = aggregated_results["avg_posterior_probs"]
        entropy = aggregated_results["avg_entropy"]
        similarity_matrix = aggregated_results["similarity_matrix"]
        stability_score = aggregated_results["stability_score"]
        valid_scales = aggregated_results["valid_scales"]

    else:
        # Otherwise extract the single result
        cluster_pred = results[0]["cluster_pred"]
        posterior_probs = results[0]["posterior_probs"]
        entropy = -np.sum(posterior_probs * np.log(posterior_probs + 1e-10), axis=1)
        similarity_matrix = None
        stability_score = None
        valid_scales = None

    # ===  Save final clustering results
    cluster_aggregation_outs = {
        "cluster_pred": cluster_pred,
        "posterior_probs": posterior_probs,
        "entropy": entropy,
        "similarity_matrix": similarity_matrix,
        "stability_score": stability_score,
        "valid_scales": valid_scales,
    }
    joblib.dump(
        cluster_aggregation_outs,
        output_dir
        / f"{reference_start_date}_{reference_end_date}_kinematic_clustering_results.joblib",
    )

    # %% # ===  POST-PROCESSING AND CLEANING OF FINAL CLUSTERING  === #
    # Retrieve data
    df_smooth = apply_2d_gaussian_filter(df, sigma=1)
    x = df_smooth["x"].to_numpy()
    y = df_smooth["y"].to_numpy()
    v = df_smooth["V"].to_numpy()
    kin_cluster = np.asarray(cluster_pred.copy())

    X, Y, kin_cluster_grid = create_2d_grid(x=x, y=y, labels=kin_cluster)

    # Filter out small clusters
    kin_cluster_grid = remove_small_grid_components(
        kin_cluster_grid, min_size=100, connectivity=8
    )

    # Split clusters along detected discontinuities
    kin_cluster_grid, split_mapping = split_disconnected_components(
        kin_cluster_grid, connectivity=8, start_label=0
    )
    kin_cluster, x, y = map_grid_to_points(X, Y, kin_cluster_grid, x, y, keep_nan=True)

    # Remove non classified points (-1 label)
    valid_mask = kin_cluster >= 0
    x = x[valid_mask]
    y = y[valid_mask]
    v = v[valid_mask]
    kin_cluster = kin_cluster[valid_mask]
    prior_probs_array = prior_probs_array[valid_mask]

    # Order clusters by median y descending (bottom = largest y first)
    clusters_ids = np.unique(kin_cluster)
    cluster_median_y = {
        int(c): float(np.median(y[kin_cluster == c])) for c in clusters_ids
    }
    ordered_clusters_ids = sorted(
        clusters_ids, key=lambda c: cluster_median_y[int(c)], reverse=True
    )

    # === Compute similarity scores with prior clusters
    # Create a "prior class" assignment based on the sector with highest probability
    sector_names = list(prior_probability.keys())
    sector_assignments = np.zeros_like(kin_cluster)
    for i, point_probs in enumerate(prior_probs_array):
        sector_assignments[i] = np.argmax(point_probs)

    # Compute similarity metrics
    ari = adjusted_rand_score(sector_assignments, kin_cluster)
    ami = adjusted_mutual_info_score(sector_assignments, kin_cluster)

    # === Make final clustering plot after cleaning
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img, alpha=0.5, cmap="gray")
    colormap = plt.get_cmap("tab10")
    for i, label in enumerate(clusters_ids):
        mask = kin_cluster == label
        ax.scatter(
            x[mask],
            y[mask],
            color=colormap(i),
            label=f"Cluster {label}",
            s=10,
            alpha=0.7,
        )
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_aspect("equal")

    if valid_scales is not None and len(valid_scales) > 1:
        stability_str = (
            f"{stability_score:.2f}" if stability_score is not None else "N/A"
        )
        title = f"Combined Clustering (scales: {valid_scales}, stability: {stability_str})\nPrior Agreement: AMI={ami:.2f}"
    else:
        title = f"Clustering (scale: {sigma_values[0]})\nPrior Agreement: AMI={ami if ami is not None else 0.0:.2f}"

    ax.set_title(title)
    plt.savefig(
        output_dir
        / f"{reference_start_date}_{reference_end_date}_kinematic_clustering.png",
        dpi=300,
        bbox_inches="tight",
    )

    # %% # ===  AUTOMATIC MORPHO-KINEMATIC SECTOR ASSIGNMENT  === #

    # Morphokinematic parameters
    morphokinematic_config = config.get("morphokinematic", {})
    minor_overlap_threshold = morphokinematic_config.get("minor_overlap_threshold", 0.9)
    base_colors = morphokinematic_config.get(
        "base_colors",
        {
            "A": "#b3140b",
            "B": "#ee9c21",
            "C": "#f1ee30",
            "D": "#5fb61c",
        },
    )

    cmap = plt.get_cmap("tab20")
    assignment = auto_assign_mk_sectors(
        x=x,
        y=y,
        kin_cluster=kin_cluster,
        ordered_clusters_ids=ordered_clusters_ids,
        overlap_threshold=minor_overlap_threshold,
    )

    mk_label_str = assignment["mk_label_str"]
    mk_label_id = assignment["mk_label_id"]
    major_clusters = assignment["major_clusters"]
    major_label_map = assignment["major_label_map"]
    minor_parent = assignment["minor_parent"]
    cluster_to_label = assignment["cluster_to_label"]
    polygons_major = assignment["polygons_major"]
    polygons_minor = assignment["polygons_minor"]

    label_order = sorted(assignment["label_to_index"].items(), key=lambda kv: kv[1])
    unique_mk = np.array([lab for lab, _ in label_order], dtype=object)
    counts = np.array([np.sum(mk_label_str == lab) for lab in unique_mk], dtype=int)

    logger.info("Major sectors: %s", major_label_map)
    if minor_parent:
        logger.info(
            "Minor cluster mapping: %s",
            {
                cluster_id: {
                    "label": cluster_to_label[cluster_id],
                    "parent": cluster_to_label[parent_id],
                    "overlap": f"{overlap_ratio:.2f}",
                }
                for cluster_id, (parent_id, overlap_ratio) in minor_parent.items()
            },
        )
    else:
        logger.info("No minor clusters detected.")

    print("Morpho-kinematic assignment summary:")
    for label, count in zip(unique_mk, counts, strict=False):
        print(f"  {label}: {count} points")

    colors = {}
    for idx, label in enumerate(unique_mk):
        major_key = "".join(filter(str.isalpha, label)) or label
        color = base_colors.get(label) or base_colors.get(major_key)
        if color is None:
            color = mcolors.to_hex(cmap(idx % cmap.N))
        colors[label] = color

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, alpha=0.5, cmap="gray")

    major_labels = [
        cluster_to_label[cid] for cid in major_clusters if cid in cluster_to_label
    ]
    for label in major_labels:
        poly = polygons_major.get(label)
        if poly is not None:
            draw_polygon(
                ax, poly, label, colors.get(label, "#444444"), fill_alpha=0.12, zorder=1
            )

    minor_labels = [label for label in unique_mk if label not in major_labels]
    for label in minor_labels:
        poly = polygons_minor.get(label)
        if poly is not None:
            draw_polygon(
                ax, poly, label, colors.get(label, "#888888"), fill_alpha=0.08, zorder=2
            )

    handles, labels = ax.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()
    for handle, name in zip(handles, labels, strict=False):
        if name in seen:
            continue
        seen.add(name)
        unique_handles.append(handle)
        unique_labels.append(name)
    if unique_handles:
        ax.legend(
            unique_handles,
            unique_labels,
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
        )

    ax.set_title("Morpho-Kinematic Sectors")
    ax.set_aspect("equal")
    fig.savefig(
        output_dir
        / f"{reference_start_date}_{reference_end_date}_mk_sectors_perimeters.png",
        dpi=300,
        bbox_inches="tight",
    )

    # %% # ===  COMPUTE AND SAVE MORPHO-KINEMATIC SECTOR STATISTICS  === #
    mk_stats = compute_mk_sector_stats(
        polygons_major,
        mk_label_str,
        x=x,
        y=y,
        v=v,
        img_shape=np.asarray(img).shape if img is not None else None,
        rasterize=True,
    )
    mk_stats.to_csv(
        output_dir / f"{reference_start_date}_{reference_end_date}_mk_sector_stats.csv",
        index=False,
    )


# %% # === ENTRY POINT  === #
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Run MCMC clustering with optional overrides."
    )
    p.add_argument(
        "--reference-date", "-d", help="Override data.reference_date (YYYY-MM-DD)."
    )
    args = p.parse_args()

    main(reference_date=args.reference_date)
