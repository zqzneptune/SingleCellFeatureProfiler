"""Legacy composite-score and K-means marker selection."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale


def legacy_select_robust_markers(
    ranked_markers_df: pd.DataFrame,
    top_n: int = 10,
    fdr_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the historical min-max composite score and K-means cutoff."""
    if ranked_markers_df.empty:
        return pd.DataFrame()

    candidates = ranked_markers_df[
        ranked_markers_df["fdr_marker"] < fdr_threshold
    ].copy()
    if len(candidates) < 2:
        if verbose:
            print(
                f"Warning: Not enough markers ({len(candidates)}) passed the "
                f"initial FDR threshold of {fdr_threshold} to perform dynamic "
                "selection."
            )
        return candidates

    specificity_column = next(
        (column for column in candidates.columns if "specificity_" in column),
        None,
    )
    if not specificity_column:
        raise ValueError("Could not find a specificity column in the DataFrame.")

    metrics = ["log2fc_all", specificity_column, "pct_expressing"]
    if "stability_score" in candidates.columns:
        metrics.append("stability_score")
    for metric in metrics:
        candidates[f"scaled_{metric}"] = minmax_scale(candidates[metric])
    candidates["marker_score"] = candidates[
        [f"scaled_{metric}" for metric in metrics]
    ].sum(axis=1)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(
        candidates[["marker_score"]].values
    )
    exceptional_label = np.argmax(kmeans.cluster_centers_)
    exceptional = candidates[kmeans.labels_ == exceptional_label]
    threshold = exceptional["marker_score"].min()

    if verbose:
        print("--- Dynamic Marker Selection via Clustering ---")
        print(f"  - Clustered {len(candidates)} candidate markers into two groups.")
        print(f"  - Identified {len(exceptional)} as 'exceptional'.")
        print(f"  - Learned marker_score threshold: {threshold:.3f}")
        print("---------------------------------------------")

    selected = candidates[candidates["marker_score"] >= threshold].copy()
    selected.sort_values(
        by=["group", "marker_score"], ascending=[True, False], inplace=True
    )
    return selected.groupby("group").head(top_n).reset_index(drop=True)
