"""
1. Central ML predicts best battery/cooling action.
2. Split buildings into 5 groups.
3. For each group:
   - if SOC too low, reduce discharge
   - if SOC too high, reduce charging
   - if indoor temperature is close to comfort limit, reduce cooling reduction
   - if group load is already high, reduce cooling/charging
4. Apply final group-specific actions.
5. Evaluate live portfolio load."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUILDING_FILE = (
    PROJECT_ROOT
    / "harish_work"
    / "outputs"
    / "ml_datasets"
    / "tx_building_ml_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_controller" / "building_clusters"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLUSTER_SUMMARY_FILE = OUTPUT_DIR / "building_cluster_summary.csv"
CLUSTER_ASSIGNMENT_FILE = OUTPUT_DIR / "building_cluster_assignments.csv"
CLUSTER_FEATURE_FILE = OUTPUT_DIR / "building_cluster_features.csv"
SILHOUETTE_FILE = OUTPUT_DIR / "building_cluster_silhouette_scores.csv"
PLOT_FILE = OUTPUT_DIR / "building_cluster_plot.png"


RANDOM_SEED = 42

# We eventually want 5 group controllers, but we also test k=2..8.
FORCED_CONTROLLER_CLUSTERS = 5


def find_col_contains(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        if all(k.lower() in col_lower for k in keywords):
            return col
    return None


def get_building_id_column(df):
    candidates = [
        "building_id",
        "building_index",
        "building",
        "Building",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    possible = find_col_contains(df, ["building", "id"])
    if possible is not None:
        return possible

    possible = find_col_contains(df, ["building", "index"])
    if possible is not None:
        return possible

    raise ValueError("Could not find building id column.")


def select_existing_columns(df, preferred_cols):
    selected = []

    for keyword_group in preferred_cols:
        col = find_col_contains(df, keyword_group)
        if col is not None and col not in selected:
            selected.append(col)

    return selected


def build_building_level_features(df, building_col):
    """
    Convert hourly building rows into one row per building.
    """

    preferred_feature_keywords = [
        ["net", "load"],
        ["true", "load"],
        ["solar"],
        ["outdoor", "temp"],
        ["rolling", "temp"],
        ["indoor", "temp"],
        ["cooling", "setpoint"],
        ["soc"],
        ["pca", "1"],
        ["pca", "2"],
        ["pca", "3"],
        ["entropy"],
    ]

    feature_cols = select_existing_columns(df, preferred_feature_keywords)

    # Add useful lag/ramp/rolling features if present.
    extra_cols = []
    for col in df.columns:
        c = col.lower()
        if any(key in c for key in ["lag", "ramp", "rolling", "std"]):
            if col not in feature_cols and pd.api.types.is_numeric_dtype(df[col]):
                extra_cols.append(col)

    # Keep this small and stable.
    feature_cols = feature_cols + extra_cols[:20]

    if not feature_cols:
        raise ValueError("No useful numeric feature columns found for clustering.")

    print("\nFeatures used for clustering:")
    for col in feature_cols:
        print(f"  - {col}")

    agg_dict = {}

    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = ["mean", "std", "min", "max"]

    grouped = df.groupby(building_col).agg(agg_dict)

    # Flatten multi-index columns.
    grouped.columns = [
        f"{feature}_{stat}" for feature, stat in grouped.columns
    ]

    grouped = grouped.reset_index()
    grouped = grouped.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return grouped


def run_kmeans_scan(x_scaled, building_features):
    rows = []

    for k in range(2, min(9, len(building_features))):
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=20,
        )

        labels = model.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)

        rows.append(
            {
                "k": k,
                "silhouette_score": score,
                "inertia": model.inertia_,
            }
        )

    scores = pd.DataFrame(rows)
    return scores


def assign_clusters(x_scaled, building_features, k):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_SEED,
        n_init=50,
    )

    labels = model.fit_predict(x_scaled)

    result = building_features.copy()
    result["cluster"] = labels

    return result, model


def main():
    print("\n=== Building clustering for hierarchical controller ===")

    df = pd.read_csv(BUILDING_FILE)
    building_col = get_building_id_column(df)

    print(f"Building id column: {building_col}")
    print(f"Raw rows: {len(df)}")
    print(f"Unique buildings: {df[building_col].nunique()}")

    building_features = build_building_level_features(df, building_col)
    building_features.to_csv(CLUSTER_FEATURE_FILE, index=False)

    feature_matrix = building_features.drop(columns=[building_col])
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(feature_matrix)

    silhouette_scores = run_kmeans_scan(x_scaled, building_features)
    silhouette_scores.to_csv(SILHOUETTE_FILE, index=False)

    print("\nSilhouette scores:")
    print(silhouette_scores)

    best_k = int(
        silhouette_scores.sort_values(
            "silhouette_score",
            ascending=False,
        ).iloc[0]["k"]
    )

    print(f"\nBest k by silhouette: {best_k}")
    print(f"Forced controller k: {FORCED_CONTROLLER_CLUSTERS}")

    # Save best-k clustering.
    best_assignments, best_model = assign_clusters(
        x_scaled,
        building_features,
        best_k,
    )

    best_assignments = best_assignments[[building_col, "cluster"]]
    best_assignments = best_assignments.rename(columns={"cluster": "best_cluster"})

    # Save forced 5-cluster grouping for controller.
    forced_assignments, forced_model = assign_clusters(
        x_scaled,
        building_features,
        FORCED_CONTROLLER_CLUSTERS,
    )

    forced_assignments = forced_assignments[[building_col, "cluster"]]
    forced_assignments = forced_assignments.rename(columns={"cluster": "controller_cluster"})

    assignments = best_assignments.merge(
        forced_assignments,
        on=building_col,
        how="left",
    )

    assignments.to_csv(CLUSTER_ASSIGNMENT_FILE, index=False)

    cluster_summary = (
        assignments
        .groupby("controller_cluster")[building_col]
        .apply(list)
        .reset_index()
    )

    cluster_summary["n_buildings"] = cluster_summary[building_col].apply(len)
    cluster_summary.to_csv(CLUSTER_SUMMARY_FILE, index=False)

    print("\nController cluster assignments:")
    print(assignments)

    print("\nController cluster summary:")
    print(cluster_summary)

    # Plot silhouette scores.
    plt.figure(figsize=(8, 5))
    plt.plot(silhouette_scores["k"], silhouette_scores["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("Building clustering silhouette scores")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()

    # Log to W&B.
    run = wandb.init(
        entity="CityLearn-TeamB",
        project="CityLearn",
        name="cluster-buildings-for-hierarchical-controller",
        job_type="building-clustering",
        config={
            "dataset": "annex96_ce1_tx_neighborhood",
            "n_buildings": int(df[building_col].nunique()),
            "forced_controller_clusters": FORCED_CONTROLLER_CLUSTERS,
            "best_k_by_silhouette": best_k,
        },
    )

    wandb.log({
        "clustering/best_k_by_silhouette": best_k,
        "clustering/best_silhouette_score": float(silhouette_scores["silhouette_score"].max()),
        "tables/silhouette_scores": wandb.Table(dataframe=silhouette_scores),
        "tables/cluster_assignments": wandb.Table(dataframe=assignments),
        "tables/cluster_summary": wandb.Table(dataframe=cluster_summary),
        "plots/silhouette_scores": wandb.Image(str(PLOT_FILE)),
    })

    artifact = wandb.Artifact(
        name="building-clusters-for-hierarchical-controller",
        type="clustering-results",
        description="Building clusters for group-aware hierarchical CityLearn controller.",
        metadata={
            "best_k_by_silhouette": best_k,
            "forced_controller_clusters": FORCED_CONTROLLER_CLUSTERS,
        },
    )

    artifact.add_file(str(CLUSTER_ASSIGNMENT_FILE))
    artifact.add_file(str(CLUSTER_SUMMARY_FILE))
    artifact.add_file(str(CLUSTER_FEATURE_FILE))
    artifact.add_file(str(SILHOUETTE_FILE))
    artifact.add_file(str(PLOT_FILE))

    run.log_artifact(artifact)
    run.finish()

    print(f"\nSaved assignments: {CLUSTER_ASSIGNMENT_FILE}")
    print(f"Saved summary:     {CLUSTER_SUMMARY_FILE}")
    print(f"Saved plot:        {PLOT_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()