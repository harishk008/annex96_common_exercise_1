from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_datasets"
ANALYSIS_DIR = ML_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

BUILDING_FILE = ML_DIR / "tx_building_ml_features.csv"
PORTFOLIO_FILE = ML_DIR / "tx_portfolio_ml_features.csv"


def select_pca_input_columns(df: pd.DataFrame) -> list[str]:
    """
    Select numeric input features for PCA interpretation.

    We exclude:
    - IDs/time counters that are not physical ML features
    - already-generated PCA/ICA columns
    - target/error columns, because those should be prediction/reward targets,
      not mixed into the PCA interpretation
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_exact = {
        "time_step",
        "step",
        "building_id",
        "building_index",
        "month",
        "day",
        "day_type",
        "hour",
    }

    exclude_substrings = [
        "pca",
        "ica",
        "tracking_error",
        "absolute_tracking_error",
        "squared_tracking_error",
        "district_target",
    ]

    selected = []

    for col in numeric_cols:
        col_lower = col.lower()

        if col_lower in exclude_exact:
            continue

        if any(pattern in col_lower for pattern in exclude_substrings):
            continue

        selected.append(col)

    return selected


def run_pca_loading_analysis(
    df: pd.DataFrame,
    dataset_name: str,
    existing_pca_prefix: str,
    n_components: int = 3,
) -> None:
    print(f"\n=== PCA loading analysis: {dataset_name} ===")

    feature_cols = select_pca_input_columns(df)

    if len(feature_cols) < n_components:
        raise ValueError(
            f"Not enough numeric feature columns for PCA in {dataset_name}. "
            f"Found only {len(feature_cols)} columns."
        )

    clean_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Rows used: {len(clean_df)}")
    print(f"Features used: {len(feature_cols)}")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(clean_df)

    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(x_scaled)

    component_names = [f"{dataset_name}_pca_{i + 1}" for i in range(n_components)]

    # PCA component weights
    weights = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=component_names,
    )

    # Loadings = component weight * sqrt(eigenvalue)
    # For standardized data, this is easier to interpret as feature-PC relation strength.
    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=feature_cols,
        columns=component_names,
    )

    explained = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": pca.explained_variance_ratio_ * 100,
        }
    )

    top_rows = []

    for component in component_names:
        sorted_loadings = loadings[component].sort_values(
            key=lambda s: s.abs(),
            ascending=False,
        )

        for rank, (feature, loading_value) in enumerate(sorted_loadings.head(15).items(), start=1):
            top_rows.append(
                {
                    "component": component,
                    "rank": rank,
                    "feature": feature,
                    "loading": loading_value,
                    "abs_loading": abs(loading_value),
                    "interpretation_hint": (
                        "positive direction"
                        if loading_value > 0
                        else "negative direction"
                    ),
                }
            )

    top_loadings = pd.DataFrame(top_rows)

    # Optional comparison with already existing PCA columns in the ML dataset.
    # If correlation is close to +/-1, the recomputed PCA matches the stored PCA.
    comparison_rows = []
    score_df = pd.DataFrame(scores, index=clean_df.index, columns=component_names)

    for i in range(n_components):
        stored_col = f"{existing_pca_prefix}_{i + 1}"
        recomputed_col = component_names[i]

        if stored_col in df.columns:
            corr = score_df[recomputed_col].corr(df.loc[clean_df.index, stored_col])
            comparison_rows.append(
                {
                    "stored_column": stored_col,
                    "recomputed_column": recomputed_col,
                    "correlation": corr,
                    "abs_correlation": abs(corr),
                    "note": "PCA sign can flip, so abs_correlation is more important.",
                }
            )

    comparison = pd.DataFrame(comparison_rows)

    weights.to_csv(ANALYSIS_DIR / f"{dataset_name}_pca_component_weights.csv")
    loadings.to_csv(ANALYSIS_DIR / f"{dataset_name}_pca_loadings.csv")
    top_loadings.to_csv(ANALYSIS_DIR / f"{dataset_name}_pca_top_loadings.csv", index=False)
    explained.to_csv(ANALYSIS_DIR / f"{dataset_name}_pca_explained_variance.csv", index=False)

    if not comparison.empty:
        comparison.to_csv(ANALYSIS_DIR / f"{dataset_name}_pca_recomputed_vs_existing.csv", index=False)

    print("\nExplained variance:")
    print(explained)

    print("\nTop loadings:")
    print(top_loadings.head(20))

    if not comparison.empty:
        print("\nComparison with stored PCA columns:")
        print(comparison)

    print(f"\nSaved PCA analysis files to: {ANALYSIS_DIR}")


def main() -> None:
    building_df = pd.read_csv(BUILDING_FILE)
    portfolio_df = pd.read_csv(PORTFOLIO_FILE)

    run_pca_loading_analysis(
        building_df,
        dataset_name="building",
        existing_pca_prefix="building_pca",
        n_components=3,
    )

    run_pca_loading_analysis(
        portfolio_df,
        dataset_name="portfolio",
        existing_pca_prefix="portfolio_pca",
        n_components=3,
    )


if __name__ == "__main__":
    main()