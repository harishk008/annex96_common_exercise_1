from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_DIR = PROJECT_ROOT / "harish_work" / "outputs" / "ml_datasets"
ANALYSIS_DIR = ML_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

BUILDING_FILE = ML_DIR / "tx_building_ml_features.csv"
PORTFOLIO_FILE = ML_DIR / "tx_portfolio_ml_features.csv"


def get_time_column(df: pd.DataFrame) -> str:
    for col in ["time_step", "step", "hour_index"]:
        if col in df.columns:
            return col

    raise ValueError("No time column found. Expected one of: time_step, step, hour_index")


def prepare_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "tracking_error" not in df.columns:
        raise ValueError("tracking_error column not found in portfolio dataset.")

    if "absolute_tracking_error" not in df.columns:
        df["absolute_tracking_error"] = df["tracking_error"].abs()

    if "squared_tracking_error" not in df.columns:
        df["squared_tracking_error"] = df["tracking_error"] ** 2

    return df


def correlation_table(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    exclude_cols = {
        target_col,
        "time_step",
        "step",
        "building_id",
        "building_index",
    }

    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    rows = []

    for feature in feature_cols:
        temp = df[[feature, target_col]].replace([np.inf, -np.inf], np.nan).dropna()

        if len(temp) < 5:
            continue

        if temp[feature].std() == 0 or temp[target_col].std() == 0:
            continue

        pearson_corr = temp[feature].corr(temp[target_col], method="pearson")
        spearman_corr = temp[feature].corr(temp[target_col], method="spearman")

        rows.append(
            {
                "dataset": dataset_name,
                "target": target_col,
                "feature": feature,
                "pearson_correlation": pearson_corr,
                "abs_pearson_correlation": abs(pearson_corr),
                "spearman_correlation": spearman_corr,
                "abs_spearman_correlation": abs(spearman_corr),
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    result = result.sort_values("abs_pearson_correlation", ascending=False)
    return result


def main() -> None:
    building_df = pd.read_csv(BUILDING_FILE)
    portfolio_df = pd.read_csv(PORTFOLIO_FILE)

    portfolio_df = prepare_error_columns(portfolio_df)

    time_col = get_time_column(portfolio_df)
    building_time_col = get_time_column(building_df)

    target_cols = [
        "tracking_error",
        "absolute_tracking_error",
        "squared_tracking_error",
    ]

    # Portfolio-level correlations
    for target in target_cols:
        corr = correlation_table(
            portfolio_df,
            target_col=target,
            dataset_name="portfolio",
        )

        out_file = ANALYSIS_DIR / f"portfolio_correlations_with_{target}.csv"
        corr.to_csv(out_file, index=False)

        print(f"\n=== Portfolio correlations with {target} ===")
        print(corr.head(20))
        print(f"Saved: {out_file}")

    # Building-level correlations need portfolio error joined by time.
    error_cols = [time_col] + target_cols
    portfolio_error_df = portfolio_df[error_cols].copy()

    if building_time_col != time_col:
        portfolio_error_df = portfolio_error_df.rename(columns={time_col: building_time_col})

    building_joined = building_df.merge(
        portfolio_error_df,
        on=building_time_col,
        how="left",
        suffixes=("", "_portfolio"),
    )

    for target in target_cols:
        corr = correlation_table(
            building_joined,
            target_col=target,
            dataset_name="building",
        )

        out_file = ANALYSIS_DIR / f"building_correlations_with_{target}.csv"
        corr.to_csv(out_file, index=False)

        print(f"\n=== Building correlations with {target} ===")
        print(corr.head(20))
        print(f"Saved: {out_file}")

    print(f"\nAll correlation analysis files saved to: {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()