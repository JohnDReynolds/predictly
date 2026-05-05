"""data_health.py"""

from typing import Any
import pandas as pd


def build_column_health_summary(
    df: pd.DataFrame,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Compute a per-column health summary for a DataFrame.

    This is intended to support a simple "Column Health" UI for uploaded
    CSV files (train.csv or test.csv). It computes a small, fixed set
    of interpretable statistics and attaches a severity level and
    human-readable messages for each column.

    For each column, the following information is produced:
      - dtype
      - n_rows
      - missing_count
      - missing_pct  (0..1)
      - unique_count
      - unique_pct   (0..1)
      - is_constant         (unique_count == 1)
      - is_almost_constant  (unique_pct < 0.01)
      - most_frequent_value, most_frequent_pct (0..1)
      - n_classes (distinct non-missing values)
      - majority_class, majority_pct (0..1)
      - minority_class, minority_pct (0..1)
      - mean, std, min, median, max (for numeric columns only)
      - severity: 'ok' | 'info' | 'warning' | 'critical'
      - messages: list of short strings for UI display

    Args:
        df:
            Input pandas DataFrame for a single uploaded file.
        target_column:
            Optional name of the target column. Used to add stricter
            checks for missing values and class imbalance.

    Returns:
        A DataFrame with one row per input column and the above fields.
        Suitable for direct JSON serialization to drive the React UI.
    """
    n_rows = len(df)
    summaries: list[dict[str, Any]] = []

    for col in df.columns:
        series = df[col]
        is_target = target_column is not None and col == target_column

        # ----- Missing values ------------------------------------------------
        missing_count = int(series.isna().sum())
        missing_pct = float(missing_count / n_rows) if n_rows > 0 else 0.0

        # Work only with non-missing values for frequencies and uniqueness.
        non_missing = series.dropna()
        non_missing_rows = len(non_missing)

        # ----- Uniqueness / constant flags ----------------------------------
        unique_count = int(non_missing.nunique())
        unique_pct = float(unique_count / n_rows) if n_rows > 0 else 0.0

        is_constant = unique_count == 1 and n_rows > 0
        is_almost_constant = unique_pct < 0.01 and n_rows > 0

        # ----- Frequencies / class balance ----------------------------------
        if non_missing_rows > 0:
            value_counts = non_missing.value_counts()

            most_frequent_value: Any = value_counts.index.tolist()[0]
            most_frequent_count = int(value_counts.iloc[0])
            most_frequent_pct = float(most_frequent_count / non_missing_rows)

            n_classes = int(len(value_counts))

            majority_class: Any = value_counts.index.tolist()[0]
            majority_pct = float(value_counts.iloc[0] / non_missing_rows)

            minority_class: Any = value_counts.index.tolist()[-1]
            minority_pct = float(value_counts.iloc[-1] / non_missing_rows)
        else:
            most_frequent_value = None
            most_frequent_count = 0
            most_frequent_pct = 0.0

            n_classes = 0
            majority_class = None
            majority_pct = 0.0
            minority_class = None
            minority_pct = 0.0

        # ----- Numeric statistics (only for numeric dtypes) ------------------
        if pd.api.types.is_numeric_dtype(series):  # pyright: ignore[reportUnknownMemberType]
            mmean = float(series.mean()) if n_rows > 0 else 0.0
            std = float(series.std()) if n_rows > 1 else 0.0
            col_min = float(series.min()) if n_rows > 0 else 0.0
            col_max = float(series.max()) if n_rows > 0 else 0.0
            median = float(series.median()) if n_rows > 0 else 0.0
        else:
            mmean = None
            std = None
            col_min = None
            col_max = None
            median = None

        # ----- Severity + messages ------------------------------------------
        severity, messages = _derive_column_severity_and_messages(
            is_target=is_target,
            missing_pct=missing_pct,
            missing_count=missing_count,
            n_rows=n_rows,
            is_constant=is_constant,
            is_almost_constant=is_almost_constant,
            n_classes=n_classes,
            majority_pct=majority_pct,
        )

        summaries.append(
            {
                "Column": col,
                "Status": severity,
                "Messages": messages,
                "Data_Type": str(series.dtype),
                "Row_Qty": n_rows,
                #
                "Missing_Qty": missing_count,
                "Missing_Pct": missing_pct,
                #
                "Unique_Qty": unique_count,
                "Unique_Pct": unique_pct,
                #
                "Is_Constant": is_constant,
                "Is_Almost_Constant": is_almost_constant,
                #
                "Most_Frequent_Value": most_frequent_value,
                "Most_Frequent_Qty": most_frequent_count,
                "Most_Frequent_Pct": most_frequent_pct,
                #
                "Classes_Qty": n_classes,
                "Majority_Class": majority_class,
                "Majority_Pct": majority_pct,
                "Minority_Class": minority_class,
                "Minority_Pct": minority_pct,
                #
                "Min": col_min,
                "Median": median,
                "Mean": mmean,
                "Max": col_max,
                "StdDev": std,
            }
        )

    return pd.DataFrame(summaries)


def _derive_column_severity_and_messages(
    *,
    is_target: bool,
    missing_pct: float,
    missing_count: int,
    n_rows: int,
    is_constant: bool,
    is_almost_constant: bool,
    n_classes: int,
    majority_pct: float,
) -> tuple[str, list[str]]:
    """Internal helper to assign a severity level and messages.

    This keeps the main function smaller and ensures the same logic
    is used for both train.csv and test.csv.
    """
    messages: list[str] = []
    severity: str = "ok"

    # Constant columns are almost always useless as features.
    if is_constant and n_rows > 0:
        severity = "warning"
        messages.append("Column is constant (only one distinct value).")

    # Almost constant columns are usually low-value and can be dropped.
    if is_almost_constant and not is_constant:
        severity = max(severity, "info", key=_severity_rank)
        messages.append("Column is almost constant (<1% unique values).")

    # Missing values.
    if missing_count > 0:
        if missing_pct >= 0.05:
            severity = "warning"
            messages.append(f"{missing_pct:.1%} of column values are missing.")
        else:
            severity = max(severity, "info", key=_severity_rank)
            messages.append(f"Column has {missing_count} missing values.")

    # Target-specific checks can be stricter.
    if is_target:
        if missing_count > 0:
            # Should never get to this point.  It is caught way upstream.
            severity = "critical"
            messages.append("Target column cannot contain missing values.")
        # Simple imbalance warning for targets.
        if n_classes >= 2 and majority_pct >= 0.95:
            severity = max(severity, "warning", key=_severity_rank)
            messages.append("Target is imbalanced (majority class ≥ 95%).")

    return severity, messages


def _severity_rank(level: str) -> int:
    """Map a severity string to a numeric rank for comparisons."""
    ranks = {"ok": 0, "info": 1, "warning": 2, "critical": 3}
    return ranks.get(level, 0)
