"""
Module: features.py
"""

# Errors to ignore
# pylint: disable=too-many-lines
# mypy: disable-error-code=import-untyped
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

# Python imports
from dataclasses import dataclass

# import sys
from typing import Any, Callable, cast, ContextManager, Optional, TypeAlias

# Third-Party imports
import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike, NDArray

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype
from scipy.stats import chi2
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# This line is needed for "IterativeImputer", which is experimental.
# pylint: disable=unused-import
from sklearn.experimental import enable_iterative_imputer  # type: ignore[unused-ignore]

# pylint: enable-unused-import
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    # PolynomialFeatures,
)

# Project imports
import tabular.utilities as util
from tabular.utilities import (
    # TaskType,
    # AppError,
    NumericImputation,
    ObjectImputation,
    Option,
    Processor,
    ProcessorType,
    YSeriesType,
)

# Clipped-copy constants.  Reuse the same conservative top-K skewed numeric-column selection logic
# as _add_selective_skew_transforms(), but add clipped copies instead of replacing the originals.
_CLIP_LOWER_QUANTILE = 0.01
_CLIP_UPPER_QUANTILE = 0.99

# Feature pruning constants
# correlation_threshold, variance_quantile, importance_threshold
_FEATURE_PRUNING_THRESHOLDS = (
    (None, None, None),  # will always exclude 100% constant
    (0.98, 0.01, np.nextafter(0.0, 1.0)),  # 1 = Low
    (0.95, 0.02, "0.1*median"),  # 2 = Medium
    (0.93, 0.05, "median"),  # 3 = High
)

# If you have a lot of numeric features fed into PolynomialFeatures(), then the quantity of
# features that it can return will explode.  Setting these to 5 and 3 will create at most 30 new
# features: 5 + 10 + (3*5) = 5 + 10 + 15 = 30.  This is intentionally conservative because the
# goal is a generic 80/20 win, not feature explosion.  NON_BINARY are typically more important than
# BINARY.
_MAX_INPUT_COLS_FOR_NON_BINARY_POLY_FEATURES = 5
_MAX_INPUT_COLS_FOR_BINARY_POLY_FEATURES = 3

# Selective skew-transform constants. These are intentionally conservative because the goal is a
# generic 80/20 win, not feature explosion.
_MIN_UNIQUE_FOR_SKEW_TRANSFORM = 8
_ABS_SKEW_THRESHOLD_FOR_TRANSFORM = 1.0
_MAX_SKEW_TRANSFORM_COLS = 8

# Frequency-encoding constants.  This is intentionally conservative and additive:
# add synthetic_freq_<col> for object columns with truly high cardinality.
_MIN_HIGH_CARDINALITY_OBJECT_UNIQUES_ABS = 20

_MISSING_SUFFIX = "_is_missing"
_MISSING_VALUES = (util.NUMBER_NAN, util.STRING_NAN)

# These steps add a lot of glop and do not seem to help, and actually make things a little worse.
_RUN_FREQUENCY_ENCODING = False
_RUN_SKEW_TRANSFORM = False
_RUN_CLIPPED_FEATURES = False

# Type Aliases
_SyntheticFeaturesFunctionType: TypeAlias = Callable[
    [pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]
]


class MahalanobisOutlier(BaseEstimator, TransformerMixin):
    """Transformer that computes Mahalanobis distance and a binary outlier flag,
    then returns the original data with those two features appended.

    Captures multivariate atypicality by measuring how far each sample lies from
    the empirical center under the covariance structure. Outputs:
        * mahalanobis_distance: continuous distance.
        * mahalanobis_outlier: binary flag (1 if distance exceeds chi-square-derived threshold).
    """

    def __init__(
        self,
        features: list[str],
        quantile: float = 0.99,
    ):
        """Initialize the transformer.

        Args:
            quantile: Chi-square upper quantile used to set the outlier cutoff.
            features: Optional iterable of column names to restrict which features
                are used for Mahalanobis computation. If None, all numeric columns
                from the fit input are inferred.
        """
        self.quantile = quantile
        self.features = features  # None if features is None else list(features)

        # To be populated in fit()
        # self._selected_features_: list[str] | None = None
        self._mean_: NDArray[np.float64] | None = None
        self._cov_inv_: NDArray[np.float64] | None = None
        self._impute_means_: NDArray[np.float64] | None = None  # training-time fill means
        self.threshold_: float | None = None

    def fit(
        self, x: pd.DataFrame, y: ArrayLike | None = None  # pylint: disable=unused-argument
    ) -> "MahalanobisOutlier":
        """Estimate mean, inverse covariance, and outlier threshold from training data.

        Missing values are imputed with column means (computed on the TRAIN data)
        only for internal computations.
        """
        # Subset to selected numeric columns
        x_sel = x[self.features]

        # Training-time means (used both for centering and for NaN filling at inference)
        col_means = x_sel.mean()
        self._mean_ = col_means.to_numpy(dtype=float)
        self._impute_means_ = col_means.to_numpy(dtype=float)

        # Fill missing values with TRAIN means for covariance computation
        filled = x_sel.fillna(col_means)
        filled_numpy_array = cast(NDArray[np.float64], filled.to_numpy(dtype=float))

        # Covariance matrix and robust inversion
        # NOTE: polynomial features make many columns collinear → singular covariance.
        # Use a small diagonal regularization + pseudo-inverse for stability.
        cov_matrix = np.atleast_2d(np.cov(filled_numpy_array.T))
        eps = 1e-6 * np.eye(cov_matrix.shape[0])
        cov_reg = cov_matrix + eps
        self._cov_inv_ = np.linalg.pinv(cov_reg)

        # Degrees of freedom and threshold
        dof = len(self.features)
        if dof <= 0:
            raise RuntimeError("No features available to compute Mahalanobis distance.")
        cutoff = np.sqrt(chi2.ppf(self.quantile, df=dof))
        self.threshold_ = float(cutoff)

        return self

    def _mahalanobis_distance(self, x_sel: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Compute Mahalanobis distance for each row in ``x_sel``."""
        if self._cov_inv_ is None or self._mean_ is None or self._impute_means_ is None:
            raise RuntimeError("Transformer must be fitted before computing Mahalanobis distance.")

        if list(x_sel.columns) != list(self.features):
            raise AssertionError("Input columns do not match fitted feature order.")

        # Use TRAIN-time means for filling, not per-call means.
        fill_values = pd.Series(self._impute_means_, index=self.features)
        filled = x_sel.fillna(fill_values).infer_objects()  # (copy=False)

        # Fail fast on non-numeric data instead of letting object dtype leak into
        # the linear algebra below.
        numeric = filled.apply(pd.to_numeric, errors="raise").astype(np.float64)

        centered = numeric.to_numpy(dtype=np.float64) - self._mean_
        left = np.dot(centered, self._cov_inv_)
        mahal_sq = np.sum(left * centered, axis=1)

        return np.sqrt(np.clip(mahal_sq, a_min=0.0, a_max=None)).astype(np.float64)

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted logic and return original X with appended outlier features.

        Args:
            x: Input DataFrame.

        Returns:
            DataFrame with original columns plus:
                * mahalanobis_distance
                * mahalanobis_outlier
        """
        # if self.features is None:
        #     raise RuntimeError("fit must be called before transform.")

        # Compute distance on selected features
        x_sel = x[self.features]
        dist = self._mahalanobis_distance(x_sel)

        # Build new feature frame
        assert self.threshold_ is not None  # just for lint
        outlier_df = pd.DataFrame(
            {
                "mahalanobis_distance": dist,
                "mahalanobis_outlier": (dist > self.threshold_).astype("int8"),  # numeric 0/1
            },
            index=x.index,
        )

        # Concatenate original data with new features, preserving original order
        result = pd.concat([x, outlier_df], axis=1)
        return result

    def fit_transform(  # type: ignore[unused-ignore] pylint: disable=arguments-differ,arguments-renamed
        self,
        X: pd.DataFrame,
        y: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Fit on X and immediately transform, returning augmented DataFrame."""
        return self.fit(X, y).transform(X)


@dataclass
class _ProcessState:
    """Holds artifacts learned in process_data_fit and snapshots for leak-free apply."""

    # config/opts
    # processors: ProcessorType
    feature_names_to_exclude: set[str] | None
    # task: TaskType
    feature_pruning_threshold: int
    options: Any

    # step-specific learned artifacts
    missingness_cols: list[str]  # <col>_is_missing created in fit
    outlier: MahalanobisOutlier | None  # fitted
    column_transformer: ColumnTransformer  # fitted encode/scale (placeholder ok)
    # var_thresh: VarianceThreshold  # fitted variance filter
    pruned_column_names: list[str] | None  # final pruning list

    # final schema promised to the model
    columns_: list[str]

    # training snapshots used as reference (for leak-free apply)
    ref_impute1: pd.DataFrame  # after missingness, before custom synthetics
    ref_synth: pd.DataFrame | None  # after impute1, for custom synthetics reference
    ref_impute2: pd.DataFrame | None  # after custom synthetics, before AF/outliers/poly

    # stored callable so apply() can replay synthetics without reading options
    synthetic_features_function: Optional[_SyntheticFeaturesFunctionType] = None

    # row-level meta-features
    include_row_meta_features: bool = False

    # selective skew transforms
    skew_transform_base_cols: Optional[list[str]] = None
    skew_transform_nonnegative_cols: Optional[list[str]] = None

    # clipped feature copies
    clipped_feature_bounds: Optional[dict[str, tuple[float, float]]] = None

    # frequency encoding for high-cardinality categoricals
    frequency_encoding_maps: Optional[dict[str, dict[Any, float]]] = None

    # numeric columns used for polynomial features (degree=2)
    polynomial_numeric_cols: Optional[list[str]] = None
    binary_numeric_cols: Optional[list[str]] = None
    polynomial_use_uid_column_name: Optional[str] = None


def _add_missingness_flags(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    """Vectorized: add <base>_is_missing for bases, filling 0 if base missing."""
    if not bases:
        return df

    present = [b for b in bases if b in df.columns]

    if present:
        ind_df = df[present].isna().astype("int8")
        ind_df.columns = [f"{c}{_MISSING_SUFFIX}" for c in present]
    else:
        ind_df = pd.DataFrame(index=df.index)

    # Ensure *all* flags exist (even if some bases missing at apply time)
    all_flags = [f"{b}{_MISSING_SUFFIX}" for b in bases]
    ind_df = ind_df.reindex(columns=all_flags, fill_value=0).astype("int8", copy=False)

    # Overwrite deterministically + avoid fragmentation
    df = df.drop(columns=all_flags, errors="ignore")
    return pd.concat([df, ind_df], axis=1).copy()


def _select_skewed_numeric_candidates(
    df: pd.DataFrame,
    uid_column_name: str | None = None,
) -> list[tuple[str, float, bool]]:
    """Return the conservative top-K skewed numeric columns suitable for generic synthetics.

    Returns tuples of:
        (column_name, abs_skew, is_nonnegative)
    """
    numeric_cols, _ = util.column_names_by_dtype(df, exclude_constant_cols=True)
    candidate_rows: list[tuple[str, float, bool]] = []

    for col in numeric_cols:
        if uid_column_name is not None and col == uid_column_name:
            continue

        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        nunique = int(non_null.nunique())
        if nunique < _MIN_UNIQUE_FOR_SKEW_TRANSFORM:
            continue
        # Treat 2-or-fewer unique values as binary / near-binary and skip.
        if nunique <= 2:
            continue

        skew_val = float(non_null.skew())  # pyright: ignore[reportArgumentType]
        if not np.isfinite(skew_val):
            continue
        if abs(skew_val) < _ABS_SKEW_THRESHOLD_FOR_TRANSFORM:
            continue

        is_nonnegative = bool((non_null >= 0).all())
        candidate_rows.append((col, abs(skew_val), is_nonnegative))

    candidate_rows.sort(key=lambda item: item[1], reverse=True)
    return candidate_rows[:_MAX_SKEW_TRANSFORM_COLS]


def _add_selective_skew_transforms(
    df: pd.DataFrame,
    base_cols: list[str] | None = None,
    nonnegative_cols: list[str] | None = None,
    uid_column_name: str | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Add conservative log-like transforms for highly skewed numeric columns.

    Fit behavior:
        When ``base_cols`` is None, infer a conservative set of numeric columns from ``df``:
        * numeric only
        * exclude UID column
        * exclude binary / near-binary columns
        * require at least `_MIN_UNIQUE_FOR_SKEW_TRANSFORM` non-null unique values
        * require `abs(skew) >= _ABS_SKEW_THRESHOLD_FOR_TRANSFORM`
        * keep only the top `_MAX_SKEW_TRANSFORM_COLS` by absolute skew

    Apply behavior:
        When ``base_cols`` is provided, replay exactly those transforms. ``nonnegative_cols``
        determines which columns use plain ``log1p`` vs signed ``log1p``.

    Returns:
        (augmented_df, fitted_base_cols, fitted_nonnegative_cols)
    """
    out = df.copy()

    if base_cols is None:
        selected = _select_skewed_numeric_candidates(out, uid_column_name=uid_column_name)
        fitted_base_cols = [col for col, _, _ in selected]
        fitted_nonnegative_cols = [col for col, _, is_nonnegative in selected if is_nonnegative]
    else:
        fitted_base_cols = list(base_cols)
        fitted_nonnegative_cols = list(nonnegative_cols or [])

    if not fitted_base_cols:
        return out, fitted_base_cols, fitted_nonnegative_cols

    nonnegative_set = set(fitted_nonnegative_cols)
    new_cols: dict[str, pd.Series] = {}

    missing_base_cols = [col for col in fitted_base_cols if col not in out.columns]
    if missing_base_cols:
        raise ValueError(
            "Missing base columns required for skew transforms: " f"{sorted(missing_base_cols)}"
        )

    for col in fitted_base_cols:
        series = pd.to_numeric(out[col], errors="raise").astype(np.float64)
        if col in nonnegative_set:
            new_name = f"synthetic_log1p_{col}"
            if new_name not in out.columns and new_name not in new_cols:
                new_cols[new_name] = pd.Series(np.log1p(series), index=out.index)
        else:
            new_name = f"synthetic_signed_log1p_{col}"
            if new_name not in out.columns and new_name not in new_cols:
                new_cols[new_name] = pd.Series(
                    np.sign(series) * np.log1p(np.abs(series)),
                    index=out.index,
                )

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out, fitted_base_cols, fitted_nonnegative_cols


def _add_clipped_feature_copies(
    df: pd.DataFrame,
    clip_bounds: dict[str, tuple[float, float]] | None = None,
    uid_column_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Add clipped-copy features using train-learned quantile bounds.

    Fit behavior:
        When ``clip_bounds`` is None, choose the same conservative top-K skewed numeric columns
        used by the selective skew-transform logic, then learn 1% / 99% clipping bounds on TRAIN
        only.

    Apply behavior:
        When ``clip_bounds`` is provided, replay exactly those train-learned bounds.

    Returns:
        (augmented_df, learned_or_replayed_clip_bounds)
    """
    out = df.copy()

    if clip_bounds is None:
        selected = _select_skewed_numeric_candidates(out, uid_column_name=uid_column_name)
        learned_bounds: dict[str, tuple[float, float]] = {}

        for col, _, _ in selected:
            series = pd.to_numeric(out[col], errors="raise").astype(np.float64)
            non_null = series.dropna()
            if non_null.empty:
                continue

            lower = float(non_null.quantile(_CLIP_LOWER_QUANTILE))
            upper = float(non_null.quantile(_CLIP_UPPER_QUANTILE))
            if not np.isfinite(lower) or not np.isfinite(upper):
                continue
            if lower >= upper:
                continue

            learned_bounds[col] = (lower, upper)

        clip_bounds = learned_bounds
    else:
        clip_bounds = dict(clip_bounds)

    if not clip_bounds:
        return out, clip_bounds

    missing_base_cols = [col for col in clip_bounds if col not in out.columns]
    if missing_base_cols:
        raise ValueError(
            "Missing base columns required for clipped copies: " f"{sorted(missing_base_cols)}"
        )

    new_cols: dict[str, pd.Series] = {}
    for col, (lower, upper) in clip_bounds.items():
        new_name = f"synthetic_clipped_{col}"
        if new_name in out.columns or new_name in new_cols:
            continue

        series = pd.to_numeric(out[col], errors="raise").astype(np.float64)
        new_cols[new_name] = series.clip(lower=lower, upper=upper)

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out, clip_bounds


def _add_frequency_encoding_copies(
    df: pd.DataFrame,
    frequency_maps: dict[str, dict[Any, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[Any, float]]]:
    """Add frequency-encoding copies for high-cardinality object columns.

    Fit behavior:
        When ``frequency_maps`` is None, infer eligible object columns from ``df``:
        * object/string columns only
        * cardinality must be strictly greater than max(20, sqrt(n_rows))
        * learn normalized TRAIN frequencies only

    Apply behavior:
        When ``frequency_maps`` is provided, replay exactly those train-learned mappings.
        Unseen values map to 0.0.

    Returns:
        (augmented_df, learned_or_replayed_frequency_maps)
    """
    out = df.copy()

    if frequency_maps is None:
        n_rows = int(len(out))
        cardinality_threshold = max(
            _MIN_HIGH_CARDINALITY_OBJECT_UNIQUES_ABS,
            int(np.sqrt(max(n_rows, 1))),
        )

        _, object_cols = util.column_names_by_dtype(out)
        learned_maps: dict[str, dict[Any, float]] = {}

        for col in object_cols:
            series = out[col]
            nunique = int(series.nunique(dropna=False))
            if nunique <= cardinality_threshold:
                continue

            # Use normalized frequencies so the feature is already in [0, 1].
            freq_series = series.value_counts(normalize=True, dropna=False)
            if freq_series.empty:
                continue

            learned_maps[col] = {key: float(value) for key, value in freq_series.items()}

        frequency_maps = learned_maps
    else:
        frequency_maps = {col: dict(mapping) for col, mapping in frequency_maps.items()}

    if not frequency_maps:
        return out, frequency_maps

    missing_base_cols = [col for col in frequency_maps if col not in out.columns]
    if missing_base_cols:
        raise ValueError(
            "Missing base columns required for frequency encoding: " f"{sorted(missing_base_cols)}"
        )

    new_cols: dict[str, pd.Series] = {}
    for col, mapping in frequency_maps.items():
        new_name = f"synthetic_freq_{col}"
        if new_name in out.columns or new_name in new_cols:
            continue

        # Unseen categories at apply time map to 0.0.
        encoded = out[col].map(mapping).fillna(0.0).astype(np.float64)
        new_cols[new_name] = encoded

    if new_cols:
        out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)

    return out, frequency_maps


def _add_poly2_manual(
    df: pd.DataFrame,
    base_cols: list[str] | None,
    uid_column_name: str | None,
    binary_cols: list[str] | None,
    require_uid_date_features: bool,
) -> tuple[pd.DataFrame, str | None]:
    """Add manual degree-2 polynomial features and optional UID-derived date features."""
    out = df.copy()
    new_cols: dict[str, pd.Series | npt.NDArray[np.float64] | npt.NDArray[np.int_]] = {}

    # ------------------------------------------------------------
    # 1. Date feature extraction from uid (if valid)
    # ------------------------------------------------------------
    polynomial_use_uid_column_name = None
    if uid_column_name is not None and uid_column_name in out.columns:
        col = out[uid_column_name]

        # Must contain no missing values. Only attempt date parsing for actual datetime dtype or
        # string-like columns. Otherwise it will parse any integer into a date.
        if (not col.isna().any()) and (
            is_datetime64_any_dtype(col) or is_object_dtype(col) or is_string_dtype(col)
        ):
            parsed = pd.to_datetime(col, errors="coerce", format="mixed")

            # Only continue if ALL values parsed successfully.
            if not parsed.isna().any():
                polynomial_use_uid_column_name = uid_column_name
                prefix = f"synthetic_{uid_column_name}_"

                # Core integer date components
                day_of_week = parsed.dt.dayofweek  # 0–6
                day_of_month = parsed.dt.day  # 1–31
                month = parsed.dt.month  # 1–12
                quarter = parsed.dt.quarter  # 1–4
                year = parsed.dt.year
                day_of_year = parsed.dt.dayofyear  # 1–365/366
                week_of_year = parsed.dt.isocalendar().week.astype("int8")  # 0-52

                is_month_start = parsed.dt.is_month_start
                is_month_end = parsed.dt.is_month_end
                is_quarter_start = parsed.dt.is_quarter_start
                is_quarter_end = parsed.dt.is_quarter_end
                is_year_start = parsed.dt.is_year_start
                is_year_end = parsed.dt.is_year_end

                # Integer features
                date_features: dict[str, pd.Series] = {
                    f"{prefix}day_of_week": day_of_week,
                    f"{prefix}is_weekend": (day_of_week >= 5).astype("int8"),
                    f"{prefix}day_of_month": day_of_month,
                    f"{prefix}month": month,
                    f"{prefix}quarter": quarter,
                    f"{prefix}year": year,
                    f"{prefix}day_of_year": day_of_year,
                    f"{prefix}week_of_year": week_of_year,
                    f"{prefix}is_month_start": is_month_start.astype("int8"),
                    f"{prefix}is_month_end": is_month_end.astype("int8"),
                    f"{prefix}is_quarter_start": is_quarter_start.astype("int8"),
                    f"{prefix}is_quarter_end": is_quarter_end.astype("int8"),
                    f"{prefix}is_year_start": is_year_start.astype("int8"),
                    f"{prefix}is_year_end": is_year_end.astype("int8"),
                }

                # Cyclical encodings
                dow_rad = 2.0 * np.pi * day_of_week.to_numpy() / 7.0
                month_zero_based = month.to_numpy() - 1
                month_rad = 2.0 * np.pi * month_zero_based / 12.0
                doy_zero_based = day_of_year.to_numpy(dtype=float) - 1.0
                days_in_year = np.where(parsed.dt.is_leap_year.to_numpy(), 366.0, 365.0)
                doy_rad = 2.0 * np.pi * doy_zero_based / days_in_year

                cyclical_features: dict[str, npt.NDArray[np.float64]] = {
                    f"{prefix}sin_day_of_week": np.sin(dow_rad),
                    f"{prefix}cos_day_of_week": np.cos(dow_rad),
                    f"{prefix}sin_month": np.sin(month_rad),
                    f"{prefix}cos_month": np.cos(month_rad),
                    f"{prefix}sin_day_of_year": np.sin(doy_rad),
                    f"{prefix}cos_day_of_year": np.cos(doy_rad),
                }

                for fname, fseries in date_features.items():
                    if fname not in out.columns:
                        new_cols[fname] = fseries

                for fname, values in cyclical_features.items():
                    if fname not in out.columns:
                        new_cols[fname] = values

    if require_uid_date_features and not polynomial_use_uid_column_name:
        raise ValueError(
            f"UID column '{uid_column_name}' must be present, non-missing, and fully "
            f"parseable as dates because fit created UID-derived polynomial/date features."
        )

    # ------------------------------------------------------------
    # 2. Polynomial Feature Generation
    # ------------------------------------------------------------
    if base_cols:
        n = len(base_cols)
        binary_col_set = set(binary_cols or [])

        for i in range(n):
            col_i = base_cols[i]
            if col_i not in out.columns:
                continue

            if col_i not in binary_col_set:
                sq_name = f"synthetic_{col_i}^2"
                if sq_name not in out.columns and sq_name not in new_cols:
                    new_cols[sq_name] = out[col_i] * out[col_i]

            for j in range(i + 1, n):
                col_j = base_cols[j]
                if col_j not in out.columns:
                    continue

                if col_i in binary_col_set and col_j in binary_col_set:
                    continue

                cross_name = f"synthetic_{col_i}*{col_j}"
                if cross_name not in out.columns and cross_name not in new_cols:
                    new_cols[cross_name] = out[col_i] * out[col_j]

    if new_cols:
        new_df = pd.DataFrame(new_cols, index=out.index)
        out = pd.concat([out, new_df], axis=1)

    return out, polynomial_use_uid_column_name


def _add_row_level_meta_features(
    missing_source_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add a very small set of row-level meta-features.

    Keep this intentionally lean.

    Adds only:
        * synthetic_row_missing_count
        * synthetic_row_missing_frac

    Notes:
        * Missingness is computed from ``missing_source_df`` so it reflects the
          pre-imputation state of the current pipeline stage.
        * Features are appended to ``target_df``.
    """
    missing_count = missing_source_df.isna().sum(axis=1).astype("int32")
    denominator = max(int(missing_source_df.shape[1]), 1)
    missing_frac = (missing_count / float(denominator)).astype(np.float64)

    new_cols = pd.DataFrame(
        {
            "synthetic_row_missing_count": missing_count,
            "synthetic_row_missing_frac": missing_frac,
        },
        index=target_df.index,
    )

    return pd.concat([target_df, new_cols], axis=1)


def _assert_no_missing_values(
    x_train: pd.DataFrame | pd.Series,
    x_test: pd.DataFrame | pd.Series,
    caller: str = "Unknown Caller",
    show_column_names: bool = False,
) -> None:
    """
    Validate that neither x_train nor x_test contains:
        • NaN
        • CATEGORY_NAN sentinel ("__N_a_N__")

    Works for both DataFrames and Series.

    Raises:
        AssertionError if any missing-like values are detected.
    """
    # Convert to DataFrame for unified logic
    if isinstance(x_train, pd.Series):
        x_train = x_train.to_frame()
    if isinstance(x_test, pd.Series):
        x_test = x_test.to_frame()

    if any(
        df.isna().any().any() or df.isin(_MISSING_VALUES).any().any() for df in (x_train, x_test)
    ):
        message = f"Missing values in {caller}: None or NaN or _NUMBER_NAN or _STRING_NAN present."
        if show_column_names:
            cols = set(
                x_train.columns[(x_train.isna() | x_train.isin(_MISSING_VALUES)).any()]
            ) | set(x_test.columns[(x_test.isna() | x_test.isin(_MISSING_VALUES)).any()])
            message += f" {cols}"
        raise AssertionError(message)


def _choose_default_numeric_imputation(
    x_train: pd.DataFrame,
    numeric_cols: list[str],
    column_name: str,
) -> NumericImputation:
    """Pick a conservative automatic numeric imputation strategy.

    Heuristic goals:
        * Use MEDIAN by default.
        * Only use ITERATIVE when the problem size is modest and missingness is not extreme.

    Rationale:
        Iterative imputation can help when there are enough correlated numeric features and the
        training set is not too large, but it is slower and less predictable than median
        imputation. This helper intentionally biases toward MEDIAN unless the data looks like a
        good fit for ITERATIVE.
    """
    if column_name not in x_train.columns:
        raise AssertionError(f"Unknown numeric column '{column_name}'")

    if column_name not in numeric_cols:
        raise AssertionError(f"Column '{column_name}' must be numeric")

    n_rows = int(len(x_train))
    n_numeric = int(len(numeric_cols))
    col_missing_frac = float(x_train[column_name].isna().mean())
    overall_numeric_missing_frac = float(x_train[numeric_cols].isna().mean().mean())
    non_null_unique_count = int(x_train[column_name].dropna().nunique())

    # Fail fast on obviously bad iterative-imputation cases.
    if n_rows == 0:
        return NumericImputation.MEDIAN
    if n_numeric < 2:
        return NumericImputation.MEDIAN
    if n_rows > 5000:
        return NumericImputation.MEDIAN
    if n_numeric > 40:
        return NumericImputation.MEDIAN
    if col_missing_frac <= 0.0:
        return NumericImputation.MEDIAN
    if col_missing_frac > 0.35:
        return NumericImputation.MEDIAN
    if overall_numeric_missing_frac > 0.20:
        return NumericImputation.MEDIAN
    if non_null_unique_count <= 2:
        return NumericImputation.MEDIAN

    return NumericImputation.ITERATIVE


def _create_onehot_pipeline(min_frequency: int, max_categories: int) -> Pipeline:
    # Create a onehot_pipeline.
    # We could specify drop="first" to avoid collinearity.  ChatGPT says: Only drop one level when
    # you need a full-rank design matrix with an intercept (i.e. ordinary linear/logistic
    # regression without regularization). In all other modeling scenarios, it’s perfectly fine, and
    # often preferable to keep the complete one-hot encoding.  I did try dropping on 07/05/25 and
    # got definitely worse results.
    # ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    #
    return Pipeline(
        [
            # ("to_str", FunctionTransformer(_to_str, validate=False)),
            (
                "encoder",
                # Do not need to scale OneHot.
                OneHotEncoder(
                    # Need sparse_output=False to avoid a bunch of warnings.
                    sparse_output=False,
                    # "error" (default): raise.
                    # "ignore": output all-zeros for that feature.
                    # "infrequent_if_exist": map unknowns to the infrequent bin if it exists for
                    #  that feature; otherwise all-zeros.
                    handle_unknown="infrequent_if_exist",
                    # Any category whose frequency is below this threshold is grouped into a single
                    # “infrequent” bin.
                    min_frequency=min_frequency,
                    # Caps how many one-hot columns a feature can expand to. If a feature has more
                    # unique values, the least frequent ones are grouped into the infrequent bin so
                    # the total columns ≤ max_categories (including the infrequent bin when
                    # present).
                    max_categories=max_categories,
                ),
            ),
        ]
    )


def _create_ordinal_pipeline(categories: tuple[str, ...]) -> Pipeline:
    """xxx"""
    # Create an ordinal_pipeline.  Unknown values and missing values will be encoded as -1 because
    # you have unknown_value=-1
    # a) If categories = ["Low", "Medium", "High"], and you have "__missing__", then the
    #    "__missing__" will be considered an unknown_value (because it is not in categories), and
    #    will get a value of -1.  Low=0, Medium=1, High=2 by default.
    # b) Technically the encoded_missing_value=-1 is not needed because the encoder has already
    #    handled the missing values.
    return Pipeline(
        [
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,  # Not in categories (i.e. "__missing__")
                    encoded_missing_value=-1,  # (i.e., np.nan or pd.NA)
                    # categories must be in square brackets.  I do not know why.
                    categories=([categories]),  # if categories is not None else "auto"),
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )


def _ct_output_to_dataframe(
    ct: ColumnTransformer,
    transformed: npt.NDArray[np.float_],
    index: pd.Index,
) -> pd.DataFrame:
    """
    Rebuild the output of ColumnTransformer.transform(...) into a DataFrame.

    Args:
        ct:
            A fitted ColumnTransformer (must implement get_feature_names_out()).

        transformed:
            The ndarray returned by ct.transform(...).

        index:
            The index to assign to the resulting DataFrame (must match the
            original input DataFrame's index).

    Returns:
        A pandas DataFrame whose columns correspond to ct.get_feature_names_out(),
        and whose values come from `transformed`.
    """
    # feature names
    feature_names_arr: npt.NDArray[np.str_] = cast(
        npt.NDArray[np.str_], ct.get_feature_names_out()
    )
    feature_names: list[str] = feature_names_arr.tolist()

    # rebuild DataFrame
    df = pd.DataFrame(
        data=transformed,
        columns=feature_names,
        index=index,
    )
    return df


def _encode_and_scale(
    x_fit: pd.DataFrame,
    x_apply: pd.DataFrame,
    processors: ProcessorType,
) -> tuple[pd.DataFrame, ColumnTransformer | None]:
    """
    Build a ColumnTransformer from `processors`, fit it on `x_fit`, and transform `x_apply`.

    Returns:
        (encoded_scaled_x_apply, fitted_column_transformer | None)

        encoded_scaled_x_apply:
            A pandas DataFrame (pd.DataFrame) whose columns are the transformed feature
            names from the ColumnTransformer, and whose index matches `x_apply`.
    """
    # Identify objects/numerics from current schema
    numeric_cols, object_cols = util.column_names_by_dtype(x_fit, x_apply)

    # ============================
    # Adaptive categorical settings
    # ============================
    n_rows = int(len(x_fit))
    min_frequency = int(round(0.005 * n_rows))  # 0.5% of rows
    min_frequency = max(8, min(200, min_frequency))  # clamp to [8, 200]
    max_categories = 80 if n_rows >= 5000 else 50  # small data → 50, larger → 80

    # Category columns. Choose encoder per categorical column.
    category_pipelines: dict[str, Pipeline] = {}
    for col in object_cols:
        ranked_categories = None
        if col in processors:
            ranked_categories = processors[col].get(util.Processor.RANKED_CATEGORIES)
        category_pipelines[col] = (
            _create_onehot_pipeline(min_frequency, max_categories)
            if ranked_categories is None
            else _create_ordinal_pipeline(ranked_categories)
        )

    # Numeric columns
    transformers: list[tuple[str, Pipeline, list[str]]] = [
        ("numerics", Pipeline([("scaler", RobustScaler())]), numeric_cols)
    ]

    for feature_name, pipeline in category_pipelines.items():
        transformers.append((f"{util.STRING_PREFIX}{feature_name}", pipeline, [feature_name]))

    ct = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=True,  # False,
    )

    with util.suppress_stdout_stderr():
        # Fit on x_fit
        ct.fit(x_fit)
        # Transform x_apply and force pandas output locally so we don't depend on global set_config
        with cast(ContextManager[None], config_context(transform_output="pandas")):
            x_apply_out = ct.transform(x_apply)

    x_apply_enc_df = cast(pd.DataFrame, x_apply_out)

    # Defensive: ensure index is aligned to x_apply (it should be, but be explicit)
    x_apply_enc_df.index = x_apply.index

    return x_apply_enc_df, ct


def _print_fit(df: pd.DataFrame, message: str) -> None:
    util.print_local(
        f"#################################################### {message}: {df.shape[1]}"
    )


def _impute(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    **options: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """xxx"""
    if x_test is x_train:
        x_test = x_train.copy()

    # Convert all non-nullable bools to non-nullable ints.
    # 1. pd.read_csv() will load a column as a bool if the column contains only True/False
    #    (any case) and no nulls.
    # 2. The user CUSTOM_SYNTHETICS_FUNCTION might also create a bool column.
    # 3. There is no way that pandas will ever load a column as a nullable Boolean.
    # 4. The only way to get a nullable Boolean is if the user explicitly adds a nullable Boolean
    #    column in their custom synthetics function, but that would be a very strange thing to do
    #    and is not worth trying to handle.  If the user does create a nullable Boolean column in
    #    their custom synthetics function, then that column will be loaded as an object dtype, and
    #    the SimpleImputer() will impute the missing values with the string "missing", and then the
    #    OrdinalEncoder() will encode it as -1, which is a reasonable outcome.
    for col in x_train.select_dtypes(include="bool").columns:
        x_train[col] = x_train[col].map({True: 1, False: 0}).astype("int8")  # numeric 0/1
    for col in x_test.select_dtypes(include="bool").columns:
        x_test[col] = x_test[col].map({True: 1, False: 0}).astype("int8")  # numeric 0/1

    numeric_cols, object_cols = util.column_names_by_dtype(x_train, x_test)

    # Trainers do not like np.inf, so replace np.inf with a real number.
    x_train[numeric_cols] = x_train[numeric_cols].replace(
        [np.inf, -np.inf], [util.FLOAT_MAX, util.FLOAT_MIN]
    )
    x_test[numeric_cols] = x_test[numeric_cols].replace(
        [np.inf, -np.inf], [util.FLOAT_MAX, util.FLOAT_MIN]
    )

    # SimpleImputer() will not replace "None", only "NaN".  So replace all "None" with "NaN".
    x_train = x_train.replace({None: np.nan})
    x_test = x_test.replace({None: np.nan})

    doing_custom = options.get(Option.CUSTOM_SYNTHETICS_FUNCTION)
    processors: ProcessorType = options.get(Option.PROCESSORS, {})
    pipelines: dict[str, Pipeline] = {}
    for col in x_train.columns:
        fill_value = processors[col].get(Processor.FILL_VALUE) if col in processors else None
        default_fill_value = fill_value is None and doing_custom is None
        if col in object_cols:
            if default_fill_value:
                fill_value = ObjectImputation.MISSING
            pipelines[col] = _impute_object(x_train, x_test, col, fill_value)
        else:
            if default_fill_value:
                # fill_value = NumericImputation.MEDIAN
                fill_value = _choose_default_numeric_imputation(x_train, numeric_cols, col)
            pipelines[col] = _impute_numeric(x_train, x_test, col, fill_value)

    transformers = []
    for feature_name, pipeline in pipelines.items():
        transformers.append((f"{util.STRING_PREFIX}{feature_name}", pipeline, [feature_name]))

    transformer: ColumnTransformer = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
        remainder="passthrough",
    )

    # ---- Fit + transform train / test (force pandas output locally) ----
    with cast(ContextManager[None], config_context(transform_output="pandas")):
        x_train_out = transformer.fit_transform(x_train)
        x_test_out = transformer.transform(x_test)

    x_train = cast(pd.DataFrame, x_train_out)
    x_test = cast(pd.DataFrame, x_test_out)

    _assert_no_missing_values(x_train, x_test, "impute()", True)

    return x_train, x_test


def _impute_numeric(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    column_name: str,
    fill_value: NumericImputation | float | int | None,
) -> Pipeline:
    """xxx"""
    if isinstance(fill_value, NumericImputation):
        if fill_value == NumericImputation.ITERATIVE:
            # Note that IterativeImputer() will use the median() if it cannot find a strong enough
            # signal with all of the other features to iterate.
            imputer = IterativeImputer(
                estimator=RandomForestRegressor(),
                random_state=util.RANDOM_STATE,
                initial_strategy="median",
            )
        elif fill_value == NumericImputation.MEDIAN:
            imputer = SimpleImputer(strategy="median")
        elif fill_value == NumericImputation.MINUS_1:
            imputer = SimpleImputer(strategy="constant", fill_value=-1)
        elif fill_value == NumericImputation.ZERO:
            imputer = SimpleImputer(strategy="constant", fill_value=0)
        else:
            raise ValueError(f"Unknown NumericImputation '{fill_value}'")
    else:
        if fill_value is None:
            _assert_no_missing_values(
                x_train[column_name],
                x_test[column_name],
                f"_impute_numeric(), The column {column_name} needs a processor fill_value.",
            )
            fill_value = util.NUMBER_NAN
        assert isinstance(fill_value, (float, int))  # could be a user 'processor' spec error?
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)

    return Pipeline([("imputer", imputer)])


def _impute_object(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    column_name: str,
    fill_value: ObjectImputation | str | None,
) -> Pipeline:
    """xxx"""
    if isinstance(fill_value, ObjectImputation):
        if fill_value == ObjectImputation.MISSING:
            imputer = SimpleImputer(strategy="constant", fill_value=util.STRING_MISSING)
        elif fill_value == ObjectImputation.MOST_FREQUENT:
            imputer = SimpleImputer(strategy="most_frequent")
        else:
            raise ValueError(f"Unknown StringImputation '{fill_value}'")
    else:
        if fill_value is None:
            _assert_no_missing_values(
                x_train[column_name],
                x_test[column_name],
                f"_impute_string(), The column '{column_name}' needs a processor fill_value",
            )
            fill_value = util.STRING_NAN
        assert isinstance(fill_value, str)  # could be a user 'processor' spec error?
        imputer = SimpleImputer(strategy="constant", fill_value=fill_value)

    return Pipeline([("imputer", imputer)])


def process_data_apply(x: pd.DataFrame, state: _ProcessState) -> pd.DataFrame:
    """
    Apply the previously learned mapping to new data x (val/test) without refitting.
    Mirrors the step order from `process_data_fit`.
    """
    z = x.copy()

    # ---- 0) Missingness indicators (same bases as fit) ----
    z = _add_missingness_flags(z, state.missingness_cols)

    # Keep the current pre-imputation view for row-level missingness features.
    row_meta_missing_source = z.copy()

    # ---- 1) Impute using train ref (ref_impute1) ----
    # IMPORTANT: use a copy of the training snapshot so it is never mutated.
    _, z = _impute(state.ref_impute1.copy(), z, **state.options)

    # Copy just so you don't get defrag messages.
    z = z.copy()

    # ---- 2) Custom synthetics (replay with the TRAIN snapshot) + re-impute (ref_impute2) ----
    if state.synthetic_features_function is not None:
        assert state.ref_synth is not None  # for lint
        assert state.ref_impute2 is not None  # for lint
        # Use copies of the training snapshots to avoid mutating stored state.
        _, z = state.synthetic_features_function(state.ref_synth.copy(), z)

        # Refresh the pre-imputation view because custom synthetics may have introduced nulls.
        row_meta_missing_source = z.copy()

        _, z = _impute(state.ref_impute2.copy(), z, **state.options)

    # ---- 2a) Row-level meta-features (transform only; no refit) ----
    if state.include_row_meta_features:
        z = _add_row_level_meta_features(row_meta_missing_source, z)

    # ---- 2b) Frequency encoding copies (transform only; no refit) ----
    if state.frequency_encoding_maps:
        z, _ = _add_frequency_encoding_copies(
            z,
            frequency_maps=state.frequency_encoding_maps,
        )

    # ---- 2c) Selective skew transforms (transform only; no refit) ----
    if state.skew_transform_base_cols:
        z, _, _ = _add_selective_skew_transforms(
            z,
            base_cols=state.skew_transform_base_cols,
            nonnegative_cols=state.skew_transform_nonnegative_cols,
        )

    # ---- 2d) Clipped feature copies (transform only; no refit) ----
    if state.clipped_feature_bounds:
        z, _ = _add_clipped_feature_copies(
            z,
            clip_bounds=state.clipped_feature_bounds,
        )

    # ---- 3) Polynomial features (degree=2; transform only, manual) ----
    if state.polynomial_use_uid_column_name is not None or state.polynomial_numeric_cols:
        poly_cols = (
            [c for c in state.polynomial_numeric_cols if c in z.columns]
            if state.polynomial_numeric_cols
            else None
        )
        binary_cols = (
            [c for c in state.binary_numeric_cols if c in z.columns]
            if state.binary_numeric_cols
            else None
        )
        z, _ = _add_poly2_manual(
            z,
            poly_cols,
            state.polynomial_use_uid_column_name,
            binary_cols,
            state.polynomial_use_uid_column_name is not None,
        )

    # ---- 4) Outlier features (transform only) ----
    if state.outlier is not None:
        z = state.outlier.transform(z)

    # ---- 7) Exclusions ----
    if state.feature_names_to_exclude:
        # excl_set = set(state.feature_names_to_exclude)
        keep = [c for c in z.columns if c not in state.feature_names_to_exclude]
        z = z[keep]

    # ---- 5) Encode + scale using the *same fitted CT* from fit ----
    z_arr = cast(npt.NDArray[np.float_], state.column_transformer.transform(z))
    z = _ct_output_to_dataframe(state.column_transformer, z_arr, z.index)

    # ---- 6) Pruning: drop the same columns learned in fit ----
    if state.pruned_column_names is not None:
        drop_set = set(state.pruned_column_names)
        keep = [c for c in z.columns if c not in drop_set]
        z = z[keep]

    # Align to final train schema
    z = z.reindex(columns=state.columns_, fill_value=0)
    return z


def process_data_fit(
    x_train: pd.DataFrame,
    y_train: YSeriesType,
    feature_pruning_threshold: int,
    **options: Any,
) -> tuple[pd.DataFrame, _ProcessState]:
    """
    Learn everything from x_train/y_train and return (x_train_processed, state).
    Ensures we can replay identical transforms on VAL/TEST (no refitting).
    """
    processors: ProcessorType = options.get(Option.PROCESSORS, {})
    synthetic_features_function = options.get(Option.CUSTOM_SYNTHETICS_FUNCTION)
    uid_column_name = options.get(Option.UID_COLUMN_NAME)

    x = x_train.copy()

    # ---- 0) Missingness indicators (fit: decide which to add) ----
    na_counts = x.isna().sum()
    mask = (na_counts > 0) & (na_counts < len(x))
    missingness_bases = na_counts.index[mask].tolist()
    x = _add_missingness_flags(x, missingness_bases)
    # Store *bases* (not flags) to replay identically in apply.
    missingness_cols = missingness_bases

    # Keep the current pre-imputation view for row-level missingness features.
    row_meta_missing_source = x.copy()

    # ---- 1) Impute (ref_impute1) ----
    x, _ = _impute(x, x, **options)  # fit-on-train via first arg
    ref_impute1 = x.copy()

    # Copy just so you don't get defrag messages.
    x = x.copy()

    # ---- 2) Custom synthetics (optional) + re-impute (ref_impute2) ----
    _print_fit(x, "Features before synthetics")
    if synthetic_features_function is not None:
        # Train-only construction: pass train for both args; keep first
        x, _ = synthetic_features_function(x, x.copy())
        ref_synth = x.copy()

        # Refresh the pre-imputation view because custom synthetics may have introduced nulls.
        row_meta_missing_source = x.copy()

        x, _ = _impute(x, x, **options)
        ref_impute2 = x.copy()
    else:
        ref_synth = None
        ref_impute2 = None
    _print_fit(x, "Features after synthetics")

    # Validate column names after synthetics.
    if not all(col in x.columns for col in processors):
        raise ValueError("Unknown keys in processors")

    # ---- 2a) Row-level meta-features (fit + transform) ----
    x = _add_row_level_meta_features(row_meta_missing_source, x)
    include_row_meta_features = True
    _print_fit(x, "Features after row meta-features")

    # ---- 2b) Frequency encoding copies (fit + transform) ----
    frequency_encoding_maps: dict[str, dict[Any, float]] | None = None
    if _RUN_FREQUENCY_ENCODING:
        x, frequency_encoding_maps = _add_frequency_encoding_copies(x)
        _print_fit(x, "Features after frequency encoding")

    # ---- 2c) Selective skew transforms (fit + transform) ----
    skew_transform_base_cols: list[str] | None = None
    skew_transform_nonnegative_cols: list[str] | None = None
    if _RUN_SKEW_TRANSFORM:
        x, skew_transform_base_cols, skew_transform_nonnegative_cols = (
            _add_selective_skew_transforms(
                x,
                uid_column_name=uid_column_name,
            )
        )
        _print_fit(x, "Features after skew transforms")

    # ---- 2d) Clipped feature copies (fit + transform) ----
    clipped_feature_bounds: dict[str, tuple[float, float]] | None = None
    if _RUN_CLIPPED_FEATURES:
        x, clipped_feature_bounds = _add_clipped_feature_copies(
            x,
            uid_column_name=uid_column_name,
        )
        _print_fit(x, "Features after clipped copies")

    # ---- 3) Polynomial features (fit + transform; degree=2, manual) ----
    polynomial_numeric_cols: list[str] | None = None
    binary_numeric_cols: list[str] | None = None
    polynomial_use_uid_column_name: str | None = None
    # Get the appropriate numeric columns for creating polynomial features.
    numeric_cols, _ = util.column_names_by_dtype(x, exclude_constant_cols=True)
    # Remove uid_column_name from numeric_cols since we handle it separately in _add_poly2_manual.
    if uid_column_name:
        numeric_cols = [s for s in numeric_cols if s != uid_column_name]
    # Do not allow clipped-copy features to feed polynomial expansion in this first version.
    numeric_cols = [s for s in numeric_cols if not s.startswith("synthetic_clipped_")]
    if util.INCLUDE_POLYNOMIAL_FEATURES[options[Option.SPEED]]:
        if uid_column_name or numeric_cols:
            # ADDED: split numeric columns into binary vs non-binary numeric columns.
            binary_numeric_cols = []  # ADDED
            non_binary_numeric_cols: list[str] = []  # ADDED
            for col in numeric_cols:  # ADDED
                non_null_unique_count = x[col].dropna().nunique()  # ADDED
                if (
                    non_null_unique_count <= 2
                ):  # ADDED: treat 2-or-fewer unique numeric values as binary
                    binary_numeric_cols.append(col)  # ADDED
                else:  # ADDED
                    non_binary_numeric_cols.append(col)  # ADDED

            # MODIFIED: rank only non-binary numeric columns by variance.
            var_series = x[non_binary_numeric_cols].var().sort_values(ascending=False)  # MODIFIED

            # ADDED: rank binary numeric columns separately by variance.
            binary_var_series = x[binary_numeric_cols].var().sort_values(ascending=False)  # ADDED

            top_variance_non_binary_cols = var_series.index[
                :_MAX_INPUT_COLS_FOR_NON_BINARY_POLY_FEATURES
            ].tolist()  # ADDED

            top_variance_binary_cols = binary_var_series.index[
                :_MAX_INPUT_COLS_FOR_BINARY_POLY_FEATURES
            ].tolist()  # ADDED

            # ADDED: preserve order and avoid duplicates.
            polynomial_numeric_cols = top_variance_non_binary_cols + [  # ADDED
                col
                for col in top_variance_binary_cols
                if col not in top_variance_non_binary_cols  # ADDED
            ]

            # MODIFIED: pass binary column info so binary^2 and binary*binary can be skipped.
            x, polynomial_use_uid_column_name = _add_poly2_manual(
                x, polynomial_numeric_cols, uid_column_name, binary_numeric_cols, False
            )  # MODIFIED

    _print_fit(x, "Features after polynomial features")

    # ---- 4) Outlier features (fit + transform) ----
    # Run this step after polynomial feature generation in pipeline order so the
    # outlier features themselves are not fed back into polynomial expansion.
    # However, Mahalanobis is intentionally fitted on the pre-polynomial numeric
    # columns only, because polynomial features are highly collinear and can make
    # covariance inversion unstable.
    outlier: MahalanobisOutlier | None = None
    if util.INCLUDE_MAHALANOBIS_OUTLIER[options[Option.SPEED]]:
        if numeric_cols:
            outlier = MahalanobisOutlier(features=numeric_cols)
            x = outlier.fit_transform(x)
            _print_fit(x, "Features after MahalanOutliers")

    # ---- 7) Exclusions ----
    # Get the feature names to exclude.
    feature_names_to_exclude: set[str] = set(options.get(Option.FEATURE_NAMES_TO_EXCLUDE, []))
    if uid_column_name:
        # Always add the UID.
        feature_names_to_exclude.add(uid_column_name)
    keep = [c for c in x.columns if c not in feature_names_to_exclude]
    x = x[keep]
    _print_fit(x, "Features after user-defined droppings")

    # ---- 5) Encode + scale (fit + transform) ----
    x, ct = _encode_and_scale(x, x, processors)
    assert ct is not None, "encode/scale must return a fitted ColumnTransformer"  # programmer bug
    _print_fit(x, "Features after encoding")

    # ---- 6) Pruning (fit; keep only the drop list to replay on apply) ----
    x, pruned_column_names = _prune_features(x, y_train, feature_pruning_threshold, **options)
    _print_fit(x, "Features after pruning")

    # Set the state of x_train for each corresponding apply (transform) step for x_test.
    state = _ProcessState(
        columns_=list(x.columns),  # final schema promised to the model
        column_transformer=ct,  # << store fitted CT
        feature_names_to_exclude=feature_names_to_exclude,
        feature_pruning_threshold=feature_pruning_threshold,
        missingness_cols=missingness_cols,
        # task=task,
        options=options,
        outlier=outlier,
        pruned_column_names=pruned_column_names,
        ref_impute1=ref_impute1,
        ref_impute2=ref_impute2,
        ref_synth=ref_synth,
        synthetic_features_function=synthetic_features_function,
        include_row_meta_features=include_row_meta_features,
        frequency_encoding_maps=frequency_encoding_maps,
        skew_transform_base_cols=skew_transform_base_cols,
        skew_transform_nonnegative_cols=skew_transform_nonnegative_cols,
        clipped_feature_bounds=clipped_feature_bounds,
        polynomial_numeric_cols=polynomial_numeric_cols,
        binary_numeric_cols=binary_numeric_cols,  # ADDED
        polynomial_use_uid_column_name=polynomial_use_uid_column_name,  # ADDED
    )

    # Return (x_train_processed, state) to match WholeDataPreprocessor.fit
    return x, state


def _prune_features(
    x_train: pd.DataFrame,
    y_train: YSeriesType,
    # task: TaskType,
    feature_pruning_threshold: int,
    **options: Any,
) -> tuple[pd.DataFrame, list[str]]:
    """Automatically drop unhelpful and redundant features from a DataFrame.

    This function performs a sequence of low-risk pruning steps intended to remove
    features that are unlikely to carry predictive signal or that tend to harm generalization.

    Steps:
      0. Drop 100% constant columns.
      1. Drop the bottom `var_quantile` fraction of numeric columns by variance (quasi-constant).
      2. Drop *conservatively detected* UID/ID columns (name indicates ID + values are unique and
         non-null).  (We intentionally do NOT drop "mostly unique" numeric columns, since they can
         be valid predictors like square footage, price, timestamps-as-int, etc.)
      3. Drop exact duplicate columns.
      4. Drop 1 feature from each highly correlated numeric pair (|Pearson corr| > `corr_thresh`).
      5. Optional: model-based pruning using RandomForest + SelectFromModel.

    Returns:
        (x_train_pruned, pruned_column_names)
    """
    threshold_values = _FEATURE_PRUNING_THRESHOLDS[feature_pruning_threshold]
    corr_thresh: float | None = threshold_values[0]
    var_quantile: float | None = threshold_values[1]
    importance_threshold: float | str | None = threshold_values[2]

    original_column_names = x_train.columns

    # 0) Always drop constant columns.
    columns_to_drop: set[str] = util.get_constant_columns(x_train)

    # 1) Data-driven variance threshold for numeric columns only.
    if var_quantile is not None:
        numeric_cols, _ = util.column_names_by_dtype(x_train)
        if numeric_cols:
            x_train_numerics = x_train[numeric_cols]
            variances = x_train_numerics.var()

            # Only apply quantile-based pruning when we have enough numeric features.
            # For very small numbers of columns (e.g. 1–4), this logic tends to just
            # "drop the lowest-variance column", which is usually not what we want.
            # Using interpolation="lower" makes var_quantile for very low quantiles more
            # predictable (it will never come out above the smallest variance).
            if len(variances) >= 5:
                var_cutoff = variances.quantile(var_quantile, interpolation="lower")
                vt = VarianceThreshold(threshold=float(var_cutoff))
                try:
                    vt.fit(x_train_numerics)
                    columns_to_drop |= set(x_train_numerics.columns[~vt.get_support()])
                except ValueError:
                    # If the data is too sparse, the fit will get errors like:
                    # ValueError: No feature in X meets the variance threshold 0.32101
                    pass

    # 2) UID-like columns (conservative; name + unique + non-null).  Note that at this point, uid
    # integer columns have been probably been encoded into floats.  Even imputing might turn them
    # into floats.  So they will probably not be found.  But the real uid will be excluded in
    # step 7 exclusions.  So in reality, this step probably does nothing.  But leave it for clarity
    uid_cols = [c for c in x_train.columns if util.looks_like_uid(c, x_train[c])]
    columns_to_drop |= set(uid_cols)

    # 3) Exact duplicates.
    duplicated_mask = x_train.T.duplicated()
    columns_to_drop |= set(x_train.columns[duplicated_mask])

    # Apply these drops before checking correlation.
    if columns_to_drop:
        x_train = x_train.drop(columns=columns_to_drop)

    # 4) High correlation (numeric only; pandas corr() will ignore non-numerics).
    if corr_thresh is not None and not x_train.empty:
        corr_matrix = x_train.corr(numeric_only=True).abs()
        if not corr_matrix.empty:
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
            corr_drops = {col for col in upper.columns if (upper[col] > corr_thresh).any()}
            if corr_drops:
                x_train = x_train.drop(columns=corr_drops)
                columns_to_drop |= corr_drops

    # 5) Drop low-importance features.  Do not explode.  Use n_jobs=2 instead of n_jobs=-1.
    if util.PRUNE_FEATURES_BY_IMPORTANCE_THREHSOLDS[options[Option.SPEED]]:
        if (importance_threshold is not None) and (not x_train.empty):
            if options[Option.TASK] == "classification":
                base_model = RandomForestClassifier(random_state=util.RANDOM_STATE, n_jobs=2)
            else:
                base_model = RandomForestRegressor(random_state=util.RANDOM_STATE, n_jobs=2)

            with util.suppress_stdout_stderr():
                base_model.fit(x_train, y_train)
                selector = SelectFromModel(estimator=base_model, threshold=importance_threshold)
                # With sklearn set_config(transform_output="pandas"), this returns a DataFrame.
                # x_train = cast(pd.DataFrame, selector.fit_transform(x_train, y_train))
                with cast(ContextManager[None], config_context(transform_output="pandas")):
                    x_train = cast(pd.DataFrame, selector.fit_transform(x_train, y_train))

    pruned_column_names = sorted([c for c in original_column_names if c not in x_train.columns])
    return x_train, pruned_column_names
