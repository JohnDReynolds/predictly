"""
Module: trainers.py
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pylint: disable=broad-exception-caught disable=too-many-lines
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
# pyright: reportAttributeAccessIssue=false
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

# Python imports
from __future__ import annotations  # Needed for pandas type hints
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast, Optional

# Third-Party imports
import tabular.utilities as util

with util.suppress_stdout_stderr():
    from autogluon.features.generators import IdentityFeatureGenerator
    from autogluon.tabular import TabularPredictor
from flaml import AutoML
import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Project imports
from tabular.hyperparams import make_hp_spaces_agl, make_hp_spaces_flaml
import tabular.features as features
from tabular.utilities import (
    MLFramework,
    TaskType,
    ModelResult,
    YSeriesType,
    Option,
    YPredictionsType,
)


@dataclass
class _ProbCal:
    """One-dimensional probability calibrator.

    Attributes:
        model: Underlying calibration model, either isotonic regression or
            logistic regression for Platt scaling.
        kind: Calibration strategy. Expected values are ``"isotonic"`` and
            ``"platt"``.
    """

    model: IsotonicRegression | LogisticRegression
    kind: str

    def transform(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calibrate raw probabilities.

        Args:
            arr: One-dimensional array of raw probabilities.

        Returns:
            Calibrated probabilities clipped to the range [0.0, 1.0].
        """
        a = np.asarray(arr, dtype=float)

        if self.kind == "isotonic":
            out = self.model.predict(a)
        else:
            out = self.model.predict_proba(a.reshape(-1, 1))[:, 1]

        return np.clip(out, 0.0, 1.0)


@dataclass
class TrainingState:
    """Persistent state saved with per-framework training artifacts.

    Attributes:
        metric: User-facing metric name used during training.
        task: Task type, either classification or regression.
        n_splits: Number of cross-validation folds saved as artifacts.
        label_encoder: Optional label encoder used for classification targets.
        best_threshold: Optional tuned binary-classification threshold.
        calibrator: Optional probability calibrator for binary classification.
    """

    metric: str
    task: TaskType
    n_splits: int
    label_encoder: Optional[LabelEncoder]
    best_threshold: float | None
    calibrator: Any | None


class WholeDataPreprocessor(BaseEstimator, TransformerMixin):
    """Fold-safe adapter around fit/apply preprocessing functions.

    The adapter lets sklearn-compatible training code use the project preprocessing
    API without leaking validation/test data into fit-time preprocessing decisions.

    Behavior:
        - ``fit`` learns preprocessing from the training fold only.
        - ``transform`` replays the same learned mapping for validation/test rows.
        - Output columns are aligned to the training-preprocessed schema.
    """

    def __init__(
        self,
        *,
        fit_func: Any,
        apply_func: Any,
        feature_pruning_threshold: Any,
        **options: Any,
    ) -> None:
        self._fit_func = fit_func
        self._apply_func = apply_func
        self.feature_pruning_threshold = feature_pruning_threshold
        self.options = options

        # Stored state
        self._prep_state: Any = None  # mapping returned by fit_func
        self._x_train_prepped: Optional[pd.DataFrame] = None
        self.columns_: Optional[list[str]] = None  # training schema

    def fit(
        self,
        x: pd.DataFrame,
        y: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    ) -> "WholeDataPreprocessor":
        """Learn preprocessing state from a training fold.

        Args:
            x: Training-fold feature DataFrame.
            y: Training-fold target values.

        Returns:
            The fitted preprocessor.

        Raises:
            AssertionError: If the fit function fails to return a non-empty DataFrame.
        """
        x_tr2, prep_state = self._fit_func(
            x,
            pd.Series(y),
            self.feature_pruning_threshold,
            **self.options,
        )

        # Safety
        assert isinstance(x_tr2, pd.DataFrame), "fit_func must return a DataFrame"  # programmr bug
        assert not x_tr2.empty, "fit_func produced an empty DataFrame"  # programmer bug

        self._prep_state = prep_state
        self.columns_ = list(x_tr2.columns)
        self._x_train_prepped = x_tr2
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Apply previously learned preprocessing state.

        Args:
            x: Feature DataFrame to transform.

        Returns:
            Transformed DataFrame aligned to the fitted training schema.

        Raises:
            AssertionError: If called before ``fit`` or if the apply function
                returns invalid output.
        """
        assert (
            self._prep_state is not None and self.columns_ is not None
        ), "transform() before fit()."  # programmer bug

        x2 = self._apply_func(x, self._prep_state)

        assert isinstance(x2, pd.DataFrame), "apply_func must return a DataFrame"  # programmer bug
        assert not x2.empty, "apply_func produced an empty DataFrame"  # programmer bug

        x2 = x2.reindex(columns=self.columns_, fill_value=0)

        assert (
            list(x2.columns) == self.columns_
        ), "Transformed columns do not match training schema"  # programmer bug
        return x2

    def fit_transform(  # pylint: disable=arguments-differ # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        X: pd.DataFrame,
        y: npt.NDArray[np.float64] | npt.NDArray[np.int64],
    ) -> pd.DataFrame:
        """Fit preprocessing state and return transformed training data.

        Args:
            X: Training-fold feature DataFrame.
            y: Training-fold target values.

        Returns:
            Transformed training-fold DataFrame.

        Raises:
            AssertionError: If fitting does not produce transformed training data.
        """
        self.fit(X, y)
        assert (
            self._x_train_prepped is not None
        ), "fit() did not produce training-prepped data"  # programmer bug
        return self._x_train_prepped


# Constants
_AGL_Y_LABEL = "__y_label__"
_FLAML_CLASSIFICATION_ESTIMATORS = [
    "catboost",
    "extra_tree",
    "histgb",
    "lgbm",
    "lrl1",
    "lrl2",
    "rf",
    "xgb_limitdepth",
    "xgboost",
]
_FLAML_REGRESSION_ESTIMATORS = [
    e for e in _FLAML_CLASSIFICATION_ESTIMATORS if e not in ("lrl1", "lrl2")
]
# ChatGPT says: Beyond ~2 minutes/fold, AGL mostly just explores more redundant GBM/CAT/XGB
# variants and deeper ensembles. Diminishing returns vs. extra run time + overfitting risk.
_PER_FOLD_MAX_SECONDS_AGL: int = 120  # Makes sure it does not get too big.
_PER_FOLD_MIN_SECONDS_AGL: int = 10  # Makes sure it does not get too small.
# ChatGPT says: FLAML’s strength is fast, efficient search. Letting it run way past ~1–1.5 minutes
# per fold on 25k rows tends to produce more fragile, over-tuned configs rather than clean
# improvements.
_PER_FOLD_MAX_SECONDS_FLAML: int = 90  # Makes sure it does not get too big.
_PER_FOLD_MIN_SECONDS_FLAML: int = 8  # Makes sure it does not get too small.


def _aggregate_regression_preds(
    preds: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Average per-fold regression predictions.

    Args:
        preds: List of per-fold prediction arrays.

    Returns:
        Mean prediction for each row.
    """
    return np.mean(np.vstack(preds), axis=0)


def _agl_pos_proba_from_df(
    df: pd.DataFrame,
    le: LabelEncoder | None,
) -> npt.NDArray[np.float64]:
    """Extract positive-class probabilities from AutoGluon output.

    Args:
        df: DataFrame returned by AutoGluon ``predict_proba``.
        le: Optional label encoder used during training.

    Returns:
        One-dimensional positive-class probability array.
    """
    if le is not None and len(le.classes_) > 1 and le.classes_[1] in df.columns:
        return df[le.classes_[1]].to_numpy(dtype=float)
    if 1 in df.columns:
        return df[1].to_numpy(dtype=float)
    return df.iloc[:, -1].to_numpy(dtype=float)


def _agl_proba_matrix_from_df(
    df: pd.DataFrame,
    le: LabelEncoder | None,
) -> npt.NDArray[np.float64]:
    """Extract a class-probability matrix from AutoGluon output.

    Args:
        df: DataFrame returned by AutoGluon ``predict_proba``.
        le: Optional label encoder used during training.

    Returns:
        Probability matrix, ordered by label encoder classes when possible.
    """
    if le is not None:
        cols = list(le.classes_)
        if all(c in df.columns for c in cols):
            return df[cols].to_numpy(dtype=float)
    return df.to_numpy(dtype=float)


def _audit_float32_fit_matrix(x: pd.DataFrame, max_examples: int = 8) -> None:
    """Audit whether a feature matrix can safely be cast to float32.

    This catches non-finite and extremely large values before downstream model
    libraries fail with less-informative errors.

    Args:
        x: Feature matrix to audit.
        max_examples: Maximum number of example row values to include per issue.

    Raises:
        AssertionError: If ``x`` is not a DataFrame.
        RuntimeError: If float32 conversion fails or unsafe values are found.
    """
    if util.DO_MODAL:  # pyright: ignore[reportUnnecessaryComparison]
        return

    assert isinstance(x, pd.DataFrame), "X must be a pandas DataFrame"  # programmer bug
    report: dict[str, Any] = {}

    arr64 = x.to_numpy(copy=False)
    pre_bad = ~np.isfinite(arr64)
    if pre_bad.any():
        cols = np.where(pre_bad.any(axis=0))[0]
        items: dict[str, Any] = {}
        for j in cols:
            col = x.columns[j]
            rows = np.where(pre_bad[:, j])[0][:max_examples]
            items[col] = [(int(r), x.iloc[r, j]) for r in rows]
        report["preexisting_nonfinite"] = items

    try:
        arr32 = x.to_numpy(dtype=np.float32, copy=True)
    except Exception as e:
        raise RuntimeError(f"float32 cast failed: {e}") from e

    post_bad = ~np.isfinite(arr32)
    if post_bad.any():
        cols = np.where(post_bad.any(axis=0))[0]
        items = {}
        for j in cols:
            col = x.columns[j]
            rows = np.where(post_bad[:, j])[0][:max_examples]
            items[col] = [(int(r), x.iloc[r, j]) for r in rows]
        report["nonfinite_after_float32_cast"] = items

    arr_abs = np.abs(arr32)
    threshold = np.float32(1e30)
    huge = arr_abs > threshold
    if huge.any():
        cols = np.where(huge.any(axis=0))[0]
        items = {}
        for j in cols:
            col = x.columns[j]
            rows = np.where(huge[:, j])[0][:max_examples]
            items[col] = [
                (int(r), float(x.iloc[r, j])) for r in rows  # pyright: ignore[reportArgumentType]
            ]
        report["extremely_large_float32_values"] = items

    if report:
        util.print_local("❌ Audit found issues:")
        for k, v in report.items():
            util.print_local(f"  - {k}: {list(v.keys())}")
        raise RuntimeError("Errors in _audit_float32_fit_matrix()")


def _cls_loss(
    metric: str,
    y_true: npt.NDArray[np.int64] | npt.NDArray[np.bool_],
    y_hat_labels: npt.NDArray[np.int64] | npt.NDArray[np.bool_],
    y_hat_proba: npt.NDArray[np.float64] | npt.NDArray[np.float32],
) -> float:
    """Compute lower-is-better classification loss for a configured metric.

    Args:
        metric: Metric name.
        y_true: True labels encoded as integers or booleans.
        y_hat_labels: Predicted labels encoded as integers or booleans.
        y_hat_proba: Predicted probabilities. Binary probability metrics expect
            a one-dimensional positive-class probability array; multiclass
            ``log_loss`` expects a two-dimensional probability matrix.

    Returns:
        Loss-style metric value where lower is better.

    Raises:
        ValueError: If a probability-based metric receives an invalid probability shape.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_hat_arr = np.asarray(y_hat_labels, dtype=int)

    if metric == "log_loss":
        p = np.asarray(y_hat_proba, dtype=float)

        if p.ndim == 1:
            return float(log_loss(y_true_arr, p))

        if p.ndim == 2:
            p = np.clip(p, 0.0, 1.0)
            row_sums = p.sum(axis=1, keepdims=True)

            zero_mask = row_sums <= 0.0
            if np.any(zero_mask):
                n_classes = p.shape[1]
                p[zero_mask] = 1.0 / float(n_classes)
                row_sums = p.sum(axis=1, keepdims=True)

            p = p / row_sums
            return float(log_loss(y_true_arr, p))

        raise ValueError(f"log_loss expects 1D or 2D probability array, got shape {p.shape!r}")

    if metric == "roc_auc":
        p = np.asarray(y_hat_proba, dtype=float)
        if p.ndim != 1:
            raise ValueError(f"roc_auc expects 1D positive-class probabilities, got {p.shape!r}")
        return float(1.0 - roc_auc_score(y_true_arr, p))

    if metric == "pr_auc":
        p = np.asarray(y_hat_proba, dtype=float)
        if p.ndim != 1:
            raise ValueError(f"pr_auc expects 1D positive-class probabilities, got {p.shape!r}")
        return float(1.0 - average_precision_score(y_true_arr, p))

    if metric == "balanced_accuracy":
        return 1.0 - float(balanced_accuracy_score(y_true_arr, y_hat_arr))

    return 1.0 - float(accuracy_score(y_true_arr, y_hat_arr))


def _estimate_hpo_trials(
    x_train_fold: pd.DataFrame, per_fold_seconds: int, do_ensemble: bool
) -> int:
    """Estimate a bounded AutoGluon HPO trial count for one fold.

    The estimate scales with rows, columns, and available per-fold time, but
    remains capped so extra time does not explode the number of trials.

    Args:
        x_train_fold: Training-fold feature matrix.
        per_fold_seconds: Time budget for the fold, in seconds.
        do_ensemble: Whether ensemble training is enabled.

    Returns:
        Number of HPO trials to request.
    """
    n_rows, n_cols = x_train_fold.shape

    # Baseline cost per trial: grows gently with rows/cols
    t_trial_base = (max(n_rows, 50) / 2000.0) ** 0.7 * (max(n_cols, 1) / 40.0) ** 0.3

    # Put a floor so tiny datasets don't explode trials
    t_trial = max(0.5, 1.5 * t_trial_base)

    # Fraction of per-fold time devoted to HPO
    hpo_fraction = 0.70 if not do_ensemble else 0.50
    hpo_budget = hpo_fraction * per_fold_seconds

    approx_trials = int(hpo_budget // t_trial)

    # Simple, fixed caps: no magic breakpoints, no cliff behavior
    max_trials = 64 if not do_ensemble else 48
    min_trials = 8

    return max(min_trials, min(approx_trials, max_trials))


def _feature_importances_agl(
    predictor: TabularPredictor | None,
    preproc: WholeDataPreprocessor | None,
    x_train: pd.DataFrame,
    y_all_orig: npt.NDArray[np.float64],
    task: TaskType,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compute AutoGluon feature importances on transformed training data.

    Args:
        predictor: Best-fold AutoGluon predictor, if available.
        preproc: Best-fold fitted preprocessor, if available.
        x_train: Original training features.
        y_all_orig: Original training targets.
        task: Task type.

    Returns:
        Tuple of ``(feature_importances, transformed_training_features)``.
        Either value may be None if required inputs are unavailable.
    """
    if predictor is None or preproc is None:
        return None, None

    x_train_transformed = preproc.transform(x_train)

    y_series = pd.Series(
        y_all_orig,
        name=_AGL_Y_LABEL,
        dtype=("object" if task == "classification" else "float"),
    )
    df = pd.concat([x_train_transformed, y_series], axis=1)

    # ChatGPT says that the .reset_index(drop=True) will avoid downstream wierdnesses.
    with util.suppress_stdout_stderr():
        df = predictor.feature_importance(data=df.reset_index(drop=True), num_shuffle_sets=1)

    x_train_transformed.columns = _uniquify_column_names(pd.Series(x_train_transformed.columns))

    return _feature_importances_calculate(df), x_train_transformed


def _feature_importances_calculate(importances: pd.DataFrame) -> pd.DataFrame | None:
    """Normalize raw feature importances for UI display.

    Args:
        importances: DataFrame containing an ``importance`` column indexed by
            feature name.

    Returns:
        DataFrame with ``Feature`` and ``Model Importance`` columns, or None if
        all importances are zero or unusable.
    """
    abs_importance = importances["importance"].abs()
    total_abs = abs_importance.sum()

    if total_abs == 0:
        # All features have zero or NaN importance, so return empty json.
        # This can happen with really bad/weird/nonsensical data
        return None

    importances["importance_pct"] = abs_importance / total_abs
    assert np.isclose(importances["importance_pct"].sum(), 1.0)  # Allow for small FP error

    # Remove rows that have zero importance.
    importances = importances[importances["importance_pct"] > 0.0]

    # "feature" is the index, so move it into a new column.
    importances = importances[["importance_pct"]].reset_index(names="feature")

    # Clean up the feature names to remove all of the crazy verbose prefixes.
    importances["feature"] = _uniquify_column_names(importances["feature"])

    # Reverse the column order so importance is the first column and feature is the second column.
    # importances = importances.iloc[:, ::-1]

    # Rename the columns
    importances.columns = ["Feature", "Model Importance"]

    return importances


def _feature_importances_flaml(
    aml: AutoML | None,
    preproc: WholeDataPreprocessor | None,
    x_train_original: pd.DataFrame,
    y_train: YSeriesType,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Compute FLAML feature importances on transformed training data.

    Uses native feature importances or coefficients when available, and falls
    back to permutation importance otherwise.

    Args:
        aml: Best-fold FLAML AutoML object, if available.
        preproc: Best-fold fitted preprocessor, if available.
        x_train_original: Original training features.
        y_train: Training targets.

    Returns:
        Tuple of ``(feature_importances, transformed_training_features)``.
        Either value may be None if required inputs are unavailable.
    """
    if aml is None or preproc is None:
        return None, None

    x_train_transformed = preproc.transform(x_train_original)
    model = getattr(aml, "model", None)
    if model is None or x_train_transformed.shape[1] == 0:
        return None, None

    def _as_1d(a: Any) -> npt.NDArray[np.float64]:
        arr = np.asarray(a, dtype=float)
        return arr.ravel()

    if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
        scores = _as_1d(model.feature_importances_)
    elif hasattr(model, "coef_") and model.coef_ is not None:
        scores = _as_1d(model.coef_)
    else:
        perm = permutation_importance(
            model,
            x_train_transformed,
            y_train,
            n_repeats=3,
            random_state=util.RANDOM_STATE,
        )
        scores = _as_1d(perm.importances_mean)

    n_feats = x_train_transformed.shape[1]
    if scores.shape[0] != n_feats:
        fixed = np.zeros(n_feats, dtype=float)
        fixed[: min(n_feats, scores.shape[0])] = scores[: min(n_feats, scores.shape[0])]
        scores = fixed

    df = pd.DataFrame(
        pd.Series(scores, index=x_train_transformed.columns).sort_values(ascending=False),
        columns=["importance"],
    )

    x_train_transformed.columns = _uniquify_column_names(pd.Series(x_train_transformed.columns))

    return _feature_importances_calculate(df), x_train_transformed


def _finalize_oof_metrics(
    task: TaskType,
    metric: str,
    y_all: YSeriesType | npt.NDArray[Any],
    oof_store: npt.NDArray[Any],
    le: Optional[LabelEncoder],
    is_binary: bool,
) -> tuple[float, float | None, Any | None]:
    """Finalize out-of-fold validation loss and optional binary artifacts.

    Args:
        task: Task type.
        metric: Metric name.
        y_all: Original target values for all rows.
        oof_store: Out-of-fold predictions or probabilities.
        le: Optional label encoder for classification.
        is_binary: Whether the classification task is binary.

    Returns:
        Tuple of ``(validation_loss, best_threshold, calibrator)``.

    Raises:
        AssertionError: If classification is requested without a label encoder.
        ValueError: If metric-specific scoring receives invalid prediction shapes.
    """
    y_all_arr = np.asarray(y_all)

    if task == "classification":
        assert le is not None, "LabelEncoder must not be None for classification."

        if is_binary:
            mask_bool: npt.NDArray[np.bool_] = np.asarray(np.isfinite(oof_store), dtype=bool)
            y_encoded_all = cast(npt.NDArray[np.int64], le.transform(y_all_arr))
            y_true_all: npt.NDArray[np.int64] = y_encoded_all[mask_bool]
            p_oof_raw: npt.NDArray[np.float64] = np.asarray(oof_store, dtype=float)[mask_bool]

            if metric == "log_loss":
                calibrator = _fit_prob_calibrator(y_true_all, p_oof_raw)
                if calibrator is not None:
                    p_oof = calibrator.transform(p_oof_raw)
                else:
                    p_oof = p_oof_raw
                best_thr: float | None = 0.5
                y_hat = p_oof >= best_thr
                val_loss = _cls_loss(metric, y_true_all, y_hat, p_oof)

            elif metric in {"roc_auc", "pr_auc"}:
                # Rank-based / threshold-free metrics: do not calibrate and do not tune a threshold
                best_thr = None
                calibrator = None
                y_hat_dummy = p_oof_raw >= 0.5
                val_loss = _cls_loss(metric, y_true_all, y_hat_dummy, p_oof_raw)

            else:
                ths = np.unique(np.round(p_oof_raw, 6))
                if ths.size > 512:
                    ths = np.linspace(0.1, 0.9, 161)

                scores: list[float] = []
                y_int = y_true_all.astype(int)
                for t in ths:
                    y_hat_t = (p_oof_raw >= t).astype(int)
                    if metric == "balanced_accuracy":
                        score_t = balanced_accuracy_score(y_int, y_hat_t)
                    else:
                        score_t = accuracy_score(y_int, y_hat_t)
                    scores.append(float(score_t))

                t_star = float(ths[int(np.argmax(scores))])
                best_thr = float(np.clip(t_star, 0.35, 0.65))
                y_hat = (p_oof_raw >= best_thr).astype(int)
                val_loss = _cls_loss(metric, y_int, y_hat, p_oof_raw)
                calibrator = None

        else:
            mask_rows = ~np.isnan(oof_store).any(axis=1)
            y_encoded_all = cast(npt.NDArray[np.int64], le.transform(y_all_arr))
            y_true_all = y_encoded_all[mask_rows]
            proba_all = np.asarray(oof_store, dtype=float)[mask_rows, :]

            y_hat = np.argmax(proba_all, axis=1)
            val_loss = _cls_loss(metric, y_true_all, y_hat, proba_all)
            best_thr = None
            calibrator = None

    else:
        mask = np.asarray(np.isfinite(oof_store), dtype=bool)
        y_true = np.asarray(y_all_arr, dtype=float)[mask]
        y_hat = np.asarray(oof_store, dtype=float)[mask]
        val_loss = _regression_loss(metric, y_true, y_hat)
        best_thr = None
        calibrator = None

    return float(val_loss), best_thr, calibrator


def _finite_label_mask(y: npt.NDArray[Any], task: TaskType) -> npt.NDArray[np.bool_]:
    """Return a mask identifying usable target values.

    Args:
        y: Target values.
        task: Task type.

    Returns:
        Boolean mask where True indicates a usable label.
    """
    s = pd.Series(y)
    if task == "classification":
        return s.notna().to_numpy()
    s_num = pd.to_numeric(s, errors="coerce")
    return np.isfinite(s_num.to_numpy())


def _fit_prob_calibrator(
    y_true_bin: npt.NDArray[np.int64],
    proba: npt.NDArray[np.float64],
) -> Optional[_ProbCal]:
    """Fit a one-dimensional probability calibrator for binary OOF scores.

    Args:
        y_true_bin: Binary target values encoded as 0/1.
        proba: Raw positive-class probabilities.

    Returns:
        Fitted calibrator, or None if there is not enough data variation.
    """
    y = np.asarray(y_true_bin, dtype=int)
    p = np.asarray(proba, dtype=float)
    mask = np.isfinite(p)
    y, p = y[mask], p[mask]

    if y.size < 40 or len(np.unique(p)) < 5:
        return None

    # Rich enough data: use isotonic regression
    if y.size >= 200 and len(np.unique(p)) >= 20:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p, y)
        return _ProbCal(model=ir, kind="isotonic")

    # Otherwise, Platt scaling via logistic regression
    lr = LogisticRegression(max_iter=1000, solver="lbfgs")
    lr.fit(p.reshape(-1, 1), y)
    return _ProbCal(model=lr, kind="platt")


def _flaml_binary_scores(
    aml: AutoML,
    x: pd.DataFrame,
    metric: str,
    le: LabelEncoder | None,
) -> npt.NDArray[np.float64]:
    """Return FLAML positive-class scores for binary classification.

    Falls back to encoded labels only when ``predict_proba`` is unavailable and
    the metric does not require probabilities.

    Args:
        aml: Fitted FLAML AutoML object.
        x: Transformed feature matrix.
        metric: Metric name.
        le: Optional label encoder used during training.

    Returns:
        One-dimensional score array.

    Raises:
        RuntimeError: If probabilities are required but unavailable, if fallback
            label encoding is required without a label encoder, or if the
            probability output has an invalid shape.
    """
    proba_raw = aml.predict_proba(x)
    if proba_raw is None:
        if metric in {"log_loss", "roc_auc", "pr_auc"}:
            raise RuntimeError(
                "FLAML estimator.predict_proba returned None while a probability-based "
                "metric was requested."
            )
        if le is None:
            raise RuntimeError("LabelEncoder is required when falling back from predict_proba.")
        labels = aml.predict(x)
        encoded = le.transform(pd.Series(labels))
        return np.asarray(encoded, dtype=float)

    proba_arr = np.asarray(proba_raw, dtype=float)
    if proba_arr.ndim == 1:
        return proba_arr
    if proba_arr.shape[1] < 2:
        raise RuntimeError(
            f"predict_proba output has shape {proba_arr.shape}, "
            "expected at least 2 columns for binary classification."
        )
    return proba_arr[:, 1]


def _flaml_proba_matrix(
    aml: AutoML,
    x: pd.DataFrame,
    le: LabelEncoder | None,
) -> npt.NDArray[np.float64]:
    """Return a FLAML multi-class probability matrix.

    Args:
        aml: Fitted FLAML AutoML object.
        x: Transformed feature matrix.
        le: Optional label encoder used during training.

    Returns:
        Probability matrix, ordered by label encoder classes when possible.

    Raises:
        RuntimeError: If ``predict_proba`` returns None.
    """
    proba_raw = aml.predict_proba(x)
    if proba_raw is None:
        raise RuntimeError("predict_proba returned None for multi-class task.")
    df = util.to_df(proba_raw)
    if le is not None:
        cols = list(le.classes_)
        if all(c in df.columns for c in cols):
            return df[cols].to_numpy(dtype=float)
    return df.to_numpy(dtype=float)


def predict_model_agl_from_artifacts(
    x_test: pd.DataFrame, model_result: ModelResult
) -> YPredictionsType:
    """Load AutoGluon fold artifacts and generate test predictions.

    Classification behavior:
        - Binary ``log_loss``, ``roc_auc``, and ``pr_auc`` return positive-class probabilities.
        - Binary label metrics return thresholded labels, decoded when possible.
        - Multiclass ``log_loss`` returns a probability matrix.
        - Multiclass label metrics return class labels, decoded when possible.

    Regression behavior:
        - Returns the average prediction across saved folds.

    Args:
        x_test: Test feature DataFrame.
        model_result: ModelResult whose artifact directory contains AutoGluon folds.

    Returns:
        Prediction array or label array appropriate for the trained task/metric.

    Raises:
        AssertionError: If expected fold artifact directories or stored classification
            thresholds are missing.
        RuntimeError: If no AutoGluon fold predictions are available.
        FileNotFoundError: If required artifact files cannot be loaded.
    """
    out_dir = util.artifacts_directory_mr(model_result)
    state: TrainingState = joblib.load(out_dir / "agl_state.joblib")

    test_preds_accum: list[npt.NDArray[np.float64]] = []

    is_class = state.task == "classification"
    n_classes = (
        len(state.label_encoder.classes_) if is_class and state.label_encoder is not None else 0
    )
    is_binary = is_class and n_classes == 2

    for fold_i in range(state.n_splits):
        fold_dir = out_dir / f"agl_fold_{fold_i}"
        assert fold_dir.exists()

        preproc: WholeDataPreprocessor = joblib.load(fold_dir / "preproc.joblib")
        with util.suppress_stdout_stderr():
            predictor = TabularPredictor.load(str(fold_dir))

        x_te_use = preproc.transform(x_test)

        if state.task == "classification":
            proba_df = util.to_df(predictor.predict_proba(x_te_use))
            if is_binary:
                proba = _agl_pos_proba_from_df(proba_df, state.label_encoder)
                test_preds_accum.append(proba)
            else:
                proba_full = _agl_proba_matrix_from_df(proba_df, state.label_encoder)
                test_preds_accum.append(proba_full)
        else:
            preds = predictor.predict(x_te_use).to_numpy()
            test_preds_accum.append(preds.astype(float))

    if not test_preds_accum:
        raise RuntimeError("No AutoGluon fold artifacts found for prediction.")

    if state.task == "classification":
        if is_binary:
            all_preds = np.vstack(test_preds_accum)

            if state.metric == "log_loss":
                if state.calibrator is not None:
                    calibrated = np.vstack([state.calibrator.transform(p) for p in all_preds])
                else:
                    calibrated = all_preds
                return np.mean(calibrated, axis=0)

            if state.metric in {"roc_auc", "pr_auc"}:
                # ROC-AUC / PR-AUC require continuous positive-class scores, not thresholded labels
                return np.mean(all_preds, axis=0)

            assert (
                state.best_threshold is not None
            ), "Stored threshold is missing for classification."
            proba_med = np.median(all_preds, axis=0)
            label_idx = (proba_med >= state.best_threshold).astype(int)

            if state.label_encoder is not None:
                labels = state.label_encoder.inverse_transform(label_idx)
                return labels

            return label_idx

        # Multi-class
        all_preds_mc = np.stack(test_preds_accum, axis=0)  # (k, n, C)
        proba_mean = np.mean(all_preds_mc, axis=0)  # (n, C)

        if state.metric == "log_loss":
            return proba_mean.astype(float)

        label_idx = np.argmax(proba_mean, axis=1)
        if state.label_encoder is not None:
            labels = state.label_encoder.inverse_transform(label_idx)
            return labels
        return label_idx.astype(int)

    return _aggregate_regression_preds(test_preds_accum)


def predict_model_flaml_from_artifacts(
    x_test: pd.DataFrame, model_result: ModelResult
) -> YPredictionsType:
    """Load FLAML fold artifacts and generate test predictions.

    Classification behavior:
        - Binary ``log_loss``, ``roc_auc``, and ``pr_auc`` return positive-class probabilities.
        - Binary label metrics return thresholded labels, decoded when possible.
        - Multiclass ``log_loss`` returns a probability matrix.
        - Multiclass label metrics return class labels, decoded when possible.

    Regression behavior:
        - Returns the average prediction across saved folds.

    Args:
        x_test: Test feature DataFrame.
        model_result: ModelResult whose artifact directory contains FLAML folds.

    Returns:
        Prediction array or label array appropriate for the trained task/metric.

    Raises:
        AssertionError: If expected fold artifact directories or stored classification
            thresholds are missing.
        RuntimeError: If no FLAML fold predictions are available, or if required
            probabilities cannot be produced for a probability-based metric.
        FileNotFoundError: If required artifact files cannot be loaded.
    """
    out_dir = util.artifacts_directory_mr(model_result)
    state: TrainingState = joblib.load(out_dir / "flaml_state.joblib")

    test_preds_accum: list[npt.NDArray[np.float64]] = []

    is_class = state.task == "classification"
    n_classes = (
        len(state.label_encoder.classes_) if is_class and state.label_encoder is not None else 0
    )
    is_binary = is_class and n_classes == 2

    for fold_i in range(state.n_splits):
        fold_dir = out_dir / f"flaml_fold_{fold_i}"
        assert fold_dir.exists()

        preproc: WholeDataPreprocessor = joblib.load(fold_dir / "preproc.joblib")
        aml: AutoML = joblib.load(fold_dir / "automl.joblib")

        x_te_use = preproc.transform(x_test)

        if state.task == "classification":
            if is_binary:
                proba = _flaml_binary_scores(
                    aml,
                    x_te_use,
                    state.metric,
                    state.label_encoder,
                )
                test_preds_accum.append(proba)
            else:
                proba_full = _flaml_proba_matrix(aml, x_te_use, state.label_encoder)
                test_preds_accum.append(proba_full)
        else:
            preds = aml.predict(x_te_use)
            test_preds_accum.append(cast(npt.NDArray[np.float64], preds))

    if not test_preds_accum:
        raise RuntimeError("No FLAML fold artifacts found for prediction.")

    if state.task == "classification":
        if is_binary:
            all_preds = np.vstack(test_preds_accum)

            if state.metric == "log_loss":
                if state.calibrator is not None:
                    calibrated = np.vstack([state.calibrator.transform(p) for p in all_preds])
                else:
                    calibrated = all_preds
                return np.mean(calibrated, axis=0)

            if state.metric in {"roc_auc", "pr_auc"}:
                # ROC-AUC / PR-AUC require continuous positive-class scores, not thresholded labels
                return np.mean(all_preds, axis=0)

            assert (
                state.best_threshold is not None
            ), "Stored threshold is missing for classification."
            proba_med = np.median(all_preds, axis=0)
            label_idx = (proba_med >= state.best_threshold).astype(int)

            if state.label_encoder is not None:
                labels = state.label_encoder.inverse_transform(label_idx)
                return labels

            return label_idx

        # Multi-class
        all_preds_mc = np.stack(test_preds_accum, axis=0)  # (k, n, C)
        proba_mean = np.mean(all_preds_mc, axis=0)

        if state.metric == "log_loss":
            return proba_mean.astype(float)

        label_idx = np.argmax(proba_mean, axis=1)
        if state.label_encoder is not None:
            labels = state.label_encoder.inverse_transform(label_idx)
            return labels
        return label_idx.astype(int)

    return _aggregate_regression_preds(test_preds_accum)


def _regression_loss(
    metric: str,
    y_true: npt.NDArray[np.float64],
    y_hat: npt.NDArray[np.float64],
) -> float:
    """Compute lower-is-better regression loss for a configured metric.

    Args:
        metric: Metric name.
        y_true: True numeric target values.
        y_hat: Predicted numeric target values.

    Returns:
        Loss-style metric value where lower is better.
    """
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_hat))

    # MSE support (explicit — do NOT fall through to RMSE)
    if metric == "mse":
        return float(mean_squared_error(y_true, y_hat))

    if metric in util.HIGHER_IS_BETTER_METRICS:
        # R² is higher-is-better; convert to a lower-is-better loss
        return float(1.0 - r2_score(y_true, y_hat))

    # Default / RMSE-style metric
    return float(root_mean_squared_error(y_true, y_hat))


def train_model_agl_store_artifacts(
    x_train: pd.DataFrame,
    y_train: YSeriesType,
    random_seed: int,
    feature_pruning_threshold: int,
    s_strict: float,
    do_ensemble: bool,
    do_early_stop: bool | None,
    task_index: int,
    **options: Any,
) -> ModelResult:
    """Train AutoGluon with out-of-fold validation and persist artifacts.

    Args:
        x_train: Training feature DataFrame.
        y_train: Training target values.
        random_seed: Random seed for splitting and model training.
        feature_pruning_threshold: Feature-pruning strictness level.
        s_strict: Continuous overfitting-discouragement strictness in [0, 1].
        do_ensemble: Whether to enable AutoGluon ensembling.
        do_early_stop: Must be None for the AutoGluon path.
        task_index: Index of this training task within the broader run.
        **options: Runtime options including task, metric, folds, paths, and run ID.

    Returns:
        ModelResult containing fold metrics, artifact metadata, feature importances,
        and OOF predictions/probabilities.

    Raises:
        AssertionError: If AutoGluon-specific invariants are violated.
        RuntimeError: If float32 matrix auditing or downstream training fails.
        ValueError: If metric-specific scoring receives invalid prediction shapes.
    """
    assert (
        do_early_stop is None
    ), "AGL path currently assumes do_early_stop is None."  # programmer bug

    (
        artifacts_directory,
        discourage_overfitting,
        metric,
        task,
        n_splits,
        per_fold_seconds,
        presets,
        splitter,
    ) = _train_parameters(
        x_train,
        MLFramework.AUTOGLUON,
        random_seed,
        feature_pruning_threshold,
        s_strict,
        do_ensemble,
        do_early_stop,
        **options,
    )

    y_all_orig = np.asarray(y_train)
    if task == "classification":
        le = LabelEncoder()
        y_all_enc = le.fit_transform(y_all_orig)
        n_classes = len(le.classes_)
        is_binary = n_classes == 2
        y_for_split = y_all_enc
        ag_problem_type = "binary" if is_binary else "multiclass"
    else:
        le = None
        y_all_enc = np.empty(0, dtype=int)
        n_classes = 0
        is_binary = False
        y_for_split = y_all_orig
        ag_problem_type = "regression"

    eval_metric: str = util.METRICS_AGL_PRIMARY.get(metric, metric)

    if do_ensemble:
        # Use dynamic_stacking (DyStack) only for binary classification.
        # Multiclass + DyStack + accuracy can trigger the error:
        #   "Classification metrics can't handle a mix of unknown and multiclass targets"
        #   inside AutoGluon's experimental DyStack code.
        dynamic_stacking = task == "classification" and is_binary
        num_bag_folds = 3
        num_bag_sets = 1
        num_stack_levels = 1
        use_bag_holdout = True
    else:
        dynamic_stacking = False
        num_bag_folds = 0
        num_bag_sets = 0
        num_stack_levels = 0
        use_bag_holdout = False

    n = len(y_all_orig)
    if task == "classification" and not is_binary:
        oof_store: npt.NDArray[np.float32] = np.full((n, n_classes), np.nan, dtype=np.float32)
    else:
        oof_store = np.full(n, np.nan, dtype=np.float32)

    train_losses: list[float] = []
    fold_results: list[util.FoldResult] = []

    best_fold_predictor: Optional[TabularPredictor] = None
    best_fold_preproc: Optional[WholeDataPreprocessor] = None
    best_fold_val_loss = float("inf")

    for fold_i, (tr_idx, va_idx) in enumerate(splitter.split(x_train, y_for_split)):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr_orig, y_va_orig = y_all_orig[tr_idx], y_all_orig[va_idx]

        tr_mask = _finite_label_mask(y_tr_orig, task)
        va_mask = _finite_label_mask(y_va_orig, task)
        if not tr_mask.all() or not va_mask.all():
            x_tr, y_tr_orig = x_tr.iloc[tr_mask], y_tr_orig[tr_mask]
            x_va, y_va_orig = x_va.iloc[va_mask], y_va_orig[va_mask]
        va_idx_filt = va_idx[va_mask]

        if task == "classification":
            if len(np.unique(y_tr_orig)) < 2 or len(np.unique(y_va_orig)) < 2:
                continue

        preproc = WholeDataPreprocessor(
            fit_func=features.process_data_fit,
            apply_func=features.process_data_apply,
            feature_pruning_threshold=feature_pruning_threshold,
            **options,
        )
        x_tr_use = preproc.fit_transform(x_tr, y_tr_orig)
        x_va_use = preproc.transform(x_va)

        y_col = _AGL_Y_LABEL
        train_df = x_tr_use.copy()
        train_df[y_col] = pd.Series(
            y_tr_orig,
            index=x_tr_use.index,
            name=y_col,
        ).astype("object" if task == "classification" else "float")
        train_df = train_df.reset_index(drop=True)

        val_df = x_va_use.copy()
        val_df[y_col] = pd.Series(
            y_va_orig,
            index=x_va_use.index,
            name=y_col,
        ).astype("object" if task == "classification" else "float")
        val_df = val_df.reset_index(drop=True)

        if 0.0 < s_strict:
            # Bug in AGL for multiclass and balanced_accuracy, so tell it to skip_xgb
            skip_xgb = (ag_problem_type == "multiclass") and (metric == "balanced_accuracy")
            hyperparameters = make_hp_spaces_agl(
                s_strict,
                include_xgb=not skip_xgb,
            )

            hpo_kwargs: Optional[dict[str, Any]] = {
                "searcher": "random",
                "scheduler": "local",
                "num_trials": _estimate_hpo_trials(
                    x_tr_use,
                    per_fold_seconds,
                    do_ensemble,
                ),
                "time_limit": max(1, per_fold_seconds - 2),
                "num_workers": 1,
            }
        else:
            hyperparameters = {"GBM": {}, "CAT": {}, "XGB": {}, "RF": {}, "XT": {}, "LR": {}}
            hpo_kwargs = None

        fold_dir = artifacts_directory / f"agl_fold_{fold_i}"
        util.print_local(
            f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AGL fold_dir: {fold_dir}"
        )

        predictor = TabularPredictor(
            label=y_col,
            eval_metric=eval_metric,
            problem_type=ag_problem_type,
            verbosity=0 if util.DO_MODAL else 1,
            path=str(fold_dir),
        )

        _audit_float32_fit_matrix(x_tr_use)
        _audit_float32_fit_matrix(x_va_use)

        assert presets is not None  # programmer bug
        tsx = util.suppress_stdout_stderr() if util.DO_MODAL else nullcontext()
        with tsx:
            predictor.fit(
                dynamic_stacking=dynamic_stacking,
                feature_generator=IdentityFeatureGenerator(),
                hyperparameters=hyperparameters,
                hyperparameter_tune_kwargs=hpo_kwargs,
                num_bag_folds=num_bag_folds,
                num_bag_sets=num_bag_sets,
                num_cpus=1,
                num_stack_levels=num_stack_levels,
                presets=presets,
                raise_on_no_models_fitted=True,
                refit_full=False,
                set_best_to_refit_full=False,
                time_limit=per_fold_seconds,
                train_data=train_df,
                tuning_data=val_df,
                ag_args_fit={"num_cpus": 1, "random_seed": random_seed},
                use_bag_holdout=use_bag_holdout,
            )

        joblib.dump(preproc, fold_dir / "preproc.joblib")

        if task == "classification":
            proba_va_df = util.to_df(predictor.predict_proba(x_va_use))
            proba_tr_df = util.to_df(predictor.predict_proba(x_tr_use))

            if is_binary:
                proba_va = _agl_pos_proba_from_df(proba_va_df, le)
                proba_tr = _agl_pos_proba_from_df(proba_tr_df, le)
                oof_store[va_idx_filt] = proba_va

                assert le is not None  # programmer bug
                y_tr_bin = cast(npt.NDArray[np.int64], le.transform(y_tr_orig))  # lint
                train_loss_fold = _cls_loss(metric, y_tr_bin, proba_tr >= 0.5, proba_tr)

                y_va_bin = cast(npt.NDArray[np.int64], le.transform(y_va_orig))  # lint
                y_va_hat_bool = oof_store[va_idx_filt] >= 0.5
                val_loss_fold = _cls_loss(
                    metric,
                    y_va_bin,
                    y_va_hat_bool,
                    oof_store[va_idx_filt],
                )
            else:
                assert le is not None  # programmer bug
                proba_va_full = _agl_proba_matrix_from_df(proba_va_df, le)
                proba_tr_full = _agl_proba_matrix_from_df(proba_tr_df, le)

                oof_store[va_idx_filt, :] = proba_va_full

                y_tr_enc = cast(npt.NDArray[np.int64], le.transform(y_tr_orig))  # lint
                y_tr_hat = np.argmax(proba_tr_full, axis=1)
                train_loss_fold = _cls_loss(metric, y_tr_enc, y_tr_hat, proba_tr_full)

                y_va_enc = cast(npt.NDArray[np.int64], le.transform(y_va_orig))  # lint
                proba_va_used = oof_store[va_idx_filt, :]
                y_va_hat = np.argmax(proba_va_used, axis=1)
                val_loss_fold = _cls_loss(metric, y_va_enc, y_va_hat, proba_va_used)
        else:
            preds_va = predictor.predict(x_va_use).to_numpy()
            oof_store[va_idx_filt] = preds_va

            preds_tr = predictor.predict(x_tr_use).to_numpy()
            train_loss_fold = _regression_loss(
                metric,
                y_tr_orig.astype(float),
                preds_tr.astype(float),
            )
            val_loss_fold = _regression_loss(
                metric,
                y_va_orig.astype(float),
                preds_va.astype(float),
            )

        train_losses.append(float(train_loss_fold))
        fold_results.append(
            util.FoldResult(
                fold=fold_i,
                train_metric=float(train_loss_fold),
                val_metric=float(val_loss_fold),
            )
        )

        if val_loss_fold < best_fold_val_loss:
            best_fold_val_loss = float(val_loss_fold)
            best_fold_predictor = predictor
            best_fold_preproc = preproc

    # Shared finalization of OOF-based val_loss / threshold / calibrator
    val_loss, best_thr, calibrator = _finalize_oof_metrics(
        task=task,
        metric=metric,
        y_all=y_all_orig,
        oof_store=oof_store,
        le=le,
        is_binary=is_binary,
    )

    # ------------------------------------------------------------------
    # Build per-row predictions / probabilities for downstream reports
    # ------------------------------------------------------------------
    if task == "classification":
        assert le is not None  # programmer bug
        # oof_pred_proba: 1D for binary, 2D for multiclass
        oof_pred_proba: npt.NDArray[np.floating[Any]] | None = np.asarray(oof_store, dtype=float)

        if is_binary:
            # Positive-class prob already in oof_store
            y_pred_enc = (oof_pred_proba >= 0.5).astype(int)
        else:
            # Multiclass: pick argmax of probability vector
            y_pred_enc = np.argmax(oof_pred_proba, axis=1)

        # Back to original label space, aligned with y_train
        oof_predictions = le.inverse_transform(y_pred_enc)
    else:
        # Regression: predictions are already in oof_store
        oof_pred_proba = None
        oof_predictions = np.asarray(oof_store, dtype=float)

    train_loss = float(np.nanmean(train_losses))
    feature_importances, x_train_transformed = _feature_importances_agl(
        best_fold_predictor,
        best_fold_preproc,
        x_train,
        y_all_orig,
        task,
    )

    state = TrainingState(
        metric=metric,
        task=task,
        n_splits=n_splits,
        label_encoder=le,
        best_threshold=best_thr,
        calibrator=calibrator,
    )
    joblib.dump(state, artifacts_directory / "agl_state.joblib")

    model_result = ModelResult(
        run_id=options[Option.RUN_ID],
        random_seed=random_seed,
        feature_pruning_threshold=feature_pruning_threshold,
        discourage_overfitting=discourage_overfitting,
        do_ensemble=do_ensemble,
        do_early_stop=do_early_stop,
        raw_train_metric=train_loss,
        raw_val_metric=val_loss,
        raw_ratio_metric=util.safe_div(val_loss, train_loss),
        cv_train_metric=util.NUMBER_NAN,
        cv_val_metric=util.NUMBER_NAN,
        cv_ratio_metric=util.NUMBER_NAN,
        cv_score_penalized=util.NUMBER_NAN,
        feature_importances=feature_importances,
        fold_results=fold_results,
        model=f"AGL AutoGluon: {n_splits} folds; ~{per_fold_seconds}s/fold",
        task_index=task_index,
        ml_framework=MLFramework.AUTOGLUON,
        x_train=x_train_transformed,
        y_train=y_train,
        train_stars=util.NUMBER_NAN,
        val_stars=util.NUMBER_NAN,
        robustness_score=None,
        robustness_stars=None,
        validation_stability=None,
        baseline_comparison=None,
        segmented_performance=None,
        oof_predictions=oof_predictions,
        oof_pred_proba=oof_pred_proba,
        options=options,
    )
    model_result.validate(n_splits)

    return model_result


def train_model_flaml_store_artifacts(
    x_train: pd.DataFrame,
    y_train: YSeriesType,
    random_seed: int,
    feature_pruning_threshold: int,
    s_strict: float,
    do_ensemble: bool,
    do_early_stop: bool | None,
    task_index: int,
    **options: Any,
) -> ModelResult:
    """Train FLAML with out-of-fold validation and persist artifacts.

    Args:
        x_train: Training feature DataFrame.
        y_train: Training target values.
        random_seed: Random seed for splitting and model training.
        feature_pruning_threshold: Feature-pruning strictness level.
        s_strict: Continuous overfitting-discouragement strictness in [0, 1].
        do_ensemble: Whether to enable FLAML ensembling.
        do_early_stop: Whether FLAML early stopping is enabled. Must not be None.
        task_index: Index of this training task within the broader run.
        **options: Runtime options including task, metric, folds, paths, and run ID.

    Returns:
        ModelResult containing fold metrics, artifact metadata, feature importances,
        and OOF predictions/probabilities.

    Raises:
        AssertionError: If FLAML-specific invariants are violated.
        RuntimeError: If required probabilities cannot be produced for scoring.
        ValueError: If metric-specific scoring receives invalid prediction shapes.
    """
    assert do_early_stop is not None, "FLAML requires explicit do_early_stop."  # programmer bug

    (
        artifacts_directory,
        discourage_overfitting,
        metric,
        task,
        n_splits,
        per_fold_seconds,
        _,
        splitter,
    ) = _train_parameters(
        x_train,
        MLFramework.FLAML,
        random_seed,
        feature_pruning_threshold,
        s_strict,
        do_ensemble,
        do_early_stop,
        **options,
    )

    metric_flaml = util.METRICS_FLAML_ALIASES.get(metric, metric)

    y_arr = np.asarray(y_train)

    if task == "classification":
        le = LabelEncoder()
        y_for_split = le.fit_transform(y_arr)
        n_classes = len(le.classes_)
        is_binary = n_classes == 2
    else:
        le = None
        y_for_split = y_arr.astype(float)
        n_classes = 0
        is_binary = False

    n = len(y_arr)
    if task == "classification" and not is_binary:
        oof_store: npt.NDArray[np.float32] = np.full((n, n_classes), np.nan, dtype=np.float32)
    else:
        oof_store = np.full(n, np.nan, dtype=np.float32)

    fold_results: list[util.FoldResult] = []
    train_losses: list[float] = []

    best_fold_val_loss = float("inf")
    best_fold_preproc: Optional[WholeDataPreprocessor] = None
    best_fold_aml: Optional[AutoML] = None

    estimator_list = (
        _FLAML_CLASSIFICATION_ESTIMATORS
        if task == "classification"
        else _FLAML_REGRESSION_ESTIMATORS
    )

    fold_settings_base: dict[str, Any] = {
        "early_stop": do_early_stop,
        "ensemble": do_ensemble,
        "estimator_list": estimator_list,
        "eval_method": "holdout",
        "hpo_method": "cfo",
        "log_training_metric": True,
        "metric": metric_flaml,
        "n_concurrent_trials": 1,
        "n_jobs": 1,
        "retrain_full": False,
        "sample": (x_train.shape[0] > 100_000),
        "seed": random_seed,
        "task": task,
        "time_budget": per_fold_seconds,
        "verbose": 0 if util.DO_MODAL else 2,
    }

    for fold_i, (tr_idx, va_idx) in enumerate(splitter.split(x_train, y_for_split)):
        x_tr, x_va = x_train.iloc[tr_idx], x_train.iloc[va_idx]
        y_tr, y_va = y_arr[tr_idx], y_arr[va_idx]

        tr_mask = _finite_label_mask(y_tr, task)
        va_mask = _finite_label_mask(y_va, task)
        if not tr_mask.all() or not va_mask.all():
            x_tr, y_tr = x_tr.iloc[tr_mask], y_tr[tr_mask]
            x_va, y_va = x_va.iloc[va_mask], y_va[va_mask]
        va_idx_filt = va_idx[va_mask]

        if task == "classification":
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
                continue

        preproc = WholeDataPreprocessor(
            fit_func=features.process_data_fit,
            apply_func=features.process_data_apply,
            feature_pruning_threshold=feature_pruning_threshold,
            **options,
        )
        x_tr_use = preproc.fit_transform(x_tr, y_tr)
        x_va_use = preproc.transform(x_va)

        fold_settings = dict(fold_settings_base)
        if 0.0 < s_strict:
            custom = make_hp_spaces_flaml(s_strict)
            fold_settings["custom_hp"] = custom
            fold_settings["estimator_list"] = list(custom.keys())

        aml = AutoML()
        with util.suppress_stdout_stderr():
            aml.fit(x_tr_use, y_tr, X_val=x_va_use, y_val=y_va, **fold_settings)

        fold_dir = artifacts_directory / f"flaml_fold_{fold_i}"
        fold_dir.mkdir(parents=True)
        joblib.dump(preproc, fold_dir / "preproc.joblib")
        joblib.dump(aml, fold_dir / "automl.joblib")
        util.print_local(
            f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FLAML fold_dir: {fold_dir}"
        )

        if task == "classification":
            if is_binary:
                proba_va = _flaml_binary_scores(aml, x_va_use, metric, le)
                proba_tr = _flaml_binary_scores(aml, x_tr_use, metric, le)
                oof_store[va_idx_filt] = proba_va

                assert le is not None  # programmer bug
                y_tr_bin = cast(npt.NDArray[np.int64], le.transform(y_tr))  # lint
                train_loss_fold = _cls_loss(metric, y_tr_bin, proba_tr >= 0.5, proba_tr)

                y_va_bin = cast(npt.NDArray[np.int64], le.transform(y_va))  # lint
                y_va_hat_bool = oof_store[va_idx_filt] >= 0.5
                val_loss_fold = _cls_loss(
                    metric,
                    y_va_bin,
                    y_va_hat_bool,
                    oof_store[va_idx_filt],
                )
            else:
                assert le is not None  # programmer bug
                proba_va_full = _flaml_proba_matrix(aml, x_va_use, le)
                proba_tr_full = _flaml_proba_matrix(aml, x_tr_use, le)

                oof_store[va_idx_filt, :] = proba_va_full

                y_tr_enc = cast(npt.NDArray[np.int64], le.transform(y_tr))  # lint
                y_tr_hat = np.argmax(proba_tr_full, axis=1)
                train_loss_fold = _cls_loss(metric, y_tr_enc, y_tr_hat, proba_tr_full)

                y_va_enc = cast(npt.NDArray[np.int64], le.transform(y_va))  # lint
                proba_va_used = oof_store[va_idx_filt, :]
                y_va_hat = np.argmax(proba_va_used, axis=1)
                val_loss_fold = _cls_loss(metric, y_va_enc, y_va_hat, proba_va_used)
        else:
            preds_va = aml.predict(x_va_use)
            oof_store[va_idx_filt] = cast(npt.NDArray[np.float64], preds_va)

            preds_tr = aml.predict(x_tr_use)
            y_tr_float = np.asarray(y_tr, dtype=float)
            train_loss_fold = _regression_loss(
                metric,
                y_tr_float,
                np.asarray(preds_tr, dtype=float),
            )
            val_loss_fold = _regression_loss(
                metric,
                np.asarray(y_va, dtype=float),
                oof_store[va_idx_filt].astype(float),
            )

        train_losses.append(float(train_loss_fold))
        fold_results.append(
            util.FoldResult(
                fold=fold_i,
                train_metric=float(train_loss_fold),
                val_metric=float(val_loss_fold),
            )
        )

        if val_loss_fold < best_fold_val_loss:
            best_fold_val_loss = float(val_loss_fold)
            best_fold_preproc = preproc
            best_fold_aml = aml

    # Shared finalization of OOF-based val_loss / threshold / calibrator
    val_loss, best_thr, calibrator = _finalize_oof_metrics(
        task=task,
        metric=metric,
        y_all=y_arr,
        oof_store=oof_store,
        le=le,
        is_binary=is_binary,
    )

    # ------------------------------------------------------------------
    # Build per-row predictions / probabilities for downstream reports
    # ------------------------------------------------------------------
    if task == "classification":
        assert le is not None  # programmer bug

        # For classification, oof_store is:
        # - 1D array (n,) of positive-class probabilities for binary
        # - 2D array (n, n_classes) of class probabilities for multiclass
        oof_pred_proba = np.asarray(oof_store, dtype=float)

        if is_binary:
            # Positive-class prob already in oof_store
            y_pred_enc = (oof_pred_proba >= 0.5).astype(int)
        else:
            # Multiclass: pick argmax of probability vector
            y_pred_enc = np.argmax(oof_pred_proba, axis=1)

        # Map back to original label space
        oof_predictions = le.inverse_transform(y_pred_enc)
    else:
        # Regression: OOF predictions are already in oof_store
        oof_pred_proba = None
        oof_predictions = np.asarray(oof_store, dtype=float)

    train_loss = float(np.nanmean(train_losses))
    feature_importances, x_train_transformed = _feature_importances_flaml(
        best_fold_aml, best_fold_preproc, x_train, y_train
    )

    state = TrainingState(
        metric=metric,
        task=task,
        n_splits=n_splits,
        label_encoder=le,
        best_threshold=best_thr,
        calibrator=calibrator,
    )
    joblib.dump(state, artifacts_directory / "flaml_state.joblib")

    model_result = ModelResult(
        run_id=options[Option.RUN_ID],
        random_seed=random_seed,
        feature_pruning_threshold=feature_pruning_threshold,
        discourage_overfitting=discourage_overfitting,
        do_ensemble=do_ensemble,
        do_early_stop=do_early_stop,
        raw_train_metric=train_loss,
        raw_val_metric=val_loss,
        raw_ratio_metric=util.safe_div(val_loss, train_loss),
        cv_train_metric=util.NUMBER_NAN,
        cv_val_metric=util.NUMBER_NAN,
        cv_ratio_metric=util.NUMBER_NAN,
        cv_score_penalized=util.NUMBER_NAN,
        feature_importances=feature_importances,
        fold_results=fold_results,
        model=f"FLAML AutoML: {n_splits} folds; ~{per_fold_seconds}s/fold",
        task_index=task_index,
        ml_framework=MLFramework.FLAML,
        x_train=x_train_transformed,
        y_train=y_train,
        train_stars=util.NUMBER_NAN,
        val_stars=util.NUMBER_NAN,
        robustness_score=None,
        robustness_stars=None,
        validation_stability=None,
        baseline_comparison=None,
        segmented_performance=None,
        oof_predictions=oof_predictions,
        oof_pred_proba=oof_pred_proba,
        options=options,
    )
    model_result.validate(n_splits)

    return model_result


def _train_parameters(
    x_train: pd.DataFrame,
    ml_framework: MLFramework,
    random_seed: int,
    feature_pruning_threshold: int,
    s_strict: float,
    do_ensemble: bool,
    do_early_stop: bool | None,
    **options: Any,
) -> tuple[Path, int, str, TaskType, int, int, str | None, KFold | StratifiedKFold]:
    """Derive framework-specific training parameters for one task.

    Args:
        x_train: Training feature DataFrame.
        ml_framework: Framework to train with.
        random_seed: Random seed for cross-validation splitting.
        feature_pruning_threshold: Feature-pruning strictness level.
        s_strict: Continuous overfitting-discouragement strictness in [0, 1].
        do_ensemble: Whether ensemble training is enabled.
        do_early_stop: Whether early stopping is enabled, if applicable.
        **options: Runtime options including task, metric, folds, paths, speed, and run ID.

    Returns:
        Tuple containing artifact directory, integer strictness code, metric, task,
        fold count, per-fold seconds, optional AutoGluon presets, and CV splitter.
    """
    # --------------------------------------------------------------
    # 1) CV splitter and minimal safety budget
    # --------------------------------------------------------------
    qty_folds = options[Option.QTY_FOLDS]
    # qty_folds = util.QTY_FOLDSS[options[Option.SPEED]]
    # task: TaskType = options[Option.TASK]
    # if qty_folds == 1:
    #     splitter = None
    # else:
    splitter = (
        StratifiedKFold(n_splits=qty_folds, shuffle=True, random_state=random_seed)
        if options[Option.TASK] == "classification"
        else KFold(n_splits=qty_folds, shuffle=True, random_state=random_seed)
    )

    # --------------------------------------------------------------
    # 2) Framework-dependent base minutes (before dataset scaling)
    # --------------------------------------------------------------
    time_factor = util.TIME_FACTORS[options[Option.SPEED]]
    if ml_framework == MLFramework.AUTOGLUON:
        # For larger time budgets use more aggressive 'best_quality' recipe.  "best_quality" is
        # typically not worth it.  The docs say it is best when using "special hardware".
        presets = "high_quality" if time_factor <= 1 else "best_quality"
        # ChatGPT recommends 1.3 - 1.7, but thought that 1.55 was enough.  Go for the max.
        base_minutes = 1.70 if do_ensemble else 1.0
        per_fold_seconds_max = _PER_FOLD_MAX_SECONDS_AGL
        per_fold_seconds_min = _PER_FOLD_MIN_SECONDS_AGL
    else:
        # FLAML does not use presets.
        presets = None
        # ChatGPT recommends 1.10 - 1.25, but thought that 1.15 was enough.  Go for the max.
        base_minutes = 1.25 if do_ensemble else 1.0
        per_fold_seconds_max = _PER_FOLD_MAX_SECONDS_FLAML
        per_fold_seconds_min = _PER_FOLD_MIN_SECONDS_FLAML
    base_minutes *= time_factor
    per_fold_seconds_max *= time_factor
    per_fold_seconds_min *= time_factor

    # Get n_rows and n_cols.  Note that this is only an approximation of the qty of rows and cols
    # because the actual qty of rows is determined per fold after adding synthetics, pruning and
    # dropping.  ChatGPT says not to worry about it since these are broad-brush scaling/cap rules.
    n_rows, n_cols = x_train.shape

    # --------------------------------------------------------------
    # 3) Dataset-size scaling for total budget
    # --------------------------------------------------------------
    # Reference dataset: ~2000 rows, 40 features.
    # - Observations scale with exponent ~0.9 (stronger effect).
    # - Features scale with exponent ~0.5 (weaker but non-trivial effect).
    scale_observations = (max(n_rows, 10) / 2000.0) ** 0.9
    scale_features = (max(n_cols, 1) / 40.0) ** 0.5
    total_minutes = base_minutes * scale_observations * scale_features
    per_fold_seconds = (total_minutes * 60) // qty_folds

    # --------------------------------------------------------------
    # 4) Adjust per_fold_seconds based on ml_framework and hardware.
    # --------------------------------------------------------------
    if ml_framework == MLFramework.AUTOGLUON:
        per_fold_seconds *= 1.30  # AGL needs more resources than FLAML
    per_fold_seconds = min(max(per_fold_seconds, per_fold_seconds_min), per_fold_seconds_max)
    # if util.DO_GOOGLE_CLOUD:
    #     per_fold_seconds *= 1.40  # Google Cloud hardawre is pathetic
    per_fold_seconds = int(round(per_fold_seconds))

    # --------------------------------------------------------------
    # 5) Artifact directory for this run
    # --------------------------------------------------------------
    discourage_overfitting = round(100000 * s_strict)
    artifacts_directory = util.artifacts_directory(
        options[Option.DATA_DIRECTORY],
        options[Option.RUN_ID],
        # util.TIME_FACTOR,  # Not needed
        random_seed,
        feature_pruning_threshold,
        discourage_overfitting,
        do_ensemble,
        do_early_stop,
    )

    # --------------------------------------------------------------
    # 6) Return full training-parameter tuple
    # --------------------------------------------------------------
    return (
        artifacts_directory,
        discourage_overfitting,
        options[Option.METRIC],
        options[Option.TASK],
        qty_folds,
        per_fold_seconds,
        presets,
        splitter,
    )


def _uniquify_column_names(values: pd.Series) -> pd.Series:
    """Simplify transformed feature names while preserving uniqueness.

    Args:
        values: Series of unique feature names.

    Returns:
        Series of simplified feature names.

    Raises:
        ValueError: If input values are not unique or cannot be made unique after
            simplification.
    """
    if not values.is_unique:
        dupes = values[values.duplicated()].unique()
        raise ValueError(f"Column contains non-unique values; examples: {dupes[:5]}")

    parts = values.astype(str).str.partition("__")  # 0=head, 1=sep, 2=tail
    head = parts[0]
    sep = parts[1]
    tail = parts[2]

    prefix = head + sep  # either "head__" or ""

    # ------------------------------------------------------------
    # Step 1: start with tail-only
    candidate = tail

    # ------------------------------------------------------------
    # Step 2: try removing '_sklearn' suffix where safe
    stripped = candidate.str.removesuffix("_sklearn")

    # keep stripped only if it stays unique
    can_strip = ~stripped.duplicated(keep=False)
    candidate = candidate.where(~can_strip, stripped)

    # ------------------------------------------------------------
    # Step 3: minimal fix — only non-first occurrences get prefixed back
    needs_prefix = candidate.duplicated(keep="first")
    new_values = candidate.where(~needs_prefix, prefix + tail)

    # ------------------------------------------------------------
    # Sanity check
    if not new_values.is_unique:
        collisions = new_values[new_values.duplicated(keep=False)]
        raise ValueError(
            f"Unable to make values unique; collisions include: {collisions.head(10).tolist()}"
        )

    return new_values
