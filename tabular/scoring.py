"""scoring.py"""

# Errors to ignore
# pylint: disable=broad-exception-caught, disable=too-many-lines
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

# Python imports
import math
from typing import Any, Final  # ,TypeAlias, TYPE_CHECKING

# Third-party imports
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.metrics import average_precision_score

# Project imports
import tabular.utilities as util
from tabular.utilities import FloatSeriesType, MetricType, ModelResult, Option, TaskType


# Constants
_EPSILON = 1e-9
_SCORE_STRENGTH: int = 1  # 0=Lenient, 1=Medium, 2=Strict


def _build_baseline_comparison_for_candidate(
    candidate_cv_mean: float,
    baseline_info: dict[str, Any],
    val_stars: float,
) -> dict[str, Any] | None:
    """Compare this candidate's CV metric to the dataset baseline.

    Args:
        candidate_cv_mean: Mean validation *loss* for this candidate
            (lower-is-better, including log_loss and 1 - score for higher-is-better metrics).
        baseline_info: Dataset-level baseline information.
        val_stars: Star rating for this candidate's validation performance
            (used as relative_improvement_stars).

    Returns:
        dict[str, Any] with baseline vs model comparison, or None if
        a baseline is not available.
    """
    if not baseline_info.get("available", False):
        return None

    orientation = baseline_info.get("orientation")
    metric: MetricType | None = baseline_info.get("metric")
    baseline_value = baseline_info.get("baseline_value")
    baseline_loss = baseline_info.get("baseline_loss")
    baseline_type = baseline_info.get("baseline_type")

    if orientation is None or metric is None or baseline_value is None or baseline_loss is None:
        return None

    # ------------------------------------------------------------
    # 0) log_loss: classification metric but lower-is-better.
    #     - candidate_cv_mean is log_loss (loss-style).
    #     - baseline_value / baseline_loss are baseline log_loss.
    # ------------------------------------------------------------
    if metric == "log_loss":
        model_loss = float(candidate_cv_mean)
        baseline_logloss = float(baseline_value)

        abs_improvement = baseline_logloss - model_loss
        denom = max(abs(baseline_logloss), _EPSILON)
        rel_improvement_pct = abs_improvement / denom

        return {
            "task": "classification",
            "metric": metric,
            "orientation": "lower_is_better",
            "baseline_type": baseline_type,
            "baseline_value": baseline_logloss,
            "baseline_loss": float(baseline_loss),
            "model_value": model_loss,
            "model_loss": model_loss,
            "absolute_improvement": float(abs_improvement),
            "relative_improvement_percent": float(rel_improvement_pct),
            "relative_improvement_stars": val_stars,
            "n_samples": baseline_info.get("n_samples"),
            "n_classes": baseline_info.get("n_classes"),
        }

    # ------------------------------------------------------------
    # 1) R²: special-case on the *metric* (not on orientation).
    #    - candidate_cv_mean is loss-style: 1 - R².
    #    - baseline_value is the baseline R² (0.0 for mean predictor).
    # ------------------------------------------------------------
    if metric == "r2":
        model_r2 = 1.0 - float(candidate_cv_mean)
        baseline_r2 = float(baseline_value)

        model_r2 = max(min(model_r2, 1.0), -1.0)

        abs_improvement = model_r2 - baseline_r2

        if abs(baseline_r2) <= _EPSILON:
            rel_improvement_pct = model_r2
        else:
            denom = max(abs(baseline_r2), _EPSILON)
            rel_improvement_pct = abs_improvement / denom

        return {
            "task": "regression",
            "metric": metric,
            "orientation": "higher_is_better",
            "baseline_type": baseline_type,
            "baseline_value": baseline_r2,
            "baseline_loss": float(baseline_loss),
            "model_value": model_r2,
            "model_loss": float(candidate_cv_mean),
            "absolute_improvement": float(abs_improvement),
            "relative_improvement_percent": float(rel_improvement_pct),
            "relative_improvement_stars": val_stars,
            "n_samples": baseline_info.get("n_samples"),
            "n_classes": baseline_info.get("n_classes"),
        }

    # # ------------------------------------------------------------
    # # 2) ROC-AUC: binary classification, higher-is-better, baseline = 0.5.
    # #    candidate_cv_mean is loss-style: 1 - ROC_AUC.
    # # ------------------------------------------------------------
    # if metric == "roc_auc":
    #     model_auc = 1.0 - float(candidate_cv_mean)
    #     model_auc = float(np.clip(model_auc, 0.0, 1.0))
    #     baseline_auc = float(baseline_value)

    #     abs_improvement = model_auc - baseline_auc

    #     # For ROC-AUC, the meaningful "headroom above random" baseline is 0.5.
    #     if baseline_auc >= 1.0 - _EPSILON:
    #         rel_improvement_pct = 0.0
    #     else:
    #         rel_improvement_pct = abs_improvement / max(1.0 - baseline_auc, _EPSILON)

    #     return {
    #         "task": "classification",
    #         "metric": metric,
    #         "orientation": "higher_is_better",
    #         "baseline_type": baseline_type,
    #         "baseline_value": baseline_auc,
    #         "baseline_loss": float(baseline_loss),
    #         "model_value": model_auc,
    #         "model_loss": float(candidate_cv_mean),
    #         "absolute_improvement": float(abs_improvement),
    #         "relative_improvement_percent": float(rel_improvement_pct),
    #         "relative_improvement_stars": val_stars,
    #         "n_samples": baseline_info.get("n_samples"),
    #         "n_classes": baseline_info.get("n_classes"),
    #     }

    # ------------------------------------------------------------
    # 2) ROC-AUC / PR-AUC: binary classification, higher-is-better.
    #    candidate_cv_mean is loss-style: 1 - score.
    # ------------------------------------------------------------
    if metric in {"roc_auc", "pr_auc"}:
        model_score = 1.0 - float(candidate_cv_mean)
        model_score = float(np.clip(model_score, 0.0, 1.0))
        baseline_score = float(baseline_value)

        abs_improvement = model_score - baseline_score

        # Both ROC-AUC and PR-AUC are bounded above by 1.0, so use remaining headroom.
        if baseline_score >= 1.0 - _EPSILON:
            rel_improvement_pct = 0.0
        else:
            rel_improvement_pct = abs_improvement / max(1.0 - baseline_score, _EPSILON)

        return {
            "task": "classification",
            "metric": metric,
            "orientation": "higher_is_better",
            "baseline_type": baseline_type,
            "baseline_value": baseline_score,
            "baseline_loss": float(baseline_loss),
            "model_value": model_score,
            "model_loss": float(candidate_cv_mean),
            "absolute_improvement": float(abs_improvement),
            "relative_improvement_percent": float(rel_improvement_pct),
            "relative_improvement_stars": val_stars,
            "n_samples": baseline_info.get("n_samples"),
            "n_classes": baseline_info.get("n_classes"),
        }

    # ------------------------------------------------------------
    # 3) Classification: higher-is-better (accuracy / balanced_accuracy),
    #    internal loss is (1 - score).
    # ------------------------------------------------------------
    if orientation == "higher_is_better":
        model_score = 1.0 - float(candidate_cv_mean)
        model_score = float(np.clip(model_score, 0.0, 1.0))

        baseline_score = float(baseline_value)
        abs_improvement = model_score - baseline_score

        denom = max(abs(baseline_score), _EPSILON)
        rel_improvement_pct = abs_improvement / denom

        return {
            "task": "classification",
            "metric": metric,
            "orientation": orientation,
            "baseline_type": baseline_type,
            "baseline_value": baseline_score,
            "baseline_loss": float(baseline_loss),
            "model_value": model_score,
            "model_loss": float(candidate_cv_mean),
            "absolute_improvement": float(abs_improvement),
            "relative_improvement_percent": float(rel_improvement_pct),
            "relative_improvement_stars": val_stars,
            "n_samples": baseline_info.get("n_samples"),
            "n_classes": baseline_info.get("n_classes"),
        }

    # ------------------------------------------------------------
    # 4) Standard regression: lower-is-better (MAE, RMSE, MSE, RMSLE, etc.).
    # ------------------------------------------------------------
    if orientation == "lower_is_better":
        model_loss = float(candidate_cv_mean)
        baseline_loss_val = float(baseline_value)

        abs_improvement = baseline_loss_val - model_loss
        denom = max(abs(baseline_loss_val), _EPSILON)
        rel_improvement_pct = abs_improvement / denom

        return {
            "task": "regression",
            "metric": metric,
            "orientation": orientation,
            "baseline_type": baseline_type,
            "baseline_value": baseline_loss_val,
            "baseline_loss": float(baseline_loss),
            "model_value": model_loss,
            "model_loss": float(candidate_cv_mean),
            "absolute_improvement": float(abs_improvement),
            "relative_improvement_percent": float(rel_improvement_pct),
            "relative_improvement_stars": val_stars,
            "n_samples": baseline_info.get("n_samples"),
            "n_classes": None,
        }

    return None


def _build_segmented_performance_report_for_candidate(
    mr: ModelResult,
) -> dict[str, Any]:
    """Build a lightweight segmented performance report for one candidate.

    Slices the primary metric by:
    - prediction confidence bands (classification, if probabilities available)
    - target quantiles (regression)

    This is intentionally lightweight and defensive:
    - uses x_train / y_train
    - optionally uses predictions and prediction probabilities if present
    - caps bands to keep the object bounded
    """
    metric: MetricType = mr.options[Option.METRIC]
    task: TaskType = mr.options[Option.TASK]

    x_train = getattr(mr, "x_train", None)
    y_train = getattr(mr, "y_train", None)

    if x_train is None or y_train is None:
        return {
            "available": False,
            "reason": "Missing x_train or y_train on ModelResult.",
            "task": task or None,
            "metric": metric,
        }

    x_df = pd.DataFrame(x_train)
    y_series = util.y_pre_transformation_values(y_train, **mr.options)

    n = min(len(x_df), len(y_series))
    if n == 0:
        return {
            "available": False,
            "reason": "x_train or y_train has zero usable rows.",
            "task": task or None,
            "metric": metric,
        }

    x_df = x_df.iloc[:n, :].reset_index(drop=True)
    y_series = y_series.iloc[:n].reset_index(drop=True)

    # Try to find predictions (OOF / validation-style). If missing, we can
    # still provide target-level stats but will omit metric_value.
    preds_raw: Any | None = None
    for attr in (
        "oof_pred",
        "oof_y_pred",
        "oof_predictions",
        "val_pred",
        "predictions",
        "y_pred",
    ):
        val = getattr(mr, attr, None)
        if val is not None:
            preds_raw = val
            break

    y_pred_series: pd.Series | None
    if preds_raw is not None:
        y_pred_series = pd.Series(preds_raw).iloc[:n].reset_index(drop=True)
        y_pred_series = util.y_pre_transformation_values(y_pred_series, **mr.options)
    else:
        y_pred_series = None

    # Try to find prediction probabilities for confidence bands.
    proba_raw: Any | None = None
    for attr in (
        "oof_pred_proba",
        "oof_proba",
        "y_pred_proba",
        "pred_proba",
        "pred_probs",
    ):
        val = getattr(mr, attr, None)
        if val is not None:
            proba_raw = val
            break

    proba_2d: pd.DataFrame | None = None
    pos_proba_series: pd.Series | None = None
    conf_series: pd.Series | None = None

    if proba_raw is not None:
        proba_df = pd.DataFrame(proba_raw).iloc[:n].reset_index(drop=True)

        # Binary classifiers sometimes store only P(class=1) as a single column.
        # Convert to a 2-column probability matrix [P(class=0), P(class=1)].
        if proba_df.shape[1] == 1:
            pos = pd.to_numeric(proba_df.iloc[:, 0], errors="coerce")
            if pos.notna().all():
                pos = pos.clip(0.0, 1.0)
                neg = 1.0 - pos
                proba_2d = pd.concat([neg, pos], axis=1, ignore_index=True)
            else:
                proba_2d = None
        else:
            proba_num = proba_df.apply(pd.to_numeric, errors="coerce")
            if proba_num.notna().all().all():
                proba_2d = proba_num.clip(lower=0.0, upper=1.0)
            else:
                proba_2d = None

        if proba_2d is not None:
            row_sums = proba_2d.sum(axis=1)
            zero_mask = row_sums <= 0.0
            if bool(zero_mask.any()):
                n_classes = int(proba_2d.shape[1])
                proba_2d.loc[zero_mask, :] = 1.0 / float(n_classes)
                row_sums = proba_2d.sum(axis=1)

            proba_2d = proba_2d.div(row_sums, axis=0)
            conf_series = proba_2d.max(axis=1)

            if proba_2d.shape[1] == 2:
                pos_proba_series = proba_2d.iloc[:, 1].copy()

    def _binary_auc_from_scores(
        y_true: pd.Series,
        y_score: pd.Series,
    ) -> float | None:
        """Compute binary ROC-AUC from continuous scores without sklearn."""
        tmp_auc = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
        if tmp_auc.empty:
            return None

        y_true_arr = tmp_auc["y_true"].to_numpy()
        y_score_num = pd.to_numeric(tmp_auc["y_score"], errors="coerce")
        if y_score_num.isna().any():
            return None

        classes = pd.unique(y_true_arr)
        if len(classes) != 2:
            return None

        # Use the second sorted class as the positive class for deterministic behavior.
        classes_sorted = np.sort(classes)
        y_bin = (y_true_arr == classes_sorted[-1]).astype(int)

        n_pos = int(y_bin.sum())
        n_neg = int((1 - y_bin).sum())
        if n_pos == 0 or n_neg == 0:
            return None

        ranks = y_score_num.rank(method="average").to_numpy(dtype=float)
        sum_ranks_pos = float(ranks[y_bin == 1].sum())
        auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
        return float(auc)

    def _binary_pr_auc_from_scores(
        y_true: pd.Series,
        y_score: pd.Series,
    ) -> float | None:
        """Compute binary PR-AUC from continuous scores."""
        tmp_ap = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
        if tmp_ap.empty:
            return None

        y_true_arr = tmp_ap["y_true"].to_numpy()
        y_score_num = pd.to_numeric(tmp_ap["y_score"], errors="coerce")
        if y_score_num.isna().any():
            return None

        classes = pd.unique(y_true_arr)
        if len(classes) != 2:
            return None

        classes_sorted = np.sort(classes)
        y_bin = (y_true_arr == classes_sorted[-1]).astype(int)

        n_pos = int(y_bin.sum())
        n_neg = int((1 - y_bin).sum())
        if n_pos == 0 or n_neg == 0:
            return None

        return float(average_precision_score(y_bin, y_score_num.to_numpy(dtype=float)))

    def _compute_segment_metric(
        y_true: pd.Series,
        y_pred: pd.Series | None,
        y_proba_pos: pd.Series | None = None,
    ) -> float | None:
        if y_true.empty:
            return None

        # -------------------------
        # Classification metrics
        # -------------------------
        # if task == "classification":
        #     if metric == "roc_auc":
        #         if y_proba_pos is None:
        #             return None
        #         return _binary_auc_from_scores(y_true, y_proba_pos)

        #     if y_pred is None:
        #         return None

        #     y_true_arr = np.asarray(y_true)
        #     y_pred_arr = np.asarray(y_pred)

        #     if metric == "balanced_accuracy":
        #         classes = np.unique(y_true_arr)
        #         if classes.size == 0:
        #             return None

        #         recalls: list[float] = []
        #         for cls in classes:
        #             mask = y_true_arr == cls
        #             denom = int(mask.sum())
        #             if denom == 0:
        #                 continue
        #             recalls.append(float(np.mean(y_pred_arr[mask] == cls)))

        #         if not recalls:
        #             return None
        #         return float(np.mean(recalls))

        #     # accuracy fallback
        #     return float(np.mean(y_true_arr == y_pred_arr))
        if task == "classification":
            if metric == "roc_auc":
                if y_proba_pos is None:
                    return None
                return _binary_auc_from_scores(y_true, y_proba_pos)

            if metric == "pr_auc":
                if y_proba_pos is None:
                    return None
                return _binary_pr_auc_from_scores(y_true, y_proba_pos)

            if y_pred is None:
                return None

            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)

            if metric == "balanced_accuracy":
                classes = np.unique(y_true_arr)
                if classes.size == 0:
                    return None

                recalls: list[float] = []
                for cls in classes:
                    mask = y_true_arr == cls
                    denom = int(mask.sum())
                    if denom == 0:
                        continue
                    recalls.append(float(np.mean(y_pred_arr[mask] == cls)))

                if not recalls:
                    return None
                return float(np.mean(recalls))

            # accuracy fallback
            return float(np.mean(y_true_arr == y_pred_arr))

        # -------------------------
        # Regression metrics
        # -------------------------
        if y_pred is None:
            return None

        y_true_num = pd.to_numeric(y_true, errors="coerce")
        y_pred_num = pd.to_numeric(y_pred, errors="coerce")
        tmp_reg = pd.DataFrame({"y_true": y_true_num, "y_pred": y_pred_num}).dropna()
        if tmp_reg.empty:
            return None

        y_true_arr = tmp_reg["y_true"].to_numpy(dtype=float)
        y_pred_arr = tmp_reg["y_pred"].to_numpy(dtype=float)
        diff = y_true_arr - y_pred_arr

        if metric == "r2":
            y_mean = float(np.mean(y_true_arr))
            ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
            ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))
            if ss_tot <= 0.0:
                return None
            return float(1.0 - (ss_res / ss_tot))

        if metric == "rmse":
            return float(np.sqrt(np.mean(diff**2)))

        if metric == "mse":
            return float(np.mean(diff**2))

        if metric == "rmsle":
            y_true_clip = np.maximum(y_true_arr, 0.0)
            y_pred_clip = np.maximum(y_pred_arr, 0.0)
            return float(
                np.sqrt(
                    np.mean(
                        (np.log1p(y_true_clip) - np.log1p(y_pred_clip)) ** 2,
                    )
                )
            )

        # MAE / fallback
        return float(np.mean(np.abs(diff)))

    # -------------------------
    # 1) By prediction confidence band (classification only)
    # -------------------------
    min_segment_size = 30

    by_confidence_band: dict[str, Any]
    conf_n_used = 0

    if task == "classification" and conf_series is not None:
        bands = [
            (0.0, 0.5, "<0.50"),
            (0.5, 0.7, "0.50–0.70"),
            (0.7, 0.85, "0.70–0.85"),
            (0.85, 1.01, "≥0.85"),
        ]
        conf_segments: list[dict[str, Any]] = []

        for lo, hi, label in bands:
            mask = (conf_series >= lo) & (conf_series < hi)
            count = int(mask.sum())
            if count < min_segment_size:
                continue

            y_seg = y_series[mask]
            y_pred_seg = y_pred_series[mask] if y_pred_series is not None else None
            y_proba_seg = pos_proba_series[mask] if pos_proba_series is not None else None

            seg_dict: dict[str, Any] = {
                "band_label": label,
                "lower": float(lo),
                "upper": float(hi),
                "count": count,
            }

            metric_val = _compute_segment_metric(y_seg, y_pred_seg, y_proba_seg)
            if metric_val is not None and np.isfinite(metric_val):
                seg_dict["metric_value"] = float(metric_val)

            conf_segments.append(seg_dict)
            conf_n_used += count

        if conf_segments and conf_n_used > 0:
            for seg in conf_segments:
                seg["fraction"] = float(seg["count"] / float(conf_n_used))

            sum_fraction_conf = sum(float(seg["fraction"]) for seg in conf_segments)
            assert (
                abs(sum_fraction_conf - 1.0) < 1e-6
            ), f"by_confidence_band fractions sum to {sum_fraction_conf}, expected 1.0"

            assert 0 < conf_n_used <= n, (
                f"by_confidence_band n_samples_used={conf_n_used} exceeds n={n} "
                "or is non-positive."
            )

            by_confidence_band = {
                "available": True,
                "n_samples_used": int(conf_n_used),
                "segments": conf_segments,
            }
        else:
            by_confidence_band = {
                "available": False,
                "reason": "Not enough samples per confidence band.",
            }
    else:
        by_confidence_band = {
            "available": False,
            "reason": (
                "Confidence bands require classification task with "
                "prediction probabilities and predictions."
            ),
        }

    if metric in {"log_loss", "pr_auc"}:
        # Too confusing to show confidence bands for log_loss and PR_AUC since they are not
        # monotonic with confidence.
        by_confidence_band = {
            "available": False,
            "reason": "Confidence bands are not shown for Log_Loss and PR_AUC metrics.",
        }

    # -------------------------
    # 2) By target quantile (regression only)
    # -------------------------
    by_target_quantile: dict[str, Any]
    quant_n_used = 0

    if task == "regression" and metric != "r2":
        y_numeric = pd.to_numeric(y_series, errors="coerce")

        if int(y_numeric.notna().sum()) >= min_segment_size * 2:
            qs = np.quantile(
                y_numeric.dropna().to_numpy(),
                [0.0, 0.25, 0.5, 0.75, 1.0],
            )

            quant_segments: list[dict[str, Any]] = []

            for i in range(len(qs) - 1):
                lo = float(qs[i])
                hi = float(qs[i + 1])

                if i < len(qs) - 2:
                    mask = y_numeric.notna() & (y_numeric >= lo) & (y_numeric < hi)
                else:
                    mask = y_numeric.notna() & (y_numeric >= lo) & (y_numeric <= hi)

                count = int(mask.sum())
                if count < min_segment_size:
                    continue

                y_seg = y_series[mask]
                y_pred_seg = y_pred_series[mask] if y_pred_series is not None else None

                seg_dict: dict[str, Any] = {
                    "target_low": lo,
                    "target_high": hi,
                    "count": count,
                }

                try:
                    seg_dict["target_mean"] = float(pd.to_numeric(y_seg, errors="coerce").mean())
                except Exception:
                    seg_dict["target_mean"] = None

                metric_val = (
                    _compute_segment_metric(y_seg, y_pred_seg) if y_pred_seg is not None else None
                )
                if metric_val is not None and np.isfinite(metric_val):
                    seg_dict["metric_value"] = float(metric_val)

                quant_segments.append(seg_dict)
                quant_n_used += count

            if quant_segments and quant_n_used > 0:
                for seg in quant_segments:
                    seg["fraction"] = float(seg["count"] / float(quant_n_used))

                sum_fraction_quant = sum(float(seg["fraction"]) for seg in quant_segments)
                assert (
                    abs(sum_fraction_quant - 1.0) < 1e-6
                ), f"by_target_quantile fractions sum to {sum_fraction_quant}, expected 1.0"

                assert 0 < quant_n_used <= n, (
                    f"by_target_quantile n_samples_used={quant_n_used} exceeds n={n} "
                    "or is non-positive."
                )

                by_target_quantile = {
                    "available": True,
                    "n_samples_used": int(quant_n_used),
                    "segments": quant_segments,
                }
            else:
                by_target_quantile = {
                    "available": False,
                    "reason": "Not enough samples per target quantile segment.",
                }
        else:
            by_target_quantile = {
                "available": False,
                "reason": "Not enough usable rows to build target quantile segments.",
            }
    elif task == "regression":
        by_target_quantile = {
            "available": False,
            "reason": "Segmented target quantiles are not shown for R2 metric.",
        }
    else:
        by_target_quantile = {
            "available": False,
            "reason": "Target quantile segments are only defined for regression task.",
        }

    # -------------------------
    # Final assembly
    # -------------------------
    available_any = bool(
        by_confidence_band.get("available", False) or by_target_quantile.get("available", False)
    )

    n_samples_used_overall = 0
    if by_confidence_band.get("available", False) and "n_samples_used" in by_confidence_band:
        n_samples_used_overall = int(by_confidence_band["n_samples_used"])
    elif by_target_quantile.get("available", False) and "n_samples_used" in by_target_quantile:
        n_samples_used_overall = int(by_target_quantile["n_samples_used"])

    if available_any and n_samples_used_overall > 0:
        assert (
            n_samples_used_overall <= n
        ), f"Overall n_samples_used={n_samples_used_overall} exceeds n_samples={n}"

    return {
        "available": available_any,
        "task": task or None,
        "metric": metric,
        "n_samples": int(n),
        "n_samples_used": int(n_samples_used_overall),
        "by_confidence_band": by_confidence_band,
        "by_target_quantile": by_target_quantile,
    }


# ------------------------------------------------------------------
# Helper: lightweight per-candidate sensitivity summary.
# ------------------------------------------------------------------
def _build_sensitivity_summary_for_candidate(
    mr: ModelResult,
    top_k: int = 5,
) -> dict[str, Any]:
    """Build a lightweight descriptive sensitivity summary for one candidate."""
    task: TaskType = mr.options[Option.TASK]
    metric: MetricType = mr.options[Option.METRIC]

    if mr.feature_importances is None or mr.x_train is None:  # or mr.y_train is None:
        return {
            "available": False,
            "reason": "Missing feature importances or training data.",
            "task": task,
            "metric": metric,
        }

    fi = mr.feature_importances.copy()
    if fi.empty or "Feature" not in fi.columns or "Model Importance" not in fi.columns:
        return {
            "available": False,
            "reason": "feature_importances is missing required columns.",
            "task": task,
            "metric": metric,
        }

    fi = fi[["Feature", "Model Importance"]].copy()
    fi["Feature"] = fi["Feature"].astype(str)
    fi["Model Importance"] = pd.to_numeric(fi["Model Importance"], errors="coerce")
    fi = fi.dropna(subset=["Model Importance"])
    fi = fi.sort_values("Model Importance", ascending=False).head(top_k)

    if fi.empty:
        return {
            "available": False,
            "reason": "No usable feature importances available.",
            "task": task,
            "metric": metric,
        }

    x_df = mr.x_train.reset_index(drop=True)
    y_series = util.y_pre_transformation_values(mr.y_train, **mr.options).reset_index(drop=True)

    n = min(len(x_df), len(y_series))
    if n == 0:
        return {
            "available": False,
            "reason": "No training rows available.",
            "task": task,
            "metric": metric,
        }

    x_df = x_df.iloc[:n].copy()
    y_series = y_series.iloc[:n].copy()

    is_classification = task == "classification"
    if is_classification:
        y_numeric = pd.to_numeric(y_series, errors="coerce")
        if y_numeric.isna().all():
            y_cat = pd.Categorical(y_series)
            y_numeric = pd.Series(y_cat.codes, index=y_series.index, dtype=float)
            y_numeric = y_numeric.where(y_numeric >= 0, np.nan)
    else:
        y_numeric = pd.to_numeric(y_series, errors="coerce")

    feature_summaries: list[dict[str, Any]] = []

    for _, row in fi.iterrows():
        feature_name = str(row["Feature"])
        importance = float(row["Model Importance"])

        if feature_name not in x_df.columns:
            continue

        x_col = pd.to_numeric(x_df[feature_name], errors="coerce")
        tmp = pd.DataFrame({"x": x_col, "y": y_numeric}).dropna()

        if len(tmp) < 10:
            continue

        x_vals = tmp["x"].to_numpy(dtype=float)
        y_vals = tmp["y"].to_numpy(dtype=float)

        q_low, q_med, q_high = np.quantile(x_vals, [0.2, 0.5, 0.8])
        low_mask = x_vals <= q_low
        high_mask = x_vals >= q_high

        if low_mask.sum() < 3 or high_mask.sum() < 3:
            continue

        y_low = y_vals[low_mask]
        y_high = y_vals[high_mask]

        if is_classification:
            low_rate = float(y_low.mean())
            high_rate = float(y_high.mean())
            diff = high_rate - low_rate
            typical_change_in_predrisk = abs(diff)

            effect = {
                "type": "event_rate_difference",
                "low_bucket_value": float(q_low),
                "high_bucket_value": float(q_high),
                "low_event_rate": low_rate,
                "high_event_rate": high_rate,
                "absolute_difference": diff,
                "typical_change_in_predrisk": typical_change_in_predrisk,
            }
        else:
            low_mean = float(y_low.mean())
            high_mean = float(y_high.mean())
            diff = high_mean - low_mean
            denom = max(abs(y_vals.mean()), 1e-12)
            typical_change_in_predrisk = diff

            effect = {
                "type": "mean_difference",
                "low_bucket_value": float(q_low),
                "high_bucket_value": float(q_high),
                "low_mean_target": low_mean,
                "high_mean_target": high_mean,
                "absolute_difference": diff,
                "relative_change_vs_mean": float(diff / denom),
                "typical_change_in_predrisk": typical_change_in_predrisk,
            }

        try:
            corr = float(pd.Series(x_vals).rank().corr(pd.Series(y_vals).rank()))
        except Exception:
            corr = float("nan")

        if not np.isfinite(corr) or abs(corr) < 0.05:
            direction = "ambiguous"
        elif corr > 0:
            direction = "higher_increases_risk" if is_classification else "higher_increases_target"
        else:
            direction = "higher_decreases_risk" if is_classification else "higher_decreases_target"

        feature_summaries.append(
            {
                "feature_name": feature_name,
                "importance": importance,
                "direction": direction,
                "correlation": corr,
                "typical_range": [float(q_low), float(q_high)],
                "impactful_range": [float(q_low), float(q_high)],
                "median_value": float(q_med),
                "effect": effect,
            }
        )

    if not feature_summaries:
        return {
            "available": False,
            "reason": "No feature had enough usable data to summarize.",
            "task": task,
            "metric": metric,
        }

    feature_summaries.sort(key=lambda d: d["importance"], reverse=True)

    return {
        "available": True,
        "task": task,
        "metric": metric,
        "n_samples": int(len(y_numeric)),
        "top_k": int(len(feature_summaries)),
        "features": feature_summaries,
    }


def _build_validation_stability_report_for_candidate(
    df: pd.DataFrame,
    candidate_id: str,
    metric: MetricType,
) -> dict[str, Any]:
    """Build stability summary across folds (and seeds) for one candidate.

    Scoring rule:
        1. Compute coefficient of variation: cv = std / abs(mean)
        2. Apply a small-sample penalty when fold count is low
        3. Convert to stars with a simple linear rule:
           stars = 5 * (1 - min(adjusted_cv, 1))

    Interpretation:
        - Lower variation => more stable => more stars
        - Higher variation => less stable => fewer stars
        - Very small fold counts are penalized because they are less trustworthy

    Notes:
        - Return keys are preserved for drop-in compatibility.
        - The fields named "*_percent" continue to store ratios, matching your
          existing structure and UI formatting.
    """
    vals_loss = df.loc[df["candidate_id"] == candidate_id, "val_metric"].to_numpy()
    n_units = int(vals_loss.size)

    if n_units == 0:
        return {
            "unit_type": "fold",
            "n_units": 0,
            "mean": None,
            "std": None,
            "relative_std_percent": None,
            "absolute_range": None,
            "relative_range_percent": None,
            "stability_stars": 0.0,
            "stability_message": "No validation results available.",
            "values": [],
        }

    higher_is_better = metric in util.HIGHER_IS_BETTER_METRICS

    if higher_is_better:
        arr = 1.0 - vals_loss
    else:
        arr = vals_loss

    arr = np.asarray(arr, dtype=float)

    mean_val = float(arr.mean())
    std_val = float(arr.std(ddof=1)) if n_units > 1 else 0.0

    # eps = 1e-12
    denom = max(abs(mean_val), _EPSILON)

    # Kept as ratios for compatibility with your existing UI code, which already
    # multiplies by 100 when displaying them as percentages.
    coef_var = abs(std_val) / denom
    relative_std_percent = coef_var

    v_min = float(arr.min())
    v_max = float(arr.max())
    absolute_range = v_max - v_min
    relative_range_percent = absolute_range / denom

    # -------------------------------------------------------------------------
    # Simple 3-line stability scoring rule
    # -------------------------------------------------------------------------
    # Penalize low fold counts, but not too aggressively.
    # 1 fold: 2.00x
    # 2 folds: 1.50x
    # 3 folds: 1.30x
    # 4 folds: 1.15x
    # 5+ folds: 1.00x
    sample_penalty_by_n = {
        1: 2.00,
        2: 1.50,
        3: 1.30,
        4: 1.15,
    }
    sample_penalty = sample_penalty_by_n.get(n_units, 1.00)

    # Stricter linear mapping:
    # adjusted_cv = 0.00 -> 5 stars
    # adjusted_cv = 0.40 -> 0 stars
    adjusted_cv = coef_var * sample_penalty
    stability_ratio = max(0.0, 1.0 - min(adjusted_cv / 0.40, 1.0))
    stability_stars = round(5.0 * stability_ratio, 1)

    unit_label = "fold" if n_units == 1 else "folds"
    stability_message = f"±{100 * relative_std_percent:.1f}% across {n_units} {unit_label}"

    return {
        "unit_type": "fold",
        "n_units": n_units,
        "mean": mean_val,
        "std": std_val,
        "relative_std_percent": relative_std_percent,
        "absolute_range": absolute_range,
        "relative_range_percent": relative_range_percent,
        "stability_stars": stability_stars,
        "stability_message": stability_message,
        "values": [float(v) for v in arr.tolist()],
    }


def _compute_dataset_baseline_from_y(
    mr_list: list[ModelResult],
) -> dict[str, Any]:
    """Compute a simple baseline for the dataset using y_train and metric."""
    if not mr_list:
        return {
            "available": False,
            "task": None,
            "metric": None,
            "baseline_type": None,
            "orientation": None,
            "baseline_value": None,
            "baseline_loss": None,
            "n_samples": None,
            "n_classes": None,
        }

    baseline_source = next(
        (mr for mr in mr_list if getattr(mr, "y_train", None) is not None),
        mr_list[0],
    )
    options = getattr(baseline_source, "options", {}) or {}
    task: TaskType = options[Option.TASK]
    metric: MetricType = options[Option.METRIC]

    y_train = getattr(baseline_source, "y_train", None)
    if y_train is None:
        return {
            "available": False,
            "task": task or None,
            "metric": metric or None,
            "baseline_type": None,
            "orientation": None,
            "baseline_value": None,
            "baseline_loss": None,
            "n_samples": None,
            "n_classes": None,
        }

    y_series = util.y_pre_transformation_values(y_train, **options)

    n_samples = int(y_series.shape[0])
    if n_samples == 0:
        return {
            "available": False,
            "task": task or None,
            "metric": metric or None,
            "baseline_type": None,
            "orientation": None,
            "baseline_value": None,
            "baseline_loss": None,
            "n_samples": 0,
            "n_classes": None,
        }

    # ------------------------------------------------------------------
    # Classification baseline
    # ------------------------------------------------------------------
    if task == "classification":
        value_counts = y_series.value_counts()
        n_classes = int(value_counts.shape[0])

        if n_classes == 0:
            return {
                "available": False,
                "task": task or None,
                "metric": metric or None,
                "baseline_type": None,
                "orientation": "higher_is_better",
                "baseline_value": None,
                "baseline_loss": None,
                "n_samples": n_samples,
                "n_classes": 0,
            }

        majority_fraction = float(value_counts.iloc[0] / max(float(n_samples), 1.0))
        majority_fraction = float(np.clip(majority_fraction, 0.0, 1.0))

        if metric == "balanced_accuracy":
            baseline_score = 1.0 / max(float(n_classes), 1.0)
            baseline_score = float(np.clip(baseline_score, 0.0, 1.0))
            baseline_loss = float(1.0 - baseline_score)
            return {
                "available": True,
                "task": "classification",
                "metric": metric,
                "baseline_type": "equal classes",
                "orientation": "higher_is_better",
                "baseline_value": baseline_score,
                "baseline_loss": baseline_loss,
                "n_samples": n_samples,
                "n_classes": n_classes,
            }

        if metric == "log_loss":
            counts = value_counts.to_numpy(dtype=float)
            probs = counts / float(n_samples)
            probs = np.clip(probs, _EPSILON, 1.0)
            baseline_logloss = float(-np.sum(probs * np.log(probs)))

            return {
                "available": True,
                "task": "classification",
                "metric": metric,
                "baseline_type": "empirical prior",
                "orientation": "lower_is_better",
                "baseline_value": baseline_logloss,
                "baseline_loss": baseline_logloss,
                "n_samples": n_samples,
                "n_classes": n_classes,
            }

        if metric in {"roc_auc", "pr_auc"}:
            # ROC-AUC and PR-AUC are only handled here for binary classification.
            if n_classes != 2:
                return {
                    "available": False,
                    "task": "classification",
                    "metric": metric,
                    "baseline_type": None,
                    "orientation": "higher_is_better",
                    "baseline_value": None,
                    "baseline_loss": None,
                    "n_samples": n_samples,
                    "n_classes": n_classes,
                }

            if metric == "roc_auc":
                baseline_score = 0.5
                baseline_type = "random ranking"
            else:
                # For PR-AUC, a random ranking baseline equals prevalence of the positive class.
                baseline_score = float(value_counts.iloc[-1] / max(float(n_samples), 1.0))
                baseline_score = float(np.clip(baseline_score, 0.0, 1.0))
                baseline_type = "positive rate"

            baseline_loss = 1.0 - baseline_score
            return {
                "available": True,
                "task": "classification",
                "metric": metric,
                "baseline_type": baseline_type,
                "orientation": "higher_is_better",
                "baseline_value": baseline_score,
                "baseline_loss": baseline_loss,
                "n_samples": n_samples,
                "n_classes": n_classes,
            }

        # if metric == "roc_auc":
        #     # ROC-AUC is only well-defined here as a simple binary random-ranking baseline.
        #     if n_classes != 2:
        #         return {
        #             "available": False,
        #             "task": "classification",
        #             "metric": metric,
        #             "baseline_type": None,
        #             "orientation": "higher_is_better",
        #             "baseline_value": None,
        #             "baseline_loss": None,
        #             "n_samples": n_samples,
        #             "n_classes": n_classes,
        #         }

        #     baseline_auc = 0.5
        #     baseline_loss = 1.0 - baseline_auc
        #     return {
        #         "available": True,
        #         "task": "classification",
        #         "metric": metric,
        #         "baseline_type": "random ranking",
        #         "orientation": "higher_is_better",
        #         "baseline_value": baseline_auc,
        #         "baseline_loss": baseline_loss,
        #         "n_samples": n_samples,
        #         "n_classes": n_classes,
        #     }

        baseline_score = majority_fraction
        baseline_loss = float(1.0 - baseline_score)
        return {
            "available": True,
            "task": "classification",
            "metric": metric,
            "baseline_type": "majority class",
            "orientation": "higher_is_better",
            "baseline_value": baseline_score,
            "baseline_loss": baseline_loss,
            "n_samples": n_samples,
            "n_classes": n_classes,
        }

    # ------------------------------------------------------------------
    # Regression baseline
    # ------------------------------------------------------------------
    y_float = util.to_float(y_series)
    mean_y = float(y_float.mean())
    median_y = float(y_float.median())

    if metric == "r2":
        baseline_r2 = 0.0
        baseline_value = baseline_r2
        baseline_loss = float(1.0 - baseline_r2)

        return {
            "available": True,
            "task": "regression",
            "metric": metric,
            "baseline_type": "R2-base",
            "orientation": "higher_is_better",
            "baseline_value": baseline_value,
            "baseline_loss": baseline_loss,
            "n_samples": n_samples,
            "n_classes": None,
        }

    if metric in {"mae"}:
        mae_mean = float(np.mean(np.abs(y_float - mean_y)))
        mae_median = float(np.mean(np.abs(y_float - median_y)))
        if mae_median <= mae_mean:
            baseline_type = "median"
            baseline_loss = mae_median
        else:
            baseline_type = "mean"
            baseline_loss = mae_mean

    elif metric in {"rmse", "mse"}:
        mse = float(np.mean((y_float - mean_y) ** 2))
        baseline_type = "mean"
        baseline_loss = float(np.sqrt(mse)) if metric == "rmse" else mse

    elif metric in {"rmsle"}:
        y_clipped = np.maximum(y_float.to_numpy(), 0.0)
        log_y = np.log1p(y_clipped)
        log_mean = float(np.mean(log_y))
        preds = np.expm1(log_mean)
        baseline_type = "mean"
        baseline_loss = float(
            np.sqrt(
                np.mean(
                    (np.log1p(y_clipped) - np.log1p(preds)) ** 2,
                )
            )
        )

    else:
        baseline_type = "mean"
        baseline_loss = float(np.mean(np.abs(y_float - mean_y)))

    baseline_loss = float(max(baseline_loss, 0.0))

    return {
        "available": True,
        "task": "regression",
        "metric": metric,
        "baseline_type": baseline_type,
        "orientation": "lower_is_better",
        "baseline_value": baseline_loss,
        "baseline_loss": baseline_loss,
        "n_samples": n_samples,
        "n_classes": None,
    }


def _compute_robustness(
    metric: MetricType,
    train_metric: float,
    val_metric: float,
) -> tuple[float | None, float | None]:
    """
    Compute a robustness score and a 0-5 star rating.

    Inputs are expected to be loss-style values (lower-is-better). For metrics
    that are naturally higher-is-better, such as accuracy, balanced_accuracy,
    and r2, the displayed metric is reconstructed as (1 - loss) before
    computing robustness.
    """
    if not np.isfinite(train_metric) or not np.isfinite(val_metric):
        return None, None

    if train_metric < 0 or val_metric < 0:
        raise ValueError("Metrics must be non-negative.")

    if metric in util.HIGHER_IS_BETTER_METRICS:
        train_value = 1.0 - train_metric
        val_value = 1.0 - val_metric
    else:
        train_value = train_metric
        val_value = val_metric

    scale = max((abs(train_value) + abs(val_value)) / 2.0, _EPSILON)
    relative_gap = abs(val_value - train_value) / scale
    robustness_value = 1.0 / (1.0 + relative_gap)

    if robustness_value >= 0.95:
        robustness_stars = 5.0
    elif robustness_value >= 0.85:
        robustness_stars = 4.0 + (10.0 * (robustness_value - 0.85))
    elif robustness_value >= 0.75:
        robustness_stars = 3.0 + (10.0 * (robustness_value - 0.75))
    elif robustness_value >= 0.65:
        robustness_stars = 2.0 + (10.0 * (robustness_value - 0.65))
    elif robustness_value >= 0.55:
        robustness_stars = 1.0 + (10.0 * (robustness_value - 0.55))
    else:
        robustness_stars = robustness_value / 0.55

    robustness_stars = round(robustness_stars * 10.0) / 10.0

    return robustness_value, robustness_stars


def _extended_feature_importances(
    raw_feature_importances: pd.DataFrame,
    feature_summaries: list[dict[str, Any]],
    task: TaskType,
) -> pd.DataFrame:
    """Build feature importance table for UI use.

    Uses the full raw feature importance table as the source of truth for:
    - which features appear
    - Model Importance values

    Then enriches with Prediction Influence when sensitivity summaries are
    available for a feature.
    """
    is_classification = task == "classification"
    prediction_influence_col = "Prediction Influence"

    if raw_feature_importances.empty:
        return pd.DataFrame(columns=["Feature", "Model Importance", prediction_influence_col])

    df = raw_feature_importances.copy()

    if "Feature" not in df.columns or "Model Importance" not in df.columns:
        return pd.DataFrame(columns=["Feature", "Model Importance", prediction_influence_col])

    df = df[["Feature", "Model Importance"]].copy()
    df["Feature"] = df["Feature"].astype(str)
    df["Model Importance"] = pd.to_numeric(df["Model Importance"], errors="coerce")
    df = df.dropna(subset=["Model Importance"])

    if df.empty:
        return pd.DataFrame(columns=["Feature", "Model Importance", prediction_influence_col])

    df = df.sort_values("Model Importance", ascending=False).reset_index(drop=True)

    # Normalize displayed importances so they add up to 1.0.
    df["Model Importance"] = _normalize_to_unit_sum(df["Model Importance"])

    summary_by_feature = {str(fs["feature_name"]): fs for fs in feature_summaries}

    influences: list[float] = []
    for feature_name in df["Feature"]:
        fs = summary_by_feature.get(str(feature_name))
        if fs is None:
            influences.append(float("nan"))
            continue

        typical_change = float(fs["effect"]["typical_change_in_predrisk"])

        if is_classification:
            influences.append(abs(typical_change))
            continue

        direction = fs["direction"]
        if direction in {"higher_increases_risk", "higher_increases_target"}:
            influences.append(abs(typical_change))
        elif direction in {"higher_decreases_risk", "higher_decreases_target"}:
            influences.append(-abs(typical_change))
        else:
            influences.append(float("nan"))

    df[prediction_influence_col] = influences

    if is_classification:
        valid = df[prediction_influence_col].notna()
        if valid.any():
            df.loc[valid, prediction_influence_col] = _normalize_to_unit_sum(
                df.loc[valid, prediction_influence_col]
            )

    return df


def _iqr_75_25(s: pd.Series) -> float:
    """Interquartile range (Q3 - Q1) for a numeric Series, robust to NaNs/tiny samples."""
    s_num = pd.to_numeric(s, errors="coerce")
    s_num = s_num.dropna()

    if s_num.size == 0:
        return 0.0
    if s_num.size == 1:
        # IQR is 0 when all values are identical / single point
        return 0.0

    q75 = s_num.quantile(0.75)
    q25 = s_num.quantile(0.25)
    return float(q75 - q25)


def _normalize_to_unit_sum(values: Series) -> Series:
    """
    Normalize a pandas Series of floats so that the result sums to exactly 1.0.

    Rules / guarantees:
    - Input must be a pandas Series of finite floats
    - Sum must be strictly positive
    - Output preserves index
    - Output sums to exactly 1.0 (within floating-point limits)

    Raises:
        TypeError: if input is not a pandas Series
        ValueError: if values contain non-finite numbers or sum <= 0
    """
    # if not isinstance(values, pd.Series):
    #     raise TypeError("values must be a pandas Series")

    if values.empty:
        raise ValueError("values must not be empty")

    # Convert to float explicitly (defensive against object dtype)
    floats: FloatSeriesType = util.to_float(values)

    if not floats.map(math.isfinite).all():
        raise ValueError("values must contain only finite floats")

    total: Final[float] = float(floats.sum())

    if total <= 0.0:
        raise ValueError(f"values must sum to a positive number, got {total}")

    normalized: Series = floats / total

    # Final defensive check (important for downstream math)
    normalized_sum = float(normalized.sum())
    assert math.isfinite(normalized_sum)
    assert math.isclose(normalized_sum, 1.0, rel_tol=0.0, abs_tol=_EPSILON)

    return normalized


def score_model_results(model_results: list[ModelResult]) -> list[ModelResult]:
    """Aggregate per-fold results into a leaderboard and compute a penalized CV score.

    This function takes a list of ``ModelResult`` objects (each with per-fold
    ``FoldResult`` entries) and produces a new list in which:

    - The per-candidate cross-validation performance is summarized.
    - Several variance and overfitting diagnostics are computed.
    - A single scalar ``cv_score_penalized`` is assigned to each candidate that
      combines:
        * Mean validation loss (``cv_mean``)
        * Overfitting penalties (gap- and ratio-based)
        * Variance penalties (across folds and seeds)
        * A mild underfitting penalty
    - A "Validation Stability / Robustness Report" is computed for each
      candidate and stored in ``ModelResult.validation_stability`` for use
      in Step 4 UI.
    - A "Baseline Comparison" summary is computed for each candidate and stored
      in ``ModelResult.baseline_comparison`` to power the
      ``BaselineComparison.tsx`` card.
    - A lightweight "Sensitivity Summary" is computed for each candidate and stored
      in ``ModelResult.sensitivity_summary`` to power the ``SensitivitySummary.tsx``
      card. This is *not* full SHAP; it is a top-k, monotonic-trend style view
      over feature vs target, intended to answer "what actually moves the
      predictions?" in a way that is cheap and bounded on the backend.
    - A "Segmented Performance Report" is computed for each candidate and stored
      in ``ModelResult.segmented_performance`` to power a
      "Where the model works well vs poorly" card. It slices the primary
      metric by:
        * prediction confidence bands (classification only, when probabilities
          are available)
        * target quantiles (regression)

    The intent is:
    * **Base ranking** is by mean validation loss (`cv_mean`).
    * **Penalties** increase this base loss multiplicatively:
        ``score_penalized = cv_mean * (1.0 + penalty)``
    * Overfit/variance/underfit behave in **bounded** ways (no wild explosions),
      especially when `cv_mean` is small or when variance is large.

    Key internal quantities (all *lower is better* in terms of final score):

    - ``cv_mean``:   Average validation loss across all folds/seeds.
    - ``train_mean``: Average train loss across all folds/seeds.
    - ``gen_gap``:   ``cv_mean - train_mean`` (absolute generalization gap).
    - ``overfit_rel``: Relative gap: ``max(gen_gap, 0) / cv_mean``.
    - ``underfit_ratio``: ``train_mean / cv_mean``.
    - ``ratio_metric``:   ``cv_mean / train_mean`` (the val/train ratio you use
      elsewhere in the code to judge overfitting; e.g., in `utilities.py` and
      `train()`).

    Overfitting is penalized via two related views:

    - **Gap-based** overfit (``overfit_rel``): how much worse validation is than
      training, as a *fraction* of validation.
    - **Ratio-based** overfit (``ratio_metric``): val/train ratio, clipped and
      pushed through a smooth saturating function.

    Variance penalties:

    - ``fold_var_rel``: within-seed, across-fold variability (std / sqrt(K))
      normalized by mean cv loss.
    - ``seed_var_rel``: variability across seeds (std of per-seed cv means,
      normalized by mean cv loss).

    Underfitting penalty is mild and one-sided:

    - If ``train_mean > cv_mean`` (train worse than val), we add a small bump
      proportional to how much worse train is than val.

    The "Validation Stability / Robustness Report" includes:

    - mean ± std across folds
    - coefficient of variation (std / mean)
    - best vs worst fold performance
    - relative dispersion (% variation)
    - stability_stars: 0.0–5.0 star-style score summarizing stability
    - stability_message: e.g. "Results vary ±4.2% across folds"
    - raw per-fold values for histograms/sparklines

    The "Baseline Comparison" summary (for the primary metric) includes:

    - baseline_type: "majority_class", "mean", or "median"
    - orientation: "higher_is_better" (classification) or "lower_is_better" (regression)
    - baseline_value: metric value for the simple baseline
    - model_value: metric value for this candidate
    - absolute_improvement: model_value - baseline_value (classification) or
      baseline_value - model_value (regression)
    - relative_improvement_percent: improvement as a percentage of baseline_value

    The "Sensitivity Summary" includes, when available:

    - available: bool (False if needed inputs are missing)
    - task / metric
    - n_samples used
    - top_k: number of features summarized (up to 5)
    - features: list of dicts, each with:
        * feature_name
        * importance
        * direction: "higher_increases_risk", "higher_decreases_risk", or "ambiguous"
        * correlation: monotonic correlation between feature and target
        * typical_range: [low_quantile, high_quantile]
        * impactful_range: [low_quantile, high_quantile] (same for now; can be
          narrowed later)
        * effect:
            - classification: low/high event rate and risk ratio
            - regression: low/high mean target and difference / relative delta

    The "Segmented Performance Report" includes, when inputs are available:

    - available: bool
    - task / metric
    - n_samples: number of rows used
    - by_confidence_band (classification only, requires prediction probabilities):
        * available + segments over confidence bands with metric_value per band
    - by_target_quantile (regression only):
        * available + segments over target quantile ranges with metric_value

    Args:
        model_results: List of ``ModelResult`` objects. Each is expected to have
            ``fold_results`` populated with per-fold training and validation
            metrics (lower-is-better).

    Returns:
        list[ModelResult]: A new list in which each ``ModelResult`` has:
            - ``cv_train_metric``: aggregated mean train loss.
            - ``cv_val_metric``: aggregated mean validation loss.
            - ``cv_ratio_metric``: ``cv_val_metric / cv_train_metric``.
            - ``cv_score_penalized``: penalized score, used for ranking.
            - ``validation_stability``: dict with the stability report for the
              metric in ``options[Option.METRIC]``.
            - ``baseline_comparison``: dict with baseline vs model info for
              the primary metric, or ``None`` if a baseline cannot be computed.
            - ``segmented_performance``: dict with a segmented performance view
              (categorical feature slices, confidence bands, target quantiles) or
              a dict with ``available=False`` if inputs are missing.

    Notes:
        - All computations treat metrics as "loss-like" (lower is better).
        - Overfit penalization is *bounded* and *smooth*:
            * small overfit = tiny penalty
            * very large overfit = penalty saturates to a fixed max
        - Variance penalties are capped so tiny cv losses do not cause huge
          relative variance to dominate the score.
        - The baseline comparator is intentionally simple and cheap:
            * classification: majority-class baseline
            * regression: mean/median baseline in closed form
        - The sensitivity summary is intentionally lightweight:
            * uses top-k features only
            * operates on x_train and y_train (no SHAP, no per-sample explainer)
            * uses coarse quantiles and monotonic trends rather than full PDPs
        - The segmented performance report is also lightweight:
            * uses x_train / y_train + optional predictions/probabilities
            * caps the number of features and segments to stay bounded
        - This function does not attempt to handle empty or invalid metrics
          beyond dropping rows where essential columns are NaN.
    """
    # ---------- Flatten per-fold results ----------
    rows: list[dict[str, Any]] = []
    for mr in model_results:
        for fr in mr.fold_results:
            rows.append(
                {
                    "candidate_id": mr.candidate_id,
                    "seed": mr.random_seed,
                    "fold": int(fr.fold),
                    "train_metric": float(fr.train_metric),
                    "val_metric": float(fr.val_metric),
                }
            )
    df = pd.DataFrame(rows)
    assert not df.empty, "unexpected empty df in score_model_results()"

    # ---------- Per-seed summaries ----------
    per_seed = (
        df.groupby(["candidate_id", "seed"])
        .agg(
            cv_mean=("val_metric", "mean"),
            cv_std=("val_metric", "std"),
            cv_iqr=("val_metric", _iqr_75_25),
            fold_min=("val_metric", "min"),
            fold_max=("val_metric", "max"),
            train_mean=("train_metric", "mean"),
            folds_per_seed=("val_metric", "count"),
        )
        .reset_index()
    )
    per_seed["gen_gap"] = per_seed["cv_mean"] - per_seed["train_mean"]

    # ---------- Aggregate across seeds to per-candidate ----------
    lb = (
        per_seed.groupby("candidate_id")
        .agg(
            cv_mean=("cv_mean", "mean"),
            cv_std_fold=("cv_std", "mean"),
            cv_iqr_fold=("cv_iqr", "mean"),
            seed_std=("cv_mean", "std"),
            gen_gap=("gen_gap", "mean"),
            best_fold=("fold_min", "min"),
            worst_fold=("fold_max", "max"),
            train_mean=("train_mean", "mean"),
            seeds=("seed", "nunique"),
            folds_per_seed_mean=("folds_per_seed", "mean"),
            total_folds=("folds_per_seed", "sum"),
        )
        .reset_index()
    )

    # ---------- Numeric safety ----------
    num_cols = [
        "cv_mean",
        "cv_std_fold",
        "cv_iqr_fold",
        "seed_std",
        "gen_gap",
        "train_mean",
        "seeds",
        "folds_per_seed_mean",
        "total_folds",
    ]
    lb[num_cols] = lb[num_cols].apply(pd.to_numeric, errors="coerce")
    lb = lb.replace([np.inf, -np.inf], np.nan)
    lb = lb.dropna(subset=["cv_mean", "train_mean", "folds_per_seed_mean"])
    if lb.empty:
        return []

    cv_mean_base = lb["cv_mean"].clip(lower=_EPSILON)

    # ---------- Relative terms ----------
    lb["overfit_rel"] = lb["gen_gap"].clip(lower=0.0) / cv_mean_base
    lb["underfit_ratio"] = lb["train_mean"] / cv_mean_base
    lb["ratio_metric"] = 1.0 / lb["underfit_ratio"].clip(lower=_EPSILON)
    lb["seed_var_rel"] = lb["seed_std"].fillna(0.0) / cv_mean_base

    sqrt_k = np.sqrt(lb["folds_per_seed_mean"].clip(lower=1.0))
    lb["fold_var_rel"] = (lb["cv_std_fold"] / sqrt_k) / cv_mean_base
    lb["fold_var_rel"] = lb["fold_var_rel"].fillna(0.0)

    # ---------- Weights based on _SCORE_STRENGTH ----------
    if _SCORE_STRENGTH == 0:
        wgt_overfit_gap, wgt_overfit_ratio = 0.15, 0.20
        s_gap, s_ratio = 0.10, 0.12
        tau = 0.06
    elif _SCORE_STRENGTH == 1:
        wgt_overfit_gap, wgt_overfit_ratio = 0.20, 0.30
        s_gap, s_ratio = 0.10, 0.12
        tau = 0.05
    else:
        wgt_overfit_gap, wgt_overfit_ratio = 0.25, 0.35
        s_gap, s_ratio = 0.12, 0.15
        tau = 0.04

    wgt_fold, wgt_seed_base = 0.15, 0.10
    wgt_underfit_ratio = 0.10

    # Increase to penalize overfitting more.  2.0 and 3.0 were too high.
    overfit_multiplier = 1.0

    wgt_seed_eff = wgt_seed_base * np.clip((lb["seeds"] - 1) / 4.0, 0.0, 1.0)

    # ---------- Overfitting penalties ----------
    gap_rel = lb["overfit_rel"]

    x_gap = (gap_rel - tau).clip(lower=0.0)
    pen_overfit_gap = overfit_multiplier * wgt_overfit_gap * (x_gap / (x_gap + s_gap))

    x_ratio = (lb["ratio_metric"] - 1.0).clip(lower=0.0)
    pen_overfit_ratio = overfit_multiplier * wgt_overfit_ratio * (x_ratio / (x_ratio + s_ratio))

    # ---------- Variance & underfit penalties ----------
    cap_var = 4.0

    pen_fold = (wgt_fold * lb["fold_var_rel"]).clip(upper=cap_var)
    pen_seed = (wgt_seed_eff * lb["seed_var_rel"]).clip(upper=cap_var)

    pen_underfit_ratio = wgt_underfit_ratio * (lb["underfit_ratio"] - 1.0).clip(lower=0.0)

    penalty = pen_fold + pen_seed + pen_overfit_gap + pen_overfit_ratio + pen_underfit_ratio
    lb["score_penalized"] = cv_mean_base * (1.0 + penalty)

    # ------------------------------------------------------------------
    # Compute dataset-level baseline once, then reuse per candidate.
    # ------------------------------------------------------------------
    dataset_baseline_info = _compute_dataset_baseline_from_y(model_results)
    mr_by_id = {mr.candidate_id: mr for mr in model_results}

    # ---------- Transfer the cv scores to model_results_scored ----------
    model_results_scored: list[ModelResult] = []
    for row in lb.itertuples():
        mri = mr_by_id[str(row.candidate_id)]

        train_stars, val_stars = _star_rankings(
            mri.y_train, row.train_mean, row.cv_mean, **mri.options  # pyright: ignore
        )
        robustness_score, robustness_stars = _compute_robustness(
            mri.options[Option.METRIC], row.train_mean, row.cv_mean  # pyright: ignore
        )

        baseline_comparison = _build_baseline_comparison_for_candidate(
            candidate_cv_mean=float(row.cv_mean),  # pyright: ignore
            baseline_info=dataset_baseline_info,
            val_stars=val_stars,
        )

        validation_stability = _build_validation_stability_report_for_candidate(
            df=df,
            candidate_id=row.candidate_id,  # pyright: ignore
            metric=mri.options[Option.METRIC],
        )

        sensitivity_summary = _build_sensitivity_summary_for_candidate(mri)
        feature_importances = _extended_feature_importances(
            raw_feature_importances=(
                mri.feature_importances if mri.feature_importances is not None else pd.DataFrame()
            ),
            feature_summaries=(
                sensitivity_summary["features"] if sensitivity_summary["available"] else []
            ),
            task=mri.options[Option.TASK],
        )

        segmented_performance = _build_segmented_performance_report_for_candidate(mri)

        mri = mri._replace(
            cv_train_metric=row.train_mean,
            cv_val_metric=row.cv_mean,
            cv_ratio_metric=row.ratio_metric,
            cv_score_penalized=row.score_penalized,
            train_stars=train_stars,
            val_stars=val_stars,
            # stars=stars,
            robustness_score=robustness_score,
            robustness_stars=robustness_stars,
            validation_stability=validation_stability,
            baseline_comparison=baseline_comparison,
            # sensitivity_summary=sensitivity_summary,
            segmented_performance=segmented_performance,  # NEW: segmented performance report
            feature_importances=feature_importances,
        )
        model_results_scored.append(mri)

    return model_results_scored


def _star_rankings(
    y_train: pd.Series,
    training_score: float,
    validation_score: float,
    **options: Any,
) -> tuple[float, float]:
    """Compute star rankings (0.0-5.0) for train score and val score."""
    metric: MetricType = options[Option.METRIC]

    y_train = util.y_pre_transformation_values(y_train, **options)

    def _skill_to_stars(skill: float) -> float:
        """Map normalized skill in [0, 1] to stars in [0.0, 5.0]."""
        if skill <= 0.0:
            return 0.0
        skill_clipped = min(1.0, skill)
        stars = 5.0 * skill_clipped
        return round(stars * 10.0) / 10.0

    def _lift_to_stars(lift: float) -> float:
        """Map multiplicative lift over baseline to stars (0.0-5.0)."""
        if lift <= 1.0:
            return 0.0

        # Smooth log-scale mapping:
        # 1x baseline -> 0 stars
        # ~3x baseline -> ~2 stars
        # ~7x baseline -> ~3.5 stars
        # 20x+ baseline -> 5 stars
        stars = 5.0 * np.log10(lift) / np.log10(20.0)
        stars = max(0.0, min(5.0, stars))
        return round(stars * 10.0) / 10.0

    # ------------------------------------------------------------------
    # log_loss
    # ------------------------------------------------------------------
    if metric == "log_loss":
        value_counts = y_train.value_counts()
        n_samples = float(max(len(y_train), 1))
        if value_counts.empty:
            return 0.0, 0.0

        counts = value_counts.to_numpy(dtype=float)
        probs = counts / n_samples
        probs = np.clip(probs, _EPSILON, 1.0)
        baseline_error = float(-np.sum(probs * np.log(probs)))

        def _skill_logloss(err: float) -> float:
            if not np.isfinite(err) or err < 0.0:
                return -1.0
            if baseline_error <= _EPSILON:
                return 0.0
            return 1.0 - (err / baseline_error)

        train_skill = _skill_logloss(training_score)
        val_skill = _skill_logloss(validation_score)

        return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

    # ------------------------------------------------------------------
    # ROC-AUC / PR-AUC
    # ------------------------------------------------------------------
    if metric in {"roc_auc", "pr_auc"}:
        n_classes = int(y_train.nunique(dropna=True))
        if n_classes != 2:
            return 0.0, 0.0

        train_score = 1.0 - training_score
        val_score = 1.0 - validation_score

        train_score = max(0.0, min(1.0, train_score))
        val_score = max(0.0, min(1.0, val_score))

        if metric == "roc_auc":
            baseline_score = 0.5

            def _skill_rank_metric(score: float) -> float:
                if score <= baseline_score + _EPSILON:
                    return 0.0
                return (score - baseline_score) / max(
                    1.0 - baseline_score,
                    _EPSILON,
                )

            train_skill = _skill_rank_metric(train_score)
            val_skill = _skill_rank_metric(val_score)

            return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

        value_counts = y_train.value_counts()
        n_samples = float(max(len(y_train), 1))
        if value_counts.empty:
            return 0.0, 0.0

        # Assume the positive class is the minority class.
        baseline_score = float(value_counts.min() / n_samples)
        baseline_score = max(0.0, min(1.0, baseline_score))

        def _pr_auc_to_stars(score: float) -> float:
            """Map PR-AUC to stars using lift over the positive-rate baseline."""
            if score <= baseline_score + _EPSILON:
                return 0.0

            lift = score / max(baseline_score, _EPSILON)
            return _lift_to_stars(lift)

        return _pr_auc_to_stars(train_score), _pr_auc_to_stars(val_score)

    # ------------------------------------------------------------------
    # accuracy / balanced_accuracy
    # ------------------------------------------------------------------
    if metric in {"accuracy", "balanced_accuracy"}:
        train_score = 1.0 - training_score
        val_score = 1.0 - validation_score

        train_score = max(0.0, min(1.0, train_score))
        val_score = max(0.0, min(1.0, val_score))

        class_counts = y_train.value_counts()
        n_classes = len(class_counts)

        if n_classes == 1:
            baseline_acc = 1.0
            baseline_bal_acc = 1.0
        else:
            baseline_acc = class_counts.max() / class_counts.sum()
            baseline_bal_acc = 1.0 / float(n_classes)

        baseline_acc = max(0.0, min(1.0, float(baseline_acc)))
        baseline_bal_acc = max(0.0, min(1.0, float(baseline_bal_acc)))
        baseline = baseline_acc if metric == "accuracy" else baseline_bal_acc

        def _skill_class(score: float) -> float:
            if baseline >= 1.0 - _EPSILON:
                return 0.0
            return (score - baseline) / max(1.0 - baseline, _EPSILON)

        train_skill = _skill_class(train_score)
        val_skill = _skill_class(val_score)

        return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

    # ------------------------------------------------------------------
    # R²
    # ------------------------------------------------------------------
    if metric == "r2":
        train_r2 = 1.0 - training_score
        val_r2 = 1.0 - validation_score

        train_r2 = max(min(train_r2, 1.0), -1.0)
        val_r2 = max(min(val_r2, 1.0), -1.0)

        baseline_r2 = 0.0

        def _skill_r2(score: float) -> float:
            if score <= baseline_r2:
                return 0.0
            return (score - baseline_r2) / (1.0 - baseline_r2 + _EPSILON)

        train_skill = _skill_r2(train_r2)
        val_skill = _skill_r2(val_r2)

        return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

    # ------------------------------------------------------------------
    # Non-R² regression metrics
    # ------------------------------------------------------------------
    y_arr = y_train.to_numpy(dtype=float)
    y_mean = float(y_arr.mean())

    if metric in {"mae"}:
        diffs = y_arr - y_mean
        baseline_error = float(np.mean(np.abs(diffs)))
    elif metric in {"rmse", "mse"}:
        diffs = y_arr - y_mean
        mse = float(np.mean(diffs**2))
        baseline_error = float(np.sqrt(mse)) if metric == "rmse" else mse
    elif metric == "rmsle":
        if np.any(y_arr < 0):
            raise ValueError("RMSLE baseline is undefined for negative y_values.")
        logy = np.log1p(y_arr)
        diffs_log = logy - logy.mean()
        baseline_error = float(np.sqrt(np.mean(diffs_log**2)))
    else:
        raise ValueError(f"Unknown metric {metric}")

    def _skill_reg(error: float) -> float:
        if not np.isfinite(error) or error < 0.0:
            return -1.0
        if baseline_error <= _EPSILON:
            return 0.0
        return 1.0 - (error / baseline_error)

    train_skill = _skill_reg(training_score)
    val_skill = _skill_reg(validation_score)

    return _skill_to_stars(train_skill), _skill_to_stars(val_skill)


################################################################# OBSOLETE
# def _build_segmented_performance_report_for_candidate(
#     mr: ModelResult,
# ) -> dict[str, Any]:
#     """Build a lightweight segmented performance report for one candidate.

#     Slices the primary metric by:
#     - prediction confidence bands (classification, if probabilities available)
#     - target quantiles (regression)

#     This is intentionally lightweight and defensive:
#     - uses x_train / y_train
#     - optionally uses predictions and prediction probabilities if present
#     - caps bands to keep the object bounded
#     """

#     # options = mr.options # getattr(mr, "options", {}) or {}
#     metric: MetricType = mr.options[Option.METRIC]  # = str(options.get(Option.METRIC, "metric"))
#     task: TaskType = mr.options[Option.TASK]  # = str(options.get(Option.TASK, "") or "")

#     x_train = getattr(mr, "x_train", None)
#     y_train = getattr(mr, "y_train", None)

#     if x_train is None or y_train is None:
#         return {
#             "available": False,
#             "reason": "Missing x_train or y_train on ModelResult.",
#             "task": task or None,
#             "metric": metric,
#         }

#     x_df = pd.DataFrame(x_train)
#     y_series = util.y_pre_transformation_values(y_train, **mr.options)

#     n = min(len(x_df), len(y_series))
#     if n == 0:
#         return {
#             "available": False,
#             "reason": "x_train or y_train has zero usable rows.",
#             "task": task or None,
#             "metric": metric,
#         }

#     x_df = x_df.iloc[:n, :].reset_index(drop=True)
#     y_series = y_series.iloc[:n].reset_index(drop=True)

#     # Try to find predictions (OOF / validation-style). If missing, we can
#     # still provide target-level stats but will omit metric_value.
#     preds_raw = None
#     for attr in (
#         "oof_pred",
#         "oof_y_pred",
#         "oof_predictions",
#         "val_pred",
#         "predictions",
#         "y_pred",
#     ):
#         val = getattr(mr, attr, None)
#         if val is not None:
#             preds_raw = pd.Series(val)
#             break

#     if preds_raw is not None:
#         y_pred_series = preds_raw.iloc[:n].reset_index(drop=True)
#         y_pred_series = util.y_pre_transformation_values(y_pred_series, **mr.options)
#     else:
#         y_pred_series = None

#     # Try to find prediction probabilities for confidence bands.
#     proba_raw = None
#     for attr in (
#         "oof_pred_proba",
#         "oof_proba",
#         "y_pred_proba",
#         "pred_proba",
#         "pred_probs",
#     ):
#         val = getattr(mr, attr, None)
#         if val is not None:
#             proba_raw = val
#             break

#     if proba_raw is not None:
#         # Convert probabilities to a NumPy array.
#         proba_arr = np.asarray(proba_raw)
#         # Ensure we have a 2D array of class probabilities:
#         # - Binary classifiers sometimes return a 1D vector of P(class=1).
#         #   Convert it to a 2-column matrix: [P(class=0), P(class=1)].
#         # - Multiclass classifiers already return shape (n_samples, n_classes),
#         #   so we simply truncate to the first n rows.
#         proba_2d = (
#             np.column_stack([1.0 - proba_arr[:n], proba_arr[:n]])
#             if proba_arr.ndim == 1
#             else proba_arr[:n]
#         )
#         # Defensive checks: ensure probability matrix is well-formed.
#         assert proba_2d.ndim == 2 and proba_2d.shape[1] >= 2, "Invalid probability matrix shape."
#         if not np.all(np.isfinite(proba_2d)):
#             conf_vec = None
#         else:
#             proba_2d = np.clip(proba_2d, 0.0, 1.0)
#             # Confidence = probability of the predicted class.
#             #   For each row, the predicted class is the one with the highest probability,
#             #   so confidence is simply the maximum class probability.
#             conf_vec = np.max(proba_2d, axis=1)
#     else:
#         # No probability information available.
#         conf_vec = None

#     # -------------------------
#     # Metric computation helper
#     # -------------------------
#     def _compute_segment_metric(
#         y_true: pd.Series,
#         y_pred: pd.Series | None,
#     ) -> float | None:
#         if y_pred is None:
#             return None

#         if y_true.empty:
#             return None

#         # --- classification vs regression casting ---
#         if task == "classification":
#             # Keep original labels (strings, ints, etc.). We only use equality
#             # and np.unique, so no need to coerce to float.
#             y_true_arr = np.asarray(y_true)
#             y_pred_arr = np.asarray(y_pred)
#         else:
#             # Regression: we do numeric operations, so float is required.
#             y_true_arr = np.asarray(y_true, dtype=float)
#             y_pred_arr = np.asarray(y_pred, dtype=float)
#         # --- end casting ---

#         # -------------------------
#         # Classification metrics
#         # -------------------------
#         if task == "classification":
#             # Default to accuracy; balanced_accuracy if requested.
#             if metric == "balanced_accuracy":
#                 classes = np.unique(y_true_arr)
#                 if classes.size == 0:
#                     return None
#                 recalls: list[float] = []
#                 for cls in classes:
#                     mask = y_true_arr == cls
#                     denom = int(mask.sum())
#                     if denom == 0:
#                         continue
#                     recalls.append(float(np.mean(y_pred_arr[mask] == cls)))
#                 if not recalls:
#                     return None
#                 return float(np.mean(recalls))
#             else:
#                 # Accuracy (or generic "higher is better" 0–1 score).
#                 return float(np.mean(y_true_arr == y_pred_arr))

#         # -------------------------
#         # Regression metrics
#         # -------------------------
#         diff = y_true_arr - y_pred_arr

#         # R²: we return the standard score (higher is better, can be negative).
#         if metric == "r2":
#             # Total sum of squares
#             y_mean = float(np.mean(y_true_arr))
#             ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
#             ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))

#             # Degenerate case: no variance in y_true → R² undefined.
#             if ss_tot <= 0.0:
#                 return None

#             r2 = 1.0 - (ss_res / ss_tot)
#             return float(r2)

#         if metric == "rmse":
#             return float(np.sqrt(np.mean(diff**2)))

#         if metric == "mse":
#             return float(np.mean(diff**2))

#         if metric == "rmsle":
#             y_true_clip = np.maximum(y_true_arr, 0.0)
#             y_pred_clip = np.maximum(y_pred_arr, 0.0)
#             return float(
#                 np.sqrt(
#                     np.mean(
#                         (np.log1p(y_true_clip) - np.log1p(y_pred_clip)) ** 2,
#                     )
#                 )
#             )

#         # MAE / fallback.
#         return float(np.mean(np.abs(diff)))

#     # -------------------------
#     # 1) By prediction confidence band (classification only)
#     # -------------------------
#     min_segment_size = 30  # reused for both bands and quantiles

#     by_confidence_band: dict[str, Any] | None
#     conf_n_used = 0

#     if task == "classification" and conf_vec is not None and y_pred_series is not None:
#         conf_series = pd.Series(conf_vec, index=y_series.index)

#         # Bands roughly aligned with "low / medium / high / very high" confidence.
#         bands = [
#             (0.0, 0.5, "<0.50"),
#             (0.5, 0.7, "0.50–0.70"),
#             (0.7, 0.85, "0.70–0.85"),
#             (0.85, 1.01, "≥0.85"),
#         ]
#         conf_segments: list[dict[str, Any]] = []

#         # First pass: collect segments and accumulate n_samples_used.
#         for lo, hi, label in bands:
#             mask = (conf_series >= lo) & (conf_series < hi)
#             count = int(mask.sum())
#             if count < min_segment_size:
#                 continue

#             y_seg = y_series[mask]
#             y_pred_seg = y_pred_series[mask]

#             seg_dict: dict[str, Any] = {
#                 "band_label": label,
#                 "lower": float(lo),
#                 "upper": float(hi),
#                 "count": count,  # fraction filled later once n_samples_used is known
#             }

#             metric_val = _compute_segment_metric(y_seg, y_pred_seg)
#             if metric_val is not None and np.isfinite(metric_val):
#                 seg_dict["metric_value"] = float(metric_val)

#             conf_segments.append(seg_dict)
#             conf_n_used += count

#         if conf_segments and conf_n_used > 0:
#             # Second pass: compute fractions based on n_samples_used and assert they sum
#             # to ~1.0.
#             for seg in conf_segments:
#                 seg["fraction"] = float(seg["count"] / float(conf_n_used))

#             sum_fraction_conf = sum(seg["fraction"] for seg in conf_segments)
#             assert (
#                 abs(sum_fraction_conf - 1.0) < 1e-6
#             ), f"by_confidence_band fractions sum to {sum_fraction_conf}, expected 1.0"

#             # Safety: used samples must be within (0, n].
#             assert 0 < conf_n_used <= n, (
#                 f"by_confidence_band n_samples_used={conf_n_used} exceeds n={n} "
#                 "or is non-positive."
#             )

#             by_confidence_band = {
#                 "available": True,
#                 "n_samples_used": int(conf_n_used),
#                 "segments": conf_segments,
#             }
#         else:
#             by_confidence_band = {
#                 "available": False,
#                 "reason": "Not enough samples per confidence band.",
#             }
#     else:
#         by_confidence_band = {
#             "available": False,
#             "reason": (
#                 "Confidence bands require classification task with "
#                 "prediction probabilities and predictions."
#             ),
#         }
#     if metric == "log_loss":
#         # Don't show this for log_loss yet.  Log loss has two UX quirks:
#         #   1. It’s unbounded above and depends on the number of classes.
#         #   2. There isn’t an obvious “good vs bad” scale like accuracy (0–1)
#         by_confidence_band = {
#             "available": False,
#             "reason": "Confidence bands are not shown for Log_Loss metric.",
#         }

#     # -------------------------
#     # 2) By target quantile (regression only)
#     # -------------------------
#     by_target_quantile: dict[str, Any] | None
#     quant_n_used = 0

#     if task == "regression" and metric != "r2":
#         y_numeric = pd.to_numeric(y_series, errors="coerce")

#         if int(y_numeric.notna().sum()) >= min_segment_size * 2:
#             qs = np.quantile(
#                 y_numeric.dropna().to_numpy(),
#                 [0.0, 0.25, 0.5, 0.75, 1.0],
#             )

#             quant_segments: list[dict[str, Any]] = []

#             for i in range(len(qs) - 1):
#                 lo = float(qs[i])
#                 hi = float(qs[i + 1])

#                 if i < len(qs) - 2:
#                     mask = y_numeric.notna() & (y_numeric >= lo) & (y_numeric < hi)
#                 else:
#                     # Include max in last bin.
#                     mask = y_numeric.notna() & (y_numeric >= lo) & (y_numeric <= hi)

#                 count = int(mask.sum())
#                 if count < min_segment_size:
#                     continue

#                 y_seg = y_series[mask]
#                 y_pred_seg = y_pred_series[mask] if y_pred_series is not None else None

#                 seg_dict: dict[str, Any] = {
#                     "target_low": lo,
#                     "target_high": hi,
#                     "count": count,
#                 }

#                 try:
#                     seg_dict["target_mean"] = float(pd.to_numeric(y_seg, errors="coerce").mean())
#                 except Exception:
#                     seg_dict["target_mean"] = None

#                 metric_val = (
#                     _compute_segment_metric(y_seg, y_pred_seg) if y_pred_seg is not None else Non
#                 )
#                 if metric_val is not None and np.isfinite(metric_val):
#                     seg_dict["metric_value"] = float(metric_val)

#                 quant_segments.append(seg_dict)
#                 quant_n_used += count

#             if quant_segments and quant_n_used > 0:
#                 for seg in quant_segments:
#                     seg["fraction"] = float(seg["count"] / float(quant_n_used))

#                 sum_fraction_quant = sum(seg["fraction"] for seg in quant_segments)
#                 assert (
#                     abs(sum_fraction_quant - 1.0) < 1e-6
#                 ), f"by_target_quantile fractions sum to {sum_fraction_quant}, expected 1.0"

#                 assert 0 < quant_n_used <= n, (
#                     f"by_target_quantile n_samples_used={quant_n_used} exceeds n={n} "
#                     "or is non-positive."
#                 )

#                 by_target_quantile = {
#                     "available": True,
#                     "n_samples_used": int(quant_n_used),
#                     "segments": quant_segments,
#                 }
#             else:
#                 by_target_quantile = {
#                     "available": False,
#                     "reason": "Not enough samples per target quantile segment.",
#                 }
#         else:
#             by_target_quantile = {
#                 "available": False,
#                 "reason": "Not enough usable rows to build target quantile segments.",
#             }
#     elif task == "regression":  # and metric == "r2" (see above)
#         # You can legitimately get really bizarre numbers for segment-level quantile R2 metrics.
#         # With a R2 validation_metric of 0.87, the segments can be like -4.56 and 0.12.
#         # I had a case where all 4 segment R2 metrics were lower than the validation_metric.
#         by_target_quantile = {
#             "available": False,
#             "reason": "Segmented target quantiles are not shown for R2 metric.",
#         }
#     else:
#         by_target_quantile = {
#             "available": False,
#             "reason": "Target quantile segments are only defined for regression task.",
#         }

#     # -------------------------
#     # Final assembly
#     # -------------------------
#     available_any = bool(
#         by_confidence_band.get("available", False) or by_target_quantile.get("available", False)
#     )

#     # Derive an overall n_samples_used for convenience:
#     # - classification: from by_confidence_band
#     # - regression: from by_target_quantile
#     # - 0 if neither segmentation is available
#     n_samples_used_overall = 0
#     if by_confidence_band.get("available", False) and "n_samples_used" in by_confidence_band:
#         n_samples_used_overall = int(by_confidence_band["n_samples_used"])
#     elif by_target_quantile.get("available", False) and "n_samples_used" in by_target_quantile:
#         n_samples_used_overall = int(by_target_quantile["n_samples_used"])

#     if available_any and n_samples_used_overall > 0:
#         # Safety: top-level n_samples_used must be <= total n_samples.
#         assert (
#             n_samples_used_overall <= n
#         ), f"Overall n_samples_used={n_samples_used_overall} exceeds n_samples={n}"

#     return {
#         "available": available_any,
#         "task": task or None,
#         "metric": metric,
#         "n_samples": int(n),
#         "n_samples_used": int(n_samples_used_overall),
#         "by_confidence_band": by_confidence_band,
#         "by_target_quantile": by_target_quantile,
#     }

# def _build_segmented_performance_report_for_candidate(
#     mr: ModelResult,
# ) -> dict[str, Any]:
#     """Build a lightweight segmented performance report for one candidate.

#     Slices the primary metric by:
#     - prediction confidence bands (classification, if probabilities available)
#     - target quantiles (regression)

#     This is intentionally lightweight and defensive:
#     - uses x_train / y_train
#     - optionally uses predictions and prediction probabilities if present
#     - caps bands to keep the object bounded
#     """

#     metric: MetricType = mr.options[Option.METRIC]
#     task: TaskType = mr.options[Option.TASK]

#     x_train = getattr(mr, "x_train", None)
#     y_train = getattr(mr, "y_train", None)

#     if x_train is None or y_train is None:
#         return {
#             "available": False,
#             "reason": "Missing x_train or y_train on ModelResult.",
#             "task": task or None,
#             "metric": metric,
#         }

#     x_df = pd.DataFrame(x_train)
#     y_series = util.y_pre_transformation_values(y_train, **mr.options)

#     n = min(len(x_df), len(y_series))
#     if n == 0:
#         return {
#             "available": False,
#             "reason": "x_train or y_train has zero usable rows.",
#             "task": task or None,
#             "metric": metric,
#         }

#     x_df = x_df.iloc[:n, :].reset_index(drop=True)
#     y_series = y_series.iloc[:n].reset_index(drop=True)

#     preds_raw = None
#     for attr in (
#         "oof_pred",
#         "oof_y_pred",
#         "oof_predictions",
#         "val_pred",
#         "predictions",
#         "y_pred",
#     ):
#         val = getattr(mr, attr, None)
#         if val is not None:
#             preds_raw = pd.Series(val)
#             break

#     if preds_raw is not None:
#         y_pred_series = preds_raw.iloc[:n].reset_index(drop=True)
#         y_pred_series = util.y_pre_transformation_values(y_pred_series, **mr.options)
#     else:
#         y_pred_series = None

#     proba_raw = None
#     for attr in (
#         "oof_pred_proba",
#         "oof_proba",
#         "y_pred_proba",
#         "pred_proba",
#         "pred_probs",
#     ):
#         val = getattr(mr, attr, None)
#         if val is not None:
#             proba_raw = val
#             break

#     proba_2d: NDArray[Any] | None = None
#     pos_proba_1d: NDArray[Any] | None = None
#     conf_vec: NDArray[Any] | None = None

#     if proba_raw is not None:
#         proba_arr = np.asarray(proba_raw)

#         proba_2d = (
#             np.column_stack([1.0 - proba_arr[:n], proba_arr[:n]])
#             if proba_arr.ndim == 1
#             else np.asarray(proba_arr[:n], dtype=float)
#         )

#         assert proba_2d.ndim == 2 and proba_2d.shape[1] >= 2, "Invalid probability matrix shape."

#         if np.all(np.isfinite(proba_2d)):
#             proba_2d = np.clip(proba_2d, 0.0, 1.0)
#             row_sums = proba_2d.sum(axis=1, keepdims=True)
#             zero_mask = row_sums <= 0.0
#             if np.any(zero_mask):
#                 proba_2d[zero_mask] = 1.0 / float(proba_2d.shape[1])
#                 row_sums = proba_2d.sum(axis=1, keepdims=True)
#             proba_2d = proba_2d / row_sums

#             conf_vec = np.max(proba_2d, axis=1)
#             if proba_2d.shape[1] == 2:
#                 pos_proba_1d = proba_2d[:, 1]
#         else:
#             proba_2d = None
#             conf_vec = None
#             pos_proba_1d = None

#     def _binary_auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float | None:
#         """Compute binary ROC-AUC from continuous scores without sklearn."""
#         y_true = np.asarray(y_true)
#         scores = np.asarray(scores, dtype=float)

#         if y_true.size == 0 or scores.size == 0 or y_true.size != scores.size:
#             return None

#         classes = np.unique(y_true)
#         if classes.size != 2:
#             return None

#         # Positive class is the larger encoded value (works for 0/1 and bool-ish cases).
#         y_bin = (y_true == classes.max()).astype(int)
#         n_pos = int(y_bin.sum())
#         n_neg = int((1 - y_bin).sum())
#         if n_pos == 0 or n_neg == 0:
#             return None

#         score_series = pd.Series(scores)
#         ranks = score_series.rank(method="average").to_numpy(dtype=float)
#         sum_ranks_pos = float(ranks[y_bin == 1].sum())
#         auc = (sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
#         return float(auc)

#     def _compute_segment_metric(
#         y_true: pd.Series,
#         y_pred: pd.Series | None,
#         y_proba: np.ndarray | None = None,
#     ) -> float | None:
#         if y_true.empty:
#             return None

#         if task == "classification":
#             y_true_arr = np.asarray(y_true)
#             y_pred_arr = np.asarray(y_pred) if y_pred is not None else None
#         else:
#             y_true_arr = np.asarray(y_true, dtype=float)
#             y_pred_arr = np.asarray(y_pred, dtype=float) if y_pred is not None else None

#         if task == "classification":
#             if metric == "roc_auc":
#                 if y_proba is None:
#                     return None
#                 return _binary_auc_from_scores(y_true_arr, np.asarray(y_proba, dtype=float))

#             if y_pred_arr is None:
#                 return None

#             if metric == "balanced_accuracy":
#                 classes = np.unique(y_true_arr)
#                 if classes.size == 0:
#                     return None
#                 recalls: list[float] = []
#                 for cls in classes:
#                     mask = y_true_arr == cls
#                     denom = int(mask.sum())
#                     if denom == 0:
#                         continue
#                     recalls.append(float(np.mean(y_pred_arr[mask] == cls)))
#                 if not recalls:
#                     return None
#                 return float(np.mean(recalls))

#             # accuracy fallback
#             return float(np.mean(y_true_arr == y_pred_arr))

#         if y_pred_arr is None:
#             return None

#         diff = y_true_arr - y_pred_arr

#         if metric == "r2":
#             y_mean = float(np.mean(y_true_arr))
#             ss_res = float(np.sum((y_true_arr - y_pred_arr) ** 2))
#             ss_tot = float(np.sum((y_true_arr - y_mean) ** 2))
#             if ss_tot <= 0.0:
#                 return None
#             return float(1.0 - (ss_res / ss_tot))

#         if metric == "rmse":
#             return float(np.sqrt(np.mean(diff**2)))

#         if metric == "mse":
#             return float(np.mean(diff**2))

#         if metric == "rmsle":
#             y_true_clip = np.maximum(y_true_arr, 0.0)
#             y_pred_clip = np.maximum(y_pred_arr, 0.0)
#             return float(
#                 np.sqrt(
#                     np.mean(
#                         (np.log1p(y_true_clip) - np.log1p(y_pred_clip)) ** 2,
#                     )
#                 )
#             )

#         return float(np.mean(np.abs(diff)))

#     min_segment_size = 30

#     by_confidence_band: dict[str, Any] | None
#     conf_n_used = 0

#     if task == "classification" and conf_vec is not None:
#         conf_series = pd.Series(conf_vec, index=y_series.index)

#         bands = [
#             (0.0, 0.5, "<0.50"),
#             (0.5, 0.7, "0.50–0.70"),
#             (0.7, 0.85, "0.70–0.85"),
#             (0.85, 1.01, "≥0.85"),
#         ]
#         conf_segments: list[dict[str, Any]] = []

#         for lo, hi, label in bands:
#             mask = (conf_series >= lo) & (conf_series < hi)
#             count = int(mask.sum())
#             if count < min_segment_size:
#                 continue

#             y_seg = y_series[mask]
#             y_pred_seg = y_pred_series[mask] if y_pred_series is not None else None
#             y_proba_seg = pos_proba_1d[mask.to_numpy()] if pos_proba_1d is not None else None

#             seg_dict: dict[str, Any] = {
#                 "band_label": label,
#                 "lower": float(lo),
#                 "upper": float(hi),
#                 "count": count,
#             }

#             metric_val = _compute_segment_metric(y_seg, y_pred_seg, y_proba_seg)
#             if metric_val is not None and np.isfinite(metric_val):
#                 seg_dict["metric_value"] = float(metric_val)

#             conf_segments.append(seg_dict)
#             conf_n_used += count

#         if conf_segments and conf_n_used > 0:
#             for seg in conf_segments:
#                 seg["fraction"] = float(seg["count"] / float(conf_n_used))

#             sum_fraction_conf = sum(seg["fraction"] for seg in conf_segments)
#             assert abs(sum_fraction_conf - 1.0) < 1e-6, (
#                 f"by_confidence_band fractions sum to {sum_fraction_conf}, expected 1.0"
#             )

#             assert 0 < conf_n_used <= n, (
#                 f"by_confidence_band n_samples_used={conf_n_used} exceeds n={n} "
#                 "or is non-positive."
#             )

#             by_confidence_band = {
#                 "available": True,
#                 "n_samples_used": int(conf_n_used),
#                 "segments": conf_segments,
#             }
#         else:
#             by_confidence_band = {
#                 "available": False,
#                 "reason": "Not enough samples per confidence band.",
#             }
#     else:
#         by_confidence_band = {
#             "available": False,
#             "reason": (
#                 "Confidence bands require classification task with "
#                 "prediction probabilities."
#             ),
#         }

#     if metric == "log_loss":
#         by_confidence_band = {
#             "available": False,
#             "reason": "Confidence bands are not shown for Log_Loss metric.",
#         }

#     by_target_quantile: dict[str, Any] | None
#     quant_n_used = 0

#     if task == "regression" and metric != "r2":
#         y_numeric = pd.to_numeric(y_series, errors="coerce")

#         if int(y_numeric.notna().sum()) >= min_segment_size * 2:
#             qs = np.quantile(
#                 y_numeric.dropna().to_numpy(),
#                 [0.0, 0.25, 0.5, 0.75, 1.0],
#             )

#             quant_segments: list[dict[str, Any]] = []

#             for i in range(len(qs) - 1):
#                 lo = float(qs[i])
#                 hi = float(qs[i + 1])

#                 if i < len(qs) - 2:
#                     mask = y_numeric.notna() & (y_numeric >= lo) & (y_numeric < hi)
#                 else:
#                     mask = y_numeric.notna() & (y_numeric >= lo) & (y_numeric <= hi)

#                 count = int(mask.sum())
#                 if count < min_segment_size:
#                     continue

#                 y_seg = y_series[mask]
#                 y_pred_seg = y_pred_series[mask] if y_pred_series is not None else None

#                 seg_dict: dict[str, Any] = {
#                     "target_low": lo,
#                     "target_high": hi,
#                     "count": count,
#                 }

#                 try:
#                     seg_dict["target_mean"] = float(pd.to_numeric(y_seg, errors="coerce").mean())
#                 except Exception:
#                     seg_dict["target_mean"] = None

#                 metric_val = (
#                     _compute_segment_metric(y_seg, y_pred_seg) if y_pred_seg is not None else Non
#                 )
#                 if metric_val is not None and np.isfinite(metric_val):
#                     seg_dict["metric_value"] = float(metric_val)

#                 quant_segments.append(seg_dict)
#                 quant_n_used += count

#             if quant_segments and quant_n_used > 0:
#                 for seg in quant_segments:
#                     seg["fraction"] = float(seg["count"] / float(quant_n_used))

#                 sum_fraction_quant = sum(seg["fraction"] for seg in quant_segments)
#                 assert abs(sum_fraction_quant - 1.0) < 1e-6, (
#                     f"by_target_quantile fractions sum to {sum_fraction_quant}, expected 1.0"
#                 )

#                 assert 0 < quant_n_used <= n, (
#                     f"by_target_quantile n_samples_used={quant_n_used} exceeds n={n} "
#                     "or is non-positive."
#                 )

#                 by_target_quantile = {
#                     "available": True,
#                     "n_samples_used": int(quant_n_used),
#                     "segments": quant_segments,
#                 }
#             else:
#                 by_target_quantile = {
#                     "available": False,
#                     "reason": "Not enough samples per target quantile segment.",
#                 }
#         else:
#             by_target_quantile = {
#                 "available": False,
#                 "reason": "Not enough usable rows to build target quantile segments.",
#             }
#     elif task == "regression":
#         by_target_quantile = {
#             "available": False,
#             "reason": "Segmented target quantiles are not shown for R2 metric.",
#         }
#     else:
#         by_target_quantile = {
#             "available": False,
#             "reason": "Target quantile segments are only defined for regression task.",
#         }

#     available_any = bool(
#         by_confidence_band.get("available", False) or by_target_quantile.get("available", False)
#     )

#     n_samples_used_overall = 0
#     if by_confidence_band.get("available", False) and "n_samples_used" in by_confidence_band:
#         n_samples_used_overall = int(by_confidence_band["n_samples_used"])
#     elif by_target_quantile.get("available", False) and "n_samples_used" in by_target_quantile:
#         n_samples_used_overall = int(by_target_quantile["n_samples_used"])

#     if available_any and n_samples_used_overall > 0:
#         assert n_samples_used_overall <= n, (
#             f"Overall n_samples_used={n_samples_used_overall} exceeds n_samples={n}"
#         )

#     return {
#         "available": available_any,
#         "task": task or None,
#         "metric": metric,
#         "n_samples": int(n),
#         "n_samples_used": int(n_samples_used_overall),
#         "by_confidence_band": by_confidence_band,
#         "by_target_quantile": by_target_quantile,
#     }

# ------------------------------------------------------------------
# Helper: build per-candidate baseline comparison dictionary.
# ------------------------------------------------------------------
# def _build_baseline_comparison_for_candidate(
#     candidate_cv_mean: float,
#     baseline_info: dict[str, Any],
#     val_stars: float,
# ) -> dict[str, Any] | None:
#     """Compare this candidate's CV metric to the dataset baseline.

#     Args:
#         candidate_cv_mean: Mean validation *loss* for this candidate
#             (lower-is-better, including log_loss and 1 - R² for R²).
#         baseline_info: Dataset-level baseline information.
#         val_stars: Star rating for this candidate's validation performance
#             (used as relative_improvement_stars).

#     Returns:
#         dict[str, Any] with baseline vs model comparison, or None if
#         a baseline is not available.
#     """
#     if not baseline_info.get("available", False):
#         return None

#     orientation = baseline_info.get("orientation")
#     metric: MetricType | None = baseline_info.get("metric")
#     baseline_value = baseline_info.get("baseline_value")
#     baseline_loss = baseline_info.get("baseline_loss")
#     baseline_type = baseline_info.get("baseline_type")

#     if orientation is None or metric is None or baseline_value is None or baseline_loss is None:
#         return None

#     # ------------------------------------------------------------
#     # 0) log_loss: classification metric but lower-is-better.
#     #     - candidate_cv_mean is log_loss (loss-style).
#     #     - baseline_value / baseline_loss are baseline log_loss.
#     # ------------------------------------------------------------
#     if metric == "log_loss":
#         model_loss = float(candidate_cv_mean)
#         baseline_logloss = float(baseline_value)

#         abs_improvement = baseline_logloss - model_loss
#         denom = max(abs(baseline_logloss), _EPSILON)
#         rel_improvement_pct = abs_improvement / denom

#         return {
#             "task": "classification",
#             "metric": metric,
#             "orientation": "lower_is_better",
#             "baseline_type": baseline_type,
#             "baseline_value": baseline_logloss,  # user-facing log_loss
#             "baseline_loss": float(baseline_loss),
#             "model_value": model_loss,
#             "model_loss": model_loss,
#             "absolute_improvement": float(abs_improvement),
#             "relative_improvement_percent": float(rel_improvement_pct),
#             "relative_improvement_stars": val_stars,
#             "n_samples": baseline_info.get("n_samples"),
#             "n_classes": baseline_info.get("n_classes"),
#         }

#     # ------------------------------------------------------------
#     # 1) R²: special-case on the *metric* (not on orientation).
#     #    - candidate_cv_mean is loss-style: 1 - R².
#     #    - baseline_value is the baseline R² (0.0 for mean predictor).
#     # ------------------------------------------------------------
#     if metric == "r2":
#         # Convert loss-style back to R² for user-facing values.
#         model_r2 = 1.0 - float(candidate_cv_mean)
#         baseline_r2 = float(baseline_value)

#         # Clamp R² to a reasonable range.
#         model_r2 = max(min(model_r2, 1.0), -1.0)

#         abs_improvement = model_r2 - baseline_r2

#         # If baseline R² ≈ 0, "relative improvement" is just "percent variance explained".
#         if abs(baseline_r2) <= _EPSILON:
#             rel_improvement_pct = model_r2
#         else:
#             denom = max(abs(baseline_r2), _EPSILON)
#             rel_improvement_pct = abs_improvement / denom

#         return {
#             "task": "regression",
#             "metric": metric,
#             "orientation": "higher_is_better",
#             "baseline_type": baseline_type,
#             "baseline_value": baseline_r2,  # Baseline R² (typically 0.0 for a mean predictor).
#             # Internal loss-style baseline (1 - baseline_r2, typically 1.0).
#             "baseline_loss": float(baseline_loss),
#             # Model R² and loss.
#             "model_value": model_r2,
#             "model_loss": float(candidate_cv_mean),
#             "absolute_improvement": float(abs_improvement),
#             "relative_improvement_percent": float(rel_improvement_pct),
#             "relative_improvement_stars": val_stars,
#             "n_samples": baseline_info.get("n_samples"),
#             "n_classes": baseline_info.get("n_classes"),
#         }

#     # ------------------------------------------------------------
#     # 2) Classification: higher-is-better (accuracy / balanced_accuracy),
#     #    internal loss is (1 - score).
#     # ------------------------------------------------------------
#     if orientation == "higher_is_better":
#         # Convert loss-style to user-facing score.
#         model_score = 1.0 - float(candidate_cv_mean)
#         model_score = float(np.clip(model_score, 0.0, 1.0))

#         baseline_score = float(baseline_value)
#         abs_improvement = model_score - baseline_score

#         denom = max(abs(baseline_score), _EPSILON)
#         rel_improvement_pct = abs_improvement / denom

#         return {
#             "task": "classification",
#             "metric": metric,
#             "orientation": orientation,
#             "baseline_type": baseline_type,
#             "baseline_value": baseline_score,
#             "baseline_loss": float(baseline_loss),
#             "model_value": model_score,
#             "model_loss": float(candidate_cv_mean),
#             "absolute_improvement": float(abs_improvement),
#             "relative_improvement_percent": float(rel_improvement_pct),
#             "relative_improvement_stars": val_stars,
#             "n_samples": baseline_info.get("n_samples"),
#             "n_classes": baseline_info.get("n_classes"),
#         }

#     # ------------------------------------------------------------
#     # 3) Standard regression: lower-is-better (MAE, RMSE, MSE, RMSLE, etc.).
#     #    baseline_value is already the baseline loss.
#     # ------------------------------------------------------------
#     if orientation == "lower_is_better":
#         model_loss = float(candidate_cv_mean)
#         baseline_loss_val = float(baseline_value)

#         abs_improvement = baseline_loss_val - model_loss
#         denom = max(abs(baseline_loss_val), _EPSILON)
#         rel_improvement_pct = abs_improvement / denom

#         return {
#             "task": "regression",
#             "metric": metric,
#             "orientation": orientation,
#             "baseline_type": baseline_type,
#             "baseline_value": baseline_loss_val,
#             "baseline_loss": float(baseline_loss),
#             "model_value": model_loss,
#             "model_loss": float(candidate_cv_mean),
#             "absolute_improvement": float(abs_improvement),
#             "relative_improvement_percent": float(rel_improvement_pct),
#             "relative_improvement_stars": val_stars,
#             "n_samples": baseline_info.get("n_samples"),
#             "n_classes": None,
#         }

#     # Unknown / unsupported orientation.
#     return None


# def _compute_dataset_baseline_from_y(
#     mr_list: list[ModelResult],
# ) -> dict[str, Any]:
#     """Compute a simple baseline for the dataset using y_train and metric.

#     Returns:
#         dict[str, Any]: Baseline info:

#             {
#                 "available": bool,
#                 "task": str | None,
#                 "metric": str | None,
#                 "baseline_type": str | None,      # "majority_class", "mean", "median", ...
#                 "orientation": str | None,        # "higher_is_better" | "lower_is_better"
#                 "baseline_value": float | None,   # user-facing metric value
#                 "baseline_loss": float | None,    # internal loss-like value
#                 "n_samples": int | None,
#                 "n_classes": int | None,          # classification only
#             }

#         If a baseline cannot be computed, "available" is False and other
#         fields may be None.
#     """
#     if not mr_list:
#         return {
#             "available": False,
#             "task": None,
#             "metric": None,
#             "baseline_type": None,
#             "orientation": None,
#             "baseline_value": None,
#             "baseline_loss": None,
#             "n_samples": None,
#             "n_classes": None,
#         }

#     # Pick the first candidate with a non-empty y_train and options.
#     baseline_source = next(
#         (mr for mr in mr_list if getattr(mr, "y_train", None) is not None),
#         mr_list[0],
#     )
#     options = getattr(baseline_source, "options", {}) or {}
#     task: TaskType = options[Option.TASK]  # str(options.get(Option.TASK, "") or "")
#     metric: MetricType = options[Option.METRIC]  # str(options.get(Option.METRIC, "metric"))

#     # y_train is used only for dataset-level baselines; we ignore folds.
#     y_train = getattr(baseline_source, "y_train", None)
#     if y_train is None:
#         return {
#             "available": False,
#             "task": task or None,
#             "metric": metric or None,
#             "baseline_type": None,
#             "orientation": None,
#             "baseline_value": None,
#             "baseline_loss": None,
#             "n_samples": None,
#             "n_classes": None,
#         }

#     # Convert np.log(y) back to original y.
#     y_series = util.y_pre_transformation_values(y_train, **options)

#     n_samples = int(y_series.shape[0])
#     if n_samples == 0:
#         return {
#             "available": False,
#             "task": task or None,
#             "metric": metric or None,
#             "baseline_type": None,
#             "orientation": None,
#             "baseline_value": None,
#             "baseline_loss": None,
#             "n_samples": 0,
#             "n_classes": None,
#         }

#     # Classification vs regression orientation.
#     # is_classification_metric = metric in util.valid_metrics("classification")
#     # is_regression_metric = metric in util.valid_metrics("regression")

#     # ------------------------------------------------------------------
#     # Classification baseline
#     # ------------------------------------------------------------------
#     if task == "classification":  # and is_classification_metric:
#         value_counts = y_series.value_counts()
#         n_classes = int(value_counts.shape[0])

#         if n_classes == 0:
#             return {
#                 "available": False,
#                 "task": task or None,
#                 "metric": metric or None,
#                 "baseline_type": None,
#                 "orientation": "higher_is_better",
#                 "baseline_value": None,
#                 "baseline_loss": None,
#                 "n_samples": n_samples,
#                 "n_classes": 0,
#             }

#         majority_fraction = float(value_counts.iloc[0] / max(float(n_samples), 1.0))
#         majority_fraction = float(np.clip(majority_fraction, 0.0, 1.0))

#         # ----- balanced_accuracy baseline -----
#         if metric == "balanced_accuracy":
#             baseline_score = 1.0 / max(float(n_classes), 1.0)
#             baseline_score = float(np.clip(baseline_score, 0.0, 1.0))
#             baseline_loss = float(1.0 - baseline_score)
#             return {
#                 "available": True,
#                 "task": "classification",
#                 "metric": metric,
#                 "baseline_type": "equal classes",
#                 "orientation": "higher_is_better",
#                 "baseline_value": baseline_score,
#                 "baseline_loss": baseline_loss,
#                 "n_samples": n_samples,
#                 "n_classes": n_classes,
#             }

#         # ----- log_loss baseline -----
#         if metric == "log_loss":
#             # Use empirical class distribution as a constant predictor:
#             # p_k = n_k / n, baseline log_loss = -sum_k p_k * log(p_k).
#             counts = value_counts.to_numpy(dtype=float)
#             probs = counts / float(n_samples)
#             probs = np.clip(probs, _EPSILON, 1.0)
#             baseline_logloss = float(-np.sum(probs * np.log(probs)))

#             return {
#                 "available": True,
#                 "task": "classification",
#                 "metric": metric,
#                 "baseline_type": "empirical prior",
#                 "orientation": "lower_is_better",
#                 "baseline_value": baseline_logloss,  # user-facing == loss
#                 "baseline_loss": baseline_logloss,  # internal loss-style
#                 "n_samples": n_samples,
#                 "n_classes": n_classes,
#             }

#         # ----- accuracy-style baseline (and any other higher-is-better cls metrics) -----
#         baseline_score = majority_fraction
#         baseline_loss = float(1.0 - baseline_score)
#         return {
#             "available": True,
#             "task": "classification",
#             "metric": metric,
#             "baseline_type": "majority class",
#             "orientation": "higher_is_better",
#             "baseline_value": baseline_score,
#             "baseline_loss": baseline_loss,
#             "n_samples": n_samples,
#             "n_classes": n_classes,
#         }

#     # ------------------------------------------------------------------
#     # Regression baseline: global mean/median predictor.
#     # Special-case R² so the user-facing value is the R² baseline (0.0) and
#     # the internal loss is 1 - R² (matching our loss-style convention).
#     # ------------------------------------------------------------------
#     y_float = util.to_float(y_series)
#     mean_y = float(y_float.mean())
#     median_y = float(y_float.median())

#     # ----- R²: baseline is constant-mean predictor → R² = 0, loss = 1 - R².
#     if metric == "r2":
#         baseline_r2 = 0.0
#         baseline_value = baseline_r2
#         baseline_loss = float(1.0 - baseline_r2)

#         return {
#             "available": True,
#             "task": "regression",
#             "metric": metric,
#             "baseline_type": "R2-base",
#             "orientation": "higher_is_better",
#             "baseline_value": baseline_value,
#             "baseline_loss": baseline_loss,
#             "n_samples": n_samples,
#             "n_classes": None,
#         }

#     # ----- Standard loss-style regression metrics.
#     if metric in {"mae"}:
#         mae_mean = float(np.mean(np.abs(y_float - mean_y)))
#         mae_median = float(np.mean(np.abs(y_float - median_y)))
#         if mae_median <= mae_mean:
#             baseline_type = "median"
#             baseline_loss = mae_median
#         else:
#             baseline_type = "mean"
#             baseline_loss = mae_mean

#     elif metric in {"rmse", "mse"}:
#         mse = float(np.mean((y_float - mean_y) ** 2))
#         baseline_type = "mean"
#         baseline_loss = float(np.sqrt(mse)) if metric == "rmse" else mse

#     elif metric in {"rmsle"}:
#         # Very defensive RMSLE baseline: clip to positive, log1p, predict log-mean.
#         y_clipped = np.maximum(y_float.to_numpy(), 0.0)
#         log_y = np.log1p(y_clipped)
#         log_mean = float(np.mean(log_y))
#         preds = np.expm1(log_mean)
#         baseline_type = "mean"
#         baseline_loss = float(
#             np.sqrt(
#                 np.mean(
#                     (np.log1p(y_clipped) - np.log1p(preds)) ** 2,
#                 )
#             )
#         )

#     else:
#         # Fallback: treat as MAE on the mean predictor.
#         baseline_type = "mean"
#         baseline_loss = float(np.mean(np.abs(y_float - mean_y)))

#     baseline_loss = float(max(baseline_loss, 0.0))

#     return {
#         "available": True,
#         "task": "regression",
#         "metric": metric,
#         "baseline_type": baseline_type,
#         "orientation": "lower_is_better",
#         "baseline_value": baseline_loss,
#         "baseline_loss": baseline_loss,
#         "n_samples": n_samples,
#         "n_classes": None,
#     }

#     # # If we end up here, just say no baseline.
#     # return {
#     #     "available": False,
#     #     "task": task or None,
#     #     "metric": metric or None,
#     #     "baseline_type": None,
#     #     "orientation": None,
#     #     "baseline_value": None,
#     #     "baseline_loss": None,
#     #     "n_samples": n_samples,
#     #     "n_classes": None,
#     # }


# def _compute_robustness(
#     metric: str,
#     train_metric: float,
#     val_metric: float,
# ) -> tuple[float | None, float | None]:
#     """
#     Compute a robustness score and a 0-5 star rating.

#     Inputs are expected to be loss-style values (lower-is-better). For metrics
#     that are naturally higher-is-better, such as accuracy, balanced_accuracy,
#     and r2, the displayed metric is reconstructed as (1 - loss) before
#     computing robustness.
#     """
#     if not np.isfinite(train_metric) or not np.isfinite(val_metric):
#         return None, None

#     if train_metric < 0 or val_metric < 0:
#         raise ValueError("Metrics must be non-negative.")

#     if metric in util.HIGHER_IS_BETTER_METRICS:
#         train_value = 1.0 - train_metric
#         val_value = 1.0 - val_metric
#     else:
#         train_value = train_metric
#         val_value = val_metric

#     scale = max((abs(train_value) + abs(val_value)) / 2.0, _EPSILON)
#     relative_gap = abs(val_value - train_value) / scale
#     robustness_value = 1.0 / (1.0 + relative_gap)

#     if robustness_value >= 0.95:
#         robustness_stars = 5.0
#     elif robustness_value >= 0.85:
#         robustness_stars = 4.0 + (10.0 * (robustness_value - 0.85))
#     elif robustness_value >= 0.75:
#         robustness_stars = 3.0 + (10.0 * (robustness_value - 0.75))
#     elif robustness_value >= 0.65:
#         robustness_stars = 2.0 + (10.0 * (robustness_value - 0.65))
#     elif robustness_value >= 0.55:
#         robustness_stars = 1.0 + (10.0 * (robustness_value - 0.55))
#     else:
#         robustness_stars = robustness_value / 0.55

#     robustness_stars = round(robustness_stars * 10.0) / 10.0

#     return float(robustness_value), float(robustness_stars)


# def _compute_robustness(
#     metric: str,
#     train_metric: float,
#     val_metric: float,
# ) -> tuple[float | None, float | None]:
#     """
#     Compute a metric-agnostic robustness score and a 0-5 star rating.

#     Robustness measures how similar training and validation performance are,
#     relative to the typical scale of the metric. It is independent of whether
#     higher or lower values are better.

#     Parameters
#     ----------
#     train_metric : float
#         Metric value on the training set. Always lower-is-better.
#     val_metric : float
#         Metric value on the validation set. Always lower-is-better.

#     Returns
#     -------
#     robustness : float
#         Robustness score in the range (0, 1], where 1 indicates identical
#         train and validation performance.
#     stars : float
#         Star rating from 0.0 to 5.0.
#     """
#     if not np.isfinite(train_metric) or not np.isfinite(val_metric):
#         return None, None

#     if train_metric < 0 or val_metric < 0:
#         raise ValueError("Metrics must be non-negative.")

#     scale = max((abs(train_metric) + abs(val_metric)) / 2.0, _EPSILON)
#     relative_gap = abs(val_metric - train_metric) / scale
#     robustness_value = 1.0 / (1.0 + relative_gap)

#     if robustness_value >= 0.95:
#         robustness_stars = 5.0
#     elif robustness_value >= 0.85:
#         robustness_stars = 4.0 + (10.0 * (robustness_value - 0.85))
#     elif robustness_value >= 0.75:
#         robustness_stars = 3.0 + (10.0 * (robustness_value - 0.75))
#     elif robustness_value >= 0.65:
#         robustness_stars = 2.0 + (10.0 * (robustness_value - 0.65))
#     elif robustness_value >= 0.55:
#         robustness_stars = 1.0 + (10.0 * (robustness_value - 0.55))
#     else:
#         robustness_stars = robustness_value / 0.55

#     robustness_stars = round(robustness_stars * 10.0) / 10.0

#     return float(robustness_value), float(robustness_stars)


# def _extended_feature_importances(
#     feature_summaries: list[dict[str, Any]], task: TaskType
# ) -> pd.DataFrame:
#     """Build a simplified feature importance table for UI use."""
#     is_classification = task == "classification"
#     prediction_influence_col = "Prediction Influence"

#     feature_importances: list[dict[str, Any]] = []
#     for fs in feature_summaries:
#         importance = float(fs["importance"])
#         typical_change = float(fs["effect"]["typical_change_in_predrisk"])

#         fi: dict[str, Any] = {
#             "Feature": fs["feature_name"],
#             "Model Importance": importance,
#         }

#         if is_classification:
#             fi[prediction_influence_col] = abs(typical_change)
#         else:
#             direction = fs["direction"]
#             if direction in {"higher_increases_risk", "higher_increases_target"}:
#                 sign = 1.0
#             elif direction in {"higher_decreases_risk", "higher_decreases_target"}:
#                 sign = -1.0
#             else:
#                 sign = 0.0

#             fi[prediction_influence_col] = (
#                 float("nan") if sign == 0.0 else sign * abs(typical_change)
#             )

#         feature_importances.append(fi)

#     df = pd.DataFrame(feature_importances)

#     if is_classification and not df.empty:
#         df[prediction_influence_col] = _normalize_to_unit_sum(df[prediction_influence_col])

#     return df


# def _star_rankings(
#     y_train: pd.Series,
#     training_score: float,
#     validation_score: float,
#     **options: Any,
# ) -> tuple[float, float]:
#     """Compute star rankings (0.0-5.0) for train score and val score.

#     Notes:
#     - For classification metrics ("accuracy", "balanced_accuracy"), the provided
#       training_score and validation_score are assumed to be LOSS-style:
#           score_input = 1 - true_score
#       and are converted internally.
#     - For log_loss, training_score and validation_score are the loss values.
#     - For regression metrics (MAE, RMSE, MSE, RMSLE), lower is better as usual and
#       training_score / validation_score are loss-style errors.
#     - For R², training_score / validation_score are loss-style: 1 - R².
#     """
#     metric: MetricType = options[Option.METRIC]

#     # Convert np.log(y) back to original y.
#     y_train = util.y_pre_transformation_values(y_train, **options)

#     # ------------------------------------------------------------------ helpers
#     def _skill_to_stars(skill: float) -> float:
#         """Map relative skill to [0.0, 5.0] stars.

#         Semantics:
#             - skill <= 0.0: 0 stars (baseline or worse)
#             - 0.0 < skill < 1.0: linearly between 0 and 5
#             - skill >= 1.0: 5 stars
#         """
#         if skill <= 0.0:
#             return 0.0

#         skill_clipped = min(1.0, skill)  # [0, 1]
#         stars = 5.0 * skill_clipped
#         return round(stars * 10.0) / 10.0  # round to 0.1

#     # ------------------------------------------------------------------
#     # log_loss branch: lower is better, relative to a log_loss baseline.
#     # ------------------------------------------------------------------
#     if metric == "log_loss":
#         value_counts = y_train.value_counts()
#         n_samples = float(max(len(y_train), 1))
#         if value_counts.empty:
#             # Degenerate case: no labels; treat as zero skill.
#             return 0.0, 0.0

#         counts = value_counts.to_numpy(dtype=float)
#         probs = counts / n_samples
#         probs = np.clip(probs, _EPSILON, 1.0)
#         baseline_error = float(-np.sum(probs * np.log(probs)))

#         def _skill_logloss(err: float) -> float:
#             if not np.isfinite(err) or err < 0.0:
#                 return -1.0
#             if baseline_error <= _EPSILON:
#                 return 0.0
#             # Relative improvement vs baseline loss.
#             return 1.0 - (err / baseline_error)

#         train_skill = _skill_logloss(training_score)
#         val_skill = _skill_logloss(validation_score)

#         train_stars = _skill_to_stars(train_skill)
#         val_stars = _skill_to_stars(val_skill)

#         return train_stars, val_stars

#     # ------------------------------------------------------------------
#     # Classification branch (accuracy / balanced_accuracy).
#     # ------------------------------------------------------------------
#     if metric in {"accuracy", "balanced_accuracy"}:
#         # Inputs are loss-style: loss = 1 - score
#         train_score = 1.0 - training_score
#         val_score = 1.0 - validation_score

#         # Clamp into [0, 1] defensively.
#         train_score = max(0.0, min(1.0, train_score))
#         val_score = max(0.0, min(1.0, val_score))

#         # Compute baseline accuracy / balanced_accuracy from y_train.
#         class_counts = y_train.value_counts()
#         n_classes = len(class_counts)

#         if n_classes == 1:
#             baseline_acc = 1.0
#             baseline_bal_acc = 1.0
#         else:
#             baseline_acc = class_counts.max() / class_counts.sum()
#             baseline_bal_acc = 1.0 / float(n_classes)

#         baseline_acc = max(0.0, min(1.0, float(baseline_acc)))
#         baseline_bal_acc = max(0.0, min(1.0, float(baseline_bal_acc)))
#         baseline = baseline_acc if metric == "accuracy" else baseline_bal_acc

#         def _skill_class(score: float) -> float:
#             if baseline >= 1.0 - _EPSILON:
#                 # Degenerate case: data is trivial, treat as zero skill.
#                 return 0.0
#             return (score - baseline) / max(1.0 - baseline, _EPSILON)

#         train_skill = _skill_class(train_score)
#         val_skill = _skill_class(val_score)

#         train_stars = _skill_to_stars(train_skill)
#         val_stars = _skill_to_stars(val_skill)

#         return train_stars, val_stars

#     # ------------------------------------------------------------------
#     # Regression-like branch (mae, rmse, mse, rmsle, r2)
#     # ------------------------------------------------------------------
#     # ----- R²: handle separately, no baseline_error involved.
#     if metric == "r2":
#         # Loss-style: loss = 1 - R²
#         train_r2 = 1.0 - training_score
#         val_r2 = 1.0 - validation_score

#         # Clamp to a reasonable range (R² can be negative, but not insane).
#         train_r2 = max(min(train_r2, 1.0), -1.0)
#         val_r2 = max(min(val_r2, 1.0), -1.0)

#         baseline_r2 = 0.0  # constant-mean predictor

#         def _skill_r2(score: float) -> float:
#             if score <= baseline_r2:
#                 return 0.0
#             return (score - baseline_r2) / (1.0 - baseline_r2 + _EPSILON)

#         train_skill = _skill_r2(train_r2)
#         val_skill = _skill_r2(val_r2)

#         train_stars = _skill_to_stars(train_skill)
#         val_stars = _skill_to_stars(val_skill)

#         return train_stars, val_stars

#     # ----- Non-R² regression metrics: use baseline_error + 1 - (err / baseline)
#     # Compute a dataset-level baseline error for a simple mean/variance-based
#     # predictor, then scale model error relative to that.
#     y_arr = y_train.to_numpy(dtype=float)
#     y_mean = float(y_arr.mean())

#     if metric in {"mae"}:
#         diffs = y_arr - y_mean
#         baseline_error = float(np.mean(np.abs(diffs)))
#     elif metric in {"rmse", "mse"}:
#         diffs = y_arr - y_mean
#         mse = float(np.mean(diffs**2))
#         baseline_error = float(np.sqrt(mse)) if metric == "rmse" else mse
#     elif metric == "rmsle":
#         if np.any(y_arr < 0):
#             raise ValueError("RMSLE baseline is undefined for negative y_values.")
#         logy = np.log1p(y_arr)
#         diffs_log = logy - logy.mean()
#         baseline_error = float(np.sqrt(np.mean(diffs_log**2)))
#     else:
#         raise ValueError(f"Unknown metric {metric}")

#     def _skill_reg(error: float) -> float:
#         if not np.isfinite(error) or error < 0.0:
#             return -1.0
#         if baseline_error <= _EPSILON:
#             return 0.0
#         return 1.0 - (error / baseline_error)

#     train_skill = _skill_reg(training_score)
#     val_skill = _skill_reg(validation_score)

#     train_stars = _skill_to_stars(train_skill)
#     val_stars = _skill_to_stars(val_skill)

#     return train_stars, val_stars

# if metric == "roc_auc":
#     n_classes = int(y_train.nunique(dropna=True))
#     if n_classes != 2:
#         return 0.0, 0.0

#     train_auc = 1.0 - training_score
#     val_auc = 1.0 - validation_score

#     train_auc = max(0.0, min(1.0, train_auc))
#     val_auc = max(0.0, min(1.0, val_auc))

#     baseline_auc = 0.5

#     def _skill_auc(score: float) -> float:
#         if score <= baseline_auc + _EPSILON:
#             return 0.0
#         return (score - baseline_auc) / max(1.0 - baseline_auc, _EPSILON)

#     train_skill = _skill_auc(train_auc)
#     val_skill = _skill_auc(val_auc)

#     return _skill_to_stars(train_skill), _skill_to_stars(val_skill)


# # ------------------------------------------------------------------
# # ROC-AUC / PR-AUC
# # ------------------------------------------------------------------
# if metric in {"roc_auc", "pr_auc"}:
#     n_classes = int(y_train.nunique(dropna=True))
#     if n_classes != 2:
#         return 0.0, 0.0

#     train_score = 1.0 - training_score
#     val_score = 1.0 - validation_score

#     train_score = max(0.0, min(1.0, train_score))
#     val_score = max(0.0, min(1.0, val_score))

#     if metric == "roc_auc":
#         baseline_score = 0.5
#     else:
#         value_counts = y_train.value_counts()
#         n_samples = float(max(len(y_train), 1))
#         if value_counts.empty:
#             return 0.0, 0.0
#         baseline_score = float(value_counts.iloc[-1] / n_samples)
#         baseline_score = max(0.0, min(1.0, baseline_score))

#     def _skill_rank_metric(score: float) -> float:
#         if score <= baseline_score + _EPSILON:
#             return 0.0
#         return (score - baseline_score) / max(1.0 - baseline_score, _EPSILON)

#     train_skill = _skill_rank_metric(train_score)
#     val_skill = _skill_rank_metric(val_score)

#     return _skill_to_stars(train_skill), _skill_to_stars(val_skill)


# def _star_rankings(
#     y_train: pd.Series,
#     training_score: float,
#     validation_score: float,
#     **options: Any,
# ) -> tuple[float, float]:
#     """Compute star rankings (0.0-5.0) for train score and val score."""
#     metric: MetricType = options[Option.METRIC]

#     y_train = util.y_pre_transformation_values(y_train, **options)

#     def _lift_to_stars(lift: float) -> float:
#         """Map multiplicative lift over baseline to stars (0.0-5.0)."""
#         if lift <= 1.0:
#             return 0.0

#         # Smooth log-scale mapping:
#         # 1x baseline -> 0 stars
#         # ~3x baseline -> ~2 stars
#         # ~7x baseline -> ~3.5 stars
#         # 20x+ baseline -> 5 stars
#         stars = 5.0 * np.log10(lift) / np.log10(20.0)
#         stars = max(0.0, min(5.0, stars))
#         return round(stars * 10.0) / 10.0

#     def _skill_to_stars(skill: float) -> float:
#         if skill <= 0.0:
#             return 0.0
#         skill_clipped = min(1.0, skill)
#         stars = 5.0 * skill_clipped
#         return round(stars * 10.0) / 10.0

#     # ------------------------------------------------------------------
#     # log_loss
#     # ------------------------------------------------------------------
#     if metric == "log_loss":
#         value_counts = y_train.value_counts()
#         n_samples = float(max(len(y_train), 1))
#         if value_counts.empty:
#             return 0.0, 0.0

#         counts = value_counts.to_numpy(dtype=float)
#         probs = counts / n_samples
#         probs = np.clip(probs, _EPSILON, 1.0)
#         baseline_error = float(-np.sum(probs * np.log(probs)))

#         def _skill_logloss(err: float) -> float:
#             if not np.isfinite(err) or err < 0.0:
#                 return -1.0
#             if baseline_error <= _EPSILON:
#                 return 0.0
#             return 1.0 - (err / baseline_error)

#         train_skill = _skill_logloss(training_score)
#         val_skill = _skill_logloss(validation_score)

#         return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

#     # ------------------------------------------------------------------
#     # ROC-AUC / PR-AUC
#     # ------------------------------------------------------------------
#     if metric in {"roc_auc", "pr_auc"}:
#         n_classes = int(y_train.nunique(dropna=True))
#         if n_classes != 2:
#             return 0.0, 0.0

#         train_score = 1.0 - training_score
#         val_score = 1.0 - validation_score

#         train_score = max(0.0, min(1.0, train_score))
#         val_score = max(0.0, min(1.0, val_score))

#         if metric == "roc_auc":
#             baseline_score = 0.5

#             def _roc_auc_to_stars(score: float) -> float:
#                 """Map ROC-AUC to stars using lift over the 0.5 baseline."""
#                 if score <= baseline_score + _EPSILON:
#                     return 0.0

#                 skill = (score - baseline_score) / max(
#                     1.0 - baseline_score,
#                     _EPSILON,
#                 )
#                 lift = 1.0 + skill
#                 return _lift_to_stars(lift)

#             return _roc_auc_to_stars(train_score), _roc_auc_to_stars(val_score)

#         value_counts = y_train.value_counts()
#         n_samples = float(max(len(y_train), 1))
#         if value_counts.empty:
#             return 0.0, 0.0

#         baseline_score = float(value_counts.iloc[-1] / n_samples)
#         baseline_score = max(0.0, min(1.0, baseline_score))

#         def _pr_auc_to_stars(score: float) -> float:
#             """Map PR-AUC to stars using lift over the positive-rate baseline."""
#             if score <= baseline_score + _EPSILON:
#                 return 0.0

#             lift = score / max(baseline_score, _EPSILON)
#             return _lift_to_stars(lift)

#         return _pr_auc_to_stars(train_score), _pr_auc_to_stars(val_score)

#     # ------------------------------------------------------------------
#     # accuracy / balanced_accuracy
#     # ------------------------------------------------------------------
#     if metric in {"accuracy", "balanced_accuracy"}:
#         train_score = 1.0 - training_score
#         val_score = 1.0 - validation_score

#         train_score = max(0.0, min(1.0, train_score))
#         val_score = max(0.0, min(1.0, val_score))

#         class_counts = y_train.value_counts()
#         n_classes = len(class_counts)

#         if n_classes == 1:
#             baseline_acc = 1.0
#             baseline_bal_acc = 1.0
#         else:
#             baseline_acc = class_counts.max() / class_counts.sum()
#             baseline_bal_acc = 1.0 / float(n_classes)

#         baseline_acc = max(0.0, min(1.0, float(baseline_acc)))
#         baseline_bal_acc = max(0.0, min(1.0, float(baseline_bal_acc)))
#         baseline = baseline_acc if metric == "accuracy" else baseline_bal_acc

#         def _skill_class(score: float) -> float:
#             if baseline >= 1.0 - _EPSILON:
#                 return 0.0
#             return (score - baseline) / max(1.0 - baseline, _EPSILON)

#         train_skill = _skill_class(train_score)
#         val_skill = _skill_class(val_score)

#         return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

#     # ------------------------------------------------------------------
#     # R²
#     # ------------------------------------------------------------------
#     if metric == "r2":
#         train_r2 = 1.0 - training_score
#         val_r2 = 1.0 - validation_score

#         train_r2 = max(min(train_r2, 1.0), -1.0)
#         val_r2 = max(min(val_r2, 1.0), -1.0)

#         baseline_r2 = 0.0

#         def _skill_r2(score: float) -> float:
#             if score <= baseline_r2:
#                 return 0.0
#             return (score - baseline_r2) / (1.0 - baseline_r2 + _EPSILON)

#         train_skill = _skill_r2(train_r2)
#         val_skill = _skill_r2(val_r2)

#         return _skill_to_stars(train_skill), _skill_to_stars(val_skill)

#     # ------------------------------------------------------------------
#     # Non-R² regression metrics
#     # ------------------------------------------------------------------
#     y_arr = y_train.to_numpy(dtype=float)
#     y_mean = float(y_arr.mean())

#     if metric in {"mae"}:
#         diffs = y_arr - y_mean
#         baseline_error = float(np.mean(np.abs(diffs)))
#     elif metric in {"rmse", "mse"}:
#         diffs = y_arr - y_mean
#         mse = float(np.mean(diffs**2))
#         baseline_error = float(np.sqrt(mse)) if metric == "rmse" else mse
#     elif metric == "rmsle":
#         if np.any(y_arr < 0):
#             raise ValueError("RMSLE baseline is undefined for negative y_values.")
#         logy = np.log1p(y_arr)
#         diffs_log = logy - logy.mean()
#         baseline_error = float(np.sqrt(np.mean(diffs_log**2)))
#     else:
#         raise ValueError(f"Unknown metric {metric}")

#     def _skill_reg(error: float) -> float:
#         if not np.isfinite(error) or error < 0.0:
#             return -1.0
#         if baseline_error <= _EPSILON:
#             return 0.0
#         return 1.0 - (error / baseline_error)

#     train_skill = _skill_reg(training_score)
#     val_skill = _skill_reg(validation_score)

#     return _skill_to_stars(train_skill), _skill_to_stars(val_skill)
