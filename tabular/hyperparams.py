"""hyperparams.py

Dynamic hyperparameter-space construction for AutoGluon (AGL) and FLAML.

The main idea:
    - s_strict is a continuous strictness parameter s ∈ [0, 1].
    - We then shape all capacity / regularization knobs (depth, leaves,
      min_samples_leaf, subsampling, rounds, learning rate, L1/L2, etc.)
      as smooth functions of s.

Key properties:
    - No hard cliffs like "if s >= 0.8 then completely different space".
      s=0.79 and s=0.80 produce similar spaces.
    - As s increases:
        * Model capacity shrinks (shallower trees, fewer leaves,
          fewer rounds).
        * Regularization grows (higher L2, bigger min_samples_leaf).
        * Subsampling becomes tighter.
    - Works generically for regression, binary classification, and
      multi-class classification.
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false

# Python imports
from __future__ import annotations
from typing import Any, cast, Dict, Sequence, TypedDict, TypeVar

# Third-party imports
from autogluon.common import space as ag
from flaml import tune

# Project imports
import tabular.utilities as util

# Conservative global caps for model capacity (FLAML)
_MAX_CAT_DEPTH = 8
_MAX_CAT_ESTIMATORS = 400

_MAX_LGBM_LEAVES = 63
_MAX_LGBM_ESTIMATORS = 400

_MAX_XGB_DEPTH = 8
_MAX_XGB_ESTIMATORS = 400

_MAX_RF_ESTIMATORS = 400
_MAX_XT_ESTIMATORS = 400

_MAX_HISTGB_DEPTH = 8
_MAX_HISTGB_ITERS = 400


TNum = TypeVar("TNum", int, float)  # pylint: disable=invalid-name


def _cap_choice_seq(
    values: list[TNum],
    v_min: float | None = None,
    v_max: float | None = None,
) -> list[TNum]:
    """Clamp numeric values into a bounded range.

    Each value is clipped to the interval [v_min, v_max] (if provided),
    preserving the original numeric type (int or float).

    This is primarily used for FLAML to prevent unbounded model growth
    when search budgets are large.

    Args:
        values: List of numeric values (int or float).
        v_min: Optional lower bound.
        v_max: Optional upper bound.

    Returns:
        List of clamped values with original types preserved.
    """
    out: list[TNum] = []

    for v in values:
        # v is int or float
        new = float(v)
        if v_min is not None and new < v_min:
            util.print_local(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FLAML HPO CAP: {new} < {v_min}"
            )
            new = v_min
        if v_max is not None and new > v_max:
            util.print_local(
                f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FLAML HPO CAP: {new} > {v_max}"
            )
            new = v_max

        # Cast back to the original numeric type (int or float)
        # out.append(cast(TNum, type(v)(new)))
        out.append(type(v)(new))

    return out


# ============================================================================
# Baseline (unscaled) hyperparameters (v: slightly stronger regularization)
# ============================================================================
BASE: Dict[str, Dict[str, Any]] = {
    # CatBoost-style baseline
    "CAT": {
        "depth": 6,
        "l2_leaf_reg": 10.0,  # was 8.0 in original BASE
        "rsm": 0.90,
        # was 0.90 – a touch more sampling noise / regularization
        "subsample": 0.85,
        "iterations": 300,
        "learning_rate": 0.05,
        "bootstrap_type": "Bernoulli",
    },
    # LightGBM-style baseline
    "GBM": {
        "min_data_in_leaf": 20,
        # was 0.90 – slightly more column subsampling
        "feature_fraction": 0.85,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        # was 1.5 in previous tweak – slightly stronger L2
        "lambda_l2": 1.8,
        "num_boost_round": 300,
        "learning_rate": 0.05,
        "lambda_l1": 6.0,  # was 5.0 original
        "min_gain_to_split": 2.5,  # was 2.0 original
        "num_leaves": 31,  # strictness scaling still shrinks this
    },
    # XGBoost-style baseline
    "XGB": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        # was 5 – slightly heavier leaves
        "min_child_weight": 6,
        # was 0.90 – slightly more column subsampling
        "colsample_bytree": 0.85,
        "subsample": 0.90,
        "reg_lambda": 1.5,  # was 1.0 original
        "reg_alpha": 6.0,  # was 5.0 original
        "gamma": 4.5,  # was 4.0 original
        "max_depth": 6,  # still the same; strictness scaling shrinks effective depth
    },
    # RandomForest-style baseline
    "RF": {
        "n_estimators": 300,
        "min_samples_leaf": 3,  # was 2 original
        "max_features": 0.90,
        "bootstrap": True,
        "max_samples": 0.90,
        "min_impurity_decrease": 7e-3,  # was 5e-3 original
    },
    # ExtraTrees-style baseline
    "XT": {
        "n_estimators": 300,
        "min_samples_leaf": 3,  # was 2 original
        "max_features": 0.90,
        "bootstrap": True,
        "max_samples": 0.90,
        "min_impurity_decrease": 7e-3,  # was 5e-3 original
    },
}


# ============================================================================
# Scaling helpers
# All helpers assume s ∈ [0,1] and behave *smoothly* in s (no cliffs).
# ============================================================================
def _clip_lr(x: float) -> float:
    """Clamp and round a learning rate.

    Args:
        x: Raw learning rate value.

    Returns:
        Learning rate clipped to [0.005, 0.2] and rounded.
    """
    return max(0.005, min(0.2, round(x, 4)))


def _subsamp_pair(base: float, s: float, floor_: float) -> list[float]:
    """Row/feature subsampling choices.

    This introduces stochasticity (via subsampling) to reduce overfitting,
    with stronger effects at higher strictness.

    Args:
        base: Baseline subsampling value.
        s: Strictness parameter in [0, 1].
        floor_: Minimum allowable subsampling.

    Returns:
        Sorted list of subsampling candidates.
    """
    s = float(max(0.0, min(1.0, s)))
    # Up to -0.30 at s=1 (tighter subsampling)
    lo = max(floor_, round(base - (0.10 + 0.20 * s), 3))
    hi = round(base, 3)
    return sorted({lo, hi})


def _depth_choices(base: int, s: float) -> list[int]:
    """Tree depth choices that shrink smoothly with strictness.

    This enforces a continuous bias toward shallower trees as strictness
    increases, reducing overfitting without introducing hard thresholds.

    Args:
        base: Baseline depth.
        s: Strictness parameter in [0, 1].

    Returns:
        Sorted list of candidate depths.
    """
    s = float(max(0.0, min(1.0, s)))

    # Shrink from ~1.0 down to ~0.4 as s increases
    min_factor = 1.0 - 0.6 * s  # 1.0 -> 0.4
    max_factor = 1.0 - 0.3 * s  # 1.0 -> 0.7

    d_min = max(2, int(round(base * min_factor)))
    d_max = max(d_min, int(round(base * max_factor)))
    return sorted({d_min, d_max})


def _leaves_pair(base: int, s: float) -> list[int]:
    """Leaf-count choices for boosted trees.

    This balances exploration vs. regularization:
    - Low strictness allows larger trees (higher capacity).
    - High strictness biases toward smaller trees.

    Args:
        base: Baseline number of leaves.
        s: Strictness parameter in [0, 1].

    Returns:
        Sorted list of candidate leaf counts.
    """
    s = float(max(0.0, min(1.0, s)))

    down_factor = 1.0 - 0.6 * s  # 1.0 -> 0.4
    up_factor = 1.0 + 0.5 * (1.0 - s)  # 1.5 at s=0, 1.0 at s=1

    n_down = max(4, int(round(base * down_factor)))
    n_up = max(n_down, int(round(base * up_factor)))
    return sorted({n_down, n_up})


def _leafsize_choices(base: int, s: float) -> list[int]:
    """Min-samples-per-leaf choices.

    Larger leaf sizes smooth predictions and reduce variance. This function
    gradually enforces that behavior as strictness increases.

    Args:
        base: Baseline minimum samples per leaf.
        s: Strictness parameter in [0, 1].

    Returns:
        Sorted list of candidate leaf sizes.
    """
    s = float(max(0.0, min(1.0, s)))

    mid_scale = 1.5 + 1.5 * s  # 1.5 -> 3.0
    hi_scale = 2.0 + 2.0 * s  # 2.0 -> 4.0

    leaf_base = max(2, base)
    mid = max(2, int(round(leaf_base * mid_scale)))
    hi = max(2, int(round(leaf_base * hi_scale)))
    return sorted({leaf_base, mid, hi})


def _rounds_choices(s: float) -> list[int]:
    """Boosting iteration choices.

    This limits ensemble size as strictness increases, encouraging simpler
    models and earlier convergence.

    Args:
        s: Strictness parameter in [0, 1].

    Returns:
        Sorted list of candidate iteration counts.
    """
    s = float(max(0.0, min(1.0, s)))

    # lower bound: 150–200 as s goes 1 -> 0
    low = int(round(150 + 50 * (1.0 - s)))  # s=0 -> 200, s=1 -> 150
    mid = 250
    # upper bound: 250–450 as s goes 1 -> 0
    high = int(round(250 + 200 * (1.0 - s)))  # s=0 -> 450, s=1 -> 250

    return sorted({low, mid, high})


def _lr_choices(base: float, s: float) -> list[float]:
    """Learning-rate choices scaled by strictness.

    Lower learning rates at higher strictness encourage more stable,
    conservative training and reduce overfitting.

    Args:
        base: Baseline learning rate.
        s: Strictness parameter in [0, 1].

    Returns:
        Sorted list of candidate learning rates.
    """
    s = float(max(0.0, min(1.0, s)))

    target = base * (1.0 - 0.60 * s)  # to 0.4× at s=1
    cands = {
        _clip_lr(target * 0.85),
        _clip_lr(target),
        _clip_lr(target * 1.10),
    }
    return sorted(cands)


def _l2_choices(base: float, s: float) -> list[float]:
    """L2 regularization choices.

    This increases regularization strength as strictness increases,
    helping control model complexity.

    Args:
        base: Baseline L2 value.
        s: Strictness parameter in [0, 1].

    Returns:
        Sorted list of candidate L2 values.
    """
    s = float(max(0.0, min(1.0, s)))

    center = round(base * (1.0 + 3.0 * s), 6)  # up to 4×
    return sorted(
        {
            center,
            round(center * 1.5, 6),
            round(max(center / 1.5, 1e-12), 6),
        }
    )


def _l1_duo(base: float, s: float) -> list[float]:
    """L1-style regularization choices.

    This allows L1 penalties to scale with strictness, enabling stronger
    sparsity and feature selection at higher strictness levels.

    Args:
        base: Baseline L1 value.
        s: Strictness parameter in [0, 1].

    Returns:
        List of candidate L1 values.
    """
    s = float(max(0.0, min(1.0, s)))
    mult = 1.0 + 0.75 * s  # up to 1.75×

    if base <= 0.0:
        return [0.0, round(0.5 * s, 6)]  # allow turning on from zero

    return sorted({round(base, 6), round(base * mult, 6)})


# ============================================================================
# FLAML typed wrapper (for tune.choice)
# ============================================================================
class TuneParam(TypedDict):
    """FLAML hyperparameter domain wrapper.

    Attributes:
        domain: Underlying FLAML search space (e.g., tune.choice).
    """

    domain: object


def _ch(vals: Sequence[object]) -> TuneParam:
    """Wrap values in a FLAML choice domain.

    This exists purely to satisfy FLAML's expected TypedDict format
    while keeping static type checkers happy.

    Args:
        vals: Sequence of candidate values.

    Returns:
        TuneParam containing a tune.choice domain.
    """
    # No easy way to avoid the generic 'object' here.
    return {"domain": cast(object, tune.choice(list(vals)))}


# ============================================================================
# AutoGluon hyperparameter spaces
# ============================================================================
def make_hp_spaces_agl(
    s: float,
    *,
    include_xgb: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Construct AutoGluon hyperparameter spaces from strictness.

    The strictness parameter ``s`` ∈ [0, 1] controls the trade-off between
    model capacity and regularization:

        - Lower s → larger models, weaker regularization
        - Higher s → smaller models, stronger regularization

    The returned dictionary maps model types to AutoGluon-compatible
    hyperparameter search spaces.

    Args:
        s: Strictness parameter in [0, 1].
        include_xgb: Whether to include XGBoost search space.

    Returns:
        Dictionary of model → hyperparameter space mappings.

    Raises:
        ValueError: If input assumptions are violated (should not occur under normal usage).
    """
    s = float(max(0.0, min(1.0, s)))
    b = BASE
    out: Dict[str, Dict[str, Any]] = {}

    # --- CatBoost ---
    out["CAT"] = {
        "iterations": ag.Categorical(*_rounds_choices(s)),
        "depth": ag.Categorical(*_depth_choices(int(b["CAT"]["depth"]), s)),
        "learning_rate": ag.Categorical(*_lr_choices(float(b["CAT"]["learning_rate"]), s)),
        "l2_leaf_reg": ag.Categorical(*_l2_choices(float(b["CAT"]["l2_leaf_reg"]), s)),
        "rsm": ag.Categorical(*_subsamp_pair(float(b["CAT"]["rsm"]), s, 0.3)),
        "subsample": ag.Categorical(*_subsamp_pair(float(b["CAT"]["subsample"]), s, 0.3)),
        "bootstrap_type": b["CAT"]["bootstrap_type"],
    }

    # --- LightGBM-style GBM ---
    gbm: Dict[str, Any] = {
        "num_boost_round": ag.Categorical(*_rounds_choices(s)),
        "learning_rate": ag.Categorical(*_lr_choices(float(b["GBM"]["learning_rate"]), s)),
        "min_data_in_leaf": ag.Categorical(
            *_leafsize_choices(int(b["GBM"]["min_data_in_leaf"]), s)
        ),
        "feature_fraction": ag.Categorical(
            *_subsamp_pair(float(b["GBM"]["feature_fraction"]), s, 0.3)
        ),
        "bagging_fraction": ag.Categorical(
            *_subsamp_pair(float(b["GBM"]["bagging_fraction"]), s, 0.3)
        ),
        "lambda_l2": ag.Categorical(*_l2_choices(float(b["GBM"]["lambda_l2"]), s)),
        "bagging_freq": b["GBM"]["bagging_freq"],
    }
    if "num_leaves" in b["GBM"]:
        gbm["num_leaves"] = ag.Categorical(*_leaves_pair(int(b["GBM"]["num_leaves"]), s))
    if "lambda_l1" in b["GBM"]:
        gbm["lambda_l1"] = ag.Categorical(*_l1_duo(float(b["GBM"]["lambda_l1"]), s))
    if "min_gain_to_split" in b["GBM"]:
        gbm["min_gain_to_split"] = ag.Categorical(
            *_l1_duo(float(b["GBM"]["min_gain_to_split"]), s)
        )
    out["GBM"] = gbm

    # --- XGBoost-style (optional) ---
    if include_xgb:
        xgb: Dict[str, Any] = {
            "n_estimators": ag.Categorical(*_rounds_choices(s)),
            "learning_rate": ag.Categorical(*_lr_choices(float(b["XGB"]["learning_rate"]), s)),
            "min_child_weight": ag.Categorical(
                *_leafsize_choices(int(b["XGB"]["min_child_weight"]), s)
            ),
            "colsample_bytree": ag.Categorical(
                *_subsamp_pair(float(b["XGB"]["colsample_bytree"]), s, 0.3)
            ),
            "subsample": ag.Categorical(*_subsamp_pair(float(b["XGB"]["subsample"]), s, 0.3)),
            "reg_lambda": ag.Categorical(*_l2_choices(float(b["XGB"]["reg_lambda"]), s)),
        }
        if "max_depth" in b["XGB"]:
            xgb["max_depth"] = ag.Categorical(*_depth_choices(int(b["XGB"]["max_depth"]), s))
        if "reg_alpha" in b["XGB"]:
            xgb["reg_alpha"] = ag.Categorical(*_l1_duo(float(b["XGB"]["reg_alpha"]), s))
        if "gamma" in b["XGB"]:
            xgb["gamma"] = ag.Categorical(*_l1_duo(float(b["XGB"]["gamma"]), s))
        out["XGB"] = xgb

    # --- RandomForest-style ---
    out["RF"] = {
        "n_estimators": ag.Categorical(*_rounds_choices(s)),
        "min_samples_leaf": ag.Categorical(
            *_leafsize_choices(int(b["RF"]["min_samples_leaf"]), s)
        ),
        "max_features": ag.Categorical(*_subsamp_pair(float(b["RF"]["max_features"]), s, 0.2)),
        "bootstrap": b["RF"]["bootstrap"],
        "max_samples": ag.Categorical(*_subsamp_pair(float(b["RF"]["max_samples"]), s, 0.3)),
        "min_impurity_decrease": ag.Categorical(
            *_l1_duo(float(b["RF"]["min_impurity_decrease"]), s)
        ),
    }

    # --- ExtraTrees-style ---
    out["XT"] = {
        "n_estimators": ag.Categorical(*_rounds_choices(s)),
        "min_samples_leaf": ag.Categorical(
            *_leafsize_choices(int(b["XT"]["min_samples_leaf"]), s)
        ),
        "max_features": ag.Categorical(*_subsamp_pair(float(b["XT"]["max_features"]), s, 0.2)),
        "bootstrap": b["XT"]["bootstrap"],
        "max_samples": ag.Categorical(*_subsamp_pair(float(b["XT"]["max_samples"]), s, 0.3)),
        "min_impurity_decrease": ag.Categorical(
            *_l1_duo(float(b["XT"]["min_impurity_decrease"]), s)
        ),
    }

    return out


# ============================================================================
# FLAML hyperparameter spaces
# ============================================================================
def make_hp_spaces_flaml(s: float) -> Dict[str, Dict[str, TuneParam]]:
    """Construct FLAML hyperparameter spaces from strictness.

    Produces FLAML-compatible ``custom_hp`` dictionaries where all
    hyperparameters are expressed as ``tune.choice`` domains.

    The strictness parameter ``s`` ∈ [0, 1] smoothly controls:

        - Model capacity (depth, estimators, leaves)
        - Regularization (L1/L2, min leaf size)
        - Subsampling intensity

    Additionally, all major capacity parameters are clamped to
    conservative global limits to prevent excessive model growth.

    Args:
        s: Strictness parameter in [0, 1].

    Returns:
        Dictionary of model → FLAML hyperparameter space mappings.

    Raises:
        ValueError: If input assumptions are violated (should not occur under normal usage).
    """
    s = float(max(0.0, min(1.0, s)))
    b = BASE
    out: Dict[str, Dict[str, TuneParam]] = {}

    # --- CatBoost ---
    cat_rounds = _cap_choice_seq(_rounds_choices(s), v_min=50, v_max=_MAX_CAT_ESTIMATORS)
    cat_depth = _cap_choice_seq(
        _depth_choices(int(b["CAT"]["depth"]), s),
        v_min=3,
        v_max=_MAX_CAT_DEPTH,
    )

    out["catboost"] = {
        "n_estimators": _ch(cat_rounds),
        "depth": _ch(cat_depth),
        "learning_rate": _ch(_lr_choices(float(b["CAT"]["learning_rate"]), s)),
        "l2_leaf_reg": _ch(_l2_choices(float(b["CAT"]["l2_leaf_reg"]), s)),
        "rsm": _ch(_subsamp_pair(float(b["CAT"]["rsm"]), s, 0.3)),
        "subsample": _ch(_subsamp_pair(float(b["CAT"]["subsample"]), s, 0.3)),
        "bootstrap_type": _ch([b["CAT"]["bootstrap_type"]]),
    }

    # --- LightGBM (lgbm) ---
    lgbm_rounds = _cap_choice_seq(_rounds_choices(s), v_min=50, v_max=_MAX_LGBM_ESTIMATORS)

    lgbm: Dict[str, TuneParam] = {
        "n_estimators": _ch(lgbm_rounds),
        "learning_rate": _ch(_lr_choices(float(b["GBM"]["learning_rate"]), s)),
        "min_data_in_leaf": _ch(_leafsize_choices(int(b["GBM"]["min_data_in_leaf"]), s)),
        "feature_fraction": _ch(_subsamp_pair(float(b["GBM"]["feature_fraction"]), s, 0.3)),
        "bagging_fraction": _ch(_subsamp_pair(float(b["GBM"]["bagging_fraction"]), s, 0.3)),
        "lambda_l2": _ch(_l2_choices(float(b["GBM"]["lambda_l2"]), s)),
        "bagging_freq": _ch([b["GBM"]["bagging_freq"]]),
    }
    if "num_leaves" in b["GBM"]:
        lgbm_leaves = _cap_choice_seq(
            _leaves_pair(int(b["GBM"]["num_leaves"]), s),
            v_min=8,
            v_max=_MAX_LGBM_LEAVES,
        )
        lgbm["num_leaves"] = _ch(lgbm_leaves)
    if "lambda_l1" in b["GBM"]:
        lgbm["lambda_l1"] = _ch(_l1_duo(float(b["GBM"]["lambda_l1"]), s))
    if "min_gain_to_split" in b["GBM"]:
        lgbm["min_gain_to_split"] = _ch(_l1_duo(float(b["GBM"]["min_gain_to_split"]), s))
    out["lgbm"] = lgbm

    # --- XGBoost (xgboost) ---
    xgb_rounds = _cap_choice_seq(_rounds_choices(s), v_min=50, v_max=_MAX_XGB_ESTIMATORS)

    xgb: Dict[str, TuneParam] = {
        "n_estimators": _ch(xgb_rounds),
        "learning_rate": _ch(_lr_choices(float(b["XGB"]["learning_rate"]), s)),
        "min_child_weight": _ch(_leafsize_choices(int(b["XGB"]["min_child_weight"]), s)),
        "colsample_bytree": _ch(_subsamp_pair(float(b["XGB"]["colsample_bytree"]), s, 0.3)),
        "subsample": _ch(_subsamp_pair(float(b["XGB"]["subsample"]), s, 0.3)),
        "reg_lambda": _ch(_l2_choices(float(b["XGB"]["reg_lambda"]), s)),
    }
    if "max_depth" in b["XGB"]:
        xgb_depth = _cap_choice_seq(
            _depth_choices(int(b["XGB"]["max_depth"]), s),
            v_min=3,
            v_max=_MAX_XGB_DEPTH,
        )
        xgb["max_depth"] = _ch(xgb_depth)
    if "reg_alpha" in b["XGB"]:
        xgb["reg_alpha"] = _ch(_l1_duo(float(b["XGB"]["reg_alpha"]), s))
    if "gamma" in b["XGB"]:
        xgb["gamma"] = _ch(_l1_duo(float(b["XGB"]["gamma"]), s))
    out["xgboost"] = xgb

    # --- RandomForest (rf) ---
    rf_rounds = _cap_choice_seq(_rounds_choices(s), v_min=50, v_max=_MAX_RF_ESTIMATORS)

    out["rf"] = {
        "n_estimators": _ch(rf_rounds),
        "min_samples_leaf": _ch(_leafsize_choices(int(b["RF"]["min_samples_leaf"]), s)),
        "max_features": _ch(_subsamp_pair(float(b["RF"]["max_features"]), s, 0.2)),
        "bootstrap": _ch([b["RF"]["bootstrap"]]),
        "max_samples": _ch(_subsamp_pair(float(b["RF"]["max_samples"]), s, 0.3)),
        "min_impurity_decrease": _ch(_l1_duo(float(b["RF"]["min_impurity_decrease"]), s)),
    }

    # --- ExtraTrees (extra_tree) ---
    xt_rounds = _cap_choice_seq(_rounds_choices(s), v_min=50, v_max=_MAX_XT_ESTIMATORS)

    out["extra_tree"] = {
        "n_estimators": _ch(xt_rounds),
        "min_samples_leaf": _ch(_leafsize_choices(int(b["XT"]["min_samples_leaf"]), s)),
        "max_features": _ch(_subsamp_pair(float(b["XT"]["max_features"]), s, 0.2)),
        "bootstrap": _ch([b["XT"]["bootstrap"]]),
        "max_samples": _ch(_subsamp_pair(float(b["XT"]["max_samples"]), s, 0.3)),
        "min_impurity_decrease": _ch(_l1_duo(float(b["XT"]["min_impurity_decrease"]), s)),
    }

    # --- HistGradientBoosting (histgb) baseline ---
    #
    # We reuse the same smooth helpers and apply conservative caps to keep
    # trees and iterations in a reasonable range even for very large budgets.
    hist_rounds = _cap_choice_seq(_rounds_choices(s), v_min=50, v_max=_MAX_HISTGB_ITERS)
    hist_depth = _cap_choice_seq(
        _depth_choices(6, s),
        v_min=3,
        v_max=_MAX_HISTGB_DEPTH,
    )

    out["histgb"] = {
        "max_iter": _ch(hist_rounds),
        "learning_rate": _ch(_lr_choices(0.05, s)),
        "max_depth": _ch(hist_depth),
        "min_samples_leaf": _ch(_leafsize_choices(20, s)),
        "l2_regularization": _ch(_l2_choices(0.05, s)),
    }

    return out
