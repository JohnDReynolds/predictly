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
    """
    Only needed for FLAML, not AGL.  ChatGPT has all sorts of good reasons why FLAML inherently
    needs these caps, but AGL does not.  With more time, FLAML just keeps trying deeper trees with
    the objective of getting the best training metric possible (e.g. overfitting).  AGL does not
    have this problem.  Note that there is also special limits of processing time for FLAML in
    trainers.py._train_parameters().

    Clamp numeric choice lists into [v_min, v_max].

    Assumptions (true in make_hp_spaces_flaml):
      - values is always a list[int] or list[float].
      - We return a list of the same numeric type (int -> int, float -> float).

    This keeps mypy/pyright/pylint happy and fits TuneParam usage.
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
    """Clip learning rate to a reasonable band and round."""
    return max(0.005, min(0.2, round(x, 4)))


def _subsamp_pair(base: float, s: float, floor_: float) -> list[float]:
    """Row/feature subsampling choices.

    As s increases, we allow more aggressive subsampling but never go
    below `floor_`.

    Example:
        base = 0.9, floor_ = 0.3
        s=0.0 -> {0.8, 0.9}
        s=1.0 -> {0.6, 0.9}
    """
    s = float(max(0.0, min(1.0, s)))
    # Up to -0.30 at s=1 (tighter subsampling)
    lo = max(floor_, round(base - (0.10 + 0.20 * s), 3))
    hi = round(base, 3)
    return sorted({lo, hi})


def _depth_choices(base: int, s: float) -> list[int]:
    """Tree depth choices that shrink smoothly with strictness.

    s=0.0  -> around [base-1, base]
    s=1.0  -> around [~0.4*base, ~0.7*base] (min 2)

    This ensures that at high s, we strongly prefer shallow trees, while
    at low s we still let the model explore deeper structures.
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

    s=0.0 -> {base, ~1.5*base}   (high capacity allowed)
    s=1.0 -> {~0.4*base, base}   (strongly constrained)

    This is symmetric: low s can explore larger models; high s
    encourages smaller models but never collapses to a single value.
    """
    s = float(max(0.0, min(1.0, s)))

    down_factor = 1.0 - 0.6 * s  # 1.0 -> 0.4
    up_factor = 1.0 + 0.5 * (1.0 - s)  # 1.5 at s=0, 1.0 at s=1

    n_down = max(4, int(round(base * down_factor)))
    n_up = max(n_down, int(round(base * up_factor)))
    return sorted({n_down, n_up})


def _leafsize_choices(base: int, s: float) -> list[int]:
    """Min-samples-per-leaf choices.

    As s increases, we gradually force larger leaves, which smooths
    predictions and reduces variance.

    s=0.0 -> {base, ~1.5*base, ~2.0*base}
    s=1.0 -> {base, ~3.0*base, ~4.0*base}
    """
    s = float(max(0.0, min(1.0, s)))

    mid_scale = 1.5 + 1.5 * s  # 1.5 -> 3.0
    hi_scale = 2.0 + 2.0 * s  # 2.0 -> 4.0

    leaf_base = max(2, base)
    mid = max(2, int(round(leaf_base * mid_scale)))
    hi = max(2, int(round(leaf_base * hi_scale)))
    return sorted({leaf_base, mid, hi})


def _rounds_choices(s: float) -> list[int]:
    """Boosting-iteration choices that decrease upper bound with strictness.

    s=0.0 -> {200, 300, 450}
    s=1.0 -> {150, 250} (via {150, 250, 250})

    This keeps iterations in a sensible band and shrinks the top end as s
    increases, encouraging earlier stopping / simpler ensembles.
    """
    s = float(max(0.0, min(1.0, s)))

    # lower bound: 150–200 as s goes 1 -> 0
    low = int(round(150 + 50 * (1.0 - s)))  # s=0 -> 200, s=1 -> 150
    mid = 250
    # upper bound: 250–450 as s goes 1 -> 0
    high = int(round(250 + 200 * (1.0 - s)))  # s=0 -> 450, s=1 -> 250

    return sorted({low, mid, high})


def _lr_choices(base: float, s: float) -> list[float]:
    """Learning-rate choices scaled smoothly by strictness.

    As s increases, we reduce the *target* learning rate and create
    a small band around it.

    s=0.0 -> ~{0.85*base, base, 1.10*base}
    s=1.0 -> ~{0.85*0.4*base, 0.4*base, 1.10*0.4*base}
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

    As s increases, we center L2 around a larger value:
        center = base * (1 + 3*s)  up to 4× at s=1
    and then explore around that center.
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
    """L1-style regularization or penalties.

    For s>0 and base>0, we scale base upward with s.
    If base==0, we allow turning L1 on gradually from zero.

    s=0.0 -> {base}
    s=1.0 -> ~{base, 1.75*base}
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
    """TypedDict used to satisfy FLAML's custom_hp format."""

    domain: object


def _ch(vals: Sequence[object]) -> TuneParam:
    """
    Wrap a sequence of candidate values in FLAML's tune.choice.

    We keep the TypedDict wrapper to make mypy/pyright happy.
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
    """Build AutoGluon Tabular hyperparameter spaces based on strictness s.

    The returned dict has the form:
        {
            "CAT": {... space ...},
            "GBM": {... space ...},
            "XGB": {... space ...},  # only if include_xgb is True
            "RF":  {... space ...},
            "XT":  {... space ...},
        }

    where each inner dict is a valid AGL hyperparameter search space.
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
    """Build FLAML AutoML custom_hp spaces based on strictness s.

    Returns a dict:
        {
            "catboost": {...},
            "lgbm": {...},
            "xgboost": {...},
            "rf": {...},
            "extra_tree": {...},
            "histgb": {...},  # scikit-learn HistGradientBoosting
        }
    suitable for passing as AutoML(..., custom_hp=...).

    Notes:
      * We keep your existing strictness logic via _rounds_choices/_depth_choices/etc.
      * We additionally clamp key capacity knobs (n_estimators, depth, num_leaves, etc.)
        to conservative global caps so that very large time budgets cannot grow single
        models arbitrarily large. Extra time should go into more trials, not much
        deeper/wider trees.
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
