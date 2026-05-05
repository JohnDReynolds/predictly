"""
tester.py
"""

# Errors to ignore
# pylint: disable=unused-import, disable=wrong-import-order
# pylint: disable=unidiomatic-typecheck
# pyright: reportUnusedImport=false
# pyright: reportArgumentType=false, reportUnknownArgumentType=false


# MUST be first real import: sets env vars / limits before numpy, autogluon, flaml, etc.
import tabular.env_setup

# Python imports
import json
import math
from pathlib import Path
from typing import Any

# Project Imports
import tabular.predict as predict
import tabular.utilities as util
from tabular.utilities import AppError, Option

# Custom project imports
# from tests.forest_cover import forest_cover
from tests.heart_disease import heart_disease

# from tests.spy import spy
from tests.house_prices import house_prices
from tests.spaceship_titanic import spaceship_titanic
from tests.titanic import titanic

_DEFAULT_TOLERANCE = 0.0025

_IGNORES: list[dict[str, set[str]]] = [
    # SPEED == 0
    {
        "customer_churn": {
            "feature_importances",
            "segments",
            "validation_stability",
            "y_predictions",
        },
        "kirinyaga": {
            "feature_importances",
            "validation_stability",
            "y_predictions",
        },
        "minimal_multiclass_text": {
            "baseline_comparison",
            "feature_importances",
            "segments",
            "validation_stability",
            "y_predictions",
        },
        "olympiad": {
            "segments",
            "validation_stability",
            "y_predictions",
        },
        "sigaida": {
            "feature_importances",
            "validation_stability",
            "y_predictions",
        },
        "statmod": {
            "feature_importances",
        },
        "valuation_prediction": {
            "y_predictions",
        },
    },
    # SPEED == 1
    {
        "kirinyaga": {
            "feature_importances",
            "validation_stability",
            "y_predictions",
        },
        "minimal_multiclass_text": {
            "baseline_comparison",
            "feature_importances",
            "validation_stability",
            "y_predictions",
        },
        "olympiad": {
            "baseline_comparison",
            "feature_importances",
            "segments",
            "validation_stability",
            "y_predictions",
        },
        "spaceship_titanic": {
            "segments",
            "y_predictions",
        },
        "statmod": {
            "y_predictions",
        },
        "valuation_prediction": {
            "validation_stability",
            "y_predictions",
        },
    },
    # SPEED == 2
    {},
]


# Score ranges by user_id, indexed by options[Option.SPEED]
_TOLERANCES: dict[str, tuple[dict[str, float], dict[str, float], dict[str, float]]] = {
    "ai_wweek": ({}, {}, {}),
    "cirrhosis": ({}, {}, {}),
    "customer_churn": (
        {  # SPEED == 0
            "model_loss": 0.0082,  # baseline_comparison
            "train_metric_stars": 0.0244,
        },
        {},
        {},
    ),
    "heart_disease": ({}, {}, {}),
    "house_prices": ({}, {}, {}),
    "kirinyaga": (
        {},
        {  # SPEED == 1
            "model_loss": 0.0029,  # baseline_comparison
            "train_metric_stars": 0.0205,
        },
        {},
    ),
    "kyle_rhombus": ({}, {}, {}),
    "minimal_multiclass_text": (
        {},
        {  # SPEED == 1
            "metric_value": 0.0059,  # segments
            "score_penalized": 0.0569,
            "train_metric": 0.0734,
            "train_metric_stars": 0.3334,
            "validation_metric": 0.0059,
            "validation_train_ratio": 0.0636,
            "validation_train_ratio_stars": 0.0639,
        },
        {},
    ),
    "olympiad": (
        {  # SPEED == 0
            "model_loss": 0.0252,  # baseline_comparison
            "score_penalized": 0.0424,
        },
        {  # SPEED == 1
            "model_loss": 0.0133,  # baseline_comparison
            "n_samples_used": 0.0034,  # confidence_band
            "score_penalized": 0.3722,
            "train_metric": 0.0029,
            "validation_metric": 0.0036,
        },
        {},
    ),
    "ramadan": ({}, {}, {}),
    "sigaida": (
        {  # SPEED == 0
            "score_penalized": 0.0029,
        },
        {},
        {},
    ),
    "spaceship_titanic": ({}, {}, {}),
    "statmod": ({}, {}, {}),
    "titanic": ({}, {}, {}),
    "valuation_prediction": ({}, {}, {}),
    "wage_predictions": ({}, {}, {}),
}


_USERS: dict[str, Any] = {
    # AppErrors
    "apperror0_feature_mismatch_between_training_and_prediction": None,
    "apperror0_invalid_csv_file": None,
    "apperror0_prediction_file_has_target_column": None,
    "apperror0_target_column_has_missing_values": None,
    "apperror0_too_few_minority_class_samples": None,
    "apperror0_too_few_per_class_for_folds": None,
    "apperror0_too_few_rows_classification_training_file": None,
    "apperror0_too_few_rows_regression_training_file": None,
    "apperror0_too_few_samples_per_class": None,
    "apperror0_too_few_target_column_values": None,
    "apperror0_train_prediction_column_mismatch": None,
    "apperror1_train_prediction_column_mismatch": None,
    #
    # Maximal/Minimal edge cases (just speed == 0)
    "maximal_columns": None,
    "maximal_rows": None,
    "minimal_binary": None,
    "minimal_multiclass_numbers": None,
    "minimal_regression": None,
    #
    # Classification
    "ai_wweek": None,  # balanced_accuracy (binary), kaggle 0=1/3, 1=1/3, 2=1/3
    "cirrhosis": None,  # log_loss, kaggle: 0=4/5, 1=4/5,
    "customer_churn": None,  # roc_auc (binary), kaggle 0=1614/2074, 1=1499/1913
    "heart_disease": (heart_disease.set_options,),  # accuracy (binary, default), kaggle 0=25/39
    "kyle_rhombus": None,  # pr_auc
    "minimal_multiclass_text": None,  # accuracy (multiclass, default)
    "olympiad": None,  # balanced_accuracy (multi-class), too cumbersome to submit to Kaggle
    "ramadan": None,  # roc_auc (binary), kaggle 0=3/8, 1=2/8
    "spaceship_titanic": (spaceship_titanic.set_options,),  # accuracy kaggle 0=746/2008 1=643/1991
    "titanic": (titanic.set_options,),  # accuracy, kaggle 0=8/12 1=916/12051
    #
    # Regression
    "house_prices": (house_prices.set_options,),  # rmsle, kaggle 0=2394/4082 1=2432/4101
    "kirinyaga": None,  # r2, Could not figure out how to submit to Kaggle
    "sigaida": None,  # mae (default) kaggle0 1/5
    "statmod": None,  # rmse, kaggle 0=1/3 1=1/3
    "valuation_prediction": None,  # mse kaggle0 1/4
    "wage_predictions": None,  # mse kaggle0 5/14
    #
    # Misc/Other
    # "forest_cover": (forest_cover.set_options,),
    # "spy": (spy.set_options,),
}

# These users are too large and slow, so we skip them entirely at the speed.
_USERS_SKIP: list[set[str]] = [
    set(),  # SPEED = 0
    {  # SPEED = 1
        "customer_churn",
    },
    {  # SPEED = 2
        "cirrhosis",
        "customer_churn",
        "kirinyaga",
        "sigaida",
        "statmod",
        "valuation_prediction",
        "wage_predictions",
    },
]


#################################################################################################
def _compare_dicts(
    parent: str,
    dict1: dict[str, Any],
    dict2: dict[str, Any],
    ignores: set[str],
    pct_diff_tolerances: dict[str, float],
) -> list[str]:
    """Compare two dictionaries with strict type and numeric rules.

    Rules:
      1) Keys must match exactly (order irrelevant).
      2) Non-(str|numeric) values are skipped.
      3) String values must match exactly.
      4) String vs numeric mismatches are errors.
      5) Numeric values:
         - int vs float type differences are errors.
         - If key in `ranges`, both values must be within [lo, hi].
         - Otherwise values must match within 1e-12.

    Returns:
      None if dictionaries match under the rules.
      List[str] of error messages otherwise.
    """
    errors: list[str] = []

    # ---- Step 1: key equality ----
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    if keys1 != keys2:
        missing_in_2 = sorted(keys1 - keys2)
        missing_in_1 = sorted(keys2 - keys1)
        errors.append(
            "Key mismatch. "
            f"Missing in dict2: {missing_in_2}. "
            f"Missing in dict1: {missing_in_1}."
        )

    shared_keys = sorted(keys1 & keys2)

    # ---- Value comparison ----
    for k in shared_keys:
        if k in ignores:
            continue

        v1 = dict1[k]
        v2 = dict2[k]

        if type(v1) != type(v2):
            errors.append(f"diff types {type(v1)} vs {type(v2)} at key '{k}'")
            continue

        type_12: type = type(v1)  # pyright: ignore[reportUnknownVariableType]
        if type_12 == dict:
            errors.extend(_compare_dicts(k, v1, v2, ignores, pct_diff_tolerances))
            continue
        elif type_12 == list:
            errors.extend(_compare_lists(v1, v2, k, ignores, pct_diff_tolerances))
            continue

        v1_is_str = _is_string(v1)
        v2_is_str = _is_string(v2)
        v1_is_num = _is_number(v1)
        v2_is_num = _is_number(v2)

        # Skip non-string / non-numeric values
        if not (v1_is_str or v1_is_num) or not (v2_is_str or v2_is_num):
            continue

        # ---- String handling ----
        if v1_is_str and v2_is_str:
            if v1 != v2:
                errors.append(f"{parent} at key '{k}': {v1!r} != {v2!r}")
            continue

        # ---- String vs numeric mismatch ----
        if (v1_is_str and v2_is_num) or (v1_is_num and v2_is_str):
            errors.append(f"{parent} at key '{k}': {type(v1).__name__} vs {type(v2).__name__}")
            continue

        # ---- Numeric handling ----
        if v1_is_num and v2_is_num:
            # int vs float type difference is an error
            if type(v1) is not type(v2):
                errors.append(
                    f"{parent} numeric type mismatch at key '{k}': "
                    f"{type(v1).__name__} vs {type(v2).__name__}"
                )
                continue

            f1 = float(v1)
            f2 = float(v2)

            pct_diff = _compare_pct_diff(
                f1, f2, (pct_diff_tolerances or {}).get(k, _DEFAULT_TOLERANCE)
            )
            if pct_diff is not None:
                errors.append(f"{parent} '{k}': {pct_diff} -- {f1} vs {f2}")

    return errors


def _compare_lists(
    v1: list[Any],
    v2: list[Any],
    key: str,
    anys: dict[str, Any],
    pct_diff_tolerances: dict[str, float] | None,
) -> list[str]:
    """Compare two lists under the same strict rules as dict values."""
    errors: list[str] = []

    if len(v1) != len(v2):
        errors.append(f"List length mismatch at key '{key}': {len(v1)} != {len(v2)}")
        return errors

    if not v1 and not v2:
        return errors  # both empty -> ok

    # If key is "any", skip list contents entirely (mirrors your dict-level skip).
    if key in anys:
        return errors

    tol = (pct_diff_tolerances or {}).get(key, _DEFAULT_TOLERANCE)

    for i, (a, b) in enumerate(zip(v1, v2)):
        path = f"{key}[{i}]"

        # dict elements
        if isinstance(a, dict) and isinstance(b, dict):
            errors.extend(_compare_dicts(key, a, b, anys, pct_diff_tolerances=pct_diff_tolerances))
            continue

        a_is_str = _is_string(a)
        b_is_str = _is_string(b)
        a_is_num = _is_number(a)
        b_is_num = _is_number(b)

        # String vs string
        if a_is_str and b_is_str:
            if a != b:
                errors.append(f"String mismatch at '{path}': {a!r} != {b!r}")
            continue

        # String vs numeric mismatch
        if (a_is_str and b_is_num) or (a_is_num and b_is_str):
            errors.append(f"Type mismatch at '{path}': {type(a).__name__} vs {type(b).__name__}")
            continue

        # Numeric vs numeric
        if a_is_num and b_is_num:
            # int vs float mismatch is an error (and also catches cases like int vs numpy.float64)
            if type(a) is not type(b):
                errors.append(
                    f"Numeric type mismatch at '{path}': "
                    f"{type(a).__name__} vs {type(b).__name__}"
                )
                continue

            pct_diff = _compare_pct_diff(a, b, tol)
            if pct_diff is not None:
                errors.append(f"{key} '{path}': {pct_diff} -- {a} vs {b}")
            continue

        # If you reach here, these are not (dict | str | number) in a strict way.
        # Your dict logic would "skip" non-(str|numeric); for lists, skipping can hide bugs,
        # so fail loud.
        errors.append(
            f"Unsupported list element type at '{path}': "
            f"{type(a).__name__} vs {type(b).__name__}"
        )

    return errors


def _compare_pct_diff(
    num1: float | int, num2: float | int, pct_diff_tolerance: float
) -> float | None:
    """
    Compare two numeric values by absolute percentage difference.

    The absolute percentage difference is defined as:
        abs(num1 - num2) / min(abs(num1), abs(num2))

    Args:
        num1: First numeric value.
        num2: Second numeric value.
        pct_diff_tolerance: Maximum allowed percentage difference
            (expressed as a decimal, not multiplied by 100).

    Returns:
        None if the absolute percentage difference is within tolerance.
        Otherwise, the absolute percentage difference rounded *up*
        to 4 decimal places (not multiplied by 100).

    Raises:
        AssertionError: If tolerance is negative.
        ZeroDivisionError: If both numbers are zero.
    """
    assert pct_diff_tolerance >= 0, "pct_diff_tolerance must be non-negative"

    abs1 = abs(float(num1))
    abs2 = abs(float(num2))
    denom = min(abs1, abs2)

    if denom == 0:
        if abs1 == 0 and abs2 == 0:
            return None  # identical zeros → no difference
        return float("inf")
        # raise ZeroDivisionError("Cannot compute percentage difference when one value is zero.")

    abs_pct_diff = abs(num1 - num2) / denom

    if abs_pct_diff <= pct_diff_tolerance:
        return None

    # Round *up* to 4 decimal places
    return math.ceil(abs_pct_diff * 10_000) / 10_000


def _is_number(v: Any) -> bool:
    # Explicitly exclude bool
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_string(v: Any) -> bool:
    return isinstance(v, str)


def _read_dict_from_json(path: Path) -> dict[str, Any]:
    """Read a dictionary from a JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_dict_to_json(path: Path, data: dict[str, Any]) -> None:
    """Write a dictionary to disk as JSON."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)


def _do_it() -> None:
    start_at_user_id = None  # "heart_disease"  # None or user_id to start at (for after failure)

    results: dict[str, Any] | None = None
    for speed in (0, 1):  # (0, 1, 2)
        # for user_id, parameters in _USERS.items():
        for user_id, _ in _USERS.items():
            if start_at_user_id:
                if user_id == start_at_user_id:
                    start_at_user_id = None
                else:
                    print(f"Skipping '{user_id}' until we reach '{start_at_user_id}'...")
                    continue

            # # TODO: NO!!
            # if user_id not in ("kyle_rhombus",):
            #     continue

            # Speed filtering
            if 1 <= speed and not (user_id.startswith("apperror") or user_id in _TOLERANCES):
                continue  # redundant
            if user_id in _USERS_SKIP[speed]:
                continue

            # Get options
            options: dict[str, Any] = {
                Option.DATA_DIRECTORY: f"{util.USERS_DIR}/{user_id}",
                Option.SPEED: speed,
            }

            # This works, but just confuses everything for now.  Including customizations does mot
            # really help or hurt much.  Pretty much the same.
            # if parameters:
            #     options = parameters[0](**options)

            # OK, DOIT!
            try:
                actual_app_error_type = None
                actual_app_error_msg = None
                results = predict.train_and_predict(user_id, always_train=True, **options)
                results = predict.train_and_predict(user_id, **options)
            except AppError as exc:
                actual_app_error_type = exc.error_type
                actual_app_error_msg = str(exc)

            # Check app_error_type
            expected_app_error_type = user_id[10:] if user_id.startswith("apperror") else None
            if expected_app_error_type:
                if actual_app_error_type != expected_app_error_type:
                    raise ValueError(
                        (
                            f"Incorrect app_error_type for '{user_id}': "
                            f"actual=='{actual_app_error_type}' vs "
                            f"expected=='{expected_app_error_type}'"
                        )
                    )
                print(f"OK: {actual_app_error_msg}")
                continue

            if actual_app_error_type:
                print(actual_app_error_msg)
                raise util.AppError(actual_app_error_type, actual_app_error_msg, None)

            # Check expected_results
            if speed <= 1 and user_id in _TOLERANCES:
                assert results is not None
                directory = Path(f"{options[Option.DATA_DIRECTORY]}/results/")
                directory.mkdir(exist_ok=True)
                results_path = directory / f"{speed}.json"
                if not results_path.exists():
                    _write_dict_to_json(results_path, results)
                expected_results = _read_dict_from_json(results_path)
                if results != expected_results:
                    ignores = _IGNORES[speed].get(user_id, set())
                    # ignores.add("stability_stars")  # NO!!!
                    tolerances = _TOLERANCES[user_id][speed]
                    errors = _compare_dicts("root", expected_results, results, ignores, tolerances)
                    if errors:
                        msg = f"{user_id} speed={speed} root    {sorted(errors)}\n"
                        # print(msg)
                        with open("t.test.txt", "a", encoding="utf-8") as f:
                            f.write(msg)


########################
_do_it()
