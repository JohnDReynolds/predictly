"""
Module: predict.py
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

# Python imports
from __future__ import annotations  # Needed for pandas type hints
from collections.abc import Mapping
import math
import os
from pathlib import Path
import shutil
from statistics import mean
import time
from typing import Any, Callable, cast, Iterable, TypeAlias
from uuid import uuid4

# Third-Party imports
from joblib import Parallel, delayed
import numpy as np
import numpy.typing as npt
import pandas as pd

# Project imports
from tabular.data_health import build_column_health_summary
from tabular.scoring import score_model_results
from tabular.trainers import (
    predict_model_agl_from_artifacts,
    predict_model_flaml_from_artifacts,
    train_model_agl_store_artifacts,
    train_model_flaml_store_artifacts,
)
import tabular.utilities as util
from tabular.utilities import MLFramework, ModelResult, Option, YPredictionsType, YSeriesType

# Type Aliases
_TrainingTaskType: TypeAlias = tuple[
    pd.DataFrame,
    YSeriesType,
    int,
    int,
    float,  # s_strict
    bool,  # do_ensemble
    bool | None,  # do_early_stop
    Any,  # **options
]
_YTransformationFunctionType: TypeAlias = Callable[[YSeriesType], YSeriesType]


def _anys_equal(a: Any, b: Any) -> bool:
    """Recursively compare two config-like structures.

    Treats NaN values as equal and supports Mapping and sequence containers.
    For all other types, falls back to normal equality.

    Args:
        a: First object to compare.
        b: Second object to compare.

    Returns:
        True if the two objects are recursively equal; False otherwise.
    """
    if a is b:
        return True

    # Handle floats (NaN == NaN semantics for configs)
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b

    # Handle mappings (dict-like)
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            return False
        return all(_anys_equal(a[k], b[k]) for k in a.keys())

    # Handle sequences (but not str/bytes)
    seq_types = (list, tuple)
    if isinstance(a, seq_types) and isinstance(b, seq_types):
        if len(a) != len(b):
            return False
        return all(_anys_equal(x, y) for x, y in zip(a, b))

    # Fallback: normal equality
    return a == b


def _get_y_predictions(**options: Any) -> tuple[str | None, ModelResult]:
    """Generate prediction output from stored model results.

    Loads acceptable model results from disk, reads the prediction file,
    ensembles predictions across retained models, writes a local submission
    CSV when running outside Modal, and returns JSON for the UI.

    Args:
        **options: Runtime options containing data paths, target column,
            metric, task, UID column, and submission configuration.

    Returns:
        A tuple containing:
            - JSON prediction records, or None if unavailable.
            - The first retained ModelResult used for display metrics.

    Raises:
        AppError: Propagated from CSV validation or data loading.
        ValueError: If prediction output has an unsupported shape.
        OSError: If model results or submission files cannot be read or written.
    """

    # -------------------------------------------------------------------------
    # 1. Get y_pred_df and write submissions.csv.
    # -------------------------------------------------------------------------
    util.print_local("Creating y_prediction submissions...")

    # 1 might be too much of "putting all of your eggs in one basket".
    # 3 sometimes gets 3 from the same s_strict and the same do_ensemble.
    # 5 is probably a good balance.
    # 7 is probably too many.
    model_results = ModelResult.load_model_results(
        f"{options[Option.DATA_DIRECTORY]}{os.sep}{util.MODEL_RESULTS_PKL}"
    )
    x_test, _ = util.read_xy(options[Option.TEST_FILE_PATH], **options)
    y_predictions = _median_y_predictions(x_test, model_results)
    # Do not need this anymore.  Moved to the bottom of train()
    # y_predictions = _median_y_predictions(
    #     x_test,
    #     model_results[: util.MODEL_QTY_FOR_MEDIAN_PREDICTION_TO_AVERAGES[options[Option.SPEED]]],
    # )

    # Convert np.log(y) back to original y.
    y_predictions = util.y_pre_transformation_values(y_predictions, **options)  # pyright: ignore

    # Convert 0/1 to False/True
    if options.get(Option.SUBMIT_TRUE_FALSE):
        y_predictions = y_predictions.astype(bool)

    # The file might not have UID, so add it as the 0-based index.
    uid = options.get(Option.UID_COLUMN_NAME, _new_column_name(x_test.columns, "ID"))
    if uid not in x_test.columns:
        x_test[uid] = x_test.index

    # Make the predictions
    y_pred_df = _make_prediction_df(
        uid=uid,
        x_test=x_test,
        y_pred=y_predictions,
        y_colname=options[Option.Y_COLUMN_NAME],
    )

    # Write the submission file.
    if not util.DO_MODAL:
        # submission.0.csv
        y_pred_df.to_csv(
            f"{options[Option.SUBMISSION_FILE_PATH]}.{options[Option.SPEED]}.csv", index=False
        )

    return y_pred_df.to_json(orient="records"), model_results[0]


def _make_prediction_df(
    uid: str,
    x_test: pd.DataFrame,
    y_pred: npt.ArrayLike,
    y_colname: str,
) -> pd.DataFrame:
    """Build a prediction DataFrame from model predictions.

    The output contains one UID column and either one prediction column
    for 1D predictions or multiple numbered prediction columns for 2D
    multi-output predictions.

    Args:
        uid: Name of the UID column to include in the output.
        x_test: Prediction input DataFrame containing the UID column.
        y_pred: Prediction array-like object.
        y_colname: Base name for prediction column(s).

    Returns:
        DataFrame containing UID values and predictions.

    Raises:
        ValueError: If y_pred is not 1D or 2D.
        KeyError: If uid is not present in x_test.
    """

    # Convert to ndarray (defensive, and zero-op if already an array)
    y_pred = np.asarray(y_pred)

    # Case 1: 1D predictions → shape (n_samples,)
    if y_pred.ndim == 1:
        pred_df = pd.DataFrame(
            {
                uid: x_test[uid].values,
                y_colname: y_pred,
            }
        )
        return pred_df

    # Case 2: 2D predictions → shape (n_samples, n_outputs)
    if y_pred.ndim == 2:
        n_outputs = y_pred.shape[1]
        colnames = [f"{y_colname}_{i+1}" for i in range(n_outputs)]
        pred_df = pd.DataFrame(
            y_pred,
            columns=colnames,
        )
        pred_df.insert(0, uid, x_test[uid].values)  # pyright: ignore
        return pred_df

    raise ValueError(f"Unsupported prediction array shape '{y_pred.shape}'")


def _median_y_predictions(
    x_test: pd.DataFrame, model_results: list[ModelResult]
) -> YPredictionsType:
    """Combine predictions from multiple trained models.

    Uses median aggregation for numeric predictions and probabilities.
    Uses simple majority vote for non-numeric class labels.

    Supports:
        - Regression: 1D float predictions.
        - Binary classification: 1D integer or float predictions.
        - Multiclass classification: 1D labels or 2D probabilities.

    Args:
        x_test: Prediction input DataFrame.
        model_results: Retained model results whose artifacts are used for
            prediction.

    Returns:
        NumPy array with the same shape as a single model's predictions.

    Raises:
        AssertionError: If stacked label predictions have an unexpected shape.
    """
    # Collect predictions from each model as numpy arrays
    preds_list: list[npt.NDArray[np.generic]] = [
        np.asarray(
            predict_model_agl_from_artifacts(x_test, mr)
            if mr.ml_framework == MLFramework.AUTOGLUON
            else predict_model_flaml_from_artifacts(x_test, mr)
        )
        for mr in model_results
    ]

    base = preds_list[0]
    base_dtype = base.dtype

    # ----- Case 1: integer labels (binary or multiclass), 1D -----
    if base.ndim == 1 and np.issubdtype(base_dtype, np.integer):
        stacked_int = np.stack(preds_list, axis=0)
        stacked_sorted = np.sort(stacked_int, axis=0)
        k = stacked_sorted.shape[0] // 2
        result_int = stacked_sorted[k, :]
        return result_int.astype(base_dtype, copy=False)

    # 1D non-float outputs (e.g., string labels) → majority vote per sample.
    # This covers the case where frameworks return original string class labels
    # like "Northville"/"Southtown" instead of integer codes or probabilities.
    if base.ndim == 1 and not np.issubdtype(base_dtype, np.floating):
        stacked = np.stack(preds_list, axis=0)  # (n_models, n_samples)

        if stacked.ndim != 2:
            raise AssertionError(
                "Expected 2D stacked predictions for label ensembling, "
                f"got shape {stacked.shape!r}."
            )

        _, n_samples = stacked.shape
        voted = np.empty(n_samples, dtype=base_dtype)

        for j in range(n_samples):
            col = stacked[:, j]  # labels from each model for sample j
            unique, counts = np.unique(col, return_counts=True)
            voted[j] = unique[int(np.argmax(counts))]  # simple majority vote

        return voted

    # ----- Case 2: everything else: floats, probabilities, multi-output -----
    stacked_float: npt.NDArray[np.float64] = np.stack(
        [a.astype(float, copy=False) for a in preds_list],
        axis=0,
    )
    median: npt.NDArray[np.float64] = np.median(stacked_float, axis=0)
    return median


def _need_to_train(**options: Any) -> bool:
    """Determine whether cached model results can be reused.

    Cached results are reused only when model_results.pkl exists and the
    stored options match the current options, ignoring RUN_ID.

    Args:
        **options: Current runtime options.

    Returns:
        True if training is required; False if cached results can be reused.

    Raises:
        OSError: If cached model results exist but cannot be read.
        KeyError: If required options are missing.
    """
    directory_path = Path(options[Option.DATA_DIRECTORY])
    if not (directory_path / util.MODEL_RESULTS_PKL).exists():
        return True

    # Get the first ModelResult
    mr: ModelResult = ModelResult.load_model_results(
        f"{options[Option.DATA_DIRECTORY]}{os.sep}{util.MODEL_RESULTS_PKL}"
    )[0]

    del mr.options[Option.RUN_ID]

    if not _anys_equal(options, mr.options):
        return True

    return False


def _new_column_name(old_names: Iterable[str] | list[str], new_name: str) -> str:
    """Return a non-conflicting column name.

    If new_name already exists, alternately prefixes and suffixes underscores
    until the generated name is unique.

    Args:
        old_names: Existing column names.
        new_name: Desired column name.

    Returns:
        A column name that does not appear in old_names.
    """
    before_after: bool = True
    while new_name in old_names:
        new_name = f"_{new_name}" if before_after else f"{new_name}_"
        before_after = not before_after
    return new_name


def _process_tasks(
    tasks: list[_TrainingTaskType],
    train_function: Callable[..., ModelResult],
) -> list[ModelResult]:
    """Run training tasks in parallel.

    Args:
        tasks: Training task tuples containing data, training parameters,
            and runtime options.
        train_function: Training function to execute for each task.

    Returns:
        List of ModelResult objects returned by train_function.

    Raises:
        Exception: Propagates exceptions raised by train_function or joblib.
    """
    jobs = [
        delayed(train_function)(
            x_train_use,
            y_train,
            random_seed,
            feature_pruning_threshold,
            s_strict,
            do_ensemble,
            do_early_stop,
            task_idx,
            **options,
        )
        for task_idx, (
            x_train_use,
            y_train,
            random_seed,
            feature_pruning_threshold,
            s_strict,
            do_ensemble,
            do_early_stop,
            options,
        ) in enumerate(tasks)
    ]

    # FLAML was getting the error: "OSError: [Errno 24] Too many open files". By default, joblib
    # memmaps large numpy arrays to temp files per process. Setting max_nbytes=None and
    # mmap_mode=None will disable it. If you need that again, uncomment the second call.
    return cast(
        list[ModelResult],
        Parallel(n_jobs=util.TRAIN_NJOBS)(jobs),
    )


def _train(**options: Any) -> dict[str, Any]:
    """Train and persist acceptable model results.

    Infers and validates options, skips training when reusable cached results
    exist, otherwise trains candidate AutoGluon/FLAML models, scores them,
    retains acceptable models, saves model_results.pkl, and removes unused
    artifacts.

    Args:
        **options: Runtime options. If incomplete, options are inferred and
            validated before training.

    Returns:
        Normalized runtime options.

    Raises:
        AppError: Propagated from option/data validation or CSV loading.
        ValueError: Propagated from invalid options or downstream training setup.
        OSError: If model artifacts cannot be read, written, or removed.
    """

    # Infer and validate the options.
    options, _, _, _, _ = util.infer_and_validate_options(ready_to_train=True, **options)

    # See if you need to train
    if not _need_to_train(**options):
        return options

    # Start with a clean slate.  Good housekeeping.
    util.remove_training_results(options[Option.DATA_DIRECTORY])

    # Set the run ID just for this training run.
    options[Option.RUN_ID] = uuid4().hex

    # Cache some widely-used option values with appropriate defaults.
    data_directory: str = options[Option.DATA_DIRECTORY]
    ratio_range = util.ratio_range(**options)
    speed: int = options[Option.SPEED]
    train_file_path: str = options[Option.TRAIN_FILE_PATH]

    # Read the training data.
    x_train, y_train = util.read_xy(train_file_path, pop_y_column=True, **options)

    def _run_one_s_strict(
        s_stricts: list[float],
        ml_framework: MLFramework,
        do_ensemble: bool,
        random_seed_index: int,
    ) -> tuple[list[ModelResult], int]:
        """Train candidate models for one strictness batch.

        Args:
            s_stricts: Strictness values to evaluate.
            ml_framework: Framework to train with.
            do_ensemble: Whether framework-level ensembling is enabled.
            random_seed_index: Index into the configured random-seed groups.

        Returns:
            A tuple containing scored model results and the next random-seed
            group index.
        """
        tasks: list[_TrainingTaskType] = []
        for random_seed in util.RANDOM_SEEDSSS[speed][random_seed_index]:
            for feature_pruning_threshold in util.FEATURE_PRUNINGSS[speed]:
                for s_strict in s_stricts:
                    base_task = (
                        x_train,
                        y_train,
                        random_seed,
                        feature_pruning_threshold,
                        s_strict,
                        do_ensemble,
                        None,
                        options,
                    )
                    if ml_framework == MLFramework.AUTOGLUON:
                        tasks.append(base_task)
                    elif ml_framework == MLFramework.FLAML:
                        for do_early_stop in util.DO_EARLY_STOPS[speed]:
                            tasks.append(base_task[:6] + (do_early_stop,) + base_task[7:])

        raw_results = (
            _process_tasks(tasks, train_model_agl_store_artifacts)
            if ml_framework == MLFramework.AUTOGLUON
            else _process_tasks(tasks, train_model_flaml_store_artifacts)
        )
        scored_results = score_model_results(raw_results)
        model_results.extend(scored_results)
        util.ModelResult.log_model_results(scored_results, False, **options)

        # Alternate random_seed_index between 0 and 1
        return scored_results, 1 if random_seed_index == 0 else 0

    model_results: list[ModelResult] = []
    random_seed_index = 0
    util.ModelResult.log(
        f"\n########## {train_file_path}, METRIC={options[Option.METRIC]}, SPEED={speed}"
    )

    for ml_framework_int in util.ML_FRAMEWORKSS[speed]:
        ml_framework = util.MLFramework(ml_framework_int)
        if ml_framework == MLFramework.FLAML and options[Option.METRIC] == "balanced_accuracy":
            # FLAML does not support balanced_accuracy.  You would need to write a custom metric
            # function which ChatGPT says is doable.  But not worth it for now.
            continue
        for do_ensemble in util.DO_ENSEMBLES[speed]:
            overfit_tries = 0
            # Binary search on s_strict in [0.0, 1.0]
            s_lo: float = 0.0
            s_hi: float = 1.0
            # Start at 75% since it's almost always better than 50%
            s_strict_curr = util.OVERFIT_STARTING_PCT

            ratios_pct_ok = False
            while (not ratios_pct_ok) and overfit_tries < util.OVERFIT_TRIES_MAXIMUMS[speed]:
                overfit_tries += 1
                use_s_strict: list[float] = [s_strict_curr]
                if s_strict_curr == util.OVERFIT_STARTING_PCT:
                    use_s_strict.append(0.0)  # start with 75%, 0%
                    if speed == 0:
                        use_s_strict.extend((0.375, 0.875))  # add 37.5% and 87.5%

                try_model_results, random_seed_index = _run_one_s_strict(
                    use_s_strict, ml_framework, do_ensemble, random_seed_index
                )

                # ratios_pct_ok if the pct of qualifying models > util.RATIO_OK_PCTS[speed].
                count_ok = sum(
                    [
                        ratio_range[0] <= mr.cv_ratio_metric <= ratio_range[1]
                        for mr in try_model_results
                    ]
                )
                pct_ok = count_ok / len(try_model_results)
                ratios_pct_ok = util.RATIO_OK_PCTS[speed] < pct_ok

                # Assumption: larger s_strict → more regularization → lower ratio.
                ratio_curr = float(
                    np.mean(
                        [
                            mr.cv_ratio_metric
                            for mr in try_model_results
                            if mr.discourage_overfitting != 0
                        ]
                    )
                )
                if ratio_curr > mean(ratio_range):
                    s_lo = s_strict_curr  # too overfit → increase strictness
                else:
                    s_hi = s_strict_curr  # underfit / OK → relax strictness

                s_next = 0.5 * (s_lo + s_hi)

                # Snap near the endpoints for cleanliness.
                if s_next < 0.04:
                    s_next = 0.0
                elif s_next > 0.96:
                    s_next = 1.0

                # No-progress guard: if midpoint equals current, we’re done searching.
                if s_next == s_strict_curr:
                    break

                s_strict_curr = s_next

    # Sort the model results.
    model_results = sorted(
        model_results, key=lambda mr: (mr.cv_score_penalized, mr.cv_ratio_metric)
    )

    # Log only the acceptable model results.
    acceptable_results = util.ModelResult.log_model_results(model_results, True, **options)

    # After you have logged them, only retain the ones you need later to compute the median.
    acceptable_results = acceptable_results[
        : util.MODEL_QTY_FOR_MEDIAN_PREDICTION_TO_AVERAGES[options[Option.SPEED]]
    ]

    # Save to disk
    util.to_pickle(f"{data_directory}{os.sep}{util.MODEL_RESULTS_PKL}", acceptable_results)

    # Delete unused artifacts_directories.
    for mr in model_results:
        if mr not in acceptable_results:
            shutil.rmtree(util.artifacts_directory_mr(mr))

    return options


def train_and_predict(user_id: str, always_train: bool = False, **options: Any) -> dict[str, Any]:
    """Train models as needed and return prediction results for the UI.

    This is the public entry point for the train-and-predict workflow. It
    resolves the user's data directory when options are not supplied, optionally
    forces retraining, trains or reuses cached models, generates predictions,
    computes data-health summaries, and returns JSON-serializable results.

    Args:
        user_id: User identifier used to locate the default data directory.
        always_train: Whether to remove cached results and force retraining.
        **options: Optional runtime options. If omitted, DATA_DIRECTORY is
            inferred from user_id.

    Returns:
        Dictionary containing status, display metrics, feature importances,
        validation diagnostics, predictions, and data-health summaries.

    Raises:
        AppError: If user-provided files, schema, options, or data are invalid.
        ValueError: If options are invalid or prediction shapes are unsupported.
        OSError: If files or model artifacts cannot be read, written, or removed.
    """
    start_time = time.time()
    if not options:
        # options: dict[str, Any] = {Option.DATA_DIRECTORY: os.path.join(util.USERS_DIR, user_id)}
        options[Option.DATA_DIRECTORY] = os.path.join(util.USERS_DIR, user_id)

    # Remove model_results and artifacts to force training if coming from test.py.
    if always_train:
        util.remove_training_results(options[Option.DATA_DIRECTORY])

    # Train
    options = _train(**options)
    util.log_modal_memory("after _train()")

    # Predict
    y_predictions, model_result = _get_y_predictions(**options)
    util.log_modal_memory("after _get_y_predictions()")

    # Clean up values for display.
    if options[Option.METRIC] in util.HIGHER_IS_BETTER_METRICS:
        train_metric = 1.0 - model_result.cv_train_metric
        validation_metric = 1.0 - model_result.cv_val_metric
    else:
        train_metric = model_result.cv_train_metric
        validation_metric = model_result.cv_val_metric

    # Get the data health
    df, _ = util.read_xy(options[Option.TRAIN_FILE_PATH])
    x_train_health = build_column_health_summary(df, target_column=options[Option.Y_COLUMN_NAME])
    df, _ = util.read_xy(options[Option.TEST_FILE_PATH])
    x_test_health = build_column_health_summary(df)

    util.print_both(f"Time for train_and_predict({user_id}) {time.time() - start_time}")

    return {
        # singletons
        "status": "ok",
        "display_task": options[Option.TASK].capitalize(),
        "display_metric": util.METRICS_DISPLAY[options[Option.METRIC]],
        "score_penalized": model_result.cv_score_penalized,
        "train_metric": train_metric,
        "validation_metric": validation_metric,
        "validation_train_ratio": model_result.robustness_score,  # validation_train_ratio,
        "train_metric_stars": model_result.train_stars,
        "validation_metric_stars": model_result.val_stars,
        "validation_train_ratio_stars": model_result.robustness_stars,  # model_result.stars[2],
        # dicts and dfs
        "feature_importances": (
            None
            if model_result.feature_importances is None
            else model_result.feature_importances.to_json(orient="records")
        ),
        "validation_stability": model_result.validation_stability,
        "baseline_comparison": model_result.baseline_comparison,
        "segmented_performance": model_result.segmented_performance,
        "y_predictions": y_predictions,
        "x_train_health": x_train_health.to_json(orient="records"),
        "x_test_health": x_test_health.to_json(orient="records"),
    }
