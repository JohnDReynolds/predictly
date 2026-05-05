"""
Module: utilities.py
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pylint: disable=broad-exception-caught
# pylint: disable=too-many-lines
# pyright: reportConstantRedefinition=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownLambdaType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

# Python imports
from __future__ import annotations  # Needed for pandas type hints
import contextlib
import csv
from datetime import datetime
from enum import Enum, auto, StrEnum
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
from typing import (
    Any,
    Callable,
    cast,
    get_args,
    Iterator,
    Literal,
    NamedTuple,
    Set,
    TextIO,
    TYPE_CHECKING,
    TypeAlias,
    TypedDict,
    Union,
)

# Third-Party imports.  Do NOT import any ML libraries here.  Needs to be lean.
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
import pandas as pd
from pandas import Series
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import psutil

# Constants General
DEFAULT_SPEED = 0  # 0=fast, 1=medium, 2=slow
DO_DEBUG = False
DO_MODAL = True  # Set to True before building Modal
ENCODING = "utf-8"
_DEVNULL: TextIO = open(os.devnull, "w", encoding=ENCODING)  # intentionally never closed
FLOAT_MAX = np.sqrt(np.float32(np.finfo(np.float32).max)) - 2  # Sqrt and a tolerance of 2.
FLOAT_MIN = -FLOAT_MAX
HIGHER_IS_BETTER_METRICS = {"accuracy", "balanced_accuracy", "r2", "roc_auc", "pr_auc"}
MODEL_RESULTS_PKL = "model_results.pkl"
NUMBER_NAN = -sys.maxsize
OPTIONS_FILE_NAME = "options.json"
OVERFIT_STARTING_PCT = 0.75
RANDOM_STATE = 424242
SESSION_EXPIRED_MESSAGE = (
    "Unexpected session error.  Please refresh the page or open a new browser window."
)
STRING_PREFIX = "--"
STRING_MISSING = f"{STRING_PREFIX}m-i-s-s-i-n-g{STRING_PREFIX}"
STRING_NAN = "__N_a_N__"
TEST_FILE_NAME = "test.csv"
TRAIN_FILE_NAME = "train.csv"
TRAIN_NJOBS = 4  # Can set to 1 for easier debugging
_UID_NAME_RE = re.compile(r"(^|_)(date|id|uid|uuid|guid|key|identifier)($|_)", re.IGNORECASE)
_UID_SUFFIXES_CASE_SENSITIVE = (
    "Date",
    "DATE",
    "Id",
    "ID",
    "Uid",
    "UID",
    "Uuid",
    "UUID",
    "Guid",
    "GUID",
    "Key",
    "KEY",
    "Identifier",
    "IDENTIFIER",
)
USERS_DIR = "/data/users" if DO_MODAL else "tests"

# SPEED constants (general)
# ALWAYS_DO_FLAMLS_UNCONDITIONALLY = (False, False, True)  # Even if ratios_pct_ok == True
DO_EARLY_STOPS = ((True,), (True,), (True, False))
DO_ENSEMBLES = ((False,), (False,), (False, True))  # Adding True works but mostly underwhelming.
FEATURE_PRUNINGSS = ((1,), (1, 2), (0, 1, 2, 3))
INCLUDE_POLYNOMIAL_FEATURES = (True, True, True)  # False so you can advertise it as a paid feature
INCLUDE_MAHALANOBIS_OUTLIER = (True, True, True)  # False so you can advertise it as a paid feature
MAX_QTY_FOLDSS = (3, 5, 5)  # Use 3-folds for SPEED == 0.  2-folds yields unpredictable results.
ML_FRAMEWORKSS = ((1,), (1,), (1, 2))  # 1 == AGL, 2 == FLAML
MODEL_QTY_FOR_MEDIAN_PREDICTION_TO_AVERAGES = [1, 1, 3]  # Need 1 if showing feature importances.
OVERFIT_TRIES_MAXIMUMS = (1, 3, 5)  # Includes 0.0 no overfitting
PRUNE_FEATURES_BY_IMPORTANCE_THREHSOLDS = (False, True, True)  # CPU-Intensive, so skip for SPEED=0
RANDOM_SEEDSSS = (((7,), (11,)), ((7, 11), (77, 22)), ((7, 11), (77, 22)))
_RATIO_RANGESSS: tuple[  # Ratio Ranges from ChatGPT indexed by [QTY_FOLDS][SPEED]
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...
] = (
    ((1.0, -1.0), (1.0, -1.0), (1.0, -1.0)),  # 0
    ((0.88, 1.18), (0.93, 1.13), (0.97, 1.07)),  # 1
    ((0.90, 1.16), (0.94, 1.12), (0.97, 1.06)),  # 2
    ((0.90, 1.15), (0.95, 1.10), (0.98, 1.05)),  # 3
    ((0.92, 1.16), (0.95, 1.11), (0.98, 1.06)),  # 4
    ((0.95, 1.18), (0.98, 1.08), (0.98, 1.08)),  # 5
)
RATIO_OK_PCTS = (0.01, 0.24, 0.49)
TIME_FACTORS = (1, 1, 1)  # Increased to (3, 3, 3), but it does not help.

# Modal
# Running test/maximal_columns (largest dataset) at speed 0, 600 seconds:
#  1. Importing predict moves memory from 561.6 MiB to RSS=4640.7 MiB. After that, only 4711.4 MiB
#      pdly: [in container MEM] in _train_and_predict_base() before importing predict RSS=561.6 MiB
#      pdly: [in container MEM] in _train_and_predict_base() after importing predict RSS=4640.7 MiB
#      pdly: [in train_and_predict MEM] final RSS=4711.4 MiB
#  2. Cost:
#      3 CPUs × 600 s × $0.0000110 = $0.0198
#      5 GiB ×  600 s × $0.0000025 = $0.0075
#                       Total cost = $0.0273
#  3. If you need more cpu than what was requested, your job will typically just run slower, not
#     crash.  But it might give heartbeat errors if it is pegged at requested.
#  4. Modal will allocate memory above the requested 3072 if it is available. Otherwise the job
#     will crash.
#  5. Modal will bill based on the higher of requested vs allocated (for both memory and cpu.)
MODAL_HOURS_TO_KEEP_USER_DIRECTORIES = 31
MODAL_CPUS = (12, 12)  # Normally below 10, can spike up to 22.  Might be able to bump down to 10.
MODAL_MEMORY = (32768, 32768)  # 35478 is the max observed, average 27000
MODAL_TIMEOUT_MINUTES = (14, 16)  # maximal_columns longest at 786 secs, average 300 secs

# Constant metrics.  The user can specify any of the keys.
METRICS_AGL_PRIMARY = {
    # Classification
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "log_loss": "log_loss",
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    # Regression
    "mae": "mean_absolute_error",
    "mse": "mean_squared_error",
    "r2": "r2",  # r-squared
    "rmse": "root_mean_squared_error",
    "rmsle": "root_mean_squared_error",  # log/exp
}
# Minimal aliasing so FLAML always sees a metric name it understands.
# User-facing Option.METRIC can still be any of these synonyms.
METRICS_FLAML_ALIASES: dict[str, str] = {
    # classification
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "log_loss": "log_loss",
    "pr_auc": "ap",
    "roc_auc": "roc_auc",
    # regression
    "mae": "mae",
    "mse": "mse",
    "r2": "r2",  # r-squared
    "rmse": "rmse",
    "rmsle": "rmse",  # log/exp
}
assert set(METRICS_AGL_PRIMARY) == set(METRICS_FLAML_ALIASES)  # programmer bug

# Tasks and Metrics for UI to display
METRICS_DISPLAY = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced_Accuracy",
    "log_loss": "Log_Loss",
    "mae": "MAE",
    "mse": "MSE",
    "pr_auc": "PR_AUC",
    "r2": "R2",
    "roc_auc": "ROC_AUC",
    "rmse": "RMSE",
    "rmsle": "RMSLE",
}

# Type Aliases
MetricType: TypeAlias = Literal[
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "mae",
    "mse",
    "pr_auc",
    "r2",
    "rmse",
    "rmsle",
    "roc_auc",
]
TaskType: TypeAlias = Literal["classification", "regression"]
if TYPE_CHECKING:
    # Only evaluated by mypy/pyright/etc
    FloatSeriesType: TypeAlias = Series[float]
    YSeriesType: TypeAlias = Series[Union[float, str]]
else:
    # At runtime just bind to the real class
    FloatSeriesType: TypeAlias = Series
    YSeriesType: TypeAlias = Series


YPredictionsType: TypeAlias = NDArray[np.generic]
_YTransformationFunctionType: TypeAlias = Callable[[YSeriesType], YSeriesType]


class AppError(Exception):
    """User-facing application error with structured metadata.

    This exception is used for all expected errors caused by user input,
    configuration, or data issues. It optionally includes a preview of
    the test dataset to aid debugging in the UI.

    Args:
        message: Human-readable error message.
        error_type: Stable machine-readable error code.
        x_test: Optional test DataFrame used to generate a preview.
    """

    def __init__(self, message: str, error_type: str, x_test: pd.DataFrame | None):
        super().__init__(message)
        self.error_type = error_type
        if x_test is not None:
            self.data, self.data_description = _get_preview(x_test, "Prediction FIle")
        else:
            self.data = self.data_description = None


class FoldResult(NamedTuple):
    """The results of a fold."""

    fold: int
    train_metric: float
    val_metric: float


class MetricInfo(TypedDict):
    """Metadata describing a metric.

    Attributes:
        requires_proba: Whether the metric requires predicted probabilities
            instead of class labels (e.g., log_loss, roc_auc).
    """

    requires_proba: bool


class MLFramework(Enum):
    """AutoGLuon or Flaml"""

    AUTOGLUON = 1
    FLAML = 2


class ModelResult(NamedTuple):
    """The results of a model."""

    # Training parameters
    run_id: str
    random_seed: int
    feature_pruning_threshold: int
    discourage_overfitting: int
    do_ensemble: bool
    do_early_stop: bool | None
    # Metrics
    raw_train_metric: float
    raw_val_metric: float
    raw_ratio_metric: float
    cv_train_metric: float
    cv_val_metric: float
    cv_ratio_metric: float
    cv_score_penalized: float
    # Other
    feature_importances: pd.DataFrame | None
    fold_results: list[FoldResult]
    model: str
    task_index: int
    ml_framework: MLFramework
    x_train: pd.DataFrame | None
    y_train: YSeriesType
    train_stars: float
    val_stars: float
    robustness_score: float | None
    robustness_stars: float | None
    validation_stability: dict[str, Any] | None
    baseline_comparison: dict[str, Any] | None
    segmented_performance: dict[str, Any] | None
    oof_predictions: npt.NDArray[Any] | pd.Series | None
    oof_pred_proba: npt.NDArray[np.floating[Any]] | None
    options: dict[str, Any]

    def _candidate_id(self) -> str:
        """Construct a unique identifier for this model configuration.

        The identifier is derived from key training parameters that define a
        candidate model. This is useful for grouping or deduplicating runs.

        Returns:
            A string uniquely identifying the model configuration.
        """
        return (
            f"{self.random_seed}-{self.feature_pruning_threshold}-{self.discourage_overfitting}-"
            f"{self.do_ensemble}-{self.do_early_stop}"
        )

    @property
    def candidate_id(self) -> str:
        """Public accessor for the model's candidate identifier.

        Returns:
            A string uniquely identifying the model configuration.
        """
        return self._candidate_id()

    def display(self, train_file_path: str) -> Any:
        """Format model result for logging or display.

        Produces a tuple of key attributes suitable for CSV-style logging.

        Args:
            train_file_path: Path to the training file (used for context display).

        Returns:
            A tuple containing formatted model metadata and metrics.
        """
        return (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            train_file_path[:20],
            self.random_seed,
            self.feature_pruning_threshold,
            self.discourage_overfitting,
            self.do_ensemble,
            self.do_early_stop,
            f"{self.cv_score_penalized:.4f}",
            f"{self.cv_train_metric:.4f}",
            f"{self.cv_val_metric:.4f}",
            f"{self.cv_ratio_metric:.4f}",
            self.options[Option.METRIC],
            self.model,
        )

    @staticmethod
    def load_model_results(path: str | Path) -> list[ModelResult]:
        """Load a list of ModelResult from disk."""
        p = Path(path)
        with p.open("rb") as f:
            return cast(list[ModelResult], pickle.load(f))

    @staticmethod
    def log(line: str) -> None:
        """Write a log line to the appropriate output.

        In Modal environments, logs are printed to stdout for visibility.
        Otherwise, logs are appended to a local CSV file.

        Args:
            line: The log line to write.
        """
        if DO_MODAL:
            # print in Modal so you can see these in the log
            print_modal(line.replace("\n", ""))
        else:
            # log to local file
            with open("t.log.csv", "a", encoding=ENCODING) as f:
                f.write(f"{line}")

    @staticmethod
    def log_model_results(
        all_model_results: list[ModelResult],
        acceptable_only: bool,
        **options: Any,
    ) -> list[ModelResult]:
        """Filter and log model results.

        Optionally filters model results based on acceptable validation ratio ranges,
        then logs the selected results in CSV format.

        Args:
            all_model_results: List of all model results.
            acceptable_only: If True, filter results based on ratio thresholds.
            **options: Runtime options including SPEED and TRAIN_FILE_PATH.

        Returns:
            The filtered list of model results.
        """
        model_results: list[ModelResult] = []
        if acceptable_only:
            minimum_acceptable_quantity = min(
                len(all_model_results),
                MODEL_QTY_FOR_MEDIAN_PREDICTION_TO_AVERAGES[options[Option.SPEED]],
            )
            for min_max in reversed(_ratio_ranges(**options)):
                model_results = [
                    mr
                    for mr in all_model_results
                    if min_max[0] <= mr.cv_ratio_metric <= min_max[1]
                ]
                if minimum_acceptable_quantity <= len(model_results):
                    break
        if not model_results:
            model_results = all_model_results

        header = (
            "\n# Time,                  Directory,            Seed,Prn,OvFt,Ensmbl,ErlySt,"
            " Score,TrainMetric,ValMetric,RatioMetric,Metric,Model\n"
        )
        ModelResult.log(header)
        for mr in model_results:
            ModelResult.log(f"{mr.display(options[Option.TRAIN_FILE_PATH])}\n")

        return model_results

    def validate(self, n_splits: int) -> None:
        """Validate internal consistency of model results.

        Performs defensive checks to ensure:
            - Fold count matches expected splits.
            - Metrics are finite.
            - Ratio metrics are consistent with raw values.

        Args:
            n_splits: Expected number of cross-validation folds.

        Raises:
            AssertionError: If any validation condition fails.
        """
        assert (
            len(self.fold_results) == n_splits
        ), "fold_results length must equal n_splits"  # programmer bug

        # numbers must be finite
        for v in (
            self.raw_train_metric,
            self.raw_val_metric,
            self.raw_ratio_metric,
            self.cv_train_metric,
            self.cv_val_metric,
            self.cv_ratio_metric,
            self.cv_score_penalized,
        ):
            assert (v is None) or (
                isinstance(v, (int, float)) and math.isfinite(v)
            ), "non-finite metric"  # programmer/model bug

        # per-fold metrics must be finite
        for fr in self.fold_results:
            assert math.isfinite(fr.train_metric) and math.isfinite(
                fr.val_metric
            ), "non-finite fold metric"  # programmer/model bug

        # ratio consistency if both are positive
        if self.raw_train_metric > 0:
            r = self.raw_val_metric / self.raw_train_metric
            # allow tiny numeric drift
            assert abs(r - self.raw_ratio_metric) <= 1e-6 * max(
                1.0, r
            ), "raw_ratio_metric mismatch"  # programmer bug


class NumericImputation(Enum):
    """Determines which numeric imputation strategy will be used in pre-processing."""

    ITERATIVE = auto()
    MEDIAN = auto()
    MINUS_1 = auto()
    ZERO = auto()


class ObjectImputation(Enum):
    """Determines which object imputation strategy will be used in pre-processing."""

    MISSING = auto()
    MOST_FREQUENT = auto()


class Option(StrEnum):
    """Enumeration of all configurable runtime options.

    These options control data paths, model behavior, preprocessing,
    and training configuration throughout the pipeline.
    """

    CUSTOM_SYNTHETICS_FUNCTION = "custom_synthetics_function"  # function pointer
    DATA_DIRECTORY = "data_directory"
    FEATURE_NAMES_TO_EXCLUDE = "feature_names_to_exclude"  # list[str]
    METRIC = "metric"  # str
    PROCESSORS = "processors"  # dict[str, Processor]
    QTY_FOLDS = "qty_folds"  # int
    RUN_ID = "run_id"  # str
    SPEED = "speed"  # int
    SUBMIT_TRUE_FALSE = "submit_true_false"  # bool
    SUBMISSION_FILE_PATH = "submission_file_path"  # str
    TASK = "task"  # str
    TEST_FILE_PATH = "test_file_path"  # str
    TRAIN_FILE_PATH = "train_file_path"  # str
    UID_COLUMN_NAME = "uid_column_name"  # str
    X_TRAIN_STARTING_ROW_INDEX = "x_train_starting_row_index"  # int
    X_TRAIN_ENDING_ROW_INDEX = "x_train_ending_row_index"  # int
    Y_COLUMN_NAME = "y_column_name"  # str
    Y_TRANSFORMATION_FUNCTION_POST = "y_transformation_function_post"
    Y_TRANSFORMATION_FUNCTION_PRE = "y_transformation_function_pre"


class Processor(Enum):
    """Enumeration of preprocessing operations applied to columns.

    Values:
        RANKED_CATEGORIES: Encode categories based on ranking.
        FILL_VALUE: Fill missing values with a specified value.
        SWAP: Replace values based on user-defined mappings.
    """

    RANKED_CATEGORIES = auto()
    FILL_VALUE = auto()
    SWAP = auto()


ProcessorType: TypeAlias = dict[str, dict[Processor, Any]]


def artifacts_directory(
    data_directory: str,
    run_id: str,
    random_seed: int,
    feature_pruning_threshold: int,
    discourage_overfitting: int,
    do_ensemble: bool,
    do_early_stop: bool | None,
) -> Path:
    """Construct the directory path for storing model artifacts.

    The path encodes key training parameters to ensure uniqueness.

    Args:
        data_directory: Base data directory.
        run_id: Unique identifier for the training run.
        random_seed: Random seed used for training.
        feature_pruning_threshold: Feature pruning level.
        discourage_overfitting: Overfitting control level.
        do_ensemble: Whether ensembling is enabled.
        do_early_stop: Whether early stopping is enabled.

    Returns:
        Path to the artifacts directory.
    """
    return Path(
        f"{data_directory}/artifacts/{run_id}/{do_ensemble}_{discourage_overfitting}_"
        f"{random_seed}_{feature_pruning_threshold}_{do_early_stop}"
    )


def artifacts_directory_mr(mr: ModelResult) -> Path:
    """Construct the artifacts directory path from a ModelResult.

    Args:
        mr: ModelResult instance.

    Returns:
        Path to the corresponding artifacts directory.
    """
    return artifacts_directory(
        mr.options[Option.DATA_DIRECTORY],
        mr.run_id,
        mr.random_seed,
        mr.feature_pruning_threshold,
        mr.discourage_overfitting,
        mr.do_ensemble,
        mr.do_early_stop,
    )


def column_names_by_dtype(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame | None = None,
    exclude_constant_cols: bool = False,
) -> tuple[list[str], list[str]]:
    """Return column names grouped by dtype.

    Separates columns into numeric and non-numeric (object/string) types.
    Optionally excludes constant columns.

    Args:
        x_train: Training DataFrame.
        x_test: Optional test DataFrame for schema validation.
        exclude_constant_cols: Whether to exclude constant columns.

    Returns:
        A tuple of (numeric_column_names, object_column_names).

    Raises:
        AssertionError:
            - If exclude_constant_cols=True and x_test is x_train (leakage guard)
        AppError:
            - Propagated from schema validation
    """
    train_numbers, train_objects = _schema_validation(x_train, x_test)

    # Exclude all constant columns (including numbers)
    columns_to_exclude: set[str] = set()
    if exclude_constant_cols:
        assert x_test is not x_train  # (otherwise leakage)
        columns_to_exclude = get_constant_columns(x_train)

    train_numbers = [x for x in train_numbers if x not in columns_to_exclude]
    train_objects = [x for x in train_objects if x not in columns_to_exclude]

    return train_numbers, train_objects


def _convert_binary_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert binary yes/no style columns to integer 1/0.

    This function scans all object/string columns in the dataframe. If a column
    contains only recognized binary tokens (case-insensitive), it is converted
    to integers.

    The original dataframe is not modified; a copy is returned.

    Args:
        df: Input pandas DataFrame.

    Returns:
        A new DataFrame where qualifying binary text columns are converted
        to integer columns with values {1, 0, -1}, where -1 represents missing.
    """
    df_out: pd.DataFrame = df.copy()

    binary_mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "t": 1,
        "male": 1,
        "m": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "f": 0,
        "female": 0,
    }

    null_tokens = {"na", "n/a", "nan", "none", "null"}

    for col in df_out.columns:
        series = df_out[col]

        if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
            continue

        normalized_series = series.astype(str).str.strip().str.lower()

        # Treat textual null tokens as real missing values.
        series = series.mask(normalized_series.isin(null_tokens))

        non_null = series.dropna()
        if non_null.empty:
            continue

        normalized = non_null.astype(str).str.strip().str.lower()
        unique_vals = set(normalized.unique())

        if unique_vals.issubset(binary_mapping):
            df_out[col] = (
                series.astype(str)
                .str.strip()
                .str.lower()
                .map(binary_mapping)
                .astype("Int8")
                .fillna(-1)
            )

    return df_out


def _determine_qty_folds_clamped(
    x_train: pd.DataFrame, x_test: pd.DataFrame, **options: Any
) -> int:
    """
    Determine a fold count for training with clamping semantics.

    Semantics:
        - Returns an int in [2, 5].
        - 2..5 means true K-fold CV / OOF is feasible under conservative constraints.
          (Important: 1-fold is *not* cross-validation; it's a policy representation.)

    Conservative, fold-dependent feasibility constraints (same math as your validator):
        - Require n_rows >= 100 * folds
        - Classification only:
            * binary:    minority_count >= 10 * folds
            * multiclass smallest_class >= 4 * folds

    Hard invariants (raise AppError):
        - Target column exists.
        - At least one feature column exists (n_cols >= 1).
        - For classification:
            * target has at least one non-null value
            * target has at least 2 classes
    """
    # Hard cap: we never return more than this.
    max_allowed_folds = MAX_QTY_FOLDSS[options[Option.SPEED]]

    # --- Invariants: we must have a real target column and at least one feature column. ---
    y_column_name = options.get(Option.Y_COLUMN_NAME)

    # Excluding the target, we must have at least 1 feature column to define an input space.
    if x_train.shape[1] == 1:
        raise ValueError("The training file must contain at least one feature column.")

    # --- Fold feasibility bounds: convert each constraint into an upper bound on folds. ---
    n_rows: int = len(x_train)

    # Row feasibility: n_rows >= 100 * folds  =>  folds <= floor(n_rows / 100)
    max_by_rows: int = n_rows // 100

    # Default class-based bound: for non-classification, there is no class constraint.
    max_by_class: int = max_allowed_folds

    if options[Option.TASK] == "classification":  # task == "classification":
        # For classification, we compute the class-count bound.  Npte that we have already
        # determined that y_column_name has no missing values.
        y = x_train[y_column_name]
        class_counts = y.value_counts()
        n_classes: int = int(class_counts.shape[0])
        if n_classes < 2:
            raise AppError(
                (
                    f"Please ensure that the Target Column '{y_column_name}' in the Training File "
                    " contains at least 2 distinct values."
                ),
                "too_few_target_column_values",
                x_test,
            )

        # Binary feasibility:    minority_count >= 10 * folds  => folds <= floor(minority/10)
        # Multiclass feasibility: smallest_class >= 4 * folds  => folds <= floor(min/4)
        if n_classes == 2:
            minority_count = int(class_counts.min())
            max_by_class = minority_count // 10
        else:
            smallest_class_count = int(class_counts.min())
            max_by_class = smallest_class_count // 4

    # --- Combine bounds, then clamp to [1, max_allowed_folds] with a single return path. ---
    # Any violated constraint simply reduces the feasible fold count; if none are feasible,
    # we fall back to 2.  # 1 ("no CV") rather than returning 0.
    max_feasible_folds = min(max_allowed_folds, max_by_rows, max_by_class)
    clamped_folds = max(2, int(max_feasible_folds))
    return clamped_folds


def get_constant_columns(df: pd.DataFrame) -> set[str]:
    """Return columns that are constant.

    A column is considered constant if all values are identical
    (including NaN-only columns).

    Args:
        df: Input DataFrame.

    Returns:
        Set of column names with a single unique value.
    """
    return {col for col in df.columns if df[col].nunique(dropna=False) == 1}


def _get_preview(df: pd.DataFrame, file_description: str) -> tuple[str, str]:
    """Generate a JSON preview of a DataFrame.

    Returns up to the first 50 rows as JSON along with a descriptive label.

    Args:
        df: Input DataFrame.
        file_description: Description of the data source.

    Returns:
        A tuple of (json_data, description).
    """
    data = df[:50].to_json(orient="records")
    qualifier = "" if len(df) <= 50 else f" first 50 (out of {len(df)}) lines of"
    return data, f"Preview of{qualifier} {file_description}"


def get_xtrain_json(**options: Any) -> tuple[str, str]:
    """Load and preview the training dataset.

    Reads the training CSV file and returns a JSON preview.

    Args:
        **options: Runtime options containing DATA_DIRECTORY.

    Returns:
        A tuple of (json_data, description).

    Raises:
        ValueError:
            - If training file cannot be read
        AppError:
            - Propagated from CSV validation
    """
    # Make sure you can read the file.
    x_train, _ = read_xy(os.path.join(options[Option.DATA_DIRECTORY], TRAIN_FILE_NAME))
    # Return data, data_description # , data_health
    return _get_preview(x_train, "Training File")


def infer_and_validate_options(
    ready_to_train: bool,
    **options: Any,
) -> tuple[dict[str, Any], dict[str, list[str]], list[str], str, str]:
    """Infer, normalize, and validate all runtime options.

    This is the central entry point for preparing a training run. It:
        1. Resolves file paths
        2. Loads persisted options (options.json)
        3. Infers task and metric from data
        4. Validates schema and data consistency
        5. Determines cross-validation folds
        6. Prepares UI-facing metadata

    Error handling strategy:
        - AssertionError: programmer bugs
        - RuntimeError: unexpected system failures
        - AppError: user/data issues (preferred at this stage)

    Args:
        ready_to_train: Whether full validation should be enforced.
        **options: Raw input options.

    Returns:
        A tuple containing:
            - normalized options dict
            - valid metrics per task (for UI)
            - list of common unique columns (UID candidates)
            - JSON preview of test data
            - test data description

    Raises:
        AppError:
            - prediction_file_has_target_column
            - train_prediction_column_mismatch
            - target_column_has_missing_values
            - feature_mismatch_between_training_and_prediction
            - Propagated from downstream validation (data, folds, etc.)
        ValueError:
            - Unknown option
            - Invalid option type
            - Unknown metric or task
    """
    # -------------------------------------------------------------------------
    # 1. Hard-code the directories and paths.
    # -------------------------------------------------------------------------
    data_directory: str = options[Option.DATA_DIRECTORY]
    options[Option.TRAIN_FILE_PATH] = os.path.join(data_directory, TRAIN_FILE_NAME)
    options[Option.TEST_FILE_PATH] = os.path.join(data_directory, TEST_FILE_NAME)
    options[Option.SUBMISSION_FILE_PATH] = os.path.join(data_directory, "submission")

    # -------------------------------------------------------------------------
    # 2. Append/overwrite options with the ones stored in options.json.
    # -------------------------------------------------------------------------
    options_path = os.path.join(data_directory, OPTIONS_FILE_NAME)
    if os.path.isfile(options_path):
        with open(options_path, encoding=ENCODING) as f:
            options.update(json.load(f))
            # Empty strings can sometimes get in here from the UI (i.e. UID_COLUMN_NAME).  An empty
            # string really means None.  So delete them completely.
            options = {k: v for k, v in options.items() if v != ""}
    # Default SPEED
    if Option.SPEED not in options:
        options[Option.SPEED] = DEFAULT_SPEED
    # Modal can send upper-case and mixed-case, so normalize to lower-case.
    for option in (Option.METRIC, Option.TASK):
        if option in options:
            options[option] = options[option].lower()

    # -------------------------------------------------------------------------
    # 3. Validate options dtypes.
    # -------------------------------------------------------------------------
    _validate_options(None, **options)

    # -------------------------------------------------------------------------
    # 4. Read x_train and x_test just plain with no options.  Then infer and validate.
    # -------------------------------------------------------------------------
    x_train, _ = read_xy(options[Option.TRAIN_FILE_PATH])
    x_test, _ = read_xy(options[Option.TEST_FILE_PATH])

    # Determine if both files probably have column headers.
    if _probably_has_header(options[Option.TRAIN_FILE_PATH]) and _probably_has_header(
        options[Option.TEST_FILE_PATH]
    ):
        missing_headers = ""
    else:
        missing_headers = "\n\nAre you possibly missing column headers?"

    # Validate train/test column relationships.
    test_train_diff_column_names = set(x_test.columns) - set(x_train.columns)
    train_test_diff_column_names = set(x_train.columns) - set(x_test.columns)
    if len(test_train_diff_column_names) == 0 and len(train_test_diff_column_names) == 0:
        tt_diff_msg = (
            "The Prediction File cannot contain the Target Column.  Please remove the Target "
            f"Column from the Prediction File.{missing_headers}"
        )
        raise AppError(tt_diff_msg, "prediction_file_has_target_column", x_test)
    else:
        if 0 < len(test_train_diff_column_names):
            tt_diff_msg = (
                "The Prediction File has extra feature column(s) that are not in the Training "
                "File.  Please remove the following extra feature column(s) from the Prediction "
                f"File:\n{sorted(test_train_diff_column_names)}"
            )
        else:
            tt_diff_msg = ""
        if 1 < len(train_test_diff_column_names):
            if tt_diff_msg:
                tt_diff_msg += "\n\n"
            tt_diff_msg += (
                "The Prediction File is missing feature column(s) that are in the Training File.  "
                "Please add the following missing feature column(s) to the Prediction File:\n"
                f"{sorted(train_test_diff_column_names)}"
            )
        if tt_diff_msg:
            tt_diff_msg += missing_headers
            raise AppError(tt_diff_msg, "train_prediction_column_mismatch", x_test)

    # Get the target y_column_name.
    y_column_name = next(iter(train_test_diff_column_names))
    options[Option.Y_COLUMN_NAME] = y_column_name

    # Make sure that y_column_name has no missing values
    if x_train[y_column_name].isna().any():
        raise AppError(
            (
                f"Please ensure that the Target Column '{y_column_name}' in the Training File "
                f"does not have any missing values.{missing_headers}"
            ),
            "target_column_has_missing_values",
            x_test,
        )

    # Now validate that all of the features have the same data types.
    _, _ = _schema_validation(x_train, x_test, y_column_name)

    # 4b) Infer Option.METRIC and Option.TASK if missing.  Then validate.
    possible_tasks, task, default_metric = _infer_task_and_metric(
        x_train[y_column_name]
    )  # , y_column_name
    if not options.get(Option.TASK):
        options[Option.TASK] = task
    else:
        task = options[Option.TASK]

    num_unique = x_train[y_column_name].nunique(dropna=True)
    if options.get(Option.METRIC, "") not in _valid_metrics(task, num_unique):
        # It does not exist or is invalid for this task.
        options[Option.METRIC] = default_metric

    if options[Option.METRIC] == "rmsle":
        options[Option.Y_TRANSFORMATION_FUNCTION_PRE] = np.log1p
        options[Option.Y_TRANSFORMATION_FUNCTION_POST] = np.exp

    # Determine the quantity of folds based on SPEED, TASK, and Y_COLUMN_NAME.
    options[Option.QTY_FOLDS] = _determine_qty_folds_clamped(x_train, x_test, **options)
    _validate_data(x_train, x_test, ready_to_train, **options)

    # _validate_rows_cols(x_train, **options) DELETE OLD

    # 4c) Now that you have inferred metric and task based on y_column_name, pop off y_column_name
    # from x_train, and then validate that x_train and x_test have the same schema.
    x_train.pop(y_column_name)
    _, _ = column_names_by_dtype(x_train, x_test)

    # -------------------------------------------------------------------------
    # 5. Perform additional inferences and validations.
    # -------------------------------------------------------------------------
    # 5a) Infer Option.UID if missing, then validate.  Note that it is not a required option.
    if options.get(Option.UID_COLUMN_NAME) == STRING_MISSING:
        # Modal is specifically sending "<None>"
        del options[Option.UID_COLUMN_NAME]
    else:
        if (
            options.get(Option.UID_COLUMN_NAME)
            and options[Option.UID_COLUMN_NAME] not in x_train.columns
        ):
            del options[Option.UID_COLUMN_NAME]
        if options.get(Option.UID_COLUMN_NAME) is None:
            if looks_like_uid(x_train.columns[0], x_train.iloc[:, 0], True):
                # Use first column as uid.
                options[Option.UID_COLUMN_NAME] = x_train.columns[0]
        uid_column_name = options.get(Option.UID_COLUMN_NAME)
        if uid_column_name is not None:
            if not x_train[uid_column_name].is_unique:
                raise ValueError(f"Unique ID '{uid_column_name}' is not unique")

    # 5b) Option.METRIC and Option.TASK.  We know that they are not None.
    if options[Option.METRIC] not in METRICS_AGL_PRIMARY:
        raise ValueError(f"Unknown Metric '{options[Option.METRIC]}'")
    if options[Option.TASK] not in get_args(TaskType):
        raise ValueError(f"Unknown Task '{options[Option.TASK]}'")

    # -------------------------------------------------------------------------
    # 6. Validate options dtypes again.  By passing in x_train it will do complete validation.
    # -------------------------------------------------------------------------
    _validate_options(x_train, **options)

    # Create a list of the common unique columns for the UI uid picker.
    if x_train.shape[1] != 1:
        train_unique_columns = x_train.columns[x_train.nunique(dropna=False) == len(x_train)]
        test_unique_columns = x_test.columns[x_test.nunique(dropna=False) == len(x_test)]
        common_unique_columns = list(train_unique_columns.intersection(test_unique_columns))
    else:
        common_unique_columns = []  # Cannot pick the only column as a UID.

    # Get the x_test json
    x_test_data, x_test_description = _get_preview(x_test, "Prediction File")

    valid_display_task_metrics: dict[str, list[str]] = {
        t.capitalize(): _valid_metrics(t, num_unique, True) for t in possible_tasks
    }

    return (
        options,
        valid_display_task_metrics,
        common_unique_columns,
        x_test_data,
        x_test_description,
    )


def _infer_task_and_metric(
    y_column: pd.Series,
    *,
    max_unique_ratio: float = 0.05,
    max_unique_count: int = 50,
) -> tuple[list[TaskType], TaskType, MetricType]:
    """
    Infer whether a target column represents a classification or regression task, and then select
    the appropriate metric.  Note that ChatGPT gave me really complicated heuristics for
    determining the appropriate task, but it always seemed to guess wrong.  ChatGPT said that
    "domain" knowledge is probably more important than heuristics for determining the metric.
    So we keep it really simple and just default to the most common metrics.

    Returns:
        (task, metric)

        task:
            "classification" or "regression"

        metric:
            Suggested evaluation metric based on task type:
                - "accuracy" for classification
                - "mae" for regression

    Heuristics:
        1. Non-numeric dtype → classification.
        2. Numeric dtype:
             - If unique values are very few relative to the sample size
               (unique_ratio <= max_unique_ratio OR num_unique <= max_unique_count),
               → treat as classification.
             - Otherwise → regression.

    Notes for possible future intelligence:
        • Classification and extremely imbalanced → consider "f1"
        • Regression with heavy-tailed distribution → consider "mae" or "huber"
        • Regression that appears very Gaussian → consider "rmse"
        • Classification with very many classes (>100) → consider "macro_f1"

    Parameters:
        y_column:
            Target column as a pandas Series.

        max_unique_ratio:
            If (num_unique / len(y_column)) <= this threshold → classification.

        max_unique_count:
            If num_unique <= this threshold → classification.
    """

    # ---- 1. Detect numeric vs non-numeric ---------------------------------
    if not pd.api.types.is_numeric_dtype(y_column):
        # Always treat non-numeric as classification.
        return ["classification"], "classification", "accuracy"

    # ---- 2. Count unique values -------------------------------------------
    num_unique = y_column.nunique(dropna=True)
    total = len(y_column)
    unique_ratio = num_unique / max(total, 1)

    # ---- 3. Low-cardinality numeric → classification -----------------------
    if num_unique <= max_unique_count or unique_ratio <= max_unique_ratio:
        # You definitely do not want to allow for a regression if num_unique == 2.  That is clearly
        # a binary classification.  I picked 10 as the lower bound somewhat arbitrarily, but it
        # seems reasonable.  Doing a regression with < 10 seems really dubious.
        return (
            ["classification"] if num_unique < 10 else ["classification", "regression"],
            "classification",
            "accuracy",
        )

    return ["regression"], "regression", "mae"


def _is_list_of_str(x: Any) -> bool:
    """Check whether an object is a list of strings.

    Args:
        x: Object to check.

    Returns:
        True if x is a list of strings; False otherwise.
    """
    return isinstance(x, list) and all(isinstance(elem, str) for elem in x)  # type: ignore


def _is_probably_csv(path: str, sample_lines: int = 10) -> bool:
    """Heuristically determine whether a file is likely a CSV.

    This function intentionally favors *false negatives* over false positives.
    It combines three lightweight checks:
      1. Explicit JSON rejection
      2. csv.Sniffer delimiter detection
      3. Structural row-consistency validation

    Notes:
        A CSV may legitimately have only 1 column. We accept 1-column CSVs only
        if the file contains at least a header row plus one data row.

    Args:
        path: Filesystem path to the file to inspect.
        sample_lines: Maximum number of rows to parse when validating structure.

    Returns:
        True if the file is probably a CSV; False otherwise.
    """
    try:
        with open(path, encoding="utf-8", newline="") as f:
            sample: str = f.read(32_768)

            # 1) Fast JSON rejection
            try:
                json.loads(sample)
                return False
            except json.JSONDecodeError:
                pass

            # 2) csv.Sniffer heuristic (delimiter detection)
            try:
                csv.Sniffer().sniff(sample)
            except csv.Error:
                # Sniffer often fails on 1-column CSVs; don't reject yet.
                pass

            # 3) Structural parse validation
            f.seek(0)
            reader = csv.reader(f)

            column_counts: Set[int] = set()
            rows_read = 0

            for _ in range(sample_lines):
                try:
                    row = next(reader)
                except StopIteration:
                    break

                rows_read += 1
                column_counts.add(len(row))

                # Early exit: inconsistent structure
                if len(column_counts) > 1:
                    return False

            if not column_counts:
                return False

            n_cols = column_counts.pop()

            # Accept:
            # - multi-column CSVs with at least 1 row (header-only still looks like CSV)
            # - 1-column CSVs only if there's at least header + 1 data row
            if n_cols >= 2:
                return True

            return n_cols == 1 and rows_read >= 2

    except Exception:
        # Any IO / decoding failure → treat as non-CSV
        return False


def log_modal_memory(msg: str = "") -> None:
    """Log total RSS for this process and all child processes in MiB."""
    proc = psutil.Process(os.getpid())
    total_rss = proc.memory_info().rss

    for child in proc.children(recursive=True):
        try:
            total_rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    total_mb = total_rss / (1024 * 1024)
    print_modal(f"MEMORY {msg} TOTAL_RSS={total_mb:.1f} MiB", DO_DEBUG)


def looks_like_uid(column_name: str, s: pd.Series, be_lenient: bool = False) -> bool:
    """
    Conservatively decide whether a column is *intended* to be an identifier (UID) and
    therefore should not be used as a predictive feature.

    Design goals:
      - Avoid false positives (e.g., keep useful numeric features like HouseSqFt even if unique).
      - Only drop when both (a) the name strongly suggests an ID, and (b) the values behave
        like an ID (no missing + all unique).

    Notes:
      - This is intentionally conservative. Many true IDs will *not* be detected unless the
        name clearly indicates ID-ness.
      - This function does not try to infer semantics from values alone (which is impossible
        in general). It uses the column name as the primary signal.
    """
    # 0) Name must strongly indicate "identifier".
    has_uid_pattern = _UID_NAME_RE.search(column_name) is not None
    has_uid_suffix = column_name.endswith(_UID_SUFFIXES_CASE_SENSITIVE)
    if not (has_uid_pattern or has_uid_suffix):
        return False

    # 1) Must not have missing values (IDs are typically complete keys).
    if s.isna().any():
        return False

    # 2) Must be unique.
    if not s.is_unique:
        return False

    # 3) Exclude types we do not want to treat as IDs by default.
    #    - floats: often measurements
    #    - bools: never IDs
    #    - datetimes: often time; require explicit handling elsewhere if desired
    if (not be_lenient) and any(
        (
            pd.api.types.is_float_dtype(s),
            pd.api.types.is_bool_dtype(s),
            pd.api.types.is_datetime64_any_dtype(s),
        )
    ):
        return False

    # 4) For string-like IDs, require "ID-ish" string characteristics to reduce false positives.
    #    Keep this permissive enough to catch UUIDs/hashes, but avoid short categorical codes.
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        s_str = s.astype(str)
        median_len = float(s_str.str.len().median())
        if median_len < 4:
            return False  # likely categorical labels ("A", "B", "C", "NY", etc.)

    # If we reached here, we have a strong name signal + unique/non-null behavior.
    return True


def print_both(message: Any) -> None:
    """Print a message to both local and Modal outputs.

    Args:
        message: Message to print.
    """
    print_local(message)
    print_modal(message)


def print_local(message: Any) -> None:
    """Print a message to local stdout only.

    Args:
        message: Message to print.
    """
    # Only print message locally.  Normally progress kind of stuff.
    if not DO_MODAL:
        print(message, flush=True)


def print_modal(message: Any, do_print: bool = True) -> None:
    """Print a message to Modal logs.

    Args:
        message: Message to print.
        do_print: Whether to actually print the message.
    """
    # Only print message for Modal.  Use pdly tag so you can filter the log.
    if DO_MODAL and do_print:
        print(f"pdly: {message}", flush=True)


def _probably_has_header(csv_path: str) -> bool:
    """Heuristically determine whether a CSV file probably has a header row.

    This function attempts to infer whether the first row of a CSV file
    represents column headers rather than data. Because CSV files are
    inherently ambiguous, this is a *probabilistic heuristic*, not a
    guaranteed determination.

    The heuristic is intentionally conservative and based on two strong signals:
      1. The first row mostly contains strings (typical of column names).
      2. The second row mostly contains numeric values (typical of data).

    This approach works well for tabular ML datasets and avoids silent
    misinterpretation of training data.

    Args:
        csv_path:
            Path to the CSV file to inspect.

    Returns:
        True if the file likely has a header row.
        False otherwise.

    Notes:
        - Only the first two rows are read for performance and safety.
        - This function does *not* modify or fully load the dataset.
        - A user-facing workflow should still allow manual override.
    """
    # Read only the first two rows without assuming headers.
    # Using header=None ensures the first row is treated as data.
    df = pd.read_csv(csv_path, nrows=2, header=None)

    # If fewer than two rows exist, we cannot reliably infer headers.
    # In this case, default to "no header" and let the user decide.
    if df.shape[0] < 2:
        return False

    first_row = df.iloc[0]
    second_row = df.iloc[1]

    # ---- Heuristic 1: First row is mostly strings -------------------------
    # Column headers are typically strings (e.g., "age", "price", "is_active").
    # We compute the fraction of values that are strings.
    first_row_string_ratio = first_row.apply(lambda v: isinstance(v, str)).mean()

    # ---- Heuristic 2: Second row is mostly numeric ------------------------
    # Data rows often contain numeric values (especially in ML datasets).
    # We attempt numeric coercion and measure how many values succeed.
    second_row_numeric_ratio = pd.to_numeric(second_row, errors="coerce").notna().mean()

    # ---- Decision thresholds ---------------------------------------------
    # These thresholds are intentionally simple and conservative.
    # They work well in practice without overfitting edge cases.
    return first_row_string_ratio >= 0.8 and second_row_numeric_ratio >= 0.2


def ratio_range(**options: Any) -> tuple[float, float]:
    """Return the acceptable ratio range for model validation.

    Args:
        **options: Runtime options containing SPEED and QTY_FOLDS.

    Returns:
        A tuple of (min_ratio, max_ratio).
    """
    return _ratio_ranges(**options)[options[Option.SPEED]]


def _ratio_ranges(
    **options: Any,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    return _RATIO_RANGESSS[options[Option.QTY_FOLDS]]


def read_xy(
    file_path: str,
    *,
    pop_y_column: bool = False,
    **options: Any,
) -> tuple[pd.DataFrame, YSeriesType]:
    """Read a CSV file and optionally extract the target column.

    Applies lightweight validation and optional preprocessing when
    runtime options are provided.

    Args:
        file_path: Path to the CSV file.
        pop_y_column: If True, remove and return the target column.
        **options: Runtime options controlling preprocessing.

    Returns:
        A tuple of:
            - DataFrame of features
            - Target Series (empty if pop_y_column=False)

    Raises:
        AppError:
            - invalid_csv_file
        ValueError:
            - File does not exist (session expired)
    """
    # Make sure the user directory exists.
    if not Path(file_path).exists():
        raise ValueError(f"{SESSION_EXPIRED_MESSAGE} (file = {file_path})")

    # Make sure it is a csv file.
    if not _is_probably_csv(file_path):
        raise AppError("Please provide a valid csv file.", "invalid_csv_file", None)

    # Read the file
    df: pd.DataFrame = pd.read_csv(file_path)

    # If options are provided, then you are really ready to train and predict.  So apply cleanup.
    if options:
        # Possibly user-excluded rows.
        x_train_starting_row_index = options.get(Option.X_TRAIN_STARTING_ROW_INDEX)
        if x_train_starting_row_index is not None:
            x_train_ending_row_index = options.get(Option.X_TRAIN_ENDING_ROW_INDEX, sys.maxsize)
            df = df[x_train_starting_row_index:x_train_ending_row_index]

        # Converts mixed-type columns to strings; leaves uniform-type columns unchanged.
        # This avoids down-stream weirdnesses.
        df = df.apply(lambda s: s.astype(str) if s.map(type).nunique() > 1 else s)  # type: ignore

        # Convert user-specified columns to integers (like Sex == M/F), no longer needed
        # df = _boolean_columns_to_int(df, options.get(Option.BOOLEAN_COLUMNS_TO_INT))

        # Convert t/f, y/n, true/false, yes/no to nullable Int8.
        df = _convert_binary_text_columns(df)

        # Apply user-specified swaps
        processors: ProcessorType = options.get(Option.PROCESSORS, {})
        for col, steps in processors.items():
            for step, swap_values in steps.items():
                if step == Processor.SWAP:
                    df[col] = df[col].replace(swap_values[0], swap_values[1])

    # Get the y column if it exists, and then validate and transform it based on the task.
    if pop_y_column:
        y: YSeriesType = cast(YSeriesType, df.pop(options[Option.Y_COLUMN_NAME]))
        task = options.get(Option.TASK)
        if task == "classification":
            if is_bool_dtype(y):  # if y.dtype == bool:
                # Only values are True/False (already validated as no-missing target)
                y = cast(YSeriesType, y.astype("int8"))  # numeric 0/1
            elif not is_numeric_dtype(y):
                y = cast(YSeriesType, y.astype(str))
        elif task == "regression":
            y = cast(YSeriesType, to_float(y))
        y_transformation_function: _YTransformationFunctionType | None = options.get(
            Option.Y_TRANSFORMATION_FUNCTION_PRE
        )
        if y_transformation_function is not None:
            y = y_transformation_function(y)
    else:
        y = cast(YSeriesType, pd.Series([], dtype=float))

    return df, y


def remove_training_results(directory_str: str) -> None:
    """Delete cached training outputs.

    Removes the serialized model results and artifacts directory
    to force a fresh training run.

    Args:
        directory_str: Base directory containing training outputs.
    """
    directory_path = Path(directory_str)
    (directory_path / MODEL_RESULTS_PKL).unlink(missing_ok=True)
    shutil.rmtree(directory_path / "artifacts", ignore_errors=True)


def safe_div(a: float, b: float) -> float:
    """Safely divide two numbers.

    Avoids division by zero by returning extreme float bounds.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        Result of division, or FLOAT_MAX / FLOAT_MIN if b == 0.

    Notes:
        Does not raise ZeroDivisionError; returns bounded values instead.
    """
    return (a / b) if b != 0 else (FLOAT_MAX if a > 0 else FLOAT_MIN)


def _schema_validation(
    x_train: pd.DataFrame, x_test: pd.DataFrame | None = None, y_column_name: str | None = None
) -> tuple[pd.Index[str], pd.Index[str]]:
    """Validate schema consistency between training and test datasets.

    Ensures that:
        - Numeric and object columns match between datasets.
        - Target column is excluded from feature validation.

    Args:
        x_train: Training DataFrame.
        x_test: Optional test DataFrame.
        y_column_name: Name of the target column to exclude.

    Returns:
        Tuple of (numeric_columns, object_columns).

    Raises:
        AppError: If schema mismatch is detected.
    """
    train_numbers = x_train.select_dtypes(include="number").columns
    train_objects = x_train.columns.difference(train_numbers)

    # Remove y_column_name since you are only vaildating the features.
    if y_column_name:
        train_numbers = train_numbers.drop(y_column_name, errors="ignore")
        train_objects = train_objects.drop(y_column_name, errors="ignore")

    # Validate that x_train and x_test have the same schema
    if x_test is not None:
        test_numbers = x_test.select_dtypes(include="number").columns
        test_objects = x_test.columns.difference(test_numbers)
        if sorted(train_numbers) != sorted(test_numbers) or sorted(train_objects) != sorted(
            test_objects
        ):
            bad_columns = (
                (set(train_numbers) - set(test_numbers))
                | (set(test_numbers) - set(train_numbers))
                | (set(train_objects) - set(test_objects))
                | (set(test_objects) - set(train_objects))
            )
            raise AppError(
                (
                    "Please ensure that the following feature columns are in both the Training "
                    "File and the Prediction File, and that they have the same data type in both "
                    "files:\n"
                    f"{sorted(bad_columns)}"
                ),
                "feature_mismatch_between_training_and_prediction",
                x_test,
            )

    return train_numbers, train_objects


@contextlib.contextmanager
def suppress_stdout_stderr() -> Iterator[None]:
    """Suppress printing to terminal.

    Uses a long-lived devnull stream to avoid 'I/O operation on closed file'
    issues with multiprocessing / logging handlers.
    """
    with contextlib.redirect_stdout(_DEVNULL), contextlib.redirect_stderr(_DEVNULL):
        yield


def to_df(xdf: Any) -> pd.DataFrame:
    """Convert input to a pandas DataFrame if needed.

    Args:
        xdf: Input object.

    Returns:
        A pandas DataFrame.
    """
    return xdf if isinstance(xdf, pd.DataFrame) else pd.DataFrame(xdf)


def to_float(y: YSeriesType) -> FloatSeriesType:
    """Ensure y is NumPy-backed float64, minimizing copies."""
    return y.astype(np.float64)


def to_pickle(path: str | Path, data: Any) -> None:
    """Serialize data to disk.

    Raises:
        OSError:
            - If the file cannot be written
    """
    p = Path(path)
    with p.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def _valid_metrics(task: TaskType, num_unique: int = -1, display_names: bool = False) -> list[str]:
    """
    Return the list of valid metrics for a given task and num_unique.
    Use lists, not sets or tuples, because they are passed in json.
    """
    if task == "classification":
        if display_names:
            valids = ["Accuracy", "Balanced_Accuracy", "Log_Loss", "ROC_AUC", "PR_AUC"]
            if num_unique != 2:
                valids.remove("ROC_AUC")
                valids.remove("PR_AUC")
        else:
            valids = ["accuracy", "balanced_accuracy", "log_loss", "roc_auc", "pr_auc"]
            if num_unique != 2:
                valids.remove("roc_auc")
                valids.remove("pr_auc")
    else:  # regression
        if display_names:
            valids = ["MAE", "MSE", "R2", "RMSE", "RMSLE"]
        else:
            valids = ["mae", "mse", "r2", "rmse", "rmsle"]

    return valids


def _validate_data(
    x_train: pd.DataFrame, x_test: pd.DataFrame, ready_to_train: bool, **options: Any
) -> None:
    """
    Validate that training/test data meet practical minimum requirements, with
    fold-aware constraints for cross-validation feasibility.

    Assumptions:
        - x_train includes the target column (Option.Y_COLUMN_NAME).
        - x_test may or may not include the target column.
        - qty_folds must be between 2 and 5 inclusive.

    Raises:
        AppError: with stable error_type codes if any requirement is violated.
    """
    qty_folds: int = options[Option.QTY_FOLDS]
    task: TaskType = options[Option.TASK]
    y_column_name = options[Option.Y_COLUMN_NAME]

    # --------------------
    # Programmer errors
    # --------------------
    if qty_folds < 2 or qty_folds > 5:
        raise ValueError(f"qty_folds '{qty_folds}' must be between 2 and 5")
    if y_column_name not in x_train.columns:
        raise ValueError(f"Training data must include the target column '{y_column_name}'")

    n_train = int(len(x_train))
    # n_test = int(len(x_test))
    y_train = x_train[y_column_name]

    # ====================
    # Regression
    # ====================
    if task == "regression":
        if n_train < 200:
            raise AppError(
                f"Your Training File has only {n_train} rows. Predictly requires at least 200 "
                "rows to produce reasonably reliable Regression validation results.",
                "too_few_rows_regression_training_file",
                x_test,
            )

    # ====================
    # Classification
    # ====================
    else:
        # ---- Global row minimum (OOF reliability, but not crazy strict)
        if n_train < 300:
            raise AppError(
                f"Your Training File has only {n_train} rows.  Predictly requires at least 300 "
                "rows to produce reasonably reliable Classification validation results.",
                "too_few_rows_classification_training_file",
                x_test,
            )

        class_counts = y_train.value_counts(dropna=False)
        n_classes = int(len(class_counts))
        min_class_count = int(class_counts.min())

        # ---- Fold-aware feasibility (this prevents sklearn hard errors)
        if ready_to_train and min_class_count < qty_folds:
            msg = (
                f"{qty_folds}-fold stratified cross-validation requires at least {qty_folds} "
                f"samples in each class of the Target Column '{y_column_name}'.  But your "
                f"smallest class in the Target Column '{y_column_name}' of your Training File has "
                f" only {min_class_count} sample(s)."
            )
            if is_numeric_dtype(y_train):
                msg = f"{msg}  Try setting your task to Regression instead of Classification."
            raise AppError(msg, "too_few_per_class_for_folds", x_test)

        # ---- Practical minimums (model quality, but not hyper-strict)
        if n_classes == 2:
            # Binary: require a reasonably sized minority class overall
            if min_class_count < 40:
                raise AppError(
                    "A Binary Classification requires at least 40 samples in each class of the "
                    f"Target Column '{y_column_name}' in order for Predictly to produce "
                    "reasonably stable metrics.  But your smallest class in the Target Column "
                    f"'{y_column_name}' of your Training File has only {min_class_count} samples.",
                    "too_few_minority_class_samples",
                    x_test,
                )
        else:
            # Multiclass: require a smaller (but still conservative) per-class minimum
            if ready_to_train and min_class_count < 30:
                raise AppError(
                    "A Multiclass Classification requires at least 30 samples in each class of "
                    f"the Target Column '{y_column_name}' in order for Predictly to produce "
                    "reasonably stable metrics.  But your smallest class in the Target Column "
                    f"'{y_column_name}' of your Training File has only {min_class_count} samples.",
                    "too_few_samples_per_class",
                    x_test,
                )


def _validate_options(x_train: pd.DataFrame | None = None, **options: Any) -> None:
    """
    Validate all of the critical options and the ones exposed to the user.
    The rest of the options are not exposed to the user, so just trust the programmer.
    if x_train is not None, then it will do full and complete validation.
    """
    # Make sure that all of the keys are valid
    for opt in options:
        if opt not in Option._value2member_map_:  # pylint: disable=protected-access
            raise ValueError(f"Unknown option '{opt}'")

    for opt in Option:
        opt_value = options.get(opt)
        if opt_value is not None:
            # Optional options
            if opt in (Option.FEATURE_NAMES_TO_EXCLUDE):
                if not _is_list_of_str(opt_value):
                    raise ValueError(f"The option '{opt.value}' must be a list of strings")
                if x_train is not None:
                    for spec in opt_value:
                        col = spec.split(";")[0]
                        if col not in x_train.columns:
                            raise ValueError(
                                f"'{opt_value}' column '{col}' is not in the training file"
                            )
            elif opt in (Option.SUBMIT_TRUE_FALSE,):
                if not isinstance(opt_value, bool):
                    raise ValueError(f"'{opt.value}' must be a boolean")
            elif opt in (
                Option.DATA_DIRECTORY,
                Option.METRIC,
                Option.SUBMISSION_FILE_PATH,
                Option.TASK,
                Option.TEST_FILE_PATH,
                Option.TRAIN_FILE_PATH,
                Option.UID_COLUMN_NAME,
                Option.Y_COLUMN_NAME,
            ):
                if not isinstance(opt_value, str) or opt_value == "":
                    raise ValueError(f"'{opt.value}' must be a non-empty string")
            elif opt in (
                Option.Y_TRANSFORMATION_FUNCTION_POST,
                Option.Y_TRANSFORMATION_FUNCTION_PRE,
            ):
                if not isinstance(opt_value, np.ufunc):
                    raise ValueError(f"'{opt.value}' must be a numpy function")
        else:  # opt_value is None
            # Required options
            if opt in (
                Option.DATA_DIRECTORY,
                Option.SPEED,
                Option.SUBMISSION_FILE_PATH,
                Option.TEST_FILE_PATH,
                Option.TRAIN_FILE_PATH,
            ):
                raise ValueError(f"'{opt.value}' is required")
            if x_train is not None:
                if opt in (
                    Option.METRIC,
                    Option.QTY_FOLDS,
                    Option.TASK,
                    Option.Y_COLUMN_NAME,
                ):
                    raise ValueError(f"'{opt.value}' is required")


def y_pre_transformation_values(y: YSeriesType, **options: Any) -> YSeriesType:
    """Reverse any pre-training transformation applied to the target.

    For example, if log1p was applied before training, this function
    applies the inverse transformation (exp and adjustment).

    Args:
        y: Transformed predictions or target values.
        **options: Runtime options containing transformation functions.

    Returns:
        Values transformed back to the original scale.
    """
    transformation_function_post = options.get(Option.Y_TRANSFORMATION_FUNCTION_POST)
    if transformation_function_post:
        y = transformation_function_post(y)
        if transformation_function_post == np.exp:
            if options.get(Option.Y_TRANSFORMATION_FUNCTION_PRE) == np.log1p:
                y = y - 1  # pyright: ignore[reportOperatorIssue]
    return y
