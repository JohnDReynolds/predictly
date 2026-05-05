"""xxx"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportUnknownMemberType=false

# Python imports
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Project imports
import tabular.utilities as util
from tabular.utilities import NumericImputation, Option, Processor

_AGE_BIN_NUMBERS = (0, 40, 50, 60, 70, 999)  # If Age==0, it will be missing.  Force error.
_AGE_BIN_LABELS = ("Young", "Middle", "Mature", "YoungSenior", "OldSenior")
_CGBT_BP_STAGE_NUMBERS = (0, 120, 130, 140, 1000)  # If RestingBP==0, it will be missing and error.
_CGBT_BP_STAGE_LABELS = ("normal", "elevated", "stage1", "stage2")


def set_options(**options: Any) -> dict[str, Any]:
    """xxx"""
    options[Option.CUSTOM_SYNTHETICS_FUNCTION] = synthetic_features

    processors: util.ProcessorType = {}
    options[Option.PROCESSORS] = processors

    # 0% missing.  Synthetic column.  Is an ordered ranking.
    processors["_age_bin"] = {Processor.RANKED_CATEGORIES: _AGE_BIN_LABELS}

    # 0% missing.  Synthetic column.  Is an ordered ranking.  Added 11/30/25
    processors["_cgbt_bp_stage"] = {Processor.RANKED_CATEGORIES: _CGBT_BP_STAGE_LABELS}

    # 0% missing.  Is an ordered ranking.
    # TA — Typical Angina: classic exertional chest pressure/tightness, relieved by rest or
    #       nitroglycerin.
    # ATA — Atypical Angina: angina-like but doesn’t meet all classic features
    #       (e.g., location/trigger/relief not perfectly typical).
    # NAP — Non-Anginal Pain: chest pain not suggestive of myocardial ischemia
    #       (e.g., sharp, localized, positional).
    # ASY — Asymptomatic: no chest pain symptoms (may still have ischemia or disease but without
    #       pain).
    processors["ChestPainType"] = {Processor.RANKED_CATEGORIES: ("ASY", "NAP", "ATA", "TA")}

    # "Cholesterol" has 17.6% zeros in x_train.  (df["Cholesterol"] == 0).mean() = 0.176
    # This will swap the 0.0 to NaN, which will then force it to impute the NaN by findinig an
    # ITERATIVE value.
    processors["Cholesterol"] = {
        Processor.SWAP: (0, np.nan),
        Processor.FILL_VALUE: NumericImputation.ITERATIVE,
    }

    # "RestingBP" did not have any zeros in x_train, but it surprised me with 1 zero in x_test.
    # The only reason I noticed was because the calculation for "_cgbt_bp_stage" yielded
    # missing when RestingBP == 0.  This will create a "_cholesterol_is_missing" column for the
    # zeros, and then impute iterative.
    processors["RestingBP"] = {
        Processor.SWAP: (0, np.nan),
        Processor.FILL_VALUE: NumericImputation.ITERATIVE,
    }

    # 0% missing.  Is an ordered ranking.
    processors["RestingECG"] = {Processor.RANKED_CATEGORIES: ("Normal", "ST", "LVH")}

    # 0% missing.  Is an ordered ranking.
    processors["ST_Slope"] = {Processor.RANKED_CATEGORIES: ("Up", "Flat", "Down")}

    return options


def synthetic_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds a suite of synthetic features to the training and test DataFrames for the
    "Heart Disease Prediction with Dataquest" competition.

    Broad groups of features:

    1. Age and basic risk flags
       - _age_bin              : categorical age bins (Young/Middle/Mature/YoungSenior/OldSenior)
       - _high_bp_flag         : RestingBP >= 130
       - _high_chol_flag       : Cholesterol > 200
       - _predicted_max_hr     : 220 - Age
       - _hr_deficit           : predicted_max_hr - MaxHR
       - _hr_reserve           : MaxHR - RestingBP
       - _oldpeak_high         : Oldpeak > 1.0
       - _st_slope_flat_flag   : ST_Slope == "Flat"
       - _simple_risk_score    : simple count of common risk flags

    2. Clinical-ish composite features (original _cgbt_ set)
       - _cgbt_bp_stage        : blood-pressure stage from RestingBP
       - _cgbt_bp_gap          : max(RestingBP - 120, 0)
       - _cgbt_chol_high       : Cholesterol >= 240
       - _cgbt_chol_per_decade : Cholesterol scaled by age decade (Chol / (Age/10), with
                                 safe handling for small/invalid ages)
       - _cgbt_hr_reserve_frac : MaxHR / (220 - Age), with safe denominator
       - _cgbt_chronotropic_incomp : chronotropic incompetence flag (reserve_frac < 0.8)
       - _cgbt_workload_idx    : MaxHR * ST_Slope_code (Up=1, Flat=0, Down=-1)
       - _cgbt_oldpeak_flag    : Oldpeak > 2
       - _cgbt_oldpeak_x_stslope : Oldpeak * ST_Slope_code

    3. New, small set of additional interactions
       - _bp_chol_fbs_risk_sum : combined count of high BP, high cholesterol, and FastingBS
       - _age_sex_high_risk    : 1 if (male & Age >= 55) or (female & Age >= 65), else 0
       - _st_depression_load   : Oldpeak * _simple_risk_score
       - _angina_exercise_flag : 1 if ExerciseAngina == 1 and ChestPainType in {"TA","ATA"}

    All features are computed row-wise and are leak-free with respect to cross-validation.
    """
    assert x_test is not x_train  # programmer bug

    slope_map = {"Up": 1, "Flat": 0, "Down": -1}

    for df in (x_train, x_test):
        # -----------------------------
        # 1) Age bins (ordered category)
        # -----------------------------
        df["_age_bin"] = pd.cut(df["Age"], bins=_AGE_BIN_NUMBERS, labels=_AGE_BIN_LABELS)

        # -----------------------------
        # 2) Basic risk flags
        # -----------------------------
        # Blood pressure: systolic threshold
        df["_high_bp_flag"] = (df["RestingBP"] >= 130).astype(int)  # summed below

        # Cholesterol: common "borderline high" threshold
        df["_high_chol_flag"] = (df["Cholesterol"] > 200).astype(int)  # summed below

        # Predicted max HR from simple 220 - age formula
        df["_predicted_max_hr"] = 220 - df["Age"]

        # How far below predicted max HR this patient achieved
        df["_hr_deficit"] = df["_predicted_max_hr"] - df["MaxHR"]

        # Heart-rate “reserve” using RestingBP as a crude baseline
        df["_hr_reserve"] = df["MaxHR"] - df["RestingBP"]

        # ST-segment depression severity
        df["_oldpeak_high"] = (df["Oldpeak"] > 1.0).astype(int)  # summed below

        # ST slope: “Flat” often higher risk vs Up
        df["_st_slope_flat_flag"] = (df["ST_Slope"] == "Flat").astype(int)  # summed below

        # Simple composite risk score (small integer 0–6-ish)
        df["_simple_risk_score"] = df[
            [
                "FastingBS",
                "_high_bp_flag",
                "_high_chol_flag",
                "ExerciseAngina",
                "_oldpeak_high",
                "_st_slope_flat_flag",
            ]
        ].sum(axis=1)

        # --------------------------------------
        # 3) ChatGPT columns
        # --------------------------------------
        # Blood pressure stage (normal/elevated/stage1/stage2)
        df["_cgbt_bp_stage"] = pd.cut(
            df["RestingBP"],
            bins=_CGBT_BP_STAGE_NUMBERS,
            labels=_CGBT_BP_STAGE_LABELS,
        )

        # Gap above 120 mmHg (0 if <= 120)
        df["_cgbt_bp_gap"] = np.clip(df["RestingBP"] - 120, a_min=0, a_max=None)

        # Cholesterol high flag
        df["_cgbt_chol_high"] = df["Cholesterol"] >= 240

        # Cholesterol scaled by age decade; guard against tiny/invalid ages
        age_safe = df["Age"].clip(lower=1)
        df["_cgbt_chol_per_decade"] = df["Cholesterol"] / (age_safe / 10.0)

        # Heart-rate reserve fraction vs predicted; guard denominator
        hr_denom = (220 - age_safe).clip(lower=1)
        df["_cgbt_hr_reserve_frac"] = df["MaxHR"] / hr_denom

        # Chronotropic incompetence (can the heart raise rate sufficiently with age?)
        df["_cgbt_chronotropic_incomp"] = df["_cgbt_hr_reserve_frac"] < 0.8

        # Workload index: higher when patient hits high HR with uphill slope
        df["_cgbt_workload_idx"] = df["MaxHR"] * df["ST_Slope"].map(slope_map)

        # Oldpeak severity flags and interaction with ST slope
        df["_cgbt_oldpeak_flag"] = (df["Oldpeak"] > 2).astype(int)
        df["_cgbt_oldpeak_x_stslope"] = df["Oldpeak"] * df["ST_Slope"].map(slope_map)

        # # Combined metabolic risk: BP + Cholesterol + fasting blood sugar
        df["_cgbt_bp_chol_fbs_risk_sum"] = df[
            ["_high_bp_flag", "_high_chol_flag", "FastingBS"]
        ].sum(axis=1)

        # Age–sex high-risk indicator:
        #   - men >= 55
        #   - women >= 65

        # Sex is already mapped to ints via BOOLEAN_COLUMNS_TO_INT ("Sex;fF;mM")
        #   so Sex == 1 -> male, Sex == 0 -> female.
        is_male = df["Sex"] == 1
        is_female = ~is_male
        df["_cgbt_age_sex_high_risk"] = (
            (is_male & (df["Age"] >= 55)) | (is_female & (df["Age"] >= 65))
        ).astype(int)

        # ST depression weighted by overall simple risk score
        df["_cgbt_st_depression_load"] = df["Oldpeak"] * df["_simple_risk_score"]

        # Angina with exercise and “typical/atypical” chest pain
        df["_cgbt_angina_exercise_flag"] = (
            (df["ExerciseAngina"] == 1) & df["ChestPainType"].isin(["TA", "ATA"])
        ).astype(int)

    return x_train, x_test
