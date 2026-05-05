"""
Custom user functions.
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportUnknownMemberType=false

# Python imports
# import os
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd

# Project imports
import tabular.utilities as util
from tabular.utilities import Option, NumericImputation, Processor, ObjectImputation

_AGE_BIN_NUMBERS = (-1, 12, 18, 35, 60, 999)  # 0 is valid for babies
_AGE_BIN_LABELS = ("Child", "Teen", "Adult", "Mature", "Senior")
_CABIN_SPLIT_COLUMNS = ("_cabin_deck", "_cabin_number", "_cabin_side")
_EXPENDITURE_COLUMNS = ["FoodCourt", "RoomService", "ShoppingMall", "Spa", "VRDeck"]


def set_options(**options: Any) -> dict[str, Any]:
    """xxx"""
    options[Option.CUSTOM_SYNTHETICS_FUNCTION] = synthetic_features

    processors: util.ProcessorType = {}
    options[Option.PROCESSORS] = processors

    # 0% missing.  Synthetic column. Is an ordered ranking.
    processors["_age_bin"] = {Processor.RANKED_CATEGORIES: _AGE_BIN_LABELS}

    # 2% missing.  No clear most frequent.  0 is valid.
    processors["Age"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # 2% missing.  No most frequent.  Not an ordered ranking.
    processors["Cabin"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # # 2% missing.  Boolean.
    processors["CryoSleep"] = {Processor.FILL_VALUE: NumericImputation.MINUS_1}

    # Synthetic columns from splitting Cabin might have None.
    for col in _CABIN_SPLIT_COLUMNS:
        processors[col] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 2% missing.  "TRAPPIST-1e" is the most frequent, but it is only 68%, so it would be
    # presumptuous to default to it.  Not an ordered ranking.
    processors["Destination"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # The expendiure columns each have 2% missing, but no clear rhyme or reason.
    # They are most likely zero.
    for col in _EXPENDITURE_COLUMNS:
        processors[col] = {Processor.FILL_VALUE: NumericImputation.ZERO}

    # 2% missing. "Earth" is most frequent, but only at 53%.  Not an ordered ranking.
    processors["HomePlanet"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 0% missing.  No most frequent.  Not an ordered ranking.  Don't really need this, but added it
    # for testing since "Name" is a "feature_names_to_exclude" in options.json.
    processors["Name"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # # 2% missing.  Boolean.
    processors["VIP"] = {Processor.FILL_VALUE: NumericImputation.MINUS_1}

    return options


def synthetic_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add synthetic features for the Spaceship Titanic competition.

    Notes
    -----
    * New features are all prefixed with `_cgbt_` and are constructed in a leak-free way
      (no use of the target; only row-wise or feature-only group statistics).
    """
    assert x_test is not x_train  # programmer bug

    for df in (x_train, x_test):
        # ------------------------------------------------------------------
        # Existing features (kept as-is)
        # ------------------------------------------------------------------

        # Age bins (some ages are missing but will be imputed upstream).
        df["_age_bin"] = pd.cut(df["Age"], bins=_AGE_BIN_NUMBERS, labels=_AGE_BIN_LABELS)

        # Cabin breakdown (Deck / Number / Side).
        cabin_split = df["Cabin"].str.split("/", expand=True)
        df["_cabin_deck"] = cabin_split[0]
        df["_cabin_number"] = cabin_split[1]
        df["_cabin_side"] = cabin_split[2]

        # Group ID & group size (based on PassengerId prefix).
        df["_group_id"] = df["PassengerId"].str.split("_").str[0].astype(int)
        df["_group_size"] = df["_group_id"].map(df["_group_id"].value_counts())

        # Traveller type: INDIVIDUAL / GROUP / FAMILY (based on surname within group).
        df["_surname"] = df["Name"].str.split().str[-1]
        df["_surname_count"] = df.groupby(["_group_id", "_surname"])["_surname"].transform("count")
        df["_traveller_type"] = "INDIVIDUAL"
        df.loc[df["_group_size"] > 1, "_traveller_type"] = "GROUP"
        df.loc[df["_surname_count"] > 1, "_traveller_type"] = "FAMILY"

        # Total expenditure & flag for any non-zero spend.
        df["_total_expenditure"] = df[_EXPENDITURE_COLUMNS].sum(axis=1)
        df["_has_expenditure"] = df["_total_expenditure"] > 0

        # ------------------------------------------------------------------
        # New _cgbt_ features: group structure, expenditure patterns, age effects
        # ------------------------------------------------------------------
        # Group structure: singleton vs larger groups.
        df["_cgbt_is_singleton"] = (df["_group_size"] == 1).astype(int)
        df["_cgbt_is_large_group"] = (df["_group_size"] >= 3).astype(int)

        # CryoSleep & expenditure interactions:
        # - Passengers in CryoSleep should typically have no expenditure.
        # - Deviations from that pattern can be informative.
        df["_cgbt_zero_spend"] = (df["_total_expenditure"] == 0).astype(int)
        df["_cgbt_cryo_and_zero_spend"] = (
            (df["CryoSleep"].astype(bool)) & (df["_total_expenditure"] == 0)
        ).astype(int)
        df["_cgbt_awake_and_zero_spend"] = (
            (~df["CryoSleep"].astype(bool)) & (df["_total_expenditure"] == 0)
        ).astype(int)

        # Log and normalized expenditure:
        # Use log1p to soften heavy tails.
        df["_cgbt_log_total_expenditure"] = np.log1p(df["_total_expenditure"])

        # Expenditure per (approximate) adulthood factor: use age>=1 to avoid division by zero.
        age_safe = df["Age"].clip(lower=1)
        df["_cgbt_exp_per_age"] = df["_total_expenditure"] / age_safe

        # Expenditure fractions by category: if total=0, all fractions default to 0.
        total_exp = df["_total_expenditure"].replace(0, np.nan)
        for col in _EXPENDITURE_COLUMNS:
            frac_col = f"_cgbt_frac_{col.lower()}"
            df[frac_col] = df[col] / total_exp
        # Where total_expenditure was 0, everything will be NaN; set those to 0.
        frac_cols = [f"_cgbt_frac_{c.lower()}" for c in _EXPENDITURE_COLUMNS]
        df[frac_cols] = df[frac_cols].fillna(0.0)

        # Simple "leisure intensity": sum of non-essential spends (ShoppingMall + VRDeck + Spa).
        df["_cgbt_leisure_spend"] = df["ShoppingMall"] + df["VRDeck"] + df["Spa"]

        # Cabin side & deck indicators (can capture layout / hazard structure).
        df["_cgbt_is_port"] = (df["_cabin_side"] == "P").astype(int)
        df["_cgbt_is_starboard"] = (df["_cabin_side"] == "S").astype(int)

        # Deck grouping: treat unknown deck as its own category; then simple indicators.
        df["_cgbt_deck_is_unknown"] = (
            df["_cabin_deck"].isna() | (df["_cabin_deck"] == util.STRING_MISSING)
        ).astype(int)
        df["_cgbt_deck_is_top"] = df["_cabin_deck"].isin(["G", "F"]).astype(int)
        df["_cgbt_deck_is_low"] = df["_cabin_deck"].isin(["A", "B"]).astype(int)

        # Age-related flags (using already-binned _age_bin conceptually).
        df["_cgbt_is_child"] = (df["_age_bin"] == "Child").astype(int)
        df["_cgbt_is_senior"] = (df["_age_bin"] == "Senior").astype(int)

        # VIP & CryoSleep combined patterns.
        df["_cgbt_vip_and_awake"] = (
            df["VIP"].astype(bool) & (~df["CryoSleep"].astype(bool))
        ).astype(int)
        df["_cgbt_vip_and_cryo"] = (df["VIP"].astype(bool) & df["CryoSleep"].astype(bool)).astype(
            int
        )

    return x_train, x_test
