"""
Custom user functions.
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportUnknownMemberType=false

# Python imports
# import os
from typing import Any

# Third-Party imports
import numpy as np
import pandas as pd

# Project imports
import tabular.utilities as util
from tabular.utilities import NumericImputation, ObjectImputation, Option, Processor

# _AGE_BIN_NUMBERS = (-2, -1, 12, 18, 35, 60, 999)  # Missing is imputed to -1.  0 is "Child".
# _AGE_BIN_LABELS = ("Missing", "Child", "Teen", "Adult", "Mature", "Senior")
_AGE_BIN_NUMBERS = (-1, 12, 18, 35, 60, 999)  # 0 is valid: "Child".
_AGE_BIN_LABELS = ("Child", "Teen", "Adult", "Mature", "Senior")
_FAMILY_SIZE = "_family_size"
_SURNAME = "_surname"


def set_options(**options: Any) -> dict[str, Any]:
    """xxx"""
    options[Option.CUSTOM_SYNTHETICS_FUNCTION] = synthetic_features

    processors: util.ProcessorType = {}
    options[Option.PROCESSORS] = processors

    # Synthetic column.  0% missing.  No clear most frequent.
    processors["_age_bin"] = {Processor.RANKED_CATEGORIES: _AGE_BIN_LABELS}

    # 20% missing.  No most frequent.  0 is valid.
    # I tried using -1, but the implication is that they would be younger than 0.  This would
    # distort both "Age" and "_age_bin".  Although ITERATIVE might be too presumptuous, I think it
    # is the lesser of two evils and better than MEDIAN.
    processors["Age"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # 77% missing.  No most frequent.  Is not an ordered category.
    processors["Cabin"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # < 1% missing.  The most frequent value is "S" at 77%.  Not an ordered ranking.
    processors["Embarked"] = {Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT}

    # train.csv has no missings, but test has 1 missing.  train.csv has 1.6% 0 fares.  The 0 fares
    # probably don't mean that they got in free, it probably means "don't know".
    # This will create a "_fare_is_missing" column for the zeros, and then impute iterative.
    processors["Fare"] = {
        Processor.SWAP: (0, np.nan),
        Processor.FILL_VALUE: NumericImputation.ITERATIVE,
    }

    return options


def synthetic_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add common synthetic features for the Titanic competition.

    Notes
    -----
    - Uses only X (no Survived), so it is leak-free.
    - Existing features are preserved:
        * _age_bin
        * _deck
        * _family_size
        * _fare_per_person
        * _is_alone
        * _surname
        * _family_id
        * _ticket_prefix
        * _title
    - New '_cgbt_' features are added as light-weight, domain-informed indicators:
        * Sex (male/female)
        * Age bands (child/senior)
        * Family size bands (small/large)
        * Class indicators (1st / 3rd)
        * Title indicators (Mr / Mrs / Miss / Master)
    """
    assert x_test is not x_train  # programmer bug

    for df in (x_train, x_test):
        # ------------------------------------------------------------------
        # Core existing synthetic features
        # ------------------------------------------------------------------

        # AgeBin.  (If Age has missing values at this stage, they become NaN bins.)
        df["_age_bin"] = pd.cut(df["Age"], bins=_AGE_BIN_NUMBERS, labels=_AGE_BIN_LABELS)

        # Deck from Cabin (first character, e.g., 'C85' -> 'C').
        df["_deck"] = df["Cabin"].str[0]

        # Family size: passenger + siblings/spouses + parents/children.
        df[_FAMILY_SIZE] = df["SibSp"] + df["Parch"] + 1

        # Fare per person within the family.
        df["_fare_per_person"] = df["Fare"] / df[_FAMILY_SIZE]

        # IsAlone: True if no family members aboard.
        df["_is_alone"] = df[_FAMILY_SIZE] == 1

        # Surname + family size → family ID (helps capture groups travelling together).
        df[_SURNAME] = df["Name"].str.split(",").str[0]
        df["_family_id"] = df[_SURNAME] + df[_FAMILY_SIZE].astype(str)

        # Ticket prefix: strip digits and punctuation, keep any alphabetic / other prefix.
        df["_ticket_prefix"] = df["Ticket"].str.replace(r"[\d\.]", "", regex=True).str.strip()

        # Title extracted from name, e.g. "Braund, Mr. Owen" -> "Mr"
        df["_title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

        # ------------------------------------------------------------------
        # New '_cgbt_' domain-informed features
        # ------------------------------------------------------------------
        # Age bands (using raw Age, not _age_bin, so imputation/thresholds are explicit):
        # - Child: clearly young passengers
        # - Senior: older / more frail passengers
        df["_cgbt_is_child"] = (df["Age"] < 16).astype(int)
        df["_cgbt_is_senior"] = (df["Age"] >= 60).astype(int)

        # Family size bands:
        # - Small families (1–2) vs larger groups (>=4).
        df["_cgbt_is_small_family"] = (df[_FAMILY_SIZE] <= 2).astype(int)
        df["_cgbt_is_large_family"] = (df[_FAMILY_SIZE] >= 4).astype(int)

        # Class indicators:
        # - 1st vs 3rd class often capture a big survival signal.
        df["_cgbt_is_first_class"] = (df["Pclass"] == 1).astype(int)
        df["_cgbt_is_third_class"] = (df["Pclass"] == 3).astype(int)

        # Title buckets: strong proxies for gender/age/social role.
        title = df["_title"].fillna("")

        df["_cgbt_is_mr"] = (title == "Mr").astype(int)
        df["_cgbt_is_mrs"] = (title == "Mrs").astype(int)
        df["_cgbt_is_miss"] = (title == "Miss").astype(int)
        df["_cgbt_is_master"] = (title == "Master").astype(int)

        # A simple "rare title" flag can also help (Dr, Rev, Col, etc.).
        common_titles = {"Mr", "Mrs", "Miss", "Master"}
        df["_cgbt_is_rare_title"] = (~title.isin(common_titles)).astype(int)

    return x_train, x_test
