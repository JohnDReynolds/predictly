"""
Custom user functions.
"""

# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportUnknownMemberType=false

# Python imports
from typing import Any

# Third-Party imports
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project imports
import tabular.utilities as util
from tabular.utilities import NumericImputation, Option, Processor, ObjectImputation

# MsSubClass mappings
_MSSUBCLASS_MAP: dict[int, int] = {
    20: 1,  # 20	1-STORY 1946 & NEWER ALL STYLES
    30: 1,  # 30	1-STORY 1945 & OLDER
    40: 1,  # 40	1-STORY W/FINISHED ATTIC ALL AGES
    120: 1,  # 120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
    45: 2,  # 45	1-1/2 STORY - UNFINISHED ALL AGES
    50: 2,  # 50	1-1/2 STORY FINISHED ALL AGES
    150: 2,  # 150	1-1/2 STORY PUD - ALL AGES
    60: 3,  # 60	2-STORY 1946 & NEWER
    70: 3,  # 70	2-STORY 1945 & OLDER
    160: 3,  # 160	2-STORY PUD - 1946 & NEWER
    75: 4,  # 75	2-1/2 STORY ALL AGES
    80: 5,  # 80	SPLIT OR MULTI-LEVEL
    85: 5,  # 85	SPLIT FOYER
    90: 6,  # 90	DUPLEX - ALL STYLES AND AGES
    180: 7,  # 180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
    190: 8,  # 190	2 FAMILY CONVERSION - ALL STYLES AND AGES
}


def _add_area_cluster_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit KMeans on 5 area-related features of x_train, then append to both x_train and x_test:
      - __SFArea_Cluster: integer cluster label (0 to n_clusters-1)
      - __SFArea_Cluster_Centroid_{i}: distance to centroid i, for i in [0..n_clusters-1]

    Args:
        x_train: training DataFrame
        x_test:  test DataFrame

    Returns:
        (x_train, x_test): The original DataFrames with the new features joined on the right.
    """
    # n_clusters will be the quantity of area cluster buckets.  A suggested heuristic to use is:
    # n_clusters = max(2, int(math.sqrt(x_train_scaled.shape[0] / 2))).
    # But with 10,000 observations, this would create 71 area cluster buckets.  Which means that
    # there would be 71 new "Centroid" features!!  ChatGPT says that this heuristic can often
    # "overshoot" and to use "common sense".  It could be as simple as 3 (small, medium, large).
    # I will settle on 5.
    n_clusters = 5
    sfarea_features = ["LotArea", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea"]

    # Create a pipeline.  Explicitly set n_init=10 (default) to to avoid FutureWarning in sklearn.
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=util.RANDOM_STATE, n_init=10)),
        ]
    )

    # fit on train
    pipeline.fit(x_train[sfarea_features])

    # labels and distances
    new_column_name = "_sfarea_cluster"
    x_train_zero = x_train[sfarea_features].fillna(0)
    x_test_zero = x_test[sfarea_features].fillna(0)
    x_train[new_column_name] = pipeline.predict(x_train_zero)
    x_test[new_column_name] = pipeline.predict(x_test_zero)

    # Add new centroid_cols to x_train and x_test.
    centroid_cols = [f"{new_column_name}-Centroid-{i}" for i in range(n_clusters)]
    x_train[centroid_cols] = pipeline.transform(x_train_zero)
    x_test[centroid_cols] = pipeline.transform(x_test_zero)

    return x_train, x_test


def _add_median_grouped_feature(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    group_col: str,  # e.g. "Neighborhood"
    target_col: str,  # e.g. "GrLivArea"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mutates x_train and x_test by adding:
      - new_feature: the median of target_col per group_col (fallback to global train median).
    """
    # 1. Compute per-group and global medians on train, only for the observed values.
    #    By default, groupby(..., dropna=True) (the default since pandas 1.1.0) will omit any rows
    #    where the grouping key itself is NaN. Those rows simply won’t contribute to any group and
    #    you won’t get an entry for “group = NaN” in your result.  Once the rows are partitioned
    #    into groups, pandas computes the median of target_col ignoring any NaN values in that
    #    column.  If a particular group has no non‐NaN values in target_col, its median will be NaN
    #    in the resulting Series.
    group_medians = x_train.groupby(group_col, observed=True)[target_col].median()
    global_median = x_train[target_col].median()

    # 2. Map & assign to train (missing groups get the global median)
    new_column_name = f"_{group_col}{target_col}"
    x_train[new_column_name] = x_train[group_col].map(group_medians).fillna(global_median)

    # 3. Map & assign to test (unseen groups get the global median)
    x_test[new_column_name] = x_test[group_col].map(group_medians).fillna(global_median)

    return x_train, x_test


def set_options(**options: Any) -> dict[str, Any]:
    """xxx"""
    options[Option.CUSTOM_SYNTHETICS_FUNCTION] = synthetic_features

    processors: util.ProcessorType = {}
    options[Option.PROCESSORS] = processors

    # These are synthetic features that will have a zero denominator if there is no basement.
    processors["_cgbt_bsmt_finished_ratio"] = {Processor.FILL_VALUE: NumericImputation.MINUS_1}

    # 94% missing.  3% gravel and 3% paved.  It does have a category "NA" for "No alley access",
    # but it is not used, and does not imply "no alley".  Since most houses probably do not have an
    # alley, _missing_ is probably a good choice.  Not an ordered ranking.
    processors["Alley"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 3% missing.  90% "TA": Typical. "NA" is for "No Basement", but it is never used.  Is an
    # ordered ranking.
    processors["BsmtCond"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("NA", "Po", "Fa", "TA", "Gd", "Ex"),
    }

    # 3% missing.  60% "No": No exposure.  "NA" is for "No Basement", but it is never used.  Is an
    # ordered ranking.
    processors["BsmtExposure"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("NA", "No", "Mn", "Av", "Gd"),
    }

    # <1% missing.
    processors["BsmtFinSF1"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}
    processors["BsmtFinSF2"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # 3% missing.  No clear most frequent.  "NA" is for "No Basement", but it is never used.  Is an
    # ordered ranking.
    processors["BsmtFinType1"] = {
        Processor.FILL_VALUE: ObjectImputation.MISSING,
        Processor.RANKED_CATEGORIES: ("NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"),
    }

    # 3% missing.  86% "Unf": Unfinished.  "NA" is for "No Basement", but it is never used.  Is an
    # ordered ranking.
    processors["BsmtFinType2"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"),
    }

    # < 1% missing.
    processors["BsmtFullBath"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}
    processors["BsmtHalfBath"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # 3% missing.  No clear most frequent.  "NA" is for "No Basement", but it is never used.  Is an
    # ordered ranking.
    processors["BsmtQual"] = {
        Processor.FILL_VALUE: ObjectImputation.MISSING,
        Processor.RANKED_CATEGORIES: ("NA", "Po", "Fa", "TA", "Gd", "Ex"),
    }

    # < 1% missing.
    processors["BsmtUnfSF"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # <1% missing.  91% are "SBrkr": Standard Circuit Breakers & Romex.  Is an ordered ranking.
    # Note that there is only one occurence of "Mix", which is kind of ambiguous in terms of the
    # ranking.
    processors["Electrical"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("Mix", "FuseP", "FuseF", "FuseA", "SBrkr"),
    }

    # <1% missing.  No clear most frequent.  Is not an ordered ranking.
    processors["Exterior1st"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # <1% missing.  No clear most frequent.  Is not an ordered ranking.
    processors["Exterior2nd"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 81% missing, which is the most frequent.  "NA" is used for "No Fence", but since 81% are
    # missing, that is probably not a good default.  Besides, "NA" is not ever used.  Is an ordered
    # ranking.
    processors["Fence"] = {
        Processor.FILL_VALUE: ObjectImputation.MISSING,
        Processor.RANKED_CATEGORIES: ("NA", "MnWw", "GdWo", "MnPrv", "GdPrv"),
    }

    # 47% missing.  No clear most frequent.  "NA" is used for "No Fireplace", but it is never used.
    # Is an ordered ranking.
    processors["FireplaceQu"] = {
        Processor.FILL_VALUE: ObjectImputation.MISSING,
        Processor.RANKED_CATEGORIES: ("NA", "Po", "Fa", "TA", "Gd", "Ex"),
    }

    # < 1% missing.  93% are "Typ": "Typical Functionality".  This is an ordered ranking.
    processors["Functional"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"),
    }

    # < 1% missing.
    processors["GarageArea"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}
    processors["GarageCars"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # 6% missing.  TA: Typical/Average is 91%.  This is an ordered ranking.
    processors["GarageCond"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("NA", "Po", "Fa", "TA", "Gd", "Ex"),
    }

    # 6% missing.  No clear "most frequent".  This is not an ordered ranking.
    processors["GarageFinish"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 6% missing.  TA: Typical/Average is 91%.  This is an ordered ranking.
    processors["GarageQual"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("NA", "Po", "Fa", "TA", "Gd", "Ex"),
    }

    # 6% missing.  The most frequent "Attchd" is only 60%, so this is probably not a good
    # default.  If it's missing, it might be a dump.  Not an ordered ranking.
    processors["GarageType"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 6% missing.
    processors["GarageYrBlt"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # <1% missing.  The most frequent "TA" is only 50%.  Is an ordered ranking.
    processors["KitchenQual"] = {
        Processor.FILL_VALUE: ObjectImputation.MISSING,
        Processor.RANKED_CATEGORIES: ("Po", "Fa", "TA", "Gd", "Ex"),
    }

    # 18% missing.  ITERATIVE or MEDIAN might be presumptuous because it might not have a lot
    # (i.e. it might be a condo).  All valid values are > 0, so default to -1.
    processors["LotFrontage"] = {Processor.FILL_VALUE: NumericImputation.MINUS_1}

    # < 1% missing.
    processors["MasVnrArea"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # 60% missing.  There is a "None", but it is not used.  Is not an ordered ranking.
    processors["MasVnrType"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # 96% missing.  There is a "NA": None, but it is not used.  Is not an ordered ranking.
    processors["MiscFeature"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}

    # <1% missing.  79% are "RL" Residential Low Density.  Not an ordered ranking.
    processors["MSZoning"] = {Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT}

    # 99% missing.  Does have a "NA": No Pool, but it is not used.  Is an ordered ranking.
    processors["PoolQC"] = {
        Processor.FILL_VALUE: ObjectImputation.MISSING,
        Processor.RANKED_CATEGORIES: ("NA", "Fa", "TA", "Gd", "Ex"),
    }

    # <1% missing.  87% are "WD": Warranty Deed.  Is not an ordered ranking.
    processors["SaleType"] = {Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT}

    # < 1% missings.
    processors["TotalBsmtSF"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}

    # <1% missing.  99% are "AllPub": AllPublic Utilities.  Not an ordered ranking.
    processors["Utilities"] = {Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT}

    return options


def synthetic_features(
    x_train: pd.DataFrame, x_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add synthetic features for the House Prices competition.

    Notes:
      * All new features introduced in this function are prefixed with "_cgbt_".
      * All transformations are row-wise or use only x_train-derived statistics
        (via helper functions) and therefore are leak-free w.r.t. the target.
    """
    assert x_test is not x_train

    # Apply simple features to x_train and x_test that have no chance of data leakage.
    for df in (x_train, x_test):
        # ----------------------------------------------------------------------
        # Existing age / remodel / pool / area features (kept as-is)
        # ----------------------------------------------------------------------
        # Garage and house age relative to sale year.
        df["_garage_age"] = df["YrSold"] - df["GarageYrBlt"]
        df["_house_age"] = df["YrSold"] - df["YearBuilt"]

        # Whether the house has been remodeled and how long ago.
        df["_is_remod"] = df["YearRemodAdd"] != df["YearBuilt"]
        df["_remod_age"] = df["YrSold"] - df["YearRemodAdd"]

        # Has Pool (binary indicator).
        df["_has_pool"] = df["PoolArea"] > 0

        # Ratio of above-ground living area to lot area.
        df["_liv_lot_ratio"] = df["GrLivArea"] / df["LotArea"]

        # Grouping of MSSubClass into coarser buckets.
        df["_mssubclass_grouping"] = df["MSSubClass"].map(_MSSUBCLASS_MAP)

        # Count of porch *types* (fixed: use > 0 to count presence, not raw square footage).
        porch_cols = [
            "3SsnPorch",
            "EnclosedPorch",
            "OpenPorchSF",
            "ScreenPorch",
            "WoodDeckSF",
        ]
        df["_porch_types_qty"] = (df[porch_cols] > 0).sum(axis=1)

        # Spaciousness of rooms above ground.
        # (Total floor SF above ground / number of rooms).
        # If TotRmsAbvGrd is zero, result will be NaN and handled downstream by imputation.
        df["_spaciousness"] = (df["1stFlrSF"] + df["2ndFlrSF"]) / df["TotRmsAbvGrd"]

        # Total inside square footage (basement + 1st + 2nd).
        df["_total_inside_sf"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

        # Total outside square footage (porches + deck).
        df["_total_outside_sf"] = (
            df["3SsnPorch"]
            + df["EnclosedPorch"]
            + df["OpenPorchSF"]
            + df["ScreenPorch"]
            + df["WoodDeckSF"]
        )

        # Total quantity of bathrooms (with half-baths weighted at 0.5).
        df["_total_qty_baths"] = (
            df["BsmtFullBath"]
            + (0.5 * df["BsmtHalfBath"])
            + df["FullBath"]
            + (0.5 * df["HalfBath"])
        )

        # Total Quality (Overall quality + overall condition).
        df["_total_qual"] = df[["OverallQual", "OverallCond"]].sum(axis=1)

        # ----------------------------------------------------------------------
        # New _cgbt_ features: quality x size, ratios, amenity scores, etc.
        # All leak-free and row-wise.
        # ----------------------------------------------------------------------
        # Quality-weighted size measures – very important for price.
        df["_cgbt_qual_x_grliv"] = df["OverallQual"] * df["GrLivArea"]
        df["_cgbt_qual_x_total_inside"] = df["OverallQual"] * df["_total_inside_sf"]

        # Simple quadratic effect of overall quality.
        df["_cgbt_overall_qual_sq"] = df["OverallQual"] ** 2

        # Total bathrooms per bedroom (saturation of bathroom capacity).
        # Avoid division by zero by clipping bedroom count to >= 1.
        bedroom_cnt = df["BedroomAbvGr"].clip(lower=1)
        df["_cgbt_baths_per_bedroom"] = df["_total_qty_baths"] / bedroom_cnt

        # Finished basement ratio (how much of the basement is livable).
        # If TotalBsmtSF is zero, ratio becomes NaN and will be imputed downstream.
        total_bsmt = df["TotalBsmtSF"]
        finished_bsmt = df["BsmtFinSF1"] + df["BsmtFinSF2"]
        df["_cgbt_bsmt_finished_ratio"] = finished_bsmt / total_bsmt

        # Garage size relative to inside living area.
        df["_cgbt_garage_to_inside_sf"] = df["GarageArea"] / df["_total_inside_sf"]

        # Indicator: has a second story (two-story vs single-story).
        df["_cgbt_has_2nd_fl"] = (df["2ndFlrSF"] > 0).astype(int)

        # Log-transformed living area (helpful with log(SalePrice) target).
        df["_cgbt_log_grliv"] = np.log1p(df["GrLivArea"])

        # Log-transformed total inside SF.
        df["_cgbt_log_total_inside"] = np.log1p(df["_total_inside_sf"])

        # Lot frontage relative to the square root of lot area
        # (proxy for frontage "generosity" controlling for lot size).
        df["_cgbt_frontage_to_area_sqrt"] = df["LotFrontage"] / np.sqrt(df["LotArea"])

        # Amenity score: count of key positive features.
        # Each boolean is cast to int and summed.
        amenity_bits = [
            (df["Fireplaces"] > 0).astype(int),
            (df["GarageCars"] > 0).astype(int),
            df["_has_pool"].astype(int),
            (df["WoodDeckSF"] > 0).astype(int),
            (df["ScreenPorch"] > 0).astype(int),
        ]
        df["_cgbt_amenity_score"] = sum(amenity_bits)

        # Age buckets: simple flags for very new vs very old houses.
        df["_cgbt_is_newer_20"] = (df["_house_age"] <= 20).astype(int)
        df["_cgbt_is_older_60"] = (df["_house_age"] >= 60).astype(int)

    # Add the SF-Area cluster features (train-fit KMeans, apply to both).
    x_train, x_test = _add_area_cluster_features(x_train, x_test)

    # Add the median GrLivArea by neighborhood (fit on train, map to both).
    x_train, x_test = _add_median_grouped_feature(x_train, x_test, "Neighborhood", "GrLivArea")

    return x_train, x_test
