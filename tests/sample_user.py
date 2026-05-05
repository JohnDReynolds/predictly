"""
Specifications.
"""

##################################################### BEGIN FEATURE ENGINEERING ###################
# Here’s a high-level roadmap of common feature-engineering strategies you can experiment with—no
# single project uses them all, but this should give you a menu of ideas to dip into as you explore
# your data:

# 1. Data Cleaning & Imputation
#    a) The default imputer for missing values is: CategoryImputation.MISSING and
#        NumericImputattion.MISSING.
# .  b) The default is OneHotEncoding for category features <= 10 distinct values, else Ordinal.
#    c) In step 2 in tester.py, put a debug breakpoint at the end of read_xy, and then view the
#       datagrid viewer for df to decide on any custom imputations that should be defined in
#       steps 6 and 7 in tester.py.

# 2. Variable Transformation
# Scaling: standard, min-max, robust (IQR) depending on your model.
# Normalization: log/Box-Cox/Yeo-Johnson to tame skewed distributions.
# Cyclic encoding: convert hours, months, days-of-week into sine/cosine pairs so that “wrap-around”
# is respected.

# 3. Categorical Encoding
# Ordinal: for truly ordered levels (e.g. “Low/Med/High”).
# One-Hot: when no order exists or for linear models.
# Target (mean) encoding: replace each level with its average target value (with smoothing to avoid
# overfit).
# Count / frequency encoding: map each level to how often it appears.
# Embeddings: learn low-dim vectors for high-cardinality features (via neural nets or
# entity-embedding libraries).

# 4. Interaction & Combinations
# Pairwise interactions: multiply or concatenate two numeric features (e.g. age × income).
# Polynomial features: squares, cubes, or cross-terms to capture non-linear effects.
# Binning + grouping: bin continuous into categories (quantiles or domain cuts) then interact with
# another feature.
# Aggregations: if you have group keys (e.g. user ID), create per-group stats (mean, max, count)
# and merge back.

# 5. Domain-Driven Features
# Ratios: like cholesterol/age or price/area.
# Residuals: difference from a group average (actual – mean of peers).
# Flags: binary indicators for special conditions (e.g. “went to ER,” “defaulted once”).
# Temporal lags & rolling: for time-series, create lagged values, rolling means, exponential moving
# averages.

# 6. Dimensionality Reduction & Clustering
# PCA / SVD: compress many correlated features into orthogonal components.
# Autoencoders: non-linear compression via a small neural net bottleneck.
# t-SNE / UMAP: for visualization, to spot clusters or weird outliers.
# Clustering: use K-Means or DBSCAN on some features, then add the cluster label or
# distance-to-centroid as a new feature.

# 7. Feature Selection & Regularization
# Filter methods: variance threshold, correlation threshold, mutual information ranking.
# Wrapper methods: recursive feature elimination (RFE), sequential forward/backward selection.
# Embedded: L1-penalized models (Lasso, ElasticNet), tree-based feature importances, or SHAP values
# to pick the top drivers.
# Stability: test which features persist across different CV folds or random seeds.

# 8. Automated Feature Engineering
# Featuretools: automatic creation of deep relational features via “DFS.”
# tsfresh: extract hundreds of time-series descriptors in one call.
# Custom pipelines: build reusable Transformer classes (for scikit-learn Pipelines or TFX).

# . 8a. Create new features from existing features.  For instance if square-footage is a great
# .     predictive feature, but you only have length and width, then you can create square-footage
# .     by multiplying length * width.  square-footage would be called a "synthetic feature".
# .     a) x_train["square_footage"] = x_train["length"] * x_train["width"].
# .     b) Or maybe you take the logarithm or square^2 or cube^3.
# .     c) Or maybe you create a new synthetic feature that is the quantity of a group of columns
# .        that have the value "True".
# .        accidents["RoadwayFeatures"] = accidents[["col1", "col2"]].sum(axis=1)
# .     d) Or maybe you create a new synthetic feature that is the quantity of a group of columns
# .        that have a value greater than 0.
# .        concrete["Components"] = concrete[["col1", "col2"]].gt(0).sum(axis=1)
# .     e) Extracting the area code from a phone number.
# .     f) Combine 2 categorical features into a new third categorical feature.
# .     g) Grouping
#         customer["AverageIncomeByState"] = customer.groupby("State")["Income"].transform("mean")

# 9. Validation & Anti-Leakage
# Leak checks: ensure no future information seeps into features (e.g. target leakage).
# Fold-aware transforms: fit imputers/encoders only on training folds, then apply to
# validation/test.
# Baseline comparisons: always check your engineered set against “raw” features to verify you’re
# actually improving.

# Workflow tip:
# Explore univariately (histograms, boxplots) and bivariately (scatter, heatmap).
# Prototype a handful of features quickly in a notebook.
# Bake into a pipeline so you can track what’s helping.
# Iterate: drop what doesn’t work, double-down on what does, and keep your feature set as lean as
# possible.
##################################################### END FEATURE ENGINEERING #####################


############################################ BEGIN IMPUTE INSTRUCTIONS ############################
# STRINGS and CATEGORIES
# 1. If <5% are missing, then use one of the following based on practical common sense:
#    a) The most frequent value if it's a very high percentage and makes logical dense.
#    b) Maybe a built in N/A or None type category if it makes logical sense.
#    c) util.MISSING
# 2. If >5% are missing, then probably use util.MISSING.  Sparingly use a or b if they make sense.
# 3. You do have a CategoryImputation.MOST_FREQUENT, but try not to use it.  Use the actual hard
#    coded most frequent value.
# NUMERIC
# 1. If zero is not a valid value, or it looks like a mistake, then you might want to swap it to
#    missing like this:
#      processors["Cholesterol"] = {
#         Processor.SWAP: (0, np.nan),
#         Processor.FILL_VALUE: NumericImputation.ITERATIVE, (or NumericImputation.MEDIAN)
#      }
# 2. Using {Processor.FILL_VALUE: 0} is often legitimate for things like dollar amounts.
# 3. There also might be another non-zero number that makes sense, like -1 for all-positives.
# 4. Otherwise prefer ITERATIVE over MEDIAN if it makes sense.
############################################ END IMPUTE INSTRUCTIONS ##############################


# Errors to ignore
# mypy: disable-error-code=import-untyped
# pyright: reportUnknownMemberType=false

# Python imports
from typing import Any

# Third-Party imports
import numpy as np
import pandas as pd

# Project imports
import tabular.utilities as util
from tabular.utilities import ObjectImputation, Option, NumericImputation, Processor

# Age Bins
_AGE_BIN_NUMBERS = (0, 40, 50, 60, 70, 999)  # If Age==0, it will be missing.  Force error.
_AGE_BIN_LABELS = ("Young", "Middle", "Mature", "YoungSenior", "OldSenior")

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


def set_options(**options: Any) -> dict[str, Any]:
    """xxx"""
    # ----------------------------------------------------------------------
    # 1. Optionally define base options in options.json.  Otherwise they will be imputed.
    # ----------------------------------------------------------------------
    # Option.BOOLEAN_COLUMNS_TO_INT
    # Option.FEATURE_NAMES_TO_EXCLUDE
    # Option.METRIC
    # Option.ML_TASK
    # Option.SUBMISSION_FILE_PATH]
    # Option.SUBMIT_TRUE_FALSE
    # Option.TEST_FILE_PATH
    # Option.TRAIN_FILE_PATH
    # Option.UID, Option.Y_COLUMN_NAME
    # Option.Y_TRANSFORMATION_FUNCTION_POST
    # Option.Y_TRANSFORMATION_FUNCTION_PRE

    # ----------------------------------------------------------------------
    # 2. Optionally define the function to create synthetic features. It will be in this file.
    # ----------------------------------------------------------------------
    options[Option.CUSTOM_SYNTHETICS_FUNCTION] = synthetic_features

    # ----------------------------------------------------------------------
    # 3. Processing argument dictionaries, optionally one for each column.
    #    When building this portion of the specification file, it is typically best to put a
    #    breakpoint in predict.py.train() right after you call util.read_xy().  Then right-click on
    #    x_train in the upper left debugger "Variables" window and select "View Value in Data
    #    Viewer".
    # ----------------------------------------------------------------------
    processors: util.ProcessorType = {}
    options[Option.PROCESSORS] = processors

    # 3A) Processor.SWAP
    #     This is used to swap value "A" to value "B".  It is done directly after reading the
    #     file before any processing.  See utililities.py.read_xy().  It is often used for
    #     swapping values like zero "0" to missing "np.nan".  This is done when a value like zero
    #     can logically mean missing.  For instance if a bunch of values in a column "train_fare"
    #     are zero, that does not mean they legitimately paid nothing for their train fare
    #     because there are no freebies on this train.  Ditto with columns like "cholesterol" or
    #     "blood_pressure". This example will swap 0 to NaN, and then impute the NaN iteratively
    #     based on the values of other features.
    processors["Cholesterol"] = {
        Processor.SWAP: (0, np.nan),
        Processor.FILL_VALUE: NumericImputation.ITERATIVE,
    }

    # 3B) Processor.RANKED_CATEGORIES
    #     This is used when a column has ranked categorical values.  Like "Low", "Medium", "High".
    #     It will implement a ranked ordinal encoding.  See features.py._create_ordinal_pipeline().
    processors["_age_bin"] = {Processor.RANKED_CATEGORIES: _AGE_BIN_LABELS}
    processors["Quality"] = {Processor.RANKED_CATEGORIES: ("Low", "Medium", "High")}

    # 3C) Processor.FILL_VALUE
    #     This specifies how "missing" (NaN or None) values are imputed.
    #     1. Numeric Features
    #        a) NumericImputation.ITERATIVE will iteratively determine the fill value based on the
    #           values of other features.
    processors["BasementSquareFeet"] = {Processor.FILL_VALUE: NumericImputation.ITERATIVE}
    #        b) NumericImputation.MEDIAN will use the median value as the fill value.
    processors["Ranking"] = {Processor.FILL_VALUE: NumericImputation.MEDIAN}
    #        c) -1 is typically used for features that will always be non-negative.  For
    #           instance, synthetic ratios of 2 non-negative numbers divided by zero.
    processors["My_Ratio"] = {Processor.FILL_VALUE: NumericImputation.MINUS_1}
    processors["My_PoisitiveNumber"] = {Processor.FILL_VALUE: NumericImputation.MINUS_1}
    #        d) 0 (zero) might be used where it makes sense.  For instance currency amounts
    #           where "missing" most likely means 0.  This will then also allow you to use the
    #           column in a summation formula with other columns.
    processors["auto_expenses"] = {Processor.FILL_VALUE: NumericImputation.ZERO}
    #          e) Other numeric values when they make sense.
    #     2. String Features
    #        a) StringImputation.MISSING is typically be used when the most-frequent value would
    #           not make sense.
    processors["HomePlanet"] = {Processor.FILL_VALUE: ObjectImputation.MISSING}
    #        b) StrinImputatiuon.MOST_FREQUENT is used when the most-frequent value constitutes
    #           the vast majority of values and makes sense as the default.
    processors["Functional"] = {
        Processor.FILL_VALUE: ObjectImputation.MOST_FREQUENT,
        Processor.RANKED_CATEGORIES: ("Min1", "Min2", "Mod", "Maj1", "Maj2"),
    }
    #          c) Other string values when they make sense.

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
    assert x_test is not x_train  # programmer bug

    # Apply simple features to x_train and x_test that have no chance of data leakage.
    for df in (x_train, x_test):
        # ----------------------------------------------------------------------
        # 1. Your own custom features.  Names start with "_".
        # ----------------------------------------------------------------------

        # Age bins (ordered category)
        df["_age_bin"] = pd.cut(df["Age"], bins=_AGE_BIN_NUMBERS, labels=_AGE_BIN_LABELS)

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
        # 2. ChatGPT features.  Names start with "_cgbt_".
        #    Ask ChatGPT:
        #      Could you please audit the below synthetic_features() function for the Kaggle
        #      competition "sample_user".  Look for feature correctness, redundancy, leakage, and
        #      anything else that you think is appropriate.  Also provide any additional synthetic
        #      features that you think might add value.
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

    return x_train, x_test
