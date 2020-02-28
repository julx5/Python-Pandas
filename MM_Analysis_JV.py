# STANDARD
import sys

import numpy as np
import xgboost as xgb
# EXCEL
from openpyxl import load_workbook
from pandas import DataFrame
from scipy.stats import randint, reciprocal, uniform
# CLASSIFIER
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate)

# Logger
import Logger


PATH_TO_EXCEL = "TabelleMyelomAllebis2006-JV.xlsx"

# LAST ELEMENT HAS TO BE THE TARGET
# Currently all relevant features by Khandanpour till "Initiales Kreatining"
USED_COLUMNS = ["C", "D", "Y", "AH", "AJ", "AK", "AM", "AO",
                "AT", "CY"]  # The columns are used in order as defined


def readExcel():
    """Read the Excel File and create a Pandas Dataframe containing Samples including their targets."""
    file = load_workbook(PATH_TO_EXCEL)[
        "TabelleMyelom"]  # Load the Excelfile and pick the first (of the three available) Sheet with all data#

    # Parse the whole excel-sheet into the Dataframe
    data = DataFrame(file.values)
    # keep only the first 130 columns (Therapiewechsel)
    data = data.iloc[:, :130]

    # TODO handling of different Types of values --> (None, N/A, Empty), Strings?, ...
    # Replace the undefined values with np.NAN
    data = data.replace(to_replace="=#N/A", value=np.nan)
    # Drop Rows/Samples that only have NA Values
    data = data.dropna(axis=0, how="all")
    # Drop Columns/Features that only have NA Values
    data = data.dropna(axis=1, how="all")

    # Filter the data before creating feature/target
    # Use the Columns that are defined at the top of this file
    data = data.iloc[:, [column_name_to_number(
        column) for column in USED_COLUMNS]]

    # Faulty
    # data = data.dropna(axis=0, how="any") # Faulty: Drop Rows that still have NA Values --> Instead of dropping maybe replacing with default value? Decision should strongly depend on the actual column

    # Pick the first Row for the column names that are used for training
    labels = data.iloc[0, :]
    data = data.iloc[1:, :]  # Remove the first row so only the features remain
    # Currently the Values are supposed to be Numeric Categories -> Abbreviations (like drugs) should be handled separately beforehand
    data = data.astype("float64")

    # Column 'Ansprechen 1. Therapielinie'
    target = data.iloc[:, -1].to_numpy()
    # Binarization of Target
    features = data.iloc[:, 0:-2]
    Logger.info(f"Used Features: {labels}")
    Logger.info(f"Row count (Data, Target): ({len(data)}, {len(target)})")

    return features, target


def column_name_to_number(name):
    """use this method to transform column name into a number"""
    out = 0
    for next_char in name:
        out *= 26
        out += int(ord(next_char)) - 64  # A == 65
    return out-1  # Zero-Based array numbering


def run():
    # Get the Training Data
    features, target = readExcel()

    # Create Folds for CV
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=41)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)

    # Inner CV

    # Alternative for Pipelines (set  of )
    # pca = PCA(n_components=int(len(DATA[0])*0.75))
    # estimator = RandomForestClassifier()
    # pipeline = Pipeline(steps=[("pca", pca), ("rf", estimator)])

    # Create the XBBoost Classifier
    estimator = xgb.XGBClassifier(objective="binary:logistic",
                                  tree_method="auto",
                                  n_jobs=4,
                                  colsample_bytree=0.6,
                                  learning_rate=0.008,
                                  max_depth=8,
                                  n_estimators=500,
                                  subsample=0.7,
                                  gamma=0.5)

    # Hyperparameter Grid
    params = {
        'min_child_weight': [1, 5, 10],
        # 'gamma': [0.5, 1, 1.5, 2, 5],
        # 'subsample': [0.6, 0.8, 1.0],
        # 'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }

    # Embed the Classifier into a GridSearchCV (Hyperparam Optimization in inner CV)
    model = GridSearchCV(estimator=estimator, scoring="balanced_accuracy",
                         param_grid=params, cv=inner_cv, verbose=1, n_jobs=5, iid=False, refit=True)

    # Outer CV - Metrics (precision, recall, f1) are Binary -> not applicable for multi class targets (categories 1 to 6 for column CY)
    cv_score = cross_validate(model, X=features, y=target, cv=outer_cv, scoring=[
                              "balanced_accuracy", "f1_weighted"], return_train_score=False, return_estimator=True)

    # Write all relevant Results
    Logger.info(
        f"FitTime: {cv_score['fit_time'].sum()} - ScoreTime: {cv_score['score_time'].sum()}")
    Logger.result(f"Cross-Validation Test Scores:")
    out = []
    for name, key in [("Balanced-Accuracy", "test_balanced_accuracy"),
                      ("F1-Weighted", "test_f1_weighted")]:
        Logger.result(
            f"\t{name}: Minimum {cv_score[key].min():.3f} - Average {cv_score[key].mean():.3f} - Maximum {cv_score[key].max():.3f} - Std {np.std(cv_score[key]):.3f}")
        out.append(f"{name}: {cv_score[key].mean():.3f}")


if __name__ == "__main__":
    run()
    aa = 2
    aa = 3
    aa = 4
