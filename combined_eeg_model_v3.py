# ============================================================
# GAMEESSS: FOUR-MODEL NESTED SUBJECT-INDEPENDENT CV
# Combined GAMEEMO + MOBA EEG dataset
#
# Models:
#   1. Extra Trees
#   2. XGBoost
#   3. Random Forest
#   4. RBF-SVM
#
# Tasks:
#   1. Emotion / Game Condition Classification
#   2. Cognitive Workload Classification
#
# Outer CV: 5-fold GroupKFold by SubjectID
# Inner CV: 4-fold GroupKFold by SubjectID
#
# Output:
#   nested_cv_results_v2.xlsx
#   nested_cv_summary_v2.xlsx
#   nested_cv_models_v2.joblib
# ============================================================

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = Path(
    r"F:\EEG_GAMING\PROJECT\master_eeg_features.xlsx"
)

OUTPUT_DIR = DATA_PATH.parent

DETAILS_PATH = OUTPUT_DIR / "nested_cv_results_v2.xlsx"
SUMMARY_PATH = OUTPUT_DIR / "nested_cv_summary_v2.xlsx"
MODELS_PATH = OUTPUT_DIR / "nested_cv_models_v2.joblib"

SUBJECT_COLUMN = "SubjectID"

FEATURES = [
    "MeanTheta",
    "MeanAlpha",
    "MeanBeta",
    "MeanGamma",
    "MeanThetaBeta",
    "FAA",
    "MeanEngagement",
    "EngagementVar",
    "MeanHjorthActivity",
    "MeanHjorthMobility",
    "MeanHjorthComplexity",
]

TARGETS = {
    "Emotion_GameCondition": "EmotionClass",
    "Cognitive_Workload": "Cognitive Workload",
}

OUTER_FOLDS = 5
INNER_FOLDS = 4
RANDOM_STATE = 42


# ============================================================
# 2. LOAD AND CLEAN DATA
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_excel(DATA_PATH)

required_columns = (
    [SUBJECT_COLUMN]
    + FEATURES
    + list(TARGETS.values())
)

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Normalize workload labels.
workload_column = TARGETS["Cognitive_Workload"]

workload_mapping = {
    "low": "Low Workload",
    "low workload": "Low Workload",
    "moderate": "Moderate Workload",
    "moderate workload": "Moderate Workload",
    "medium": "Moderate Workload",
    "medium workload": "Moderate Workload",
    "high": "High Workload",
    "high workload": "High Workload",
}

df[workload_column] = (
    df[workload_column]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(workload_mapping)
)

print("=" * 75)
print("GAMEESSS: FOUR-MODEL NESTED SUBJECT-INDEPENDENT CV")
print("=" * 75)
print("Dataset shape:", df.shape)
print("Unique subjects:", df[SUBJECT_COLUMN].nunique())

if "DatasetType" in df.columns:
    print("\nDataset counts:")
    print(df["DatasetType"].value_counts())


# ============================================================
# 3. MODEL PIPELINES AND SEARCH SPACES
# ============================================================

def make_model_configs(number_of_classes):

    extra_trees = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])

    random_forest = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )),
    ])

    svm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            class_weight="balanced",
        )),
    ])

    xgboost = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            objective="multi:softprob",
            num_class=number_of_classes,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            tree_method="hist",
            n_jobs=1,
        )),
    ])

    return {
        "Extra Trees": {
            "pipeline": extra_trees,
            "grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", 0.8],
            },
        },

        "XGBoost": {
            "pipeline": xgboost,
            "grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [2, 3],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__subsample": [0.8],
                "classifier__colsample_bytree": [0.8],
                "classifier__min_child_weight": [1, 3],
                "classifier__reg_lambda": [1, 3],
            },
        },

        "Random Forest": {
            "pipeline": random_forest,
            "grid": {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", 0.8],
            },
        },

        "RBF-SVM": {
            "pipeline": svm,
            "grid": {
                "classifier__C": [0.1, 1, 10],
                "classifier__gamma": ["scale", 0.01],
            },
        },
    }


# ============================================================
# 4. NESTED CROSS-VALIDATION
# ============================================================

all_predictions = []
all_fold_metrics = []
all_overall_metrics = []

best_models = {}
label_encoders = {}
class_mappings = {}
selected_models = {}

for task_name, target_column in TARGETS.items():

    print("\n" + "#" * 75)
    print("TASK:", task_name)
    print("#" * 75)

    task_df = df.dropna(
        subset=[target_column, SUBJECT_COLUMN]
    ).copy()

    task_df = task_df.loc[
        ~task_df[FEATURES].isna().all(axis=1)
    ].reset_index(drop=True)

    X = task_df[FEATURES].copy()

    groups = (
        task_df[SUBJECT_COLUMN]
        .astype(str)
        .reset_index(drop=True)
    )

    original_y = (
        task_df[target_column]
        .astype(str)
        .reset_index(drop=True)
    )

    encoder = LabelEncoder()
    y = encoder.fit_transform(original_y)

    class_names = encoder.classes_
    number_of_classes = len(class_names)

    print("Samples:", len(X))
    print("Subjects:", groups.nunique())
    print("Classes:", list(class_names))
    print("\nClass counts:")
    print(original_y.value_counts())

    if number_of_classes < 2:
        print("Only one class found. Skipping task.")
        continue

    if groups.nunique() < OUTER_FOLDS:
        raise ValueError(
            f"Not enough subjects for {OUTER_FOLDS}-fold CV."
        )

    outer_cv = GroupKFold(n_splits=OUTER_FOLDS)

    outer_splits = list(
        outer_cv.split(X, y, groups=groups)
    )

    model_configs = make_model_configs(number_of_classes)
    task_results = {}

    for model_name, config in model_configs.items():

        print("\n" + "-" * 75)
        print("MODEL:", model_name)
        print("-" * 75)

        pipeline = config["pipeline"]
        parameter_grid = config["grid"]

        oof_predictions = np.full(len(y), -1, dtype=int)

        for fold_number, (train_idx, test_idx) in enumerate(
            outer_splits, start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            groups_train = groups.iloc[train_idx]
            groups_test = groups.iloc[test_idx]

            # Prevent subject leakage.
            overlap = set(groups_train).intersection(
                set(groups_test)
            )

            if overlap:
                raise RuntimeError(
                    f"Subject leakage detected: {overlap}"
                )

            inner_n_splits = min(
                INNER_FOLDS,
                groups_train.nunique()
            )

            inner_cv = GroupKFold(
                n_splits=inner_n_splits
            )

            search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid=parameter_grid,
                scoring="f1_macro",
                cv=inner_cv,
                refit=True,
                n_jobs=-1,
                verbose=0,
                error_score="raise",
            )

            search.fit(
                X_train,
                y_train,
                groups=groups_train,
            )

            fold_model = search.best_estimator_

            y_pred = fold_model.predict(X_test)
            oof_predictions[test_idx] = y_pred

            fold_accuracy = accuracy_score(y_test, y_pred)

            fold_balanced_accuracy = balanced_accuracy_score(
                y_test, y_pred
            )

            fold_macro_f1 = f1_score(
                y_test,
                y_pred,
                average="macro",
                zero_division=0,
            )

            print(
                f"Fold {fold_number}/{len(outer_splits)} | "
                f"Train subjects: {groups_train.nunique()} | "
                f"Test subjects: {groups_test.nunique()} | "
                f"Inner macro-F1: {search.best_score_:.4f} | "
                f"Accuracy: {fold_accuracy:.4f} | "
                f"Balanced accuracy: {fold_balanced_accuracy:.4f} | "
                f"Macro-F1: {fold_macro_f1:.4f}"
            )

            all_fold_metrics.append({
                "Task": task_name,
                "Model": model_name,
                "OuterFold": fold_number,
                "TrainSubjects": groups_train.nunique(),
                "TestSubjects": groups_test.nunique(),
                "InnerBestMacroF1": search.best_score_,
                "Accuracy": fold_accuracy,
                "BalancedAccuracy": fold_balanced_accuracy,
                "MacroF1": fold_macro_f1,
                "BestParameters": str(search.best_params_),
            })

        if np.any(oof_predictions < 0):
            raise RuntimeError(
                "Some samples did not receive an OOF prediction."
            )

        # Aggregate predictions from all held-out outer folds.
        overall_accuracy = accuracy_score(
            y, oof_predictions
        )

        overall_balanced_accuracy = balanced_accuracy_score(
            y, oof_predictions
        )

        overall_macro_f1 = f1_score(
            y,
            oof_predictions,
            average="macro",
            zero_division=0,
        )

        overall_weighted_f1 = f1_score(
            y,
            oof_predictions,
            average="weighted",
            zero_division=0,
        )

        majority_baseline = (
            pd.Series(y)
            .value_counts(normalize=True)
            .max()
        )

        print("\nAGGREGATED OUTER-FOLD RESULTS")
        print(f"Accuracy:          {overall_accuracy:.4f}")
        print(f"Balanced accuracy: {overall_balanced_accuracy:.4f}")
        print(f"Macro F1:          {overall_macro_f1:.4f}")
        print(f"Weighted F1:       {overall_weighted_f1:.4f}")
        print(f"Majority baseline: {majority_baseline:.4f}")

        print("\nClassification report:")
        print(classification_report(
            y,
            oof_predictions,
            labels=np.arange(number_of_classes),
            target_names=class_names,
            zero_division=0,
            digits=4,
        ))

        cm = confusion_matrix(
            y,
            oof_predictions,
            labels=np.arange(number_of_classes),
        )

        print("Confusion matrix:")
        print(pd.DataFrame(
            cm,
            index=[f"Actual: {c}" for c in class_names],
            columns=[f"Predicted: {c}" for c in class_names],
        ))

        task_results[model_name] = {
            "Accuracy": overall_accuracy,
            "BalancedAccuracy": overall_balanced_accuracy,
            "MacroF1": overall_macro_f1,
            "WeightedF1": overall_weighted_f1,
            "MajorityBaseline": majority_baseline,
        }

        all_overall_metrics.append({
            "Task": task_name,
            "Model": model_name,
            "Samples": len(y),
            "Subjects": groups.nunique(),
            "Accuracy": overall_accuracy,
            "BalancedAccuracy": overall_balanced_accuracy,
            "MacroF1": overall_macro_f1,
            "WeightedF1": overall_weighted_f1,
            "MajorityBaseline": majority_baseline,
        })

        # Save row-level out-of-fold predictions.
        output_df = task_df[
            [SUBJECT_COLUMN] +
            [
                c for c in ["Dataset", "DatasetType"]
                if c in task_df.columns
            ]
        ].copy()

        output_df["Task"] = task_name
        output_df["Model"] = model_name
        output_df["Actual"] = encoder.inverse_transform(y)
        output_df["Predicted"] = encoder.inverse_transform(
            oof_predictions
        )

        output_df["Correct"] = (
            output_df["Actual"] == output_df["Predicted"]
        ).astype(int)

        all_predictions.append(output_df)

        # Fit final model on all subjects after evaluation.
        # This fit does not contribute to the reported OOF metrics.
        final_search = GridSearchCV(
            estimator=clone(pipeline),
            param_grid=parameter_grid,
            scoring="f1_macro",
            cv=GroupKFold(
                n_splits=min(INNER_FOLDS, groups.nunique())
            ),
            refit=True,
            n_jobs=-1,
            verbose=0,
            error_score="raise",
        )

        final_search.fit(
            X,
            y,
            groups=groups,
        )

        best_models[(task_name, model_name)] = (
            final_search.best_estimator_
        )

        label_encoders[(task_name, model_name)] = encoder

        class_mappings[(task_name, model_name)] = list(
            class_names
        )

        print(
            "\nFinal all-data model parameters:",
            final_search.best_params_,
        )

    # Compare models using aggregated outer-fold macro-F1.
    comparison_df = pd.DataFrame(task_results).T

    print("\n" + "=" * 75)
    print("MODEL COMPARISON:", task_name)
    print("=" * 75)

    print(
        comparison_df
        .sort_values("MacroF1", ascending=False)
        .round(4)
    )

    selected_model = (
        comparison_df["MacroF1"].astype(float).idxmax()
    )

    selected_models[task_name] = selected_model

    print(
        f"\nBest candidate by aggregated OOF macro-F1: "
        f"{selected_model}"
    )


# ============================================================
# 5. SAVE RESULTS
# ============================================================

if not all_predictions:
    raise RuntimeError("No predictions were generated.")

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True,
)

fold_metrics_df = pd.DataFrame(
    all_fold_metrics
)

overall_metrics_df = pd.DataFrame(
    all_overall_metrics
)

summary_df = (
    fold_metrics_df
    .groupby(["Task", "Model"])
    .agg(
        MeanFoldAccuracy=("Accuracy", "mean"),
        SDFoldAccuracy=("Accuracy", "std"),
        MeanFoldBalancedAccuracy=("BalancedAccuracy", "mean"),
        SDFoldBalancedAccuracy=("BalancedAccuracy", "std"),
        MeanFoldMacroF1=("MacroF1", "mean"),
        SDFoldMacroF1=("MacroF1", "std"),
        MeanInnerBestMacroF1=("InnerBestMacroF1", "mean"),
    )
    .reset_index()
)

with pd.ExcelWriter(DETAILS_PATH) as writer:
    predictions_df.to_excel(
        writer,
        sheet_name="OOF_Predictions",
        index=False,
    )

    fold_metrics_df.to_excel(
        writer,
        sheet_name="Outer_Fold_Metrics",
        index=False,
    )

    overall_metrics_df.to_excel(
        writer,
        sheet_name="Aggregated_OOF_Metrics",
        index=False,
    )

with pd.ExcelWriter(SUMMARY_PATH) as writer:
    summary_df.to_excel(
        writer,
        sheet_name="Nested_CV_Summary",
        index=False,
    )

    overall_metrics_df.to_excel(
        writer,
        sheet_name="Aggregated_OOF_Metrics",
        index=False,
    )

joblib.dump(
    {
        "models": best_models,
        "label_encoders": label_encoders,
        "class_mappings": class_mappings,
        "selected_model_candidates": selected_models,
        "features": FEATURES,
        "targets": TARGETS,
        "subject_column": SUBJECT_COLUMN,
    },
    MODELS_PATH,
)

print("\n" + "=" * 75)
print("NESTED CROSS-VALIDATION COMPLETED")
print("=" * 75)

print("\nAggregated outer-fold metrics:")
print(
    overall_metrics_df
    .sort_values(
        ["Task", "MacroF1"],
        ascending=[True, False],
    )
    .round(4)
    .to_string(index=False)
)

print("\nSelected candidate models:")
for task, model in selected_models.items():
    print(f"{task}: {model}")

print("\nFiles saved:")
print("1.", DETAILS_PATH)
print("2.", SUMMARY_PATH)
print("3.", MODELS_PATH)

print(
    "\nIMPORTANT: Use aggregated outer-fold metrics as the "
    "subject-independent evaluation. Results are preliminary; "
    "validate on independently collected participants."
)