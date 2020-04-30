# To run this classifier, it must be placed in the same directory as the data folder.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("muted")
sns.set_style("whitegrid")

# Setting numpy seed for reproducability
np.random.seed(42)

### EXPLORATORY DATA ANALYSIS

# Identifying missing data in vle.csv
vle = pd.read_csv("../data/vle.csv")
sns.heatmap(vle.isnull(), cbar=False)
plt.savefig("./diagrams/vle_heatmap", dpi=300, bbox_inches="tight")

# Identifying missing data in studentRegistration.csv
studentRegistration = pd.read_csv("../data/studentRegistration.csv")
sns.heatmap(studentRegistration.isnull(), cbar=False)
plt.savefig("./diagrams/studentReg_heatmap", dpi=300, bbox_inches="tight")

# Identifying which students unregistered
unregistered = studentRegistration[
    studentRegistration["date_unregistration"].notna()
].loc[:, ["id_student", "code_module", "code_presentation", "date_unregistration"]]

### DATA PREPARATION

studentInfo = pd.read_csv("../data/studentInfo.csv")

# Determining which students have inconsistent data in 'date_unregistration' and 'final_result'
# e.g. they unregistered from the module but their result is 'Fail' <- IMPOSSIBLE
errors = studentInfo.merge(
    unregistered, how="left", on=["id_student", "code_module", "code_presentation"]
)
errors = errors[
    (errors["final_result"] != "Withdrawn") & (errors["date_unregistration"].notna())
]

# Editing error entries in studentInfo so that 'final_result' = Withdrawn
for error in errors[["id_student", "code_module", "code_presentation"]].values:
    studentInfo.loc[
        (studentInfo.id_student == error[0])
        & (studentInfo.code_module == error[1])
        & (studentInfo.code_presentation == error[2]),
        "final_result",
    ] = "Withdrawn"

assessments = pd.read_csv("../data/assessments.csv")


#  assessments.groupby(["code_module", "code_presentation"]).agg(
#    total_weight=("weight", sum)
# )

# assessments[
#    (assessments["code_module"] == "CCC") | (assessments["code_module"] == "GGG")
# ].groupby(["code_module", "code_presentation", "assessment_type"]).agg(
#    type_weights=("weight", sum)
# )

# Changing weight of each Exam in CCC to 50 rather than 100
assessments.loc[
    (assessments.code_module == "CCC") & (assessments.assessment_type == "Exam"),
    "weight",
] = 50

# Changing weight of each TMA and CMA in GGG to 16.66% and 8.33% respectively
assessments.loc[
    (assessments.code_module == "GGG") & (assessments.assessment_type == "TMA"),
    "weight",
] = (50 / 3)
assessments.loc[
    (assessments.code_module == "GGG") & (assessments.assessment_type == "CMA"),
    "weight",
] = (50 / 6)

# Transforming 'final_result' into 'result_class' so that the problem is binary classification
studentInfo["result_class"] = studentInfo["final_result"].apply(
    lambda x: 0 if (x == "Fail") | (x == "Withdrawn") else 1
)

sns_plot = sns.catplot(
    x="imd_band",
    data=studentInfo,
    kind="count",
    palette="muted",
    hue="result_class",
    order=[
        "0",
        "0-10%",
        "10-20",
        "20-30%",
        "30-40%",
        "40-50%",
        "50-60%",
        "60-70%",
        "70-80%",
        "80-90%",
        "90-100%",
    ],
).set_xticklabels(rotation=45)
sns_plot.savefig("./diagrams/success_by_imd_band", dpi=300)

studentAssessments = pd.read_csv("../data/studentAssessment.csv")

# Adding weights of assessments to studentAssessments
merged = pd.merge(studentAssessments, assessments, how="left", on=["id_assessment"])
# Dropping irrelevant columns
merged.drop(["date", "is_banked"], axis=1, inplace=True)

# Calculating the weighted score of each assessment
merged["score*weight"] = merged["score"] * merged["weight"]
# Calculating the sum of the weighted scores and sum of weights or each student by 'assessment_type'
marks = merged.groupby(
    ["code_module", "code_presentation", "id_student", "assessment_type"], as_index=True
).agg(total_s_w=("score*weight", "sum"), attempted_w=("weight", "sum"))

# Calculating the 'assessment_type' average for each student and rounding
marks["avg"] = marks["total_s_w"] / marks["attempted_w"]
marks.reset_index()
rounded = marks.round(1)

# Pivoting so that each student has a column for each 'assessment_type'
pivotedRounded = pd.pivot_table(
    rounded,
    index=["code_module", "code_presentation", "id_student"],
    values=["avg"],
    columns=["assessment_type"],
    fill_value=0,
)
# Squashing column hierarchy into one
pivotedRounded.columns = pivotedRounded.columns.to_series().str.join("_")
# Dropping their exam average because it's too indicative of 'final_result'
pivotedRounded.drop(["avg_Exam"], axis=1, inplace=True)

# Adding avg assessment values to studentInfo
studentInfo = studentInfo.merge(
    pivotedRounded, how="left", on=["code_module", "code_presentation", "id_student"]
).fillna(0)

studentVle = pd.read_csv("../data/studentVle.csv")
# Adding vle activity types to studentVle
studentVleWithType = studentVle.merge(
    vle, how="left", on=["code_module", "code_presentation", "id_site"]
)
# Removing irrelevant data
studentVleWithType.drop(
    ["id_site", "date", "week_from", "week_to"], axis=1, inplace=True
)
# Setting up count column for feature extraction
studentVleWithType["count"] = 1
# Calculating each student's 'sum_click' and 'count' by activity type
groupedStudentVle = (
    studentVleWithType.groupby(
        ["code_module", "code_presentation", "id_student", "activity_type"]
    )["sum_click", "count"]
    .sum()
    .reset_index()
)

# Pivoting so that each student has 40 new features (2 per activity_type (20))
# showing their overall interaction with each type of activity during the module
pivotedStudentVle = pd.pivot_table(
    groupedStudentVle,
    index=["code_module", "code_presentation", "id_student"],
    values=["sum_click", "count"],
    columns=["activity_type"],
    fill_value=0,
).swaplevel(0, 1, axis=1)
# Squashing column hierarchy
pivotedStudentVle.columns = pivotedStudentVle.columns.to_series().str.join("_")

# Adding vle data to studentInfo and filling missing values with 0
studentInfo = studentInfo.merge(
    pivotedStudentVle, how="left", on=["code_module", "code_presentation", "id_student"]
).fillna(0)

# Boxplots looking for relationship between avg_TMA/avg_CMA and result_class
f, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(
    x="result_class",
    y="avg_TMA",
    data=studentInfo.loc[studentInfo["avg_TMA"] != 0],
    whis=[5, 95],
    showfliers=True,
    ax=axes[0],
)
sns.boxplot(
    x="result_class",
    y="avg_CMA",
    data=studentInfo.loc[studentInfo["avg_CMA"] != 0],
    whis=[5, 95],
    showfliers=True,
    ax=axes[1],
)
plt.savefig("./diagrams/CMA_TMA_boxplot")

# Isolating labels
y = studentInfo["result_class"]
# Isolating input data
X = studentInfo.drop(
    ["id_student", "final_result", "result_class", "code_module", "code_presentation"],
    axis=1,
)

### DATA TRANSFORMATION

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Transformer requires strings
X["imd_band"] = X["imd_band"].astype("str")

imd_categories = [
    "0",
    "0-10%",
    "10-20",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
]
edu_categories = [
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]
age_categories = ["0-35", "35-55", "55<="]

# Setting up column transformer
# remainer='passthrough' forces unchanged columns through the transformer
column_trans = ColumnTransformer(
    [
        (
            "ord",
            OrdinalEncoder([imd_categories, edu_categories, age_categories]),
            ["imd_band", "highest_education", "age_band"],
        ),
        ("hot", OneHotEncoder(sparse=False), ["region", "gender", "disability"]),
    ],
    remainder="passthrough",
)

from sklearn.model_selection import train_test_split

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### MACHINE LEARNING

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    plot_confusion_matrix,
)

svcPipeline = Pipeline(
    [
        ("ct", column_trans),
        ("scale", StandardScaler()),
        # probability=True required for ROC curve
        # significantly increases time
        ("svc", SVC(probability=True)),
    ]
)

svcPipeline.fit(X_train, y_train)

rfcPipeline = Pipeline(
    [
        ("ct", column_trans),
        ("scale", StandardScaler()),
        ("rfc", RandomForestClassifier()),
    ]
)

rfcPipeline.fit(X_train, y_train)

# Helper function for plotting roc curves
def generateROCCurve(fittedModel, X_test, y_test, label):
    plt.figure()
    y_pred_prob = fittedModel.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    sns.lineplot([0, 1], [0, 1]).lines[0].set_linestyle("--")
    sns.lineplot(
        fpr,
        tpr,
        label="{} (auc = {:.3f})".format(label, roc_auc_score(y_test, y_pred_prob)),
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("./diagrams/ROC_" + label)


generateROCCurve(svcPipeline, X_test, y_test, "Support Vector Classifier")
generateROCCurve(rfcPipeline, X_test, y_test, "Random Forest Classifier")

# Helper function to print info about model performance
def printModelInfo(fittedModel, X_test, y_test, name):
    y_pred = fittedModel.predict(X_test)
    print("Classification report for {}:\n".format(name))
    print(classification_report(y_test, y_pred))
    print("Confusion matrix for {}:\n".format(name))
    print(confusion_matrix(y_test, y_pred))
    print(
        "\nAccuracy of {} on test set: {:.4f}\n".format(
            name, fittedModel.score(X_test, y_test)
        )
    )
    plot_confusion_matrix(
        fittedModel, X_test, y_test, normalize="all", cmap="Blues",
    )
    # plt.colorbar().remove()
    plt.grid(False)
    plt.savefig("./diagrams/matrix" + name, dpi=300)


printModelInfo(svcPipeline, X_test, y_test, "Support Vector Classifier")
printModelInfo(rfcPipeline, X_test, y_test, "Random Forest Classifier")

### HYPERPARAMETER TUNING

# Set up grid for GridSearchCV
grid = {
    "svc__C": [0.01, 0.1, 1, 10, 100],
    "svc__gamma": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1],
    "svc__probability": [False],
}

# Perform hyperparameter searching for SVC
# svcCV = GridSearchCV(estimator=svcPipeline, param_grid=grid, cv=3, verbose=2, n_jobs=4)
# svcCV.fit(X_train, y_train)
# print(svcCV.best_params_)

# Hyperparameter tuning commented out for speed
# Optimal hyperparameters: gamma=0.01, C=10

# Set new hyperparameters
svcPipelineTuned = Pipeline(
    [
        ("ct", column_trans),
        ("scale", StandardScaler()),
        ("svc", SVC(probability=True, gamma=0.01, C=10)),
    ]
)

svcPipelineTuned.fit(X_train, y_train)

# Generate new curve and info
generateROCCurve(svcPipelineTuned, X_test, y_test, "Tuned Support Vector Classifier")
printModelInfo(svcPipelineTuned, X_test, y_test, "Tuned Support Vector Classifier")

# Set up grid for RandomizedSearchCV
random_grid = {
    "rfc__bootstrap": [True, False],
    "rfc__max_depth": [30, 40, 50, 60, 70, 80, 90, None],
    "rfc__max_features": ["auto", "sqrt"],
    "rfc__min_samples_leaf": [1, 2, 4],
    "rfc__min_samples_split": [2, 5, 10],
    "rfc__n_estimators": [400, 600, 800, 1000, 1200, 1400, 1600],
}

# Perform hyperparameter searching for SVC
# rfcCV = RandomizedSearchCV(
#    estimator=rfcPipeline,
#    param_distributions=random_grid,
#    n_iter=20,
#    cv=3,
#    verbose=2,
#    n_jobs=4,
#    scoring="recall",
# )
# rfcCV.fit(X_train, y_train)
# print(rfcCV.best_params_)

# Hyperparameter tuning commented out for speed
# Optimal hyperparameters: n_estimators=800, min_samples_split=2, min_samples_leaf=2, max_features="sqrt", max_depth=80, bootstrap=False

rfcPipelineTuned = Pipeline(
    [
        ("ct", column_trans),
        ("scale", StandardScaler()),
        (
            "rfc",
            RandomForestClassifier(
                n_estimators=800,
                min_samples_split=2,
                min_samples_leaf=2,
                max_features="sqrt",
                max_depth=80,
                bootstrap=False,
            ),
        ),
    ]
)

rfcPipelineTuned.fit(X_train, y_train)

generateROCCurve(rfcPipelineTuned, X_test, y_test, "Tuned Random Forest Classifier")
printModelInfo(rfcPipelineTuned, X_test, y_test, "Tuned Random Forest Classifier")

# Helper function to generate precision-recall-threshold graphs
def generatePRT(X_test, y_test):
    y_pred_prob = svcPipelineTuned.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    sns.lineplot(
        thresholds, precisions[:-1], label="Tuned SVC Precision", palette="muted"
    )
    sns.lineplot(thresholds, recalls[:-1], label="Tuned SVC Recall", palette="muted")
    y_pred_prob = rfcPipelineTuned.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)
    sns.lineplot(
        thresholds, precisions[:-1], label="Tuned RFC Precision", palette="muted"
    )
    sns.lineplot(thresholds, recalls[:-1], label="Tuned RFC Recall", palette="muted")
    plt.xlabel("Decision Threshold")
    plt.legend()


plt.figure()
generatePRT(X_test, y_test)
plt.savefig("./diagrams/prt", dpi=300)

# Compare ROC curves for all models
def generateROCComparison(models, X_test, y_test, labels):
    plt.figure(figsize=(8, 8))
    sns.lineplot([0, 1], [0, 1]).lines[0].set_linestyle("--")
    for i in range(len(models)):
        y_pred_prob = models[i].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        sns.lineplot(
            fpr,
            tpr,
            label="{} (auc = {:.3f})".format(
                labels[i], roc_auc_score(y_test, y_pred_prob)
            ),
        )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.savefig("./diagrams/ROCComparison", dpi=300)


generateROCComparison(
    [svcPipeline, svcPipelineTuned, rfcPipeline, rfcPipelineTuned],
    X_test,
    y_test,
    [
        "Support Vector Classifier",
        "Tuned Support Vector Classifier",
        "Random Forest Classifier",
        "Tuned Random Forest Classsifier",
    ],
)

# Generating feature importances from tuned rfc model

# imp = rfcPipelineTuned["rfc"].feature_importances_
# fig, ax = plt.subplots(figsize=(15, 5))
# sns.barplot([x for x in range(len(imp))], imp, ax=ax, palette="muted")
# plt.xticks(rotation=90)
# plt.figsize = (10, 7)
#  plt.xlabel("Feature Number")
# plt.ylabel("Importance")
# plt.savefig("rfc_importances.png", dpi=300)
