# -*- coding: utf-8 -*-
# Basic libraries
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler

# Model selection and evaluation
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
    GridSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')
#Ensuring data was imported and displayed correctly
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

source_df = pd.read_csv('Datathon #3 Dataset - HAD7001.csv')

source_df['apache_3j_bodysystem'].unique()

cardio_patients_full = source_df[source_df['apache_3j_bodysystem'] == 'Cardiovascular']

targeted_columns = [
    'age',
    'gender',
    'weight',
    'pre_icu_los_days',
    'readmission_status',
    'ventilated_apache',
    'd1_heartrate_max',
    'd1_resprate_min',
    'd1_sysbp_max',
    'd1_hemaglobin_max',
    'd1_sodium_max',
    'd1_creatinine_max',
    'hepatic_failure',
    'diabetes_mellitus',
    'leukemia',
    'lymphoma',
    'solid_tumor_with_metastasis',
    'temp_apache',
    'd1_bun_max',
    'd1_glucose_max',
    'd1_wbc_max',
    'd1_hco3_max',
    'hospital_death'
    ]

categorical_columns = [
    'gender',
    'readmission_status',
    'ventilated_apache',
    'hepatic_failure',
    'diabetes_mellitus',
    'leukemia',
    'lymphoma',
    'solid_tumor_with_metastasis',
    'hospital_death' # the target y-label
]

numerical_columns = [
    'age',
    'weight',
    'pre_icu_los_days',
    'd1_heartrate_max',
    'd1_resprate_min',
    'd1_sysbp_max',
    'd1_hemaglobin_max',
    'd1_sodium_max',
    'd1_creatinine_max',
    'temp_apache',
    'd1_bun_max',
    'd1_glucose_max',
    'd1_wbc_max',
    'd1_hco3_max'
]

cardio_patients = cardio_patients_full[targeted_columns]
cardio_patients.head()

description = cardio_patients.describe()
description

# Checking for missing values
cardio_patients.isnull().sum()

# apply mode imputation after train/test split.

# inspect null value in target variable
print(cardio_patients.shape[0])
print(cardio_patients[cardio_patients['hospital_death'].isnull()].shape[0])

import matplotlib.pyplot as plt

data = cardio_patients

counts = data['hospital_death'].value_counts()

# Defining colors
colors = ['#87CEEB', '#7cb1c2']

# Ploting the pie chart
plt.figure(figsize=(8, 6))
plt.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title('Figure 2: Distribution of Classes in Hospital Death')
plt.legend(title='Hospital Death', loc='center left', bbox_to_anchor=(1, 0.5))

# Showing the plot
plt.show()

# Grouping variables into binary, ordinal, and categorical
binary_categorical = [item for item in categorical_columns if item != 'hospital_death']

#Setting Y/class

Class= 'hospital_death'

# Determining the number of subplots in each row
subplots_per_row = 3
total_rows = len(binary_categorical) // subplots_per_row + 1

# Creating bar plots
fig, axes = plt.subplots(total_rows, subplots_per_row, figsize=(18, 6 * total_rows))

for i, variable_name in enumerate(binary_categorical):
    row = i // subplots_per_row
    col = i % subplots_per_row
    sns.barplot(data=data, x=variable_name, y=Class, palette='coolwarm', ax=axes[row, col])
    axes[row, col].set_title(f"{variable_name} and {Class}")

# Adjusting the font size for unit names
    axes[row, col].set_xticklabels(axes[row, col].get_xticklabels(), fontsize=6.5)

# Adding and adjusting lables for both axes
for row in axes:
    for ax in row:
        ax.set_xlabel(ax.get_xlabel(), color='grey', fontsize=9.5)
        ax.set_ylabel(ax.get_ylabel(), color='grey', fontsize=9.5)

# Removing  empty subplots
for i in range(len(binary_categorical), total_rows * subplots_per_row):
    fig.delaxes(axes[i // subplots_per_row, i % subplots_per_row])

plt.tight_layout()
plt.show()

# categorical columns to drop
to_drop_categorical = [
    'readmission_status',
    'lymphoma',
]

for todo_remove in to_drop_categorical:
    categorical_columns.remove(todo_remove)

categorical_columns

Class = 'hospital_death'

# Determining the number of subplots in each row
subplots_per_row = 3
total_rows = len(numerical_columns) // subplots_per_row + 1

# Creating line plots
fig, axes = plt.subplots(total_rows, subplots_per_row, figsize=(18, 6 * total_rows))

for i, variable_name in enumerate(numerical_columns):
    row = i // subplots_per_row
    col = i % subplots_per_row
    sns.lineplot(data=data, x=Class, y=variable_name, palette='coolwarm', ax=axes[row, col])
    axes[row, col].set_title(f"{variable_name} and {Class}")

# Adding and adjusting labels for both axes
for row in axes:
    for ax in row:
        ax.set_xlabel(ax.get_xlabel(), color='grey', fontsize=9.5)
        ax.set_ylabel(ax.get_ylabel(), color='grey', fontsize=9.5)

# Removing empty subplots
for i in range(len(numerical_columns), total_rows * subplots_per_row):
    fig.delaxes(axes[i // subplots_per_row, i % subplots_per_row])

plt.tight_layout()
plt.show()

# Creating a DataFrame with only the selected columns
numeric_data = data[numerical_columns]

# Calculating the correlation matrix
correlation_matrix = numeric_data.corr()

# Creating a heatmap
plt.figure(figsize=(12, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numeric Columns')
plt.show()

high_correlation_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.65:
            pair = (correlation_matrix.columns[i], correlation_matrix.columns[j])
            high_correlation_pairs.append(pair)

print("Pairs of variables with correlation > 0.65:")
for pair in high_correlation_pairs:
    print(pair)

# correlation remove
numeric_variables_to_drop = [
    'd1_creatinine_max',
]

for todo_remove in numeric_variables_to_drop:
    numerical_columns.remove(todo_remove)

numerical_columns

# Determining the number of subplots in each row
subplots_per_row = 3
total_rows = len(numerical_columns) // subplots_per_row + 1

# Creating a grid of subplots
fig, axs = plt.subplots(total_rows, subplots_per_row, figsize=(18, 6 * total_rows))

# Creating a copy of the original data
data_copy = data.copy()

# Looping through each feature in numeric variables
for i, column in enumerate(numerical_columns):
    # Calculate the position for the subplot; x is the row index and y is the column index
    x = i // subplots_per_row
    y = i % subplots_per_row

    # Plot a histogram of the scaled feature split by the Diabetes class
    sns.histplot(data=data_copy, x=column, hue='hospital_death', kde=True, palette='rocket', ax=axs[x][y])

# Removing any empty subplots
for i in range(len(numerical_columns), total_rows * subplots_per_row):
    fig.delaxes(axs[i // subplots_per_row, i % subplots_per_row])

# Display all the subplots
plt.tight_layout()
plt.show()

# drop missing values
data.dropna(inplace=True)

# Defining a Z-score threshold
zscore_threshold = 4

# Initialize a dictionary to store variable names and their respective outlier counts
variables_with_outliers = {}

# Iterate through each continuous variable and identify variables with outliers
for variable_name in numerical_columns:
    data_column = data[variable_name]
    z_scores = (data_column - data_column.mean()) / data_column.std()
    outliers_count = sum(abs(z_scores) > zscore_threshold)

    # Store the variable name and outlier count in the dictionary
    variables_with_outliers[variable_name] = outliers_count

# Printing the names of variables with their respective outlier counts
print("Variables with the number of outliers beyond 4 standard deviations:")
total = 0
for variable_name, outliers_count in variables_with_outliers.items():
    print(f"{variable_name}: {outliers_count} observations")
    total += outliers_count

print("Total: ", total)

# Drop outliers
# Compute Z-scores for all numerical columns
z_scores = (data[numerical_columns] - data[numerical_columns].mean()) / data[numerical_columns].std()

# Keep only rows where all Z-scores are within the threshold
data_no_outliers = data[(np.abs(z_scores) <= zscore_threshold).all(axis=1)]

print(f"\nOriginal data shape: {data.shape}")
print(f"Data shape after outlier removal: {data_no_outliers.shape}")

data = data_no_outliers

# apply log transform to skewed columns
must_log_transform = [
    # 'pre_icu_los_days', # what does negative columns mean here??
    'd1_bun_max',
    'd1_glucose_max',
    'd1_wbc_max'
]

for column in must_log_transform:
    # Apply log transformation directly
    data[f'log_{column.lower().replace(" ", "_")}'] = np.log(data[column])

cardio_patients = data

# 1. drop missing values:
cardio_patients.dropna(inplace=True)

# 2. mode imputation [before train/test split]
# Apply mode imputation for each column with missing values
# for column in cardio_patients.columns:
#     if cardio_patients[column].isnull().sum() > 0:
#         mode_value = cardio_patients[column].mode()[0]
#         cardio_patients[column].fillna(mode_value, inplace=True)

# Check that missing values are handled
print(cardio_patients.isnull().sum())

categorical_columns_without_gender = [col for col in categorical_columns if col != 'gender']
cardio_patients[categorical_columns_without_gender] = cardio_patients[categorical_columns_without_gender].astype(int)
cardio_patients[numerical_columns] = cardio_patients[numerical_columns].astype(float)

cardio_patients = cardio_patients.drop(columns=must_log_transform, axis=1)

cardio_patients.dtypes

# RUS - faster training time, less performance

X_S = cardio_patients.drop('hospital_death', axis=1)
y_S = cardio_patients['hospital_death']

# Split the data into training and test sets
X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(X_S, y_S, test_size=0.2, random_state=42)

# Application of RUS for downsampling on the training data
rus = RandomUnderSampler()
X_train_resampled_S, y_train_resampled_S = rus.fit_resample(X_train_S, y_train_S)

ct = ColumnTransformer(
    transformers=[
        ('one_hot_encoder', OneHotEncoder(), ['gender']),
    ],
    remainder='passthrough'
)

# Applying the one-hot encoding transformation to X_resampled, X_test, X_External
X_train_resampled_encoded_S = ct.fit_transform(X_train_resampled_S)
X_test_S = ct.transform(X_test_S)

# ROS - slow training time, better performance.
from imblearn.over_sampling import RandomOverSampler

X_S = cardio_patients.drop('hospital_death', axis=1)
y_S = cardio_patients['hospital_death']

# Split the data into training and test sets
X_train_S, X_test_S, y_train_S, y_test_S = train_test_split(X_S, y_S, test_size=0.2, random_state=42)

# Apply RandomOverSampler for upsampling on the training data
ros = RandomOverSampler(random_state=42)
X_train_resampled_S, y_train_resampled_S = ros.fit_resample(X_train_S, y_train_S)

ct = ColumnTransformer(
    transformers=[
        ('one_hot_encoder', OneHotEncoder(), ['gender']),
    ],
    remainder='passthrough'
)

# Apply the one-hot encoding transformation to resampled training data and test data
X_train_resampled_encoded_S = ct.fit_transform(X_train_resampled_S, y_train_resampled_S)
X_test_S = ct.transform(X_test_S)

scaler = StandardScaler()
X_train_resampled_encoded_scaled_S = scaler.fit_transform(X_train_resampled_encoded_S)
X_test_scaled = scaler.transform(X_test_S)

# Initialize an LogisticRegression regressor
lr = LogisticRegression()

# Creating a pipeline that applies the LogisticRegression regressor
model_for_LogisticRegression = Pipeline([
    ('lr', lr)
])

# Define a dictionary of hyperparameter values to search over.
param_dist_LogisticRegression = {
    'lr__solver': ['sag'], # ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
    'lr__C': [0.2], # [i/10 for i in range(1,11)],
    'lr__penalty': ['l2'] ,# ['l1', 'l2', 'elasticnet', None],
}
# Setting up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)

# Initialize Grid Search with the model and the hyperparameters to search
random_search_LogisticRegression = GridSearchCV(
    model_for_LogisticRegression, param_grid=param_dist_LogisticRegression, cv=skf)

# Train Grid Search on the training data
random_search_LogisticRegression.fit(X_train_resampled_encoded_S, y_train_resampled_S)

# Extract results into a DataFrame
results_LogisticRegression = pd.DataFrame(random_search_LogisticRegression.cv_results_)

# Display the set of parameters that achieved the best score
results_LogisticRegression[results_LogisticRegression['rank_test_score'] == 1]
random_search_LogisticRegression.best_params_

best_config = random_search_LogisticRegression.best_params_
model_for_LogisticRegression = model_for_LogisticRegression.set_params(
    lr__C=best_config["lr__C"],
    lr__penalty=best_config["lr__penalty"],
    lr__solver=best_config["lr__solver"],
)

# Refit the model using the training data
LR_model = model_for_LogisticRegression.fit(X_train_resampled_encoded_S, y_train_resampled_S)

def evaluate_model_with_cross_val(model, X, y):
    """
    Evaluates a classification model using 5-fold cross-validation and prints performance metrics.

    Parameters:
    - model: The classification model to evaluate.
    - X: Feature set.
    - y: Target labels.

    Returns:
    - results: Dictionary containing all calculated metrics.
    """
    # Cross-validated predictions with fixed 5 folds
    predicted_labels = cross_val_predict(model, X, y, cv=5)

    # Metrics
    conf_matrix = confusion_matrix(y, predicted_labels)
    class_report = classification_report(y, predicted_labels)
    accuracy = accuracy_score(y, predicted_labels)
    precision = precision_score(y, predicted_labels)
    recall = recall_score(y, predicted_labels)
    f1 = f1_score(y, predicted_labels)

    # Print results
    print("Model Evaluation Results")
    print("A. Confusion Matrix:")
    print(conf_matrix)
    print("B. Classification Report:")
    print(class_report)
    print("C. Accuracy: {:.2f}".format(accuracy))
    print("D. Precision: {:.2f}".format(precision))
    print("E. Recall (Sensitivity): {:.2f}".format(recall))
    print("F. F1 Score: {:.2f}".format(f1))

    # Optionally return metrics
    results = {
        "confusion_matrix": conf_matrix,
        "classification_report": class_report,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return results

train_results_LR = evaluate_model_with_cross_val(
    LR_model,
    X_train_resampled_encoded_S,
    y_train_resampled_S,
)

sns.heatmap(train_results_LR["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Logistic Regression Model - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

def draw_roc(model, x , y, set_title, model_name):
  predicted_probs = model.predict_proba(x)[:, 1]
  # Calculate ROC-AUC on the test set
  roc_auc_test = roc_auc_score(y, predicted_probs)
  print(f"{set_title} ROC-AUC: {roc_auc_test:.4f}")
  # Calculate ROC curve on the test set
  fpr_test, tpr_test, _ = roc_curve(y, predicted_probs)
  # Plot ROC curve for the test set
  plt.figure(figsize=(8, 6))
  plt.plot(fpr_test, tpr_test, label=f'{set_title} ROC Curve (AUC = {roc_auc_test:.2f})')
  plt.plot([0, 1], [0, 1], 'k--', lw=2)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'{set_title} Set - {model_name} - (ROC) Curve')
  plt.legend(loc='lower right')
  plt.show()

  return roc_auc_test, fpr_test, tpr_test

def performance_metrics(model, x, y, set_title, model_name):
  y_pred_test = model.predict(x)
  # Calculating metrics
  accuracy_test = accuracy_score(y, y_pred_test)
  precision_test = precision_score(y, y_pred_test)
  recall_test = recall_score(y, y_pred_test)
  f1_test = f1_score(y, y_pred_test)
  conf_matrix_test = confusion_matrix(y, y_pred_test)

  # Printing the evaluation metrics
  print(f"{model_name} Model - {set_title} Set- Confusion Matrix/Classification Report")
  print("Accuracy: {:.2f}".format(accuracy_test))
  print("Precision: {:.2f}".format(precision_test))
  print("Recall (Sensitivity): {:.2f}".format(recall_test))
  print("F1 Score: {:.2f}".format(f1_test))
  print("Confusion Matrix:")
  print(conf_matrix_test)

  return accuracy_test, precision_test, recall_test, f1_test

roc_auc_train_LR, fpr_train_LR, tpr_train_LR = draw_roc(LR_model, X_train_resampled_encoded_S, y_train_resampled_S, 'Train', 'Logistic Regression Model')

test_results_LR = evaluate_model_with_cross_val(
    LR_model,
    X_test_scaled,
    y_test_S,
)

sns.heatmap(test_results_LR["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Logistic Regression Model - Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_test_LR, fpr_test_LR, tpr_test_LR = draw_roc(LR_model, X_test_scaled, y_test_S, 'Test', 'Logistic Regression Model')

"""KNN"""

#Creating a list to store the cross-validation scores for different values of k
k_grid = list(range(1, 30))  # Try k values from 1 to 20, for example
cv_scores = []

for k in k_grid:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    scores = cross_val_score(knn, X_train_resampled_encoded_scaled_S, y_train_resampled_S, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

#Plotting the cross-validation scores to identify the optimal K
plt.plot(k_grid, cv_scores, marker = "s")
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.title('KNN: Cross-Validation Accuracy vs. Number of Neighbors')
plt.xticks(np.arange(min(k_grid), max(k_grid) + 1, 1))
plt.show()

# Print K and the corresponding cross-validation scores
for k, score in zip(k_grid, cv_scores):
    print(f"K = {k}, Cross-Validation Accuracy = {score:.3f}")

# Grid search
# Initialize an LogisticRegression regressor

knn = KNeighborsClassifier(n_neighbors=9)

# Creating a pipeline that applies the LogisticRegression regressor
model_for_KNN = Pipeline([
    ('knn', knn)
])

# Define a dictionary of hyperparameter values to search over.
param_dist_KNN = {
    'knn__metric': ['cosine'], # ['cityblock', 'manhattan', 'cosine', 'euclidean', 'haversine', 'nan_euclidean', 'l1', 'l2'],
    'knn__algorithm': ['auto'], # ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'knn__weights': ['uniform'], # ['uniform', 'distance'],
}
# Setting up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)

# Initialize Grid Search with the model and the hyperparameters to search
random_search_KNN = GridSearchCV(
    model_for_KNN, param_grid=param_dist_KNN, cv=skf)

# Train Grid Search on the training data
random_search_KNN.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

# Extract results into a DataFrame
results_KNN = pd.DataFrame(random_search_KNN.cv_results_)

# Display the set of parameters that achieved the best score
results_KNN[results_KNN['rank_test_score'] == 1]
random_search_KNN.best_params_

# select one of the best configs

best_config = random_search_KNN.best_params_
model_for_KNN = model_for_KNN.set_params(
    knn__metric=best_config["knn__metric"],
    knn__algorithm=best_config["knn__algorithm"],
    knn__weights=best_config["knn__weights"],
)

# Refit the model using the training data
knn = model_for_KNN.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

train_results_knn = evaluate_model_with_cross_val(
    knn,
    X_train_resampled_encoded_S,
    y_train_resampled_S,
)

sns.heatmap(train_results_knn["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('KNN Model - Confusion Matrix (Train)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_train_knn, fpr_train_knn, tpr_train_knn = draw_roc(knn, X_train_resampled_encoded_S, y_train_resampled_S, 'Train', 'KNN Model')

_X = X_test_scaled
_Y = y_test_S
model = knn

test_results_knn = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(test_results_knn["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('KNN Model - Confusion Matrix (Test)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_test_knn, fpr_test_knn, tpr_test_knn = draw_roc(model, _X, _Y, 'Test', 'KNN Model')

# Initialize an SVM classifier
SVM = SVC(class_weight=None, probability=True)


# Creating a pipeline that applies the Decision Tree classifier
model_for_SVM = Pipeline([
    ('svm', SVM)
])

# Define a dictionary of hyperparameter values to search over.
param_dist_SVM = {
    'svm__C': [10], # [10**(i) for i in range(-2,2)], # [10**(i) for i in range(-3,4)],
    'svm__kernel': ['rbf'], # ['rbf', 'linear', 'poly', 'sigmoid'],
    'svm__gamma': [0.1], #[10**(i) for i in range(-3,1)] # [10**(i) for i in range(-6,2)],
}
# Setting up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)

# Initialize Grid Search with the model and the hyperparameters to search
random_search_SVM = GridSearchCV(
    model_for_SVM, param_grid=param_dist_SVM, cv=skf)

# Train Grid Search on the training data
random_search_SVM.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

# Extract results into a DataFrame
results_SVM = pd.DataFrame(random_search_SVM.cv_results_)

# Display the set of parameters that achieved the best score
results_SVM[results_SVM['rank_test_score'] == 1]
print("Best Config: ", random_search_SVM.best_params_)

best_config = random_search_SVM.best_params_

model_for_SVM = model_for_SVM.set_params(
    svm__C=best_config["svm__C"],
    svm__gamma=best_config["svm__gamma"],
    svm__kernel=best_config["svm__kernel"],
)
# Refit the model using the training data
model_for_SVM.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

_X = X_train_resampled_encoded_scaled_S
_Y = y_train_resampled_S
model = model_for_SVM

train_results_svm = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(train_results_svm["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('SVM Model - Confusion Matrix (Train)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_train_svm, fpr_train_svm, tpr_train_svm = draw_roc(model, _X, _Y, 'Train', 'SVM Model')

_X = X_test_scaled
_Y = y_test_S
model = model_for_SVM

test_results_svm = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(test_results_svm["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('SVM Model - Confusion Matrix (SVM)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_test_svm, fpr_test_svm, tpr_test_svm = draw_roc(model, _X, _Y, 'Test', 'SVM Model')

# Initializing a Decision Tree classifier
#we already have balanced class through the previous undersampling step
#thus, class weight will be set to none instead of balanced
dt = DecisionTreeClassifier(criterion='entropy', class_weight=None)

# Creating a pipeline that applies the Decision Tree classifier
model_for_dt = Pipeline([
    ('DT', dt)
])

# Fitting the model on the training data
model_for_dt.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

# Define the hyperparameter values that should be tested
param_dist_DT = {
    "DT__max_depth" : [7], #[3, 5, 7, 10, 15, 20],
    "DT__min_samples_leaf" : [20], # [3, 5, 7, 10, 15, 20],
    "DT__splitter" : ['best'], # ['best', 'random'],
    "DT__max_features" : ['sqrt'], #['sqrt', 'log2']
}

# Setting up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)

# Initialize Grid Search with the model and the hyperparameters to search
random_search_DT = GridSearchCV(
    model_for_dt, param_grid=param_dist_DT, cv=skf)

# Train Grid Search on the training data
random_search_DT.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

# Extract results into a DataFrame
results_DT = pd.DataFrame(random_search_DT.cv_results_)

# Display the set of parameters that achieved the best score
results_DT[results_DT['rank_test_score'] == 1]
# Retrieve the best hyperparameters after Grid Search
print(random_search_DT.best_params_)

# Update the model's hyperparameters to the best found during Grid Search
best_config = random_search_DT.best_params_
model_for_dt = model_for_dt.set_params(
    DT__max_depth=best_config["DT__max_depth"],
    DT__max_features=best_config["DT__max_features"],
    DT__min_samples_leaf=best_config["DT__min_samples_leaf"],
    DT__splitter=best_config["DT__splitter"],
)
# Refit the model using the training data
model_for_dt = model_for_dt.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

_X = X_train_resampled_encoded_scaled_S
_Y = y_train_resampled_S
model = model_for_dt

train_results_dt = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(train_results_dt["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Decision Tree Model - Confusion Matrix (Train)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_train_dt, fpr_train_dt, tpr_train_dt = draw_roc(model, _X, _Y, 'Train', 'Decision Tree Model')

_X = X_test_scaled
_Y = y_test_S

test_results_dt = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(test_results_dt["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Decision Tree Model - Confusion Matrix (Test)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_test_dt, fpr_test_dt, tpr_test_dt = draw_roc(model, _X, _Y, 'Test', 'Decision Tree Model')

# Initializing a Random Forest classifier
#we already have balanced class through the previous undersampling step
#thus, class weight will be set to none instead of balanced
rf = RandomForestClassifier(criterion='entropy', class_weight=None)

# Creating a pipeline that applies the Random Forest
model_for_rf = Pipeline([
    ('random_forest', rf)
])

# Define the hyperparameter values that should be tested

param_dist_RF = {
    "random_forest__n_estimators" : [200], # [100, 150, 200],
    "random_forest__max_depth" : [10], # [3, 5, 7, 10],
    "random_forest__min_samples_leaf" : [5], # [3, 5, 7, 10],
    "random_forest__max_features" : ['log2'], # ['sqrt', 'log2']
}

# Setting up Stratified K-Fold cross-validation
skf = StratifiedKFold(n_splits=5)

# Initialize Grid Search with the model and the hyperparameters to search
random_search_RF = GridSearchCV(
    model_for_rf,
    param_grid=param_dist_RF,
    cv=skf,
    # verbose=1
    )

# Train Grid Search on the training data
random_search_RF.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

# Extract results into a DataFrame
results_RF = pd.DataFrame(random_search_RF.cv_results_)

# Display the set of parameters that achieved the best score
results_RF[results_RF['rank_test_score'] == 1]
# Retrieve the best hyperparameters after Grid Search for RF
print(random_search_RF.best_params_)

# Update the model's hyperparameters to the best found during Grid Search
best_config = random_search_RF.best_params_
model_for_rf = model_for_rf.set_params(
    random_forest__max_depth=best_config["random_forest__max_depth"],
    random_forest__max_features=best_config["random_forest__max_features"],
    random_forest__min_samples_leaf=best_config["random_forest__min_samples_leaf"],
    random_forest__n_estimators=best_config["random_forest__n_estimators"],
)

# Refit the model using the training data
model_for_rf = model_for_rf.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

_X = X_train_resampled_encoded_scaled_S
_Y = y_train_resampled_S
model = model_for_rf

train_results_rf = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(train_results_rf["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Random Forest Model - Confusion Matrix (Train)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_train_rf, fpr_train_rf, tpr_train_rf = draw_roc(model, _X, _Y, 'Train', 'Random Forest Model')

_X = X_test_scaled
_Y = y_test_S

test_results_rf = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(test_results_rf["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('Random Forest Model - Confusion Matrix (Test)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_test_rf, fpr_test_rf, tpr_test_rf = draw_roc(model, _X, _Y, 'Test', 'Random Forest Model')

# Initializing the Gradient Boosting model with initial parameters
xgb = GradientBoostingClassifier()

# Creating a pipeline that first applies the column transformations and then runs the Gradient Boosting model
model_for_xgb = Pipeline([
    ('xg_boost', xgb)
])

# Fitting the model on the training data
model_for_xgb.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

### Hyperparameter tuning using GridSearchCV

param_dist_xgb = {
    "xg_boost__n_estimators" : [150], # [50, 100, 150, 200],
    "xg_boost__max_depth" : [3], # [3, 5, 7, 9],
    "xg_boost__min_samples_leaf" : [3], # [3, 5, 7, 10],
}

# Using StratifiedKFold for cross-validation, ensuring each fold has the same proportion of observations with each target value
skf = StratifiedKFold(n_splits=5)

# Setting up the GridSearchCV to find the best hyperparameters for the Gradient Boosting model
random_search_xgb = GridSearchCV(
    model_for_xgb, param_grid=param_dist_xgb, cv=skf
)

# Fitting the GridSearchCV on the training data
random_search_xgb.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

# Extract results into a DataFrame
results_xgb = pd.DataFrame(random_search_xgb.cv_results_)

# Display the set of parameters that achieved the best score
results_xgb[results_xgb['rank_test_score'] == 1]
# Retrieve the best hyperparameters after Grid Search for RF
print(random_search_xgb.best_params_)

# Update the model's hyperparameters to the best found during Grid Search
best_config = random_search_xgb.best_params_
model_for_xgb = model_for_xgb.set_params(
    xg_boost__n_estimators=best_config["xg_boost__n_estimators"],
    xg_boost__max_depth=best_config["xg_boost__max_depth"],
    xg_boost__min_samples_leaf=best_config["xg_boost__min_samples_leaf"],
)

# Refit the model using the training data
model_for_xgb = model_for_xgb.fit(X_train_resampled_encoded_scaled_S, y_train_resampled_S)

_X = X_train_resampled_encoded_scaled_S
_Y = y_train_resampled_S
model = model_for_xgb

train_results_xgb = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(train_results_xgb["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('XGBoost Model - Confusion Matrix (Train)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_train_xgb, fpr_train_xgb, tpr_train_xgb = draw_roc(model, _X, _Y, 'Train', 'XGBoost Model')

_X = X_test_scaled
_Y = y_test_S

test_results_xgb = evaluate_model_with_cross_val(
    model,
    _X,
    _Y,
)

sns.heatmap(test_results_xgb["confusion_matrix"], annot=True, fmt='d', cmap='YlGnBu')
plt.title('XGBoost Model - Confusion Matrix (Test)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

roc_auc_test_xgb, fpr_test_xgb, tpr_test_xgb = draw_roc(model, _X, _Y, 'Test', 'XGBoost Model')

# setting evaluations
# TODO add external validation

evaluations = {
    'Model': [
        'LR Training',
        'LR Test',
        'KNN Training',
        'KNN Test',
        'SVM Training',
        'SVM Test',
        'DT Training',
        'DT Test',
        'RF Training',
        'RF Test',
        'XGB train',
        'XGB test',
    ],
    'Accuracy': [
        train_results_LR["accuracy"],
        test_results_LR["accuracy"],
        train_results_knn["accuracy"],
        test_results_knn["accuracy"],
        train_results_svm["accuracy"],
        test_results_svm["accuracy"],
        train_results_dt["accuracy"],
        test_results_dt["accuracy"],
        train_results_rf["accuracy"],
        test_results_rf["accuracy"],
        train_results_xgb["accuracy"],
        test_results_xgb["accuracy"],
    ],
    'Precision': [
        train_results_LR["precision"],
        test_results_LR["precision"],
        train_results_knn["precision"],
        test_results_knn["precision"],
        train_results_svm["precision"],
        test_results_svm["precision"],
        train_results_dt["precision"],
        test_results_dt["precision"],
        train_results_rf["precision"],
        test_results_rf["precision"],
        train_results_xgb["precision"],
        test_results_xgb["precision"],
    ],
    'Recall (Sensitivity)': [
        train_results_LR["recall"],
        test_results_LR["recall"],
        train_results_knn["recall"],
        test_results_knn["recall"],
        train_results_svm["recall"],
        test_results_svm["recall"],
        train_results_dt["recall"],
        test_results_dt["recall"],
        train_results_rf["recall"],
        test_results_rf["recall"],
        train_results_xgb["recall"],
        test_results_xgb["recall"],
    ],
    'F1 Score': [
        train_results_LR["f1_score"],
        test_results_LR["f1_score"],
        train_results_knn["f1_score"],
        test_results_knn["f1_score"],
        train_results_svm["f1_score"],
        test_results_svm["f1_score"],
        train_results_dt["f1_score"],
        test_results_dt["f1_score"],
        train_results_rf["f1_score"],
        test_results_rf["f1_score"],
        train_results_xgb["f1_score"],
        test_results_xgb["f1_score"],
    ]
}

# Creating a DataFrame
results_summary = pd.DataFrame(evaluations)
results_summary.head(n=len(results_summary))

# Creating a figure
plt.figure(figsize=(8, 5))

# Plotting the ROC curves for training sets (all models)
plt.title('Summary: ROC Curves for Training Sets')
plt.plot(fpr_train_LR, tpr_train_LR, label=f'LR (AUC = {roc_auc_train_LR:.2f})')
plt.plot(fpr_train_knn, tpr_train_knn, label=f'KNN (AUC = {roc_auc_train_knn:.2f})')
plt.plot(fpr_train_svm, tpr_train_svm, label=f'SVM (AUC = {roc_auc_train_svm:.2f})')
plt.plot(fpr_train_dt, tpr_train_dt, label=f'DT (AUC = {roc_auc_train_dt:.2f})')
plt.plot(fpr_train_rf, tpr_train_rf, label=f'RF (AUC = {roc_auc_train_rf:.2f})')
plt.plot(fpr_train_xgb, tpr_train_xgb, label=f'XGBoost (AUC = {roc_auc_train_xgb:.2f})')
# Plotting the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Display the graph
plt.show()

# Creating a figure
plt.figure(figsize=(8, 5))

# Plotting the ROC curves for training sets (all models)
plt.title('Summary: ROC Curves for Test Sets')
plt.plot(fpr_test_LR, tpr_test_LR, label=f'LR (AUC = {roc_auc_test_LR:.2f})')
plt.plot(fpr_test_knn, tpr_test_knn, label=f'KNN (AUC = {roc_auc_test_knn:.2f})')
plt.plot(fpr_test_svm, tpr_test_svm, label=f'SVM (AUC = {roc_auc_test_svm:.2f})')
plt.plot(fpr_test_dt, tpr_test_dt, label=f'DT (AUC = {roc_auc_test_dt:.2f})')
plt.plot(fpr_test_rf, tpr_test_rf, label=f'RF (AUC = {roc_auc_test_rf:.2f})')
plt.plot(fpr_test_xgb, tpr_test_xgb, label=f'XGBoost (AUC = {roc_auc_test_xgb:.2f})')
# Plotting the diagonal line (random classifier)
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')

# Display the graph
plt.show()

