import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# 1. Load data
train_data = pd.read_csv("/kaggle/input/client-churn-rate/train.csv")
test_data = pd.read_csv("/kaggle/input/client-churn-rate/test.csv")

# 2. Data preprocessing
cat_columns = train_data.select_dtypes(include=["object"]).columns

# Create a new feature if the required columns exist
if "Balance" in train_data.columns and "EstimatedSalary" in train_data.columns:
    train_data["BalanceSalaryRatio"] = train_data["Balance"] / (train_data["EstimatedSalary"] + 1)
    test_data["BalanceSalaryRatio"] = test_data["Balance"] / (test_data["EstimatedSalary"] + 1)

# Define columns to exclude from the feature set
drop_cols = ["Exited"]
if "id" in train_data.columns:
    drop_cols.insert(0, "id")

X = train_data.drop(columns=drop_cols)
y = train_data["Exited"]

# Define categorical features - CatBoost requires column indices
cat_features_names = [col for col in cat_columns if col in X.columns]
cat_features_indices = [list(X.columns).index(col) for col in cat_features_names]

# 3. Split data for RandomizedSearchCV with eval_set
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define initial CatBoost model with early stopping and GPU usage
cb_clf = CatBoostClassifier(
    loss_function="Logloss",
    iterations=200,
    task_type="GPU",
    devices='0',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=30
)

# 5. Define hyperparameter search space
param_dist = {
    "learning_rate": [0.01, 0.05, 0.1],  # Learning rate
    "depth": [6, 8, 10],  # Tree depth
    "l2_leaf_reg": [1, 3, 5],  # L2 regularization to prevent overfitting
    "bagging_temperature": [0, 1, 5]  # Bagging temperature to stabilize model learning
}

# Hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=cb_clf,
    param_distributions=param_dist,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    n_jobs=1,  # Using a single thread for optimal GPU performance
    verbose=2,
    random_state=42
)

# 6. Run RandomizedSearchCV, passing cat_features and eval_set for early stopping
random_search.fit(
    X_train, y_train,
    cat_features=cat_features_indices,
    eval_set=(X_val, y_val)
)

print("\nBest hyperparameters:", random_search.best_params_)
print("Best ROC AUC from cross-validation:", random_search.best_score_)

# 7. Train the final model using the best hyperparameters
best_params = random_search.best_params_

final_model = CatBoostClassifier(
    loss_function="Logloss",
    iterations=200,
    task_type="GPU",
    devices='0',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=30,
    **best_params
)

# Split data for final training with early stopping control
X_full_train, X_early, y_full_train, y_early = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Train the final model
final_model.fit(
    X_full_train, y_full_train,
    eval_set=(X_early, y_early),
    cat_features=cat_features_indices,
    early_stopping_rounds=30
)

# 8. Create wrapper for proper categorical feature handling in cross-validation
class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None
        
    def fit(self, X, y):
        self.model = CatBoostClassifier(**self.params)
        self.model.fit(
            X, y,
            cat_features=cat_features_indices,
            verbose=0,
            early_stopping_rounds=30
        )
        # Save class labels for proper scoring compatibility
        self.classes_ = self.model.classes_
        return self
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def decision_function(self, X):
        # For ROC AUC, return probability of the positive class
        return self.model.predict_proba(X)[:, 1]

# 9. Perform 10-fold cross-validation using the wrapper
wrapped_model = CatBoostWrapper(
    loss_function="Logloss",
    iterations=200,
    task_type="GPU",
    devices='0',
    random_seed=42,
    **best_params
)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    wrapped_model,
    X, y,
    cv=cv,
    scoring='roc_auc',
    n_jobs=1,
    verbose=2
)

print("\nCross-validation ROC AUC results:")
print("Mean = {:.4f}, Standard deviation = {:.4f}".format(
    np.mean(cv_scores), np.std(cv_scores)
))

# 10. Generate predictions for the test set and save results
if "id" in test_data.columns:
    test_ids = test_data["id"]
    X_test = test_data.drop("id", axis=1)
else:
    test_ids = np.arange(len(test_data))
    X_test = test_data.copy()

test_preds = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "id": test_ids,
    "Exited": test_preds
})

submission.to_csv("submission.csv", index=False)
print("\nFile 'submission.csv' has been successfully created!")
