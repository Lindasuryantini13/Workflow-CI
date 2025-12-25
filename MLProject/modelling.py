import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import argparse

def main(data_path, experiment_name):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }

    mlflow.log_params(params)

    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        'test_accuracy': accuracy_score(y_test, y_pred_test),
        'test_precision': precision_score(y_test, y_pred_test),
        'test_recall': recall_score(y_test, y_pred_test),
        'test_f1': f1_score(y_test, y_pred_test),
        'test_roc_auc': roc_auc_score(y_test, y_pred_proba_test)
    }

    mlflow.log_metrics(metrics)

    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    mlflow.sklearn.log_model(model, "model")

    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='preprocessed_data.csv')
    parser.add_argument('--experiment', type=str, default='telco-churn-classification')
    args = parser.parse_args()

    main(data_path=args.data, experiment_name=args.experiment)
