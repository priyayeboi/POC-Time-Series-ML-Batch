
import os
import time
import pandas as pd
import xgboost as xgb
import mlflow
from mlflow.models import infer_signature
import ray.data
import ray.train
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

print("SCRIPT STARTED", flush=True)

# MLflow server started by start_monitoring.sh
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
mlflow.set_experiment("dummy-ray-xgboost")

# Model and run settings; defined here so they can be logged to MLflow
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 5,
    "learning_rate": 0.1,
    "tree_method": "hist",
    "nthread": 2,
    "seed": 42,
}
NUM_BOOST_ROUND = 500
REPORT_EVERY = 10  # rounds between metric reports to Ray (each report has a fixed cost)
NUM_WORKERS = 2
CPUS_PER_WORKER = 2

#Generate synthetic data
X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        weights=[0.8, 0.2],
        random_state=42,
    )

feature_names = [f"feature_{i}" for i in range(20)]

df = pd.DataFrame(X, columns=feature_names)
df["label"] = y

train_dataset, test_dataset = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

# Start Ray before creating any Ray objects; shutting down afterwards would delete them
# Connects to the cluster from start_monitoring.sh (metrics go to Prometheus/Grafana);
# starts a temporary local cluster instead if none is running.
ray.init()

# Convert pandas DataFrames to Ray Datasets; Ray shards these across the workers
train_ds = ray.data.from_pandas(train_dataset)
eval_ds = ray.data.from_pandas(test_dataset)

def train_func():

    rank = ray.train.get_context().get_world_rank()

    print(
        f"TRAINING WORKER {rank} STARTED",
        flush=True
    )

    # Get this worker's train and eval shards
    train_shard = ray.train.get_dataset_shard("train")
    eval_shard = ray.train.get_dataset_shard("eval")

    # Convert shards to pandas DataFrames
    train_df = train_shard.materialize().to_pandas()
    eval_df = eval_shard.materialize().to_pandas()

    # 3. Separate features and labels
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_eval = eval_df.drop(columns=["label"])
    y_eval = eval_df["label"]

    # 4. Convert to XGBoost DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    # 5. Train model; the callback reports metrics and saves the model as a checkpoint
    xgb.train(
        params=PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (deval, "eval")],
        verbose_eval=20,
        callbacks=[RayTrainReportCallback(frequency=REPORT_EVERY)],
    )

scaling_config = ray.train.ScalingConfig(num_workers=NUM_WORKERS, resources_per_worker={"CPU": CPUS_PER_WORKER})

trainer = XGBoostTrainer(
    train_func,
    scaling_config=scaling_config,
    datasets={"train": train_ds, "eval": eval_ds},
    # If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)

with mlflow.start_run() as run:
    mlflow.log_params({
        **PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "num_workers": NUM_WORKERS,
        "cpus_per_worker": CPUS_PER_WORKER,
        "train_rows": len(train_dataset),
        "test_rows": len(test_dataset),
        "train_positive_rate": round(train_dataset["label"].mean(), 4),
    })

    start_time = time.time()
    result = trainer.fit()
    training_time = time.time() - start_time

    # Learning curves: one point per report from RayTrainReportCallback
    history = result.metrics_dataframe
    for i, row in enumerate(history.itertuples(index=False)):
        step = min((i + 1) * REPORT_EVERY, NUM_BOOST_ROUND)
        mlflow.log_metrics({"train_logloss": row[history.columns.get_loc("train-logloss")],
                            "eval_logloss": row[history.columns.get_loc("eval-logloss")]}, step=step)

    model = RayTrainReportCallback.get_model(result.checkpoint)  # xgb.Booster

    # 6. Evaluate on the full test set (not just one worker's shard)
    X_test = test_dataset.drop(columns=["label"])
    y_test = test_dataset["label"]
    probabilities = model.predict(xgb.DMatrix(X_test))
    predictions = (probabilities >= 0.5).astype(int)

    test_metrics = {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions),
        "test_recall": recall_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
        "test_pr_auc": average_precision_score(y_test, probabilities),
        "training_time_s": training_time,
    }
    mlflow.log_metrics(test_metrics)
    mlflow.set_tag("ray_result_path", result.path)

    print("\nModel Evaluation")
    print("-------------------------")
    print(f"Accuracy:  {test_metrics['test_accuracy']:.4f}")
    print(f"Precision: {test_metrics['test_precision']:.4f}")
    print(f"Recall:    {test_metrics['test_recall']:.4f}")
    print(f"F1 Score:  {test_metrics['test_f1']:.4f}")
    print(f"ROC AUC:   {test_metrics['test_roc_auc']:.4f}")
    print(f"PR AUC:    {test_metrics['test_pr_auc']:.4f}")

    print(f"\nTraining time: {training_time:.2f} seconds")

    # 7. Save model: to MLflow (versioned in the model registry) and locally
    mlflow.xgboost.log_model(
        model,
        name="model",
        signature=infer_signature(X_test, probabilities),
        input_example=X_test.head(5),
        registered_model_name="dummy-ray-xgboost",
    )
    model.save_model("xgboost_model.json")

    print(f"Model saved. MLflow run: {run.info.run_id}")
