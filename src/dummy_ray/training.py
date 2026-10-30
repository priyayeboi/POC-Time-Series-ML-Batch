
import time
import pandas as pd
import xgboost as xgb
import ray.data
import ray.train
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

print("SCRIPT STARTED", flush=True)

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

    # 5. Configure model
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 5,
        "learning_rate": 0.1,
        "tree_method": "hist",
        "nthread": 2,
        "seed": 42,
    }

    # 6. Train model; the callback reports metrics and saves the model as a checkpoint
    xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=500,
        evals=[(dtrain, "train"), (deval, "eval")],
        verbose_eval=20,
        callbacks=[RayTrainReportCallback()],
    )

scaling_config = ray.train.ScalingConfig(num_workers=2, resources_per_worker={"CPU": 2})

trainer = XGBoostTrainer(
    train_func,
    scaling_config=scaling_config,
    datasets={"train": train_ds, "eval": eval_ds},
    # If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)

start_time = time.time()
result = trainer.fit()
training_time = time.time() - start_time

model = RayTrainReportCallback.get_model(result.checkpoint)  # xgb.Booster

# Evaluate on the full test set (not just one worker's shard)
X_test = test_dataset.drop(columns=["label"])
y_test = test_dataset["label"]
probabilities = model.predict(xgb.DMatrix(X_test))

# 7. Evaluate
predictions = (probabilities >= 0.5).astype(int)

print("\nModel Evaluation")
print("-------------------------")
print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
print(f"Precision: {precision_score(y_test, predictions):.4f}")
print(f"Recall:    {recall_score(y_test, predictions):.4f}")
print(f"F1 Score:  {f1_score(y_test, predictions):.4f}")

print(f"\nTraining time: {training_time:.2f} seconds")

# 8. Save model
model.save_model("xgboost_model.json")

print("Model saved successfully.")
