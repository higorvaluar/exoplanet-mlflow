import argparse
import os
import random
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from torch import nn

DATASET = "data/exoplanets.csv"
FEATURES = ["pl_orbper", "pl_eqt", "st_teff", "st_rad", "st_mass"]
TARGET = "pl_rade"
RANDOM_STATE = 42
PATIENCE = 50
MIN_DELTA = 1e-6
DEVICE = torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--experiment-name", type=str, default="NASA Exoplanets - Learning Rate")
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--hidden-layers", type=int, nargs="+", default=[64, 32])
    return parser.parse_args()


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ExoplanetMLP(nn.Module):
    def __init__(self, input_dim, hidden_layers):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_layers:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU()])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def build_preprocessor():
    orbital_period = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scaler", StandardScaler()),
    ])
    numeric = Pipeline([("scaler", StandardScaler())])

    return ColumnTransformer([
        ("orbital_period", orbital_period, ["pl_orbper"]),
        ("numeric", numeric, ["pl_eqt", "st_teff", "st_rad", "st_mass"]),
    ])


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def to_tensor(array):
    return torch.tensor(np.asarray(array), dtype=torch.float32, device=DEVICE)


def train_model(model, x_train, y_train, x_val, y_val, learning_rate, max_epochs):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = model(x_train).squeeze(1)
        train_loss = criterion(train_pred, y_train)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val).squeeze(1)
            val_loss = criterion(val_pred, y_val)

        train_value = float(train_loss.item())
        val_value = float(val_loss.item())
        train_losses.append(train_value)
        val_losses.append(val_value)

        mlflow.log_metric("train_loss", train_value, step=epoch)
        mlflow.log_metric("val_loss", val_value, step=epoch)

        if val_value < best_val_loss - MIN_DELTA:
            best_val_loss = val_value
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, len(train_losses)


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).squeeze(1).cpu().numpy()


def main():
    args = parse_args()
    hidden_layers = tuple(args.hidden_layers)
    set_seed()
    mlflow.set_experiment(args.experiment_name)

    run_name = f"lr_{args.learning_rate}_layers_{'-'.join(map(str, hidden_layers))}"

    with mlflow.start_run(run_name=run_name):
        with mlflow.start_span(name="training_pipeline"):
            with mlflow.start_span(name="data_loading"):
                print("Carregando dataset...")
                df = pd.read_csv(DATASET, usecols=FEATURES + [TARGET])
                original_rows = len(df)
                df = df.dropna().copy()
                processed_rows = len(df)
                x = df[FEATURES]
                y = df[TARGET]

                x_train, x_temp, y_train, y_temp = train_test_split(
                    x, y, test_size=0.20, random_state=RANDOM_STATE
                )
                x_val, x_test, y_val, y_test = train_test_split(
                    x_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
                )

            mlflow.log_params({
                "framework": "PyTorch",
                "optimizer": "Adam",
                "loss_function": "MSELoss",
                "activation": "ReLU",
                "learning_rate": args.learning_rate,
                "hidden_layers": str(hidden_layers),
                "max_iter": args.max_iter,
                "early_stopping_patience": PATIENCE,
                "random_state": RANDOM_STATE,
                "device": str(DEVICE),
                "dataset": "NASA Exoplanet Archive - PSCompPars",
                "target": TARGET,
                "features": ",".join(FEATURES),
                "preprocessing": "dropna + log1p(pl_orbper) + StandardScaler",
                "split": "80% train / 10% validation / 10% test",
                "train_size": len(x_train),
                "validation_size": len(x_val),
                "test_size": len(x_test),
            })

            dataset_mlflow = mlflow.data.from_pandas(
                x_train.assign(**{TARGET: y_train}),
                source=DATASET,
                name="NASA Exoplanet Archive - PSCompPars",
            )
            mlflow.log_input(dataset_mlflow, context="training")

            with mlflow.start_span(name="preprocessing"):
                preprocessor = build_preprocessor()
                x_train_np = preprocessor.fit_transform(x_train).astype(np.float32)
                x_val_np = preprocessor.transform(x_val).astype(np.float32)
                x_test_np = preprocessor.transform(x_test).astype(np.float32)

                x_train_t = to_tensor(x_train_np)
                x_val_t = to_tensor(x_val_np)
                x_test_t = to_tensor(x_test_np)
                y_train_t = to_tensor(y_train.to_numpy(dtype=np.float32))
                y_val_t = to_tensor(y_val.to_numpy(dtype=np.float32))

            model = ExoplanetMLP(len(FEATURES), hidden_layers).to(DEVICE)

            print("\nTreinando modelo PyTorch...")
            print(f"Learning rate: {args.learning_rate}")
            print(f"Camadas ocultas: {hidden_layers}")

            with mlflow.start_span(name="training"):
                train_losses, val_losses, epochs_executed = train_model(
                    model,
                    x_train_t,
                    y_train_t,
                    x_val_t,
                    y_val_t,
                    args.learning_rate,
                    args.max_iter,
                )

            with mlflow.start_span(name="evaluation"):
                val_pred = predict(model, x_val_t)
                test_pred = predict(model, x_test_t)
                val_mae, val_rmse, val_r2 = metrics(y_val, val_pred)
                test_mae, test_rmse, test_r2 = metrics(y_test, test_pred)

                mlflow.log_metrics({
                    "val_mae": val_mae,
                    "val_rmse": val_rmse,
                    "val_r2": val_r2,
                    "test_mae": test_mae,
                    "test_rmse": test_rmse,
                    "test_r2": test_r2,
                    "epochs_executed": epochs_executed,
                })

            with mlflow.start_span(name="artifact_logging"):
                os.makedirs("artifacts", exist_ok=True)

                loss_path = "artifacts/loss_curve.png"
                plt.figure()
                plt.plot(train_losses, label="Treino")
                plt.plot(val_losses, label="Validação")
                plt.xlabel("Época")
                plt.ylabel("MSE Loss")
                plt.title("Curva de perda")
                plt.legend()
                plt.savefig(loss_path, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(loss_path)

                prediction_path = "artifacts/real_vs_predicted.png"
                plt.figure()
                plt.scatter(y_test, test_pred, alpha=0.6)
                minimum = min(float(y_test.min()), float(test_pred.min()))
                maximum = max(float(y_test.max()), float(test_pred.max()))
                plt.plot([minimum, maximum], [minimum, maximum])
                plt.xlabel("Raio real")
                plt.ylabel("Raio previsto")
                plt.title("Valores reais x previstos")
                plt.savefig(prediction_path, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(prediction_path)

                preprocessor_path = "artifacts/preprocessor.joblib"
                joblib.dump(preprocessor, preprocessor_path)
                mlflow.log_artifact(preprocessor_path)

                state_dict_path = "artifacts/model_state_dict.pt"
                torch.save(model.state_dict(), state_dict_path)
                mlflow.log_artifact(state_dict_path)

                scripted_model = torch.jit.script(model.cpu())
                mlflow.pytorch.log_model(
                    scripted_model,
                    name="model",
                    serialization_format="pickle",
                )

        print("\n" + "=" * 60)
        print("RESULTADOS")
        print("=" * 60)
        print(f"\nVALIDAÇÃO\nMAE:  {val_mae:.4f}\nRMSE: {val_rmse:.4f}\nR²:   {val_r2:.4f}")
        print(f"\nTESTE\nMAE:  {test_mae:.4f}\nRMSE: {test_rmse:.4f}\nR²:   {test_r2:.4f}")
        print(f"\nÉpocas executadas: {epochs_executed}")
        print(f"Registros: {original_rows} -> {processed_rows}")


if __name__ == "__main__":
    main()
