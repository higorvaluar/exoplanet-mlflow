import argparse
import os

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATASET = "data/exoplanets.csv"

FEATURES = [
    "pl_orbper",
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
]

TARGET = "pl_rade"
RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="NASA Exoplanets - Learning Rate"
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=500
    )

    parser.add_argument(
        "--hidden-layers",
        type=int,
        nargs="+",
        default=[64, 32]
    )

    return parser.parse_args()


def calcular_metricas(y_real, y_pred):
    mae = mean_absolute_error(
        y_real,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_real,
            y_pred
        )
    )

    r2 = r2_score(
        y_real,
        y_pred
    )

    return mae, rmse, r2


def main():
    args = parse_args()

    hidden_layers = tuple(
        args.hidden_layers
    )

    # ========================================================
    # MLFLOW
    # ========================================================

    mlflow.set_experiment(
        args.experiment_name
    )

    # ========================================================
    # CARREGAMENTO DOS DADOS
    # ========================================================

    print("Carregando dataset...")

    df = pd.read_csv(
        DATASET,
        usecols=FEATURES + [TARGET]
    )

    registros_originais = len(df)

    # Remove registros com dados ausentes
    df = df.dropna().copy()

    registros_processados = len(df)

    X = df[FEATURES]
    y = df[TARGET]

    # ========================================================
    # SPLIT
    # ========================================================

    # 80% treino
    # 10% validação
    # 10% teste

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE
    )

    # ========================================================
    # PREPROCESSAMENTO
    # ========================================================

    # pl_orbper possui forte assimetria e valores extremos.
    # Primeiro aplicamos log1p e depois StandardScaler.

    log_transformer = Pipeline(
        steps=[
            (
                "log1p",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # As demais variáveis recebem apenas padronização.

    normal_transformer = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "orbital_period",
                log_transformer,
                ["pl_orbper"]
            ),
            (
                "numeric",
                normal_transformer,
                [
                    "pl_eqt",
                    "st_teff",
                    "st_rad",
                    "st_mass",
                ]
            ),
        ]
    )

    # ========================================================
    # MODELO
    # ========================================================

    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layers,
        learning_rate_init=args.learning_rate,
        max_iter=args.max_iter,
        random_state=RANDOM_STATE
    )

    # Pipeline completo:
    # dados crus -> preprocessing -> MLP

    model = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessor
            ),
            (
                "mlp",
                mlp
            ),
        ]
    )

    # ========================================================
    # RUN DO MLFLOW
    # ========================================================

    run_name = (
        f"lr_{args.learning_rate}_"
        f"layers_{'-'.join(map(str, hidden_layers))}"
    )

    with mlflow.start_run(
        run_name=run_name
    ):

        # ----------------------------------------------------
        # PARÂMETROS
        # ----------------------------------------------------

        mlflow.log_param(
            "learning_rate",
            args.learning_rate
        )

        mlflow.log_param(
            "hidden_layers",
            str(hidden_layers)
        )

        mlflow.log_param(
            "max_iter",
            args.max_iter
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "dataset",
            "NASA Exoplanet Archive - PSCompPars"
        )

        mlflow.log_param(
            "target",
            TARGET
        )

        mlflow.log_param(
            "features",
            ",".join(FEATURES)
        )

        mlflow.log_param(
            "preprocessing",
            "dropna + log1p(pl_orbper) + StandardScaler"
        )

        mlflow.log_param(
            "train_size",
            len(X_train)
        )

        mlflow.log_param(
            "validation_size",
            len(X_val)
        )

        mlflow.log_param(
            "test_size",
            len(X_test)
        )

        dataset_mlflow = mlflow.data.from_pandas(
            X_train.assign(**{TARGET: y_train}),
            source=DATASET,
            name="NASA Exoplanet Archive - PSCompPars"
        )

        mlflow.log_input(
            dataset_mlflow,
            context="training"
        )

        # ----------------------------------------------------
        # TREINAMENTO
        # ----------------------------------------------------

        print("\nTreinando modelo...")
        print(
            f"Learning rate: "
            f"{args.learning_rate}"
        )
        print(
            f"Camadas ocultas: "
            f"{hidden_layers}"
        )

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDIÇÕES
        # ----------------------------------------------------

        val_pred = model.predict(
            X_val
        )

        test_pred = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # MÉTRICAS
        # ----------------------------------------------------

        val_mae, val_rmse, val_r2 = calcular_metricas(
            y_val,
            val_pred
        )

        test_mae, test_rmse, test_r2 = calcular_metricas(
            y_test,
            test_pred
        )

        epochs_executed = (
            model.named_steps["mlp"].n_iter_
        )

        mlflow.log_metric(
            "val_mae",
            val_mae
        )

        mlflow.log_metric(
            "val_rmse",
            val_rmse
        )

        mlflow.log_metric(
            "val_r2",
            val_r2
        )

        mlflow.log_metric(
            "test_mae",
            test_mae
        )

        mlflow.log_metric(
            "test_rmse",
            test_rmse
        )

        mlflow.log_metric(
            "test_r2",
            test_r2
        )

        mlflow.log_metric(
            "epochs_executed",
            epochs_executed
        )

        # ----------------------------------------------------
        # ARTEFATOS
        # ----------------------------------------------------

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # ====================================================
        # CURVA DE PERDA
        # ====================================================

        plt.figure()

        plt.plot(
            model.named_steps["mlp"].loss_curve_
        )

        plt.xlabel(
            "Época"
        )

        plt.ylabel(
            "Loss"
        )

        plt.title(
            "Curva de perda do treinamento"
        )

        loss_path = (
            "artifacts/loss_curve.png"
        )

        plt.savefig(
            loss_path,
            bbox_inches="tight"
        )

        plt.close()

        mlflow.log_artifact(
            loss_path
        )

        # ====================================================
        # REAL X PREVISTO
        # ====================================================

        plt.figure()

        plt.scatter(
            y_test,
            test_pred,
            alpha=0.6
        )

        minimo = min(
            y_test.min(),
            test_pred.min()
        )

        maximo = max(
            y_test.max(),
            test_pred.max()
        )

        plt.plot(
            [minimo, maximo],
            [minimo, maximo]
        )

        plt.xlabel(
            "Raio real"
        )

        plt.ylabel(
            "Raio previsto"
        )

        plt.title(
            "Valores reais x previstos"
        )

        prediction_path = (
            "artifacts/real_vs_predicted.png"
        )

        plt.savefig(
            prediction_path,
            bbox_inches="tight"
        )

        plt.close()

        mlflow.log_artifact(
            prediction_path
        )

        # ----------------------------------------------------
        # MODELO COMPLETO
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=[
                "sklearn.neural_network._stochastic_optimizers.AdamOptimizer"
            ]
        )

        # ----------------------------------------------------
        # RESULTADOS
        # ----------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "RESULTADOS"
        )

        print(
            "=" * 60
        )

        print(
            "\nVALIDAÇÃO"
        )

        print(
            f"MAE:  "
            f"{val_mae:.4f}"
        )

        print(
            f"RMSE: "
            f"{val_rmse:.4f}"
        )

        print(
            f"R²:   "
            f"{val_r2:.4f}"
        )

        print(
            "\nTESTE"
        )

        print(
            f"MAE:  "
            f"{test_mae:.4f}"
        )

        print(
            f"RMSE: "
            f"{test_rmse:.4f}"
        )

        print(
            f"R²:   "
            f"{test_r2:.4f}"
        )

        print(
            f"\nÉpocas executadas: "
            f"{epochs_executed}"
        )

        print(
            f"\nRegistros: "
            f"{registros_originais} -> "
            f"{registros_processados}"
        )


if __name__ == "__main__":
    main()
