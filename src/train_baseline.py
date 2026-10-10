import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURAÇÕES
# ============================================================

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


# ============================================================
# 1. CARREGAMENTO
# ============================================================

print("Carregando dataset...")

df = pd.read_csv(
    DATASET,
    usecols=FEATURES + [TARGET]
)

print(f"Registros originais: {len(df)}")


# ============================================================
# 2. LIMPEZA
# ============================================================

df = df.dropna()

print(f"Registros após remover valores ausentes: {len(df)}")

df["pl_orbper"] = np.log1p(df["pl_orbper"])


# ============================================================
# 3. FEATURES E TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# 4. TREINO / VALIDAÇÃO / TESTE
# ============================================================

# 80% treino + 20% temporário
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE
)

# Os 20% restantes viram:
# 10% validação
# 10% teste
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE
)

print("\nDivisão do dataset:")
print(f"Treino:    {len(X_train)}")
print(f"Validação: {len(X_val)}")
print(f"Teste:     {len(X_test)}")


# ============================================================
# 5. NORMALIZAÇÃO
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. MODELO
# ============================================================

model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    learning_rate_init=0.001,
    max_iter=500,
    random_state=RANDOM_STATE
)


# ============================================================
# 7. TREINAMENTO
# ============================================================

print("\nTreinando MLP...")

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 8. VALIDAÇÃO
# ============================================================

val_predictions = model.predict(X_val_scaled)

val_mae = mean_absolute_error(
    y_val,
    val_predictions
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_predictions
    )
)

val_r2 = r2_score(
    y_val,
    val_predictions
)


# ============================================================
# 9. TESTE
# ============================================================

test_predictions = model.predict(X_test_scaled)

test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


# ============================================================
# 10. RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADOS - VALIDAÇÃO")
print("=" * 60)

print(f"MAE:  {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"R²:   {val_r2:.4f}")


print("\n" + "=" * 60)
print("RESULTADOS - TESTE")
print("=" * 60)

print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")


print("\nÉpocas executadas:")
print(model.n_iter_)

print("\n" + "=" * 60)
print("ANÁLISE DO TARGET POR PARTIÇÃO")
print("=" * 60)

for nome, valores in [
    ("Treino", y_train),
    ("Validação", y_val),
    ("Teste", y_test),
]:
    print(f"\n{nome}:")
    print(f"  mínimo:  {valores.min():.4f}")
    print(f"  mediana: {valores.median():.4f}")
    print(f"  média:   {valores.mean():.4f}")
    print(f"  máximo:  {valores.max():.4f}")

    erros_val = np.abs(y_val.to_numpy() - val_predictions)

indices = np.argsort(erros_val)[-10:][::-1]

print("\n" + "=" * 60)
print("10 MAIORES ERROS NA VALIDAÇÃO")
print("=" * 60)

for i in indices:
    print(
        f"Real: {y_val.iloc[i]:8.4f} | "
        f"Previsto: {val_predictions[i]:8.4f} | "
        f"Erro: {erros_val[i]:8.4f}"
    )

print("\n" + "=" * 60)
print("FEATURES DOS 10 MAIORES ERROS")
print("=" * 60)

for i in indices:
    original_index = y_val.index[i]

    print(f"\nErro: {erros_val[i]:.4f}")
    print(f"Real: {y_val.iloc[i]:.4f}")
    print(f"Previsto: {val_predictions[i]:.4f}")

    print(X_val.loc[original_index].to_string())

print("\n" + "=" * 60)
print("REGISTRO COM MAIOR ERRO - VALORES ESCALADOS")
print("=" * 60)

pior_i = indices[0]

print("Valores originais:")
print(X_val.iloc[pior_i])

print("\nValores após StandardScaler:")

for feature, value in zip(
    FEATURES,
    X_val_scaled[pior_i]
):
    print(f"{feature:10}: {value:12.4f}")
