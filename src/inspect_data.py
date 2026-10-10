import pandas as pd

DATASET = "data/exoplanets.csv"
FEATURES = ["pl_orbper", "pl_eqt", "st_teff", "st_rad", "st_mass"]
TARGET = "pl_rade"

df = pd.read_csv(DATASET)

print("=" * 60)
print("DIMENSÕES")
print("=" * 60)
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print("\n" + "=" * 60)
print("COLUNAS UTILIZADAS NO MODELO")
print("=" * 60)

for coluna in FEATURES + [TARGET]:
    preenchidos = df[coluna].notna().sum()
    ausentes = df[coluna].isna().sum()
    percentual = ausentes / len(df) * 100

    print(
        f"{coluna:12} | "
        f"preenchidos: {preenchidos:5} | "
        f"ausentes: {ausentes:5} | "
        f"{percentual:6.2f}% ausentes"
    )

df_modelo = df[FEATURES + [TARGET]].dropna()

print("\n" + "=" * 60)
print("REGISTROS APÓS DROPNA")
print("=" * 60)
print(f"Registros totais: {len(df)}")
print(f"Registros utilizados pelo modelo: {len(df_modelo)}")
print(f"Percentual aproveitável: {len(df_modelo) / len(df) * 100:.2f}%")

print("\n" + "=" * 60)
print("ESTATÍSTICAS DAS VARIÁVEIS DO MODELO")
print("=" * 60)
print(df_modelo.describe())
