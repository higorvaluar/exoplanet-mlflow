import pandas as pd

df = pd.read_csv("data/exoplanets.csv")

print("=" * 60)
print("DIMENSÕES")
print("=" * 60)
print(f"Linhas: {df.shape[0]}")
print(f"Colunas: {df.shape[1]}")

print("\n" + "=" * 60)
print("PRIMEIROS REGISTROS")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("COLUNAS")
print("=" * 60)
for coluna in df.columns:
    print(coluna)

colunas_interesse = [
    "pl_name",
    "pl_orbper",
    "pl_rade",
    "pl_bmasse",
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
]

print("\n" + "=" * 60)
print("ANÁLISE DAS COLUNAS DE INTERESSE")
print("=" * 60)

for coluna in colunas_interesse:
    if coluna in df.columns:
        preenchidos = df[coluna].notna().sum()
        ausentes = df[coluna].isna().sum()
        percentual = ausentes / len(df) * 100

        print(
            f"{coluna:12} | "
            f"preenchidos: {preenchidos:5} | "
            f"ausentes: {ausentes:5} | "
            f"{percentual:6.2f}% ausentes"
        )
    else:
        print(f"{coluna}: NÃO ENCONTRADA")

print("\n" + "=" * 60)
print("REGISTROS COMPLETOS")
print("=" * 60)

features = [
    "pl_orbper",
    "pl_bmasse",
    "pl_eqt",
    "st_teff",
    "st_rad",
    "st_mass",
]

target = "pl_rade"

colunas_modelo = features + [target]

existentes = [c for c in colunas_modelo if c in df.columns]

df_completo = df[existentes].dropna()

print(f"Registros totais: {len(df)}")
print(f"Registros completos nessas colunas: {len(df_completo)}")

if len(df) > 0:
    print(
        f"Percentual aproveitável: "
        f"{len(df_completo) / len(df) * 100:.2f}%"
    )

print("\nEstatísticas:")
print(df_completo.describe())
