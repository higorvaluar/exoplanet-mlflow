# NASA Exoplanets — Experimentos com MLflow

Projeto desenvolvido para a disciplina de Inteligência Artificial com o objetivo de construir um pipeline de treinamento, validação e teste utilizando MLflow para rastreamento e comparação de experimentos de Machine Learning.

O projeto utiliza dados reais do **NASA Exoplanet Archive**, especificamente a tabela **Planetary Systems Composite Parameters (PSCompPars)**.

## Objetivo

O modelo busca estimar o raio de um exoplaneta, representado pela variável `pl_rade`, a partir de características do planeta e de sua estrela hospedeira.

O problema foi tratado como uma tarefa de **regressão utilizando uma rede neural MLP (Multilayer Perceptron)**.

## Dataset

Fonte:

NASA Exoplanet Archive — Planetary Systems Composite Parameters (PSCompPars)

### Download do dataset

O arquivo CSV não está incluído diretamente no repositório devido ao seu tamanho. Os dados podem ser obtidos diretamente do NASA Exoplanet Archive por meio da API TAP oficial.

Na raiz do projeto, execute:

```bash
mkdir -p data

curl -L "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv" \
  -o data/exoplanets.csv
```

Após o download, o arquivo deverá estar disponível em:

```text
data/exoplanets.csv
```

O comando consulta a tabela `pscomppars`, correspondente à **Planetary Systems Composite Parameters (PSCompPars)**, utilizada neste projeto.

O dataset utilizado possui originalmente:

- 6.375 registros;
- 703 atributos.

Para este projeto foram selecionadas apenas as variáveis relevantes ao modelo.

### Features

| Variável | Descrição |
|---|---|
| `pl_orbper` | Período orbital do planeta |
| `pl_eqt` | Temperatura de equilíbrio do planeta |
| `st_teff` | Temperatura efetiva da estrela |
| `st_rad` | Raio da estrela |
| `st_mass` | Massa da estrela |

### Target

| Variável | Descrição |
|---|---|
| `pl_rade` | Raio do planeta em raios terrestres |

Após a remoção dos registros com valores ausentes nas variáveis utilizadas, foram mantidos **5.724 registros**.

A divisão dos dados foi realizada da seguinte forma:

- Treinamento: 4.579 registros;
- Validação: 572 registros;
- Teste: 573 registros.

Foi utilizado `random_state=42` para garantir reprodutibilidade das divisões.

## Pré-processamento

Durante a análise inicial foi identificado um valor extremamente alto na variável `pl_orbper`, referente ao período orbital.

Na execução inicial, esse valor causou uma previsão extremamente discrepante no conjunto de validação.

Para reduzir a assimetria dessa variável sem remover registros do dataset, foi aplicada a transformação:

```python
np.log1p(pl_orbper)
```

Em seguida, as variáveis utilizadas pelo modelo são padronizadas utilizando `StandardScaler`.

Todo o pré-processamento foi incorporado a um `Pipeline` do Scikit-learn, permitindo que o modelo salvo no MLflow contenha tanto as etapas de transformação quanto o MLP.

## Modelo

Foi utilizado o `MLPRegressor` do Scikit-learn.

A configuração de referência possui:

```text
hidden_layer_sizes = (64, 32)
learning_rate_init = 0.01
max_iter = 500
random_state = 42
```

## MLflow

O MLflow é utilizado para registrar:

- parâmetros;
- hiperparâmetros;
- métricas de validação e teste;
- número de épocas executadas;
- dataset utilizado no treinamento;
- gráficos;
- modelo treinado;
- pipeline de pré-processamento.

As principais métricas avaliadas foram:

- MAE — Mean Absolute Error;
- RMSE — Root Mean Squared Error;
- R² — Coeficiente de determinação.

## Experimento 1 — Learning Rate

### Hipótese

Diferentes taxas de aprendizado influenciam o desempenho e a convergência do MLP.

Foram avaliados três valores mantendo a arquitetura `(64, 32)`:

| Learning Rate | Test MAE | Test RMSE | Test R² | Épocas |
|---:|---:|---:|---:|---:|
| 0.0001 | 2.4144 | 3.3844 | 0.5747 | 500 |
| 0.001 | 2.2119 | 3.2230 | 0.6143 | 473 |
| **0.01** | **2.0448** | **3.0906** | **0.6454** | **164** |

### Resultado

A taxa de aprendizado `0.01` apresentou o melhor desempenho entre as configurações avaliadas.

Além de atingir maior R² e menores valores de MAE e RMSE no conjunto de teste, o treinamento convergiu em apenas 164 épocas.

A configuração `0.0001` atingiu o limite de 500 iterações sem convergir.

## Experimento 2 — Arquitetura

### Hipótese

Aumentar a complexidade da arquitetura do MLP melhora o desempenho do modelo.

A melhor taxa encontrada no experimento anterior (`0.01`) foi mantida, alterando apenas as camadas ocultas.

| Arquitetura | Test MAE | Test RMSE | Test R² | Épocas |
|---|---:|---:|---:|---:|
| `(32,)` | 2.3809 | 3.3621 | 0.5803 | 145 |
| **`(64, 32)`** | **2.0448** | **3.0906** | **0.6454** | **164** |
| `(128, 64, 32)` | 2.0563 | 3.2006 | 0.6197 | 162 |

### Resultado

O aumento da complexidade da rede não resultou necessariamente em melhor generalização.

A arquitetura intermediária `(64, 32)` apresentou o melhor resultado no conjunto de teste.

A arquitetura maior `(128, 64, 32)` não superou a configuração intermediária.

## Melhor configuração encontrada

```text
Learning Rate: 0.01
Hidden Layers: (64, 32)
Max Iterations: 500
Random State: 42
```

Resultados no conjunto de teste:

```text
MAE:  2.0448
RMSE: 3.0906
R²:   0.6454
```

## Estrutura do projeto

```text
exoplanet-mlflow/
├── data/
│   └── exoplanets.csv
├── src/
│   ├── inspect_data.py
│   └── train_mlflow.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Execução

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

### Executando uma run

Exemplo:

```bash
python src/train_mlflow.py \
  --experiment-name "NASA Exoplanets - Learning Rate Final" \
  --learning-rate 0.01 \
  --hidden-layers 64 32 \
  --max-iter 500
```

### Abrindo o MLflow

```bash
mlflow server --port 5000
```

A interface ficará disponível em:

```text
http://127.0.0.1:5000
```

## Tecnologias

- Python 3.12
- MLflow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- NASA Exoplanet Archive

## Conclusão

O projeto demonstrou como o MLflow pode ser utilizado para transformar treinamentos individuais em experimentos rastreáveis e comparáveis.

A comparação das execuções permitiu identificar tanto uma taxa de aprendizado mais adequada quanto uma arquitetura que apresentou melhor capacidade de generalização.

Os resultados também mostraram que aumentar a complexidade de uma rede neural não garante melhoria de desempenho e que o rastreamento das métricas e dos hiperparâmetros facilita a tomada de decisão com base em evidências.
