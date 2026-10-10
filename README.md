# NASA Exoplanets — Experimentos com PyTorch e MLflow

Projeto desenvolvido para a disciplina de Inteligência Artificial com o objetivo de construir um pipeline de treinamento, validação e teste utilizando **PyTorch** para a implementação de uma rede neural MLP e **MLflow** para rastreamento, tracing e comparação de experimentos de Machine Learning.

O projeto utiliza dados reais do **NASA Exoplanet Archive**, especificamente a tabela **Planetary Systems Composite Parameters (PSCompPars)**.

## Objetivo

O modelo busca estimar o raio de um exoplaneta, representado pela variável `pl_rade`, a partir de características do planeta e de sua estrela hospedeira.

O problema foi tratado como uma tarefa de **regressão com uma MLP (Multilayer Perceptron) implementada em PyTorch**.

## Dataset

Fonte:

**NASA Exoplanet Archive — Planetary Systems Composite Parameters (PSCompPars)**

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

O comando consulta a tabela `pscomppars`, correspondente à **Planetary Systems Composite Parameters (PSCompPars)**.

O snapshot utilizado durante o desenvolvimento deste projeto possuía:

- 6.375 registros;
- 703 atributos.

Como o NASA Exoplanet Archive é atualizado ao longo do tempo, um download realizado posteriormente pode apresentar quantidade diferente de registros.

### Features utilizadas

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

Foi utilizado `random_state=42` para manter a mesma divisão entre as diferentes rodadas experimentais.

## Pré-processamento

Durante a análise inicial foi identificado um valor extremamente alto na variável `pl_orbper`, referente ao período orbital. Esse valor gerava forte assimetria e produzia previsões discrepantes.

Para reduzir essa assimetria sem remover registros do dataset, foi aplicada a transformação:

```python
np.log1p(pl_orbper)
```

Em seguida, as variáveis utilizadas pelo modelo são padronizadas com `StandardScaler`.

O pré-processamento é implementado com componentes do Scikit-learn (`Pipeline` e `ColumnTransformer`) e é ajustado apenas com os dados de treinamento. O transformador treinado é persistido no MLflow como o artefato `preprocessor.joblib`.

## Modelo

A rede neural é implementada em **PyTorch** utilizando camadas `Linear` e ativação `ReLU`.

O treinamento utiliza:

```text
Framework: PyTorch
Optimizer: Adam
Loss: MSELoss
Activation: ReLU
Maximum epochs: 500
Early stopping patience: 50
Random state: 42
Device: CPU
```

A arquitetura e a taxa de aprendizado são parametrizadas pela linha de comando, permitindo executar diferentes configurações sem modificar o código.

## MLflow

O MLflow é utilizado para registrar e relacionar cada execução experimental.

Em cada run são persistidos:

- hiperparâmetros e configuração do treinamento;
- configuração do dataset e da divisão treino/validação/teste;
- dataset utilizado no treinamento;
- `train_loss` e `val_loss` ao longo das épocas;
- métricas finais de validação e teste;
- número de épocas executadas;
- gráficos;
- pré-processador treinado;
- `state_dict` da rede;
- modelo PyTorch;
- traces das etapas do pipeline.

As principais métricas finais avaliadas são:

- MAE — Mean Absolute Error;
- RMSE — Root Mean Squared Error;
- R² — Coeficiente de determinação.

### Tracing

Cada run possui um trace principal chamado `training_pipeline`, dividido nas etapas:

```text
training_pipeline
├── data_loading
├── preprocessing
├── training
├── evaluation
└── artifact_logging
```

Dessa forma, além dos resultados finais, o processo de execução também pode ser acompanhado no MLflow.

## Experimento 1 — Learning Rate

### Hipótese

Diferentes taxas de aprendizado influenciam o desempenho e a velocidade de treinamento da MLP.

Foram avaliados três valores mantendo fixa a arquitetura `(64, 32)`:

| Learning Rate | Test MAE | Test RMSE | Test R² | Épocas |
|---:|---:|---:|---:|---:|
| **0.01** | **2.3025** | **3.2991** | **0.5959** | **359** |
| 0.001 | 2.5629 | 3.4810 | 0.5501 | 500 |
| 0.0001 | 3.8990 | 5.9519 | -0.3152 | 500 |

### Resultado

A taxa de aprendizado `0.01` apresentou o melhor desempenho entre os valores avaliados, com menor MAE e RMSE e maior R² no conjunto de teste.

O treinamento com `0.01` foi encerrado em 359 épocas pelo mecanismo de early stopping. As taxas `0.001` e `0.0001` chegaram ao limite de 500 épocas.

A taxa `0.0001` apresentou R² negativo no teste, indicando que essa configuração foi inadequada dentro do limite de treinamento utilizado.

## Experimento 2 — Arquitetura

### Hipótese

Aumentar a capacidade da MLP, mantendo os demais elementos controlados, pode melhorar o desempenho do modelo.

A melhor taxa encontrada no experimento anterior (`0.01`) foi mantida, alterando somente as camadas ocultas:

| Arquitetura | Test MAE | Test RMSE | Test R² | Épocas |
|---|---:|---:|---:|---:|
| `(32,)` | 2.4395 | 3.3922 | 0.5728 | 500 |
| `(64, 32)` | 2.3025 | 3.2991 | 0.5959 | 359 |
| **`(128, 64, 32)`** | **2.1301** | **3.0917** | **0.6451** | **366** |

### Resultado

Entre as três arquiteturas avaliadas, a rede `(128, 64, 32)` apresentou o melhor resultado no conjunto de teste.

Nesse conjunto de experimentos, aumentar a capacidade da rede melhorou o desempenho, culminando em menor MAE e RMSE e maior R² para a maior arquitetura testada. Esse resultado sustenta a hipótese dentro das configurações avaliadas, sem implicar que arquiteturas maiores sejam sempre melhores.

## Melhor configuração encontrada

```text
Framework: PyTorch
Learning Rate: 0.01
Hidden Layers: (128, 64, 32)
Activation: ReLU
Optimizer: Adam
Loss: MSELoss
Maximum Epochs: 500
Early Stopping Patience: 50
Random State: 42
```

Resultados de validação:

```text
MAE:  2.1496
RMSE: 3.3382
R²:   0.5771
```

Resultados no conjunto de teste:

```text
MAE:  2.1301
RMSE: 3.0917
R²:   0.6451
```

Foram executadas **366 épocas** nessa rodada.

## Artefatos

Cada run gera e registra no MLflow:

```text
loss_curve.png
real_vs_predicted.png
preprocessor.joblib
model_state_dict.pt
model/
```

- `loss_curve.png`: evolução da perda de treinamento e validação;
- `real_vs_predicted.png`: comparação entre valores reais e previstos no teste;
- `preprocessor.joblib`: transformações de pré-processamento ajustadas no conjunto de treino;
- `model_state_dict.pt`: pesos do modelo PyTorch;
- `model/`: modelo PyTorch persistido pelo MLflow.

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

O diretório `data/`, o banco local do MLflow e os artefatos gerados durante os treinamentos não são versionados no Git.

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

Baixe o dataset conforme descrito na seção **Dataset**.

### Executando uma run

Exemplo:

```bash
python src/train_mlflow.py \
  --experiment-name "NASA Exoplanets - Architecture PyTorch Final" \
  --learning-rate 0.01 \
  --hidden-layers 128 64 32 \
  --max-iter 500
```

### Abrindo o MLflow

Em outro terminal, com o ambiente virtual ativado:

```bash
mlflow server --port 5000
```

A interface ficará disponível em:

```text
http://127.0.0.1:5000
```

## Tecnologias

- Python 3.12
- PyTorch
- MLflow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- NASA Exoplanet Archive

## Conclusão

O projeto demonstrou como o MLflow pode ser utilizado para transformar execuções de treinamento em experimentos rastreáveis e comparáveis.

O primeiro experimento mostrou que a taxa de aprendizado teve impacto significativo na qualidade do modelo e na duração do treinamento. Entre os valores testados, `0.01` apresentou o melhor resultado.

Mantendo essa taxa fixa, o segundo experimento comparou diferentes arquiteturas. A configuração `(128, 64, 32)` apresentou a melhor generalização no conjunto de teste, atingindo R² de aproximadamente `0.645`.

Além das métricas e hiperparâmetros, o MLflow foi utilizado para persistir o dataset de treinamento, gráficos, pré-processador, pesos, modelo PyTorch e traces das principais etapas do pipeline, permitindo recuperar e comparar o contexto de cada execução.
