# Passos Mágicos — Guia de Apresentação (Tech Challenge 5)

> Roteiro para o vídeo de apresentação do projeto de ML para previsão de risco de defasagem escolar.

---

## Visão Geral do Projeto

**Problema:** A Associação Passos Mágicos acompanha alunos de baixa renda e precisa identificar, de forma proativa, quais estudantes estão em risco de defasagem escolar (estudante na série correta ou abaixo da ideal para sua idade).

**Solução:** Um pipeline de Machine Learning treinado com dados educacionais de 2022 que classifica cada aluno em **Baixo Risco**, **Médio Risco** ou **Alto Risco** de defasagem — exposto via API REST e deployado na AWS com SageMaker.

**Target:**
- `risco = 1` → aluno **em risco** (Defasagem >= 0, ou seja, não está adiantado)
- `risco = 0` → aluno **adiantado** (Defasagem < 0)

**Dataset:** `database/base_2024.xlsx` — 860 alunos, 42 colunas, dados de 2022.

---

## Estrutura do Projeto

```
tech-challenge-5/
├── database/
│   ├── base_2024.xlsx          # Dataset principal (860 alunos, 2022)
│   └── bases_antigas.zip       # Bases históricas
│
├── notebooks/
│   └── eda_e_treinamento.ipynb # Análise exploratória + treinamento visual
│
├── src/                        # Pipeline de ML (módulos reutilizáveis)
│   ├── preprocessing.py        # Carregamento, limpeza e pré-processamento
│   ├── feature_engineering.py  # Criação de features derivadas
│   ├── train.py                # Script de treinamento completo
│   ├── evaluate.py             # Métricas e comparação de modelos
│   └── utils.py                # Logging, paths, save/load model
│
├── app/                        # API REST (FastAPI)
│   ├── main.py                 # Entrypoint FastAPI
│   ├── route.py                # Rotas: /predict, /health, /model-info, /ping, /invocations
│   └── model/
│       └── model.joblib        # Modelo treinado serializado
│
├── tests/                      # Testes automatizados (65 testes, 99% de cobertura)
│   ├── test_preprocessing.py
│   ├── test_model.py
│   └── test_train.py
│
├── infra/
│   └── cloudformation.yml      # Infraestrutura AWS (IAM, ECR, SageMaker)
│
├── .github/
│   └── workflows/
│       └── deploy.yml          # CI/CD: GitHub Actions → ECR → SageMaker
│
└── Dockerfile                  # Imagem Docker (treina + serve a API)
```

---

## O Notebook (`notebooks/eda_e_treinamento.ipynb`)

O notebook é o ambiente exploratório onde todo o raciocínio analítico é documentado visualmente. Ele reutiliza os mesmos módulos `src/` do pipeline de produção — nenhum código duplicado.

### Seções do Notebook

| # | Seção | O que mostra |
|---|-------|-------------|
| 1 | Carregamento e Visão Geral | Shape (860, 42), tipos de colunas, valores ausentes |
| 2 | Limpeza e Engenharia de Features | Aplicação de `clean_data` + `create_features` |
| 3 | Distribuição do Target | ~70% adiantados (risco=0) vs ~30% em risco (risco=1) |
| 4 | Análise Univariada — Numéricas | Histogramas com mediana de: fase, idade, INDE, IAA, IEG... |
| 5 | Análise Univariada — Categóricas | Distribuição de Pedra 22/21 e gênero |
| 6 | Análise Bivariada | Boxplots e taxa de risco por categoria |
| 7 | Matriz de Correlação | Heatmap Pearson de todas as features + target |
| 8 | Preparação para Treinamento | Split 80/20 estratificado |
| 9 | Cross-Validation | Comparação de 4 modelos com StratifiedKFold (5 folds) |
| 10 | Avaliação no Test Set | Accuracy, Precision, Recall, F1, ROC-AUC |
| 11 | Matriz de Confusão | Visualização para todos os modelos |
| 12 | Curva ROC | Comparação visual da área sob a curva |
| 13 | Importância de Features | Coeficientes (Logistic) ou Gini (árvores) |
| 14 | Resumo Final | Métricas consolidadas do modelo selecionado |

### Ponto crítico — Data Leakage
> A coluna **IAN (Índice de Adequação ao Nível)** é uma discretização direta de `Defas` (IAN=10 quando Defas≥0, IAN=5 quando Defas=-1/-2, IAN=2.5 quando Defas≤-3). Incluí-la seria vazamento de dados — ela está **explicitamente excluída** de todas as features.

---

## Os Scripts (`src/`)

### `src/preprocessing.py`

Responsável por carregar e limpar os dados brutos.

**Funções principais:**
- `load_raw_data()` — lê `database/base_2024.xlsx`
- `clean_data(df)` — renomeia colunas, calcula `anos_no_programa = 2022 - ano_ingresso + 1`, preenche nulos, cria o target `risco = (defas >= 0)`
- `build_preprocessor()` — retorna um `ColumnTransformer` sklearn com:
  - Numéricas: `SimpleImputer(median)` + `StandardScaler`
  - Categóricas: `SimpleImputer(constant)` + `OrdinalEncoder` (Pedra tem ordem: Ametista < Ágata < Quartzo < Topázio)
- `prepare_dataset()` — pipeline completo: carrega → limpa → retorna X, y

**Features utilizadas (17 no total):**

| Tipo | Colunas |
|------|---------|
| Numéricas (14) | fase, idade, anos_no_programa, inde, iaa, ieg, ips, ida, ipv, cg, cf, ct, matem, portug |
| Categóricas (3) | pedra_22, pedra_21, genero |

---

### `src/feature_engineering.py`

Cria features derivadas que enriquecem o sinal preditivo.

**`create_features(df)`** adiciona:

| Feature Derivada | Cálculo |
|-----------------|---------|
| `indice_academico_medio` | Média de: inde, iaa, ieg, ida |
| `indice_socio_medio` | Média de: ips, ipv |
| `media_notas` | Média de: matem, portug |
| `inde_desvio_fase` | INDE do aluno − mediana do INDE na sua fase |
| `ingressante_recente` | 1 se anos_no_programa ≤ 2, senão 0 |
| `pedra_rank_22` / `pedra_rank_21` | Ranking numérico da Pedra (1=Ametista, 4=Topázio) |
| `tendencia_pedra` | pedra_rank_22 − pedra_rank_21 (positivo = melhora) |

---

### `src/train.py`

Script principal de treinamento. Executado com:

```bash
python src/train.py
```

**Fluxo de execução:**

```
1. Carrega e limpa os dados (load_raw_data + clean_data)
2. Aplica engenharia de features (create_features)
3. Split treino/teste: 80%/20%, estratificado por target
4. Cross-validation (StratifiedKFold, 5 folds, métrica F1) em 4 candidatos:
   - Logistic Regression (class_weight=balanced)
   - Random Forest (200 árvores, max_depth=8)
   - Gradient Boosting (200 estimadores, lr=0.05)
   - XGBoost (200 estimadores, lr=0.05)
5. Treina o melhor modelo (por CV F1) no conjunto de treino completo
6. Avalia todos os modelos no test set
7. Serializa o pipeline vencedor em app/model/model.joblib
```

**Resultado:** Logistic Regression vence com CV F1 ≈ 0.99.

O modelo serializado é um dict com chaves `"pipeline"` e `"metadata"`.

---

### `src/evaluate.py`

Utilitários de avaliação desacoplados do treinamento.

- `evaluate_model(pipeline, X_test, y_test)` — retorna dict com accuracy, precision, recall, F1, ROC-AUC, confusion_matrix, classification_report
- `compare_models(results)` — retorna o nome do melhor modelo por F1
- `print_summary(results)` — imprime tabela comparativa formatada

---

### `src/utils.py`

Utilitários compartilhados:
- `setup_logging(name)` — logger padronizado
- `get_database_path()` — resolve path do diretório `database/`
- `save_model(pipeline, metadata)` — serializa com joblib em `app/model/model.joblib`
- `load_model()` — desserializa e retorna `(pipeline, metadata)`

---

## A API (`app/`)

FastAPI exposta na porta 8000. O modelo é carregado em memória no startup.

### Endpoints

#### `GET /health`
Liveness probe.
```json
{"status": "ok"}
```

#### `GET /ping`
Health check exigido pelo SageMaker (retorna 200).
```json
{"status": "ok"}
```

#### `GET /model-info`
Retorna metadados do modelo carregado (nome, features, métricas, data de treino).

#### `POST /predict`
Predição para um aluno. Input:

```json
{
  "fase": 3,
  "idade": 12,
  "ano_ingresso": 2018,
  "inde": 6.5,
  "iaa": 7.0,
  "ieg": 5.5,
  "ips": 6.0,
  "ida": 6.2,
  "ipv": 7.5,
  "cg": 300,
  "cf": 10,
  "ct": 8,
  "matem": 6.5,
  "portug": 6.8,
  "pedra_22": "Ametista",
  "pedra_21": "Ametista",
  "genero": "Menino"
}
```

Output:
```json
{
  "risco_defasagem": 0.8234,
  "classificacao": "Alto Risco",
  "confianca": 0.8234,
  "modelo": "Logistic Regression"
}
```

**Classificação por probabilidade:**
- `< 0.35` → Baixo Risco
- `0.35 – 0.65` → Médio Risco
- `> 0.65` → Alto Risco

#### `POST /invocations`
Mesmo comportamento do `/predict` — rota exigida pelo SageMaker para inferência.

> **Nota:** O campo `IAN` é intencionalmente omitido do input por ser data leakage. A API recalcula `anos_no_programa` internamente a partir do `ano_ingresso`.

### Como rodar localmente

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar o modelo (gera app/model/model.joblib)
python src/train.py

# Subir a API
uvicorn app.main:app --reload

# Acessar a documentação interativa
# http://localhost:8000/docs
```

---

## Docker

O Dockerfile **treina o modelo durante o build** — a imagem já contém o artefato treinado:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY app/ ./app/
COPY database/ ./database/
RUN python src/train.py          # treina e gera model.joblib
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build local
docker build -t passos-magicos .

# Run local
docker run -p 8000:8000 passos-magicos
```

---

## Infraestrutura AWS (`infra/cloudformation.yml`)

A stack CloudFormation provisiona todos os recursos necessários para servir o modelo na AWS via **SageMaker Serverless Inference**.

### Recursos criados

| Recurso | Tipo | Descrição |
|---------|------|-----------|
| `SageMakerExecutionRole` | `AWS::IAM::Role` | Role com permissões: SageMakerFullAccess + ECRReadOnly + CloudWatch Logs |
| `SageMakerModel` | `AWS::SageMaker::Model` | Aponta para a imagem Docker no ECR |
| `SageMakerEndpointConfig` | `AWS::SageMaker::EndpointConfig` | Serverless: 2048 MB RAM, até 10 invocações simultâneas |
| `SageMakerEndpoint` | `AWS::SageMaker::Endpoint` | Endpoint público do SageMaker |

### Parâmetros da stack

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `ImageTag` | `github.sha` | Tag da imagem Docker (injetada pelo CI/CD) |
| `Environment` | `prod` | Ambiente (dev ou prod) |

### Fluxo da imagem

```
ECR URI: {ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/passos-magicos:{IMAGE_TAG}
```

### Outputs da stack

- `SageMakerExecutionRoleArn` — ARN da role criada
- `SageMakerEndpointName` — Nome do endpoint ativo
- `ECRImageUri` — URI completa da imagem deployada

---

## CI/CD (`.github/workflows/deploy.yml`)

Pipeline automatizado que dispara em **todo push para `main`**.

### Jobs

```
push to main
    │
    ▼
[1] build-image
    ├── Checkout do código
    ├── Configura credenciais AWS (secrets do GitHub)
    ├── Login no Amazon ECR
    ├── Cria repositório ECR (se não existir), com scan de imagem ativado
    ├── Docker Build (tag: latest + sha do commit)
    ├── Tag da imagem com URI do ECR
    └── Push para o ECR
    │
    ▼
[2] deploy
    ├── Checkout do código
    ├── Configura credenciais AWS
    └── Deploy da stack CloudFormation
        └── aws cloudformation deploy
            --stack-name passos-magicos-infra
            --parameter-overrides ImageTag={sha} Environment=prod
    │
    ▼
[3] release
    ├── Setup Node 20
    ├── Instala semantic-release
    ├── Gera release no GitHub (changelog automático por commits)
    └── Extrai a versão do último git tag
```

### Secrets necessários no GitHub

| Secret | Descrição |
|--------|-----------|
| `AWS_ACCESS_KEY_ID` | Chave de acesso IAM |
| `AWS_SECRET_ACCESS_KEY` | Secret de acesso IAM |
| `REGISTRY` | URI do ECR (`{account_id}.dkr.ecr.us-east-1.amazonaws.com`) |

### Região

Todos os recursos são criados em **`us-east-1`** (N. Virginia).

---

## Testes (`tests/`)

```bash
# Rodar todos os testes com cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

**65 testes** distribuídos em:
- `test_preprocessing.py` — carregamento, limpeza, preprocessor, target
- `test_model.py` — pipeline sklearn, predição, serialização/deserialização
- `test_train.py` — fluxo completo de treinamento

**Cobertura: 99%** dos módulos em `src/` (exceto 2 linhas em `train.py`).

---

## Decisões de Design

| Decisão | Justificativa |
|---------|--------------|
| Logistic Regression como modelo final | CV F1 ≈ 0.99 — melhor generalização com dados tabulares bem normalizados |
| `class_weight="balanced"` | Dataset desbalanceado (~70/30) — evita viés para a classe majoritária |
| IAN excluído das features | Discretização direta do target — incluir seria data leakage |
| Treino embutido no Docker build | Garante que a imagem deployada já tem o modelo; sem dependência de artefatos externos |
| SageMaker Serverless | Custo zero quando idle; escala automaticamente; adequado para inferência esporádica |
| Rotas `/ping` e `/invocations` | Exigidas pelo protocolo do SageMaker para health check e inferência |
| `anos_no_programa` calculado na API | `clean_data` não é chamado na rota — o campo precisa ser derivado explicitamente de `ano_ingresso` |

---

## Roteiro Sugerido para o Vídeo

1. **Contexto** (1 min) — Problema da Passos Mágicos, objetivo do modelo, definição do target
2. **Dataset** (2 min) — Mostrar o Excel, colunas, alerta sobre IAN (data leakage)
3. **Notebook — EDA** (4 min) — Distribuição do target, correlações, análise bivariada
4. **Notebook — Treinamento** (3 min) — Cross-validation, comparação de modelos, métricas finais
5. **Código `src/`** (3 min) — preprocessing.py, feature_engineering.py, train.py, evaluate.py
6. **API FastAPI** (2 min) — Mostrar o Swagger UI (`/docs`), fazer uma requisição ao `/predict`
7. **Docker** (1 min) — Dockerfile, `docker build`, conceito de modelo embutido na imagem
8. **Infraestrutura AWS** (3 min) — cloudformation.yml, recursos criados, SageMaker Serverless
9. **CI/CD** (2 min) — deploy.yml, fluxo build → deploy → release, secrets
10. **Testes** (1 min) — `pytest`, cobertura 99%
11. **Demo ao vivo** (2 min) — API rodando localmente, chamada curl ou Swagger

---

## Comandos de Referência Rápida

```bash
# Treinar modelo
python src/train.py

# Subir API local
uvicorn app.main:app --reload

# Swagger UI
open http://localhost:8000/docs

# Rodar testes
pytest tests/ --cov=src --cov-report=term-missing

# Build Docker
docker build -t passos-magicos .

# Run Docker
docker run -p 8000:8000 passos-magicos

# Chamar o endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fase": 3, "idade": 12, "ano_ingresso": 2018,
    "inde": 6.5, "iaa": 7.0, "ieg": 5.5, "ips": 6.0,
    "ida": 6.2, "ipv": 7.5, "cg": 300, "cf": 10, "ct": 8,
    "matem": 6.5, "portug": 6.8,
    "pedra_22": "Ametista", "pedra_21": "Ametista", "genero": "Menino"
  }'

# Deploy CloudFormation (manual)
aws cloudformation deploy \
  --template-file infra/cloudformation.yml \
  --stack-name passos-magicos-infra \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides ImageTag=latest Environment=prod
```
