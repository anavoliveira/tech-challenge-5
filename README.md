
# Datathon Passos Mágicos — Previsão de Risco de Defasagem Escolar

> Transformando a vida de crianças e jovens por meio da educação e da tecnologia.

---

## 1. Visão Geral do Projeto

A [Associação Passos Mágicos](https://www.passosmagicos.org.br/) atua há 32 anos na transformação de vida de crianças e jovens em vulnerabilidade social, oferecendo educação de qualidade, apoio psicológico/psicopedagógico e ampliação de visão de mundo.

### Problema de Negócio

Com base em dados do desenvolvimento educacional dos anos de **2022, 2023 e 2024** (3.030 alunos), este projeto desenvolve um **modelo preditivo capaz de estimar o risco de defasagem escolar** de cada estudante — permitindo intervenções mais rápidas e direcionadas pela equipe da associação.

**Target:**
- `risco = 1` → aluno **em risco** (não está adiantado na série — `Defas >= 0`)
- `risco = 0` → aluno **adiantado** (`Defas < 0`)

### Solução Proposta

Construção de uma **pipeline completa de Machine Learning**, cobrindo desde o pré-processamento dos dados até o deploy do modelo em produção via API REST na AWS, com testes automatizados, logging estruturado e monitoramento de drift.

### Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11 |
| ML & Data | scikit-learn, XGBoost, pandas, numpy |
| API | FastAPI + Uvicorn |
| Serialização | joblib |
| Testes | pytest + pytest-cov |
| Empacotamento | Docker + Amazon ECR |
| Deploy | AWS SageMaker Real-Time Endpoint (ml.t2.medium) + API Gateway |
| CI/CD | GitHub Actions |
| Monitoramento | Logging estruturado (JSON) + Notebook de análise de drift |

---

## 2. Estrutura do Projeto

```
tech-challenge-5/
│
├── .github/workflows/
│   └── deploy.yml              # CI/CD: build → push ECR → deploy SageMaker → release
│
├── database/
│   └── base_2024.xlsx          # Dataset principal — 3 abas: PEDE2022 (860), PEDE2023 (1014), PEDE2024 (1156)
│
├── infra/
│   └── cloudformation.yml      # Stack AWS: S3 (DataCapture), IAM Roles, SageMaker Model/EndpointConfig/Endpoint, API Gateway
│
├── app/
│   ├── main.py                 # Entrypoint FastAPI
│   ├── route.py                # Rotas: /predict, /invocations, /ping, /health, /model-info
│   └── model/
│       └── model.joblib        # Modelo serializado (gerado no docker build)
│
├── src/
│   ├── preprocessing.py        # Carregamento, limpeza e pré-processamento
│   ├── feature_engineering.py  # Criação de features derivadas
│   ├── train.py                # Pipeline de treinamento e seleção de modelo
│   ├── evaluate.py             # Métricas e comparação de modelos
│   └── utils.py                # Logging, paths, save/load model
│
├── tests/
│   ├── test_preprocessing.py   # Testes de carregamento e limpeza
│   ├── test_model.py           # Testes de pipeline e predição
│   └── test_train.py           # Testes do fluxo de treinamento
│
├── notebooks/
│   ├── eda_e_treinamento.ipynb      # Análise exploratória e treinamento visual
│   ├── monitoramento_drift.ipynb   # Monitoramento de drift do modelo
│   └── setup_model_monitor.ipynb   # Configuração do SageMaker Model Monitor
│
├── serve                       # Script de entrypoint exigido pelo SageMaker
├── Dockerfile
└── requirements.txt
```

---

## 3. Pipeline de Machine Learning

### 3.1 Pré-processamento dos Dados (`src/preprocessing.py`)

- Carregamento das 3 abas do Excel (`PEDE2022`, `PEDE2023`, `PEDE2024`) com normalização de nomes de colunas por ano
- Renomeação e padronização das colunas para o formato canônico de 2022
- Conversão de valores não-numéricos (ex: `"INCLUIR"`) para NaN via `pd.to_numeric`
- Cálculo de `anos_no_programa = ano_ref - ano_ingresso + 1` (usa o ano da aba como referência)
- Preenchimento de nulos: mediana para numéricas, "Desconhecido" para categóricas
- Criação do target: `risco = 1` se `Defas >= 0`, `risco = 0` se `Defas < 0`
- **Exclusão do IAN**: índice excluído por ser discretização direta de `Defas` (data leakage)
- Preprocessor sklearn: `StandardScaler` para numéricas, `OrdinalEncoder` para categóricas (Pedra tem ordem: Ametista < Ágata < Quartzo < Topázio)

### 3.2 Engenharia de Features (`src/feature_engineering.py`)

Features derivadas criadas automaticamente:

| Feature | Cálculo |
|---------|---------|
| `indice_academico_medio` | Média de INDE, IAA, IEG, IDA |
| `indice_socio_medio` | Média de IPS, IPV |
| `media_notas` | Média de Matemática e Português |
| `inde_desvio_fase` | INDE do aluno − mediana do INDE na sua fase |
| `ingressante_recente` | 1 se anos_no_programa ≤ 2 |
| `pedra_rank_22` / `pedra_rank_21` | Ranking numérico da Pedra (1 a 4) |
| `tendencia_pedra` | pedra_rank_22 − pedra_rank_21 |

### 3.3 Treinamento e Validação (`src/train.py`)

- Split treino/teste: **80%/20%**, estratificado pelo target
- **Cross-validation**: StratifiedKFold com 5 folds, métrica F1
- Modelos avaliados: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Seleção automática do melhor modelo por CV F1
- Avaliação final no test set: Accuracy, Precision, Recall, F1, ROC-AUC

### 3.4 Seleção do Modelo

**Modelo selecionado: Logistic Regression** (`class_weight="balanced"`)

| Modelo | CV F1 |
|--------|-------|
| Logistic Regression | ~0.99 |
| Random Forest | comparado |
| Gradient Boosting | comparado |
| XGBoost | comparado |

**Por que é confiável para produção:**
- CV F1 consistente entre folds (baixo desvio padrão)
- `class_weight="balanced"` evita viés para a classe majoritária (~70% adiantados)
- Logistic Regression é interpretável: coeficientes mostram a contribuição de cada feature
- Validado em test set holdout separado do treino

### 3.5 Serialização

Modelo salvo com `joblib` em `app/model/model.joblib`:
```python
{"pipeline": sklearn_pipeline, "metadata": {...métricas, features, data de treino...}}
```

---

## 4. Instruções de Deploy

### Pré-requisitos

- Python 3.11+
- Docker instalado
- Dependências: `pip install -r requirements.txt`

### Instalação local

```bash
git clone <url-do-repositorio>
cd tech-challenge-5

pip install -r requirements.txt
```

### Treinar o modelo

```bash
python src/train.py
```

### Subir a API localmente

```bash
uvicorn app.main:app --reload
# Acesse: http://localhost:8000/docs
```

### Deploy com Docker

```bash
# Build da imagem (já treina o modelo internamente)
docker build -t passos-magicos .

# Executar o container
docker run -p 8080:8080 passos-magicos

# Acesse: http://localhost:8080/docs
```

### Deploy na AWS (automático via CI/CD)

Todo push para `main` dispara o pipeline GitHub Actions:
1. Build da imagem Docker (treina o modelo)
2. Push para o Amazon ECR
3. Deploy da stack CloudFormation:
   - S3 Bucket para DataCapture e baseline do Model Monitor
   - IAM Roles (SageMaker + API Gateway)
   - SageMaker Real-Time Endpoint (`ml.t3.medium`) com DataCapture habilitado (100%)
   - API Gateway REST público em frente ao endpoint SageMaker
4. Geração de release semântico

Secrets necessários no GitHub:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `REGISTRY` — URI do ECR (`{account_id}.dkr.ecr.us-east-1.amazonaws.com`)

### Testes unitários

```bash
# Rodar todos os testes com cobertura
pytest tests/ --cov=src --cov-report=term-missing

# Resultado esperado: 65 testes, 99% de cobertura
```

---

## 5. Exemplos de Chamadas à API

### `GET /ping` — SageMaker health check

```bash
curl http://localhost:8080/ping
```
```json
{"status": "ok"}
```

### `GET /health` — Liveness probe

```bash
curl http://localhost:8080/health
```
```json
{"status": "ok"}
```

### `GET /model-info` — Metadados do modelo

```bash
curl http://localhost:8080/model-info
```
```json
{
  "model_name": "Logistic Regression",
  "feature_columns": ["fase", "idade", "anos_no_programa", "..."],
  "trained_at": "2025-01-01T00:00:00",
  "cv_f1": 0.99,
  "target": "risco"
}
```

### `POST /predict` — Predição de risco de defasagem

**curl (produção — AWS API Gateway):**
```bash
curl -X POST https://n8c8xksefj.execute-api.us-east-1.amazonaws.com/prod/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**curl (local):**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"fase": 3, "idade": 12, "ano_ingresso": 2018, "inde": 6.5, "iaa": 7.0, "ieg": 5.5, "ips": 6.0, "ida": 6.2, "ipv": 7.5, "cg": 300, "cf": 10, "ct": 8, "matem": 6.5, "portug": 6.8, "pedra_22": "Ametista", "pedra_21": "Ametista", "genero": "Menino"}'
```

**Response:**
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

**Campos do input:**

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `fase` | int (0–8) | Fase atual no programa |
| `idade` | int (5–30) | Idade do aluno |
| `ano_ingresso` | int (2010–2024) | Ano de ingresso no programa |
| `inde` | float (0–10) | Índice de Desenvolvimento Educacional |
| `iaa` | float (0–10) | Índice de Auto Avaliação |
| `ieg` | float (0–10) | Índice de Engajamento |
| `ips` | float (0–10) | Índice Psicossocial |
| `ida` | float (0–10) | Índice de Desenvolvimento do Aprendizado |
| `ipv` | float (0–10) | Índice do Ponto de Virada |
| `cg` | float (≥0) | Conceito Geral |
| `cf` | float (≥0) | Conceito Final |
| `ct` | float (≥0) | Conceito Total |
| `matem` | float (0–10) | Nota de Matemática |
| `portug` | float (0–10) | Nota de Português |
| `pedra_22` | str | Pedra 2022: Ametista / Agata / Quartzo / Topazio |
| `pedra_21` | str (opcional) | Pedra 2021 (opcional) |
| `genero` | str | Menino / Menina / Desconhecido |

> **Nota:** O campo `IAN` é intencionalmente omitido — é data leakage direto do target.

**Script Python:**
```python
import requests

payload = {
    "fase": 3, "idade": 12, "ano_ingresso": 2018,
    "inde": 6.5, "iaa": 7.0, "ieg": 5.5, "ips": 6.0,
    "ida": 6.2, "ipv": 7.5, "cg": 300, "cf": 10, "ct": 8,
    "matem": 6.5, "portug": 6.8,
    "pedra_22": "Ametista", "pedra_21": "Ametista", "genero": "Menino"
}

response = requests.post(
    "https://n8c8xksefj.execute-api.us-east-1.amazonaws.com/prod/predict",
    json=payload,
)
print(response.json())
```

---

## 6. Monitoramento Contínuo

### Logging estruturado

Cada predição é registrada em formato JSON nos logs da aplicação, capturando:
- Timestamp da requisição
- Features de entrada (valores recebidos)
- Probabilidade e classificação de saída
- Modelo utilizado

Em produção (AWS), os logs são capturados automaticamente pelo **Amazon CloudWatch** no grupo `/aws/sagemaker/Endpoints/passos-magicos-prod`.

Exemplo de entrada de log gerada a cada predição:
```json
{
  "event": "prediction",
  "timestamp": "2025-01-01T12:00:00",
  "input": {"fase": 3, "idade": 12, "inde": 6.5, "...": "..."},
  "output": {"risco_defasagem": 0.82, "classificacao": "Alto Risco"},
  "model": "Logistic Regression"
}
```

### Análise de Drift

O notebook `notebooks/monitoramento_drift.ipynb` implementa análise de drift comparando:
- Distribuição das features de entrada vs. distribuição de treino (teste KS)
- Distribuição das predições ao longo do tempo
- Alertas automáticos quando p-value < 0.05

### SageMaker Model Monitor

O notebook `notebooks/setup_model_monitor.ipynb` configura o **SageMaker Model Monitor** para monitoramento contínuo em produção:
- Criação do baseline estatístico a partir dos dados de treino
- DataCapture habilitado no endpoint (100% das requisições capturadas no S3)
- Agendamento de jobs de monitoramento para detecção automática de desvios

---

## 7. Links

- API em produção: https://n8c8xksefj.execute-api.us-east-1.amazonaws.com/prod/predict
- Documentação interativa (local): `http://localhost:8080/docs`
- Dataset e Dicionário: `database/base_2024.xlsx`
- Site Passos Mágicos: https://www.passosmagicos.org.br/

---

## 8. Time

| Nome | GitHub |
|---|---|
| Nome 1 | [@usuario1](https://github.com/) |

---

> *"A educação é a arma mais poderosa que você pode usar para mudar o mundo."*
