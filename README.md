
# 🎓 Datathon Passos Mágicos — Previsão de Risco de Defasagem Escolar

> Transformando a vida de crianças e jovens por meio da educação e da tecnologia.

---

## 📌 Visão Geral do Projeto

A [Associação Passos Mágicos](https://www.passosmagicos.org.br/) atua há 32 anos na transformação de vida de crianças e jovens em vulnerabilidade social, oferecendo educação de qualidade, apoio psicológico/psicopedagógico e ampliação de visão de mundo.

### Problema de Negócio

Com base em dados de pesquisa extensiva do desenvolvimento educacional nos anos de **2022, 2023 e 2024**, este projeto desenvolve 
um **modelo preditivo capaz de estimar o risco de defasagem escolar** de cada estudante — permitindo intervenções mais rápidas e direcionadas.

### Solução Proposta

Construção de uma **pipeline completa de Machine Learning**, cobrindo desde o pré-processamento dos dados até o deploy do modelo 
em produção via API REST, com monitoramento contínuo e testes automatizados.

---

## 🛠️ Stack Tecnológica

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.14 |
| ML & Data | scikit-learn, pandas, numpy |
| API | FastAPI ou Flask |
| Serialização | joblib |
| Testes | pytest |
| Empacotamento | Docker + ECR |
| Deploy | AWS |
| Monitoramento | AWS Cloudwatch + Sagemaker Model Monitor |

---

## 📁 Estrutura do Projeto - OK

```
project-root/
│
├── .github/workflows/
│   └── deploy.yml           # Workflow para o deploy da infra na AWS
│
│
├── databse/
│   ├── base_2024.xlsx        # Base 2024
│   └── bases_antigas.zip     # Bases Antigas
│
├── infra/
│   └── cloudformation.py     # Entrypoint da aplicação
│
├── app/
│   ├── main.py               # Entrypoint da aplicação
│   ├── route.py              # Definição das rotas da API
│   └── model/                # Modelo serializado (.pkl / .joblib)
│
├── src/
│   ├── preprocessing.py      # Limpeza e transformação dos dados
│   ├── feature_engineering.py # Criação e seleção de features
│   ├── train.py              # Treinamento do modelo
│   ├── evaluate.py           # Avaliação e métricas
│   └── utils.py              # Funções auxiliares
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── notebooks/                # Análises exploratórias (EDA)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Pipeline de Machine Learning

### 1. Pré-processamento dos Dados
- Tratamento de valores nulos e outliers
- Normalização e encoding de variáveis categóricas
- Divisão treino/teste com estratificação

### 2. Engenharia de Features
- Criação de variáveis derivadas relevantes ao contexto educacional
- Seleção de features com base em importância e correlação

### 3. Treinamento e Validação
- Treinamento com cross-validation
- Otimização de hiperparâmetros
- Serialização do modelo com `pickle` ou `joblib`

### 4. Seleção de Modelo
- Comparação entre algoritmos (ex.: Random Forest, XGBoost, Logistic Regression)
- Justificativa da métrica de avaliação escolhida (ex.: F1-score, AUC-ROC) e confiabilidade para produção

### 5. Pós-processamento *(se aplicável)*
- Calibração de probabilidades
- Thresholding para classificação de risco

---

## 🚀 Instruções de Deploy

### Pré-requisitos

- Python 3.x
- Docker instalado
- Dependências listadas em `requirements.txt`

### Instalação local

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

# Instale as dependências
pip install -r requirements.txt
```

### Treinar o modelo

```bash
python src/train.py
```

### Subir a API localmente

```bash
uvicorn app.main:app --reload
# ou
python app/main.py
```

### Deploy com Docker

```bash
# Build da imagem
docker build -t passos-magicos-api .

# Execução do container
docker run -p 8000:8000 passos-magicos-api
```

---

## 🔌 Exemplos de Chamadas à API

### Endpoint: `POST /predict`

**Request (curl):**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 12,
    "ano_escolar": 7,
    "frequencia": 0.85,
    "nota_media": 6.2,
    "...": "..."
  }'
```

**Response:**
```json
{
  "risco_defasagem": 0.73,
  "classificacao": "Alto Risco",
  "confianca": 0.91
}
```

---

## 🧪 Testes Unitários

O projeto mantém **no mínimo 80% de cobertura** de testes unitários, validando cada componente da pipeline.

```bash
# Executar todos os testes
pytest tests/

# Com relatório de cobertura
pytest --cov=src tests/ --cov-report=term-missing
```

---

## 📊 Monitoramento Contínuo

- **Logs:** registrados via módulo `logging` do Python, com níveis `INFO`, `WARNING` e `ERROR`
- **Drift do Modelo:** painel de acompanhamento de data drift e performance ao longo do tempo
- **Métricas em produção:** monitoramento de distribuição das predições e alertas de degradação

---

## 📎 Links

- 🔗 [API em produção](#) *(substituir pelo link real)*
- 🎥 [Vídeo de apresentação (até 5 min)](#) *(substituir pelo link real)*
- 📊 [Dataset e Dicionário de Dados](#) *(substituir pelo link real)*
- 🌐 [Site Passos Mágicos](https://www.passosmagicos.org.br/)

---

## 👥 Time

| Nome | GitHub |
|---|---|
| Nome 1 | [@usuario1](https://github.com/) |
| Nome 2 | [@usuario2](https://github.com/) |

---

> *"A educação é a arma mais poderosa que você pode usar para mudar o mundo."*