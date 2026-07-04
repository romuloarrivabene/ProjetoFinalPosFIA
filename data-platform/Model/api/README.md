# API de risco de crédito

Uma única aplicação expõe o mesmo modelo de duas formas:

- `POST /predict/features`: recebe todas as features prontas;
- `POST /predict/customer/{customer_id}`: consulta o PostgreSQL e monta as features;
- `GET /model/features`: lista as features esperadas pelo modelo;
- `GET /health`: verifica se o artefato foi carregado.

Após o score do modelo, uma política separada produz uma recomendação:

- `approve`: score menor que `CREDIT_APPROVE_MAX_SCORE`;
- `manual_review`: score na faixa intermediária;
- `reject`: score igual ou maior que `CREDIT_MANUAL_REVIEW_MAX_SCORE`.

Os valores padrão (`0.35` e `0.65`) são demonstrativos e devem ser validados com custos e regras reais de negócio. Como o modelo foi treinado com `class_weight="balanced"`, o `risk_score` deve ser tratado como pontuação de ordenação até que haja calibração de probabilidade.

## Instalação

Na pasta `data-platform`:

```bash
.venv/bin/python -m pip install -r Model/api/requirements.txt
```

## Execução local

Na pasta `data-platform/Model`:

```bash
../.venv/bin/python -m uvicorn api.main:app --reload
```

Documentação interativa:

```text
http://localhost:8000/docs
```

## Configuração

```bash
export MODEL_PATH="/caminho/logistic_regression_abt.joblib"
export DATABASE_URL="postgresql+psycopg2://airflow:airflow@localhost:5432/data"
export CREDIT_APPROVE_MAX_SCORE="0.35"
export CREDIT_MANUAL_REVIEW_MAX_SCORE="0.65"
export CREDIT_POLICY_VERSION="demo-v1"
```

Dentro do Docker Compose, o host do banco deve ser `postgres`, não `localhost`.

## Exemplo usando um cliente do banco

```bash
curl -X POST http://localhost:8000/predict/customer/100002
```

## Exemplo usando features prontas

O corpo precisa incluir todos os nomes presentes em `artifact["input_features"]`:

```bash
curl -X POST http://localhost:8000/predict/features \
  -H "Content-Type: application/json" \
  -d @features.json
```

Formato de `features.json`:

```json
{
  "features": {
    "ext_source_2": 0.62,
    "age": 35,
    "years_employed": 5
  }
}
```

O exemplo abreviado acima retorna `422` até que todas as features obrigatórias sejam enviadas. Consulte `GET /model/features` para obter a lista completa e use `/docs` para testar a chamada.

## Exemplo de resposta

```json
{
  "source": "database",
  "customer_id": 100002,
  "risk_score": 0.71,
  "predicted_class": 1,
  "model_decision_threshold": 0.5,
  "policy": {
    "recommendation": "reject",
    "reason": "Score acima do limite máximo aceito pela política.",
    "policy_version": "demo-v1",
    "approve_max_score": 0.35,
    "manual_review_max_score": 0.65
  }
}
```

O modelo fornece risco; a política fornece recomendação. Em um cenário real, regras cadastrais, fraude, capacidade financeira, legislação e revisão humana ainda podem alterar a decisão final.
