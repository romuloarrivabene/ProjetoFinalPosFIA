# Frontend de análise de crédito

Interface Streamlit que consome a API de risco sem carregar o modelo diretamente.

## Instalação

Na pasta `data-platform`:

```bash
.venv/bin/python -m pip install -r Model/frontend/requirements.txt
```

## Execução

Primeiro, inicie a API na pasta `data-platform/Model`:

```bash
../.venv/bin/python -m uvicorn api.main:app --reload
```

Em outro terminal, ainda na pasta `data-platform/Model`:

```bash
../.venv/bin/python -m streamlit run frontend/app.py
```

Acesse `http://localhost:8501`. Por padrão, a tela chama a API em `http://localhost:8000`.

Para usar outro endereço:

```bash
export CREDIT_API_URL="http://endereco-da-api:8000"
../.venv/bin/python -m streamlit run frontend/app.py
```

## Funcionalidades

- formulário com todas as 32 features do modelo;
- consulta por `customer_id` no PostgreSQL;
- verificação de disponibilidade da API;
- score, classe prevista e recomendação da política;
- exibição dos JSONs enviados e recebidos;
- aviso sobre o caráter demonstrativo e a ausência de calibração do score.

