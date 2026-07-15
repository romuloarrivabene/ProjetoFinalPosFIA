# MLOps

Esta pasta operacionaliza o modelo treinado por meio de uma API FastAPI e de uma interface Streamlit. A política de crédito permanece separada do score do modelo.

## Contexto e valor do componente

Um notebook ou arquivo Pickle, isoladamente, não permite que um analista utilize o modelo de forma consistente. A camada MLOps transforma o resultado da modelagem em um serviço com contrato explícito, validação das entradas, política configurável e interface de demonstração.

O desenho resolve quatro preocupações:

- **consistência:** toda predição passa pelo mesmo serviço de modelo;
- **separação de decisão:** score, classe estatística e recomendação de negócio são conceitos distintos;
- **reuso:** frontend, scripts e outros consumidores podem usar a mesma API;
- **demonstração auditável:** resposta, thresholds e versão da política são apresentados juntos.

A implementação é deliberadamente acadêmica: demonstra o serving do modelo e a integração entre componentes. Autenticação, registry, persistência estruturada de auditoria e monitoramento produtivo permanecem evoluções futuras.

## Arquitetura funcional completa

```text
Home Credit CSVs
  application_train | previous_application | bureau | installments_payments
                                      │
                                      ▼
                         Airflow / pipeline_orchestration
              ingestão → índices → limpeza → agregações → ABT → treino
                                      │
                 ┌──────────────────┴─────────────────┐
                 ▼                                   ▼
       PostgreSQL / banco data                 Model/artifacts
    raw → clean → application_abt        lightgbm_abt.pkl + metrics.json
                 │                                   │
                 └──────────────────┬─────────────────┘
                                    ▼
                            FastAPI / serving
                 FeatureService + PredictionService + CreditPolicy
                         │                         │
             Swagger / consumidores HTTP             └→ Streamlit
                         │                              │
                         └────── score + classe + política ─────┘
```

| Camada | Componente | Responsabilidade | Saída ou contrato |
|---|---|---|---|
| Origem | CSVs Home Credit | Fornecer cadastro, propostas, bureau e parcelas. | Quatro arquivos de entrada. |
| Orquestração | Airflow | Ordenar ingestão, tratamento, ABT e treinamento. | DAG `pipeline_orchestration`. |
| Persistência | PostgreSQL | Manter dados raw, clean e a visão por cliente. | `application_abt`. |
| Modelagem | LightGBM / `train.py` | Treinar, avaliar e empacotar o contrato de inferência. | `lightgbm_abt.pkl` e `metrics.json`. |
| Acesso a dados | `FeatureService` | Recuperar da ABT as mesmas features usadas no treino. | Dicionário de features por cliente. |
| Inferência | `PredictionService` | Validar o artefato, alinhar tipos e calcular o score. | `risk_score` e `predicted_class`. |
| Decisão | `CreditPolicy` | Traduzir o score em aprovação, revisão ou rejeição. | Recomendação e versão da política. |
| Exposição | FastAPI | Publicar contratos, saúde, features e predições. | HTTP/JSON e Swagger. |
| Experiência | Streamlit | Demonstrar consulta, edição e análise de clientes. | Interface para o analista. |

O fluxo contém dois ciclos. No ciclo **offline**, a DAG reconstrói os dados, materializa a ABT e gera o artefato. No ciclo **online**, a API combina as features da ABT ou do formulário com o artefato já treinado e aplica a política sem reexecutar o pipeline.

### Arquitetura do serviço de predição

```text
Formulário de features ───────────────┐
                                     ├→ PredictionService → CreditPolicy → FastAPI
PostgreSQL / application_abt → FeatureService                         │
                                                                      └→ Streamlit
```

- `FeatureService` recupera da ABT as mesmas features usadas no treinamento;
- `PredictionService` carrega o artefato LightGBM e calcula o score;
- `CreditPolicy` converte o score em `approve`, `manual_review` ou `reject`;
- FastAPI expõe os contratos;
- Streamlit oferece preenchimento manual, recuperação editável e consulta direta de clientes.

O score é uma pontuação de ordenação de risco, não uma probabilidade calibrada de inadimplência.

### Fluxo em camadas

Cada requisição de predição atravessa camadas com **responsabilidade única** — é isso que mantém o modelo isolado da regra de negócio e da apresentação:

```text
Streamlit  (apresentação)
   │  HTTP
   ▼
FastAPI    (contrato / transporte)
   │
   ├─ FeatureService ───→ acesso a dados: recupera as features do cliente na ABT
   ├─ PredictionService → inferência: alinha o contrato do artefato e calcula o score
   └─ CreditPolicy ─────→ regra de negócio: converte o score em recomendação
```

- **acesso a dados** (`feature_service`) e **inferência** (`model_service`) não conhecem regra de negócio;
- **política** (`credit_policy`) não conhece o modelo — recebe apenas um score;
- **transporte** (FastAPI) e **apresentação** (Streamlit) não contêm lógica de crédito.

### Decisões arquiteturais

- **Consistência treino ↔ inferência pela ABT.** O `feature_service` lê a **mesma** `application_abt` usada no treinamento; as features online são idênticas às offline **por construção**. A API **não re-implementa** a engenharia de atributos do pipeline, eliminando *training/serving skew*. Custo consciente: a predição por cliente depende de a ABT estar atualizada.
- **Modelo e política desacoplados.** O modelo entrega um **score de ordenação** (estável, versionado no artefato); a **política de crédito** o traduz em recomendação por **limiares configuráveis**, que mudam sem re-treinar. Por isso `predicted_class` (limiar do modelo) e `recommendation` (política) são conceitos distintos e podem divergir.
- **Contrato dirigido pelo artefato.** A lista de features, as categorias e o threshold viajam dentro do próprio artefato; a API valida e alinha a entrada contra esse contrato antes de pontuar. `schemas.py` formaliza o contrato HTTP e o frontend o consome — uma **fonte de verdade única** que flui de **treino → artefato → API → UI**.
- **Dependências inicializadas no startup e modelo sob demanda.** O serviço de modelo e a engine de banco são criados **uma vez** no `lifespan` e guardados em `app.state`. O artefato é carregado na primeira verificação de saúde ou predição depois de ser gerado pela DAG, e então reutilizado sem recarga por chamada. O pool usa `pool_pre_ping` para resiliência a conexões ociosas.
- **Três modos de consumo sobre o mesmo núcleo.** O caminho de predição (`_predict`) é único; muda apenas a **origem das features** — fornecidas pelo consumidor, recuperadas da ABT por `sk_id_curr`, ou recuperadas e **editadas** antes de reavaliar.

### Fluxo do contrato

O mesmo contrato de features atravessa treino, artefato e serviço — nada é redefinido no caminho:

```text
train.py  ──→  artefato .pkl  ──→  PredictionService  ──→  /model/features  ──→  Streamlit / field_config
(features,      (features,          (valida e alinha         (expõe o             (renderiza os
 categorias,     categorias,         a entrada ao             contrato)             mesmos campos)
 threshold)      threshold)          contrato)
```

## Aplicações implementadas

### API FastAPI (`app/api`)

Serviço de scoring que expõe o modelo como serviço de predição:

- **carga segura sob demanda** — serviço e conexão de banco são inicializados uma vez (`lifespan`); o artefato é carregado pelo `/health` ou pela primeira operação que precisa do modelo, e sua assinatura de arquivo permite detectar remoção ou uma nova versão gerada pela DAG;
- **documentação viva** — OpenAPI/Swagger em `/docs`, gerada a partir dos contratos de `schemas.py`;
- **capacidades** — *liveness* (`/health`), metadados de features (`/model/features`), recuperação das features de um cliente (`/customers/{id}/features`) e **dois modos de predição** (por features fornecidas e por cliente armazenado na ABT);
- **validação e erros tipados** — features obrigatórias ausentes → `422` com a lista; cliente inexistente → `404`; falha de banco → `503`; sem artefato, o `/health` informa `model_loaded=false` e a predição retorna `503`;
- **rastreabilidade** — cada predição é registrada em **JSON no stdout** do container (apoio a demonstração e diagnóstico; não substitui auditoria persistente);
- **separação de decisão** — a resposta traz, junto ao score, a recomendação da política e os limiares que a produziram.

### Interface Streamlit (`app/frontend`)

Simulador para o analista de crédito, que **consome a API** e nunca acessa o modelo diretamente:

- **barra lateral** — URL da API configurável e botão **"Verificar conexão"** (checa `/health` e se o modelo está carregado);
- **três abas** — *Preencher todos os dados*, *Buscar cliente e editar* e *Consultar cliente do banco*;
- **formulário dinâmico** — campos agrupados por contexto e gerados a partir de `field_config.py` (categóricos com opções controladas, flags binárias, numéricos com limites e passos);
- **jornada de edição** — carrega as features de um cliente da ABT, permite **ajustar** os campos e reavaliar, evidenciando o efeito de mudanças no score **sem alterar a ABT**;
- **visão do resultado** — faixa de recomendação (cor/ícone), métricas de score, classe prevista e origem, barra de posição na escala de risco, legenda com limiar e política, aviso de que o score **não é probabilidade calibrada** e os **JSONs** enviado e recebido.

## Responsabilidades e limites

| Componente | Responsabilidade | Não é responsabilidade |
|---|---|---|
| `feature_service` | Recuperar uma linha da ABT e preparar suas features. | Reexecutar a engenharia de atributos sobre as fontes brutas. |
| `model_service` | Validar o artefato, alinhar tipos e calcular score/classe. | Definir aprovação ou rejeição de negócio. |
| `credit_policy` | Traduzir faixas de score em recomendação demonstrativa. | Retreinar ou calibrar o modelo. |
| FastAPI | Gerenciar ciclo de vida, contratos e erros HTTP. | Armazenar histórico definitivo das decisões. |
| Streamlit | Oferecer jornadas de demonstração e explicar o resultado. | Conter o modelo ou acessar diretamente o Pickle. |

Essa separação evita acoplar mudanças da política comercial ao treinamento do algoritmo.

## Estrutura

```text
MLOps/
├── app/
│   ├── api/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── feature_service.py
│   │   ├── model_service.py
│   │   ├── credit_policy.py
│   │   └── requirements.txt
│   └── frontend/
│       ├── app.py
│       ├── field_config.py
│       └── requirements.txt
├── tests/
├── Dockerfile.api
├── Dockerfile.frontend
├── test-requirements.txt
└── README.md
```

## Configuração

| Variável | Finalidade | Padrão no Compose |
|---|---|---|
| `MODEL_PATH` | Caminho do artefato LightGBM | `/app/Model/artifacts/lightgbm_abt.pkl` |
| `DATABASE_URL` | Conexão com o banco `data` | PostgreSQL do Compose |
| `CREDIT_APPROVE_MAX_SCORE` | Limite superior para aprovação | `0.50` |
| `CREDIT_MANUAL_REVIEW_MAX_SCORE` | Limite superior para revisão manual | `0.60` |
| `CREDIT_POLICY_VERSION` | Identificador da política | `demo-v1` |
| `CREDIT_API_URL` | URL consumida pelo frontend | `http://credit-api:8000` |

Os limites são demonstrativos e devem ser validados com custos e regras reais do negócio.

## Implementações críticas

### Carregamento do modelo

No startup da FastAPI, o `lifespan`:

1. valida os limites da política;
2. cria o `PredictionService` com `MODEL_PATH`, sem exigir que o artefato já exista;
3. cria o engine SQLAlchemy com `pool_pre_ping=True`;
4. instancia serviço de features e política;
5. registra os serviços em `app.state` para reuso pelas requisições;
6. libera o pool de conexões no shutdown.

O `/health` tenta carregar e validar o Pickle quando ele está disponível. Isso permite iniciar a plataforma antes do treinamento e faz com que o botão **Verificar conexão** do Streamlit reconheça o modelo sem exigir uma predição anterior. Se o arquivo ainda não foi gerado ou foi removido, a API permanece ativa com `model_loaded=false`; endpoints de predição retornam `503`. Quando a DAG cria ou substitui o arquivo, a API compara tamanho e data de modificação e carrega a versão atual antes de responder.

O artefato precisa conter modelo, threshold, métricas e uma lista de features. Para compatibilidade, o serviço aceita a chave atual `features` ou a chave histórica `input_features`.

### Preparação das features para inferência

Antes do `predict_proba`, o `PredictionService`:

- rejeita requisições que não contenham todas as features obrigatórias;
- reorganiza as colunas exatamente na ordem do treinamento;
- ignora campos extras durante o reindex;
- restaura `pandas.Categorical` com as categorias salvas no artefato;
- converte as demais features para tipo numérico;
- calcula `risk_score` a partir da classe positiva;
- compara o score com o threshold persistido para gerar `predicted_class`.

Restaurar categorias é essencial para o LightGBM com categóricas nativas: o mesmo texto precisa ocupar a mesma categoria lógica usada durante o ajuste.

### Recuperação do cliente

O `CustomerFeatureService` consulta diretamente `application_abt` por `sk_id_curr`. Identificador e target são removidos antes do retorno. O serviço garante ainda a presença das features de parcelas por compatibilidade com ABTs materializadas anteriormente.

Consumir a ABT evita duplicar na API as regras complexas de agregação do pipeline. A desvantagem consciente é que uma predição por cliente depende da atualização prévia da ABT.

### Política de crédito

O `CreditPolicy` recebe dois limites validados:

```text
score < approve_max_score
  → approve

approve_max_score ≤ score < manual_review_max_score
  → manual_review

score ≥ manual_review_max_score
  → reject
```

A resposta inclui limites e `policy_version`, tornando explícita a regra que produziu a recomendação. O `predicted_class` continua baseado no threshold do modelo e pode divergir da recomendação, pois atende a outra finalidade.

### Tratamento de erros

| Situação | Resposta |
|---|---|
| Cliente inexistente na ABT | HTTP `404`. |
| Falha ao consultar PostgreSQL | HTTP `503`. |
| Features obrigatórias ausentes | HTTP `422` com lista das ausências. |
| Artefato ausente ou inválido no startup | API não conclui a inicialização. |

As requisições de predição são registradas em JSON no stdout do container para apoiar demonstração e diagnóstico. Esse registro não substitui uma trilha de auditoria persistente.

## Inicialização com Docker

Na pasta [`data-platform`](../README.md):

```bash
docker compose up -d --build postgres credit-api credit-frontend
```

Para acompanhar os serviços:

```bash
docker compose logs -f credit-api credit-frontend
```

## URLs

| Serviço | URL |
|---|---|
| Documentação Swagger | http://localhost:8000/docs |
| Health check da API | http://localhost:8000/health |
| Streamlit | http://localhost:8501 |

## Endpoints

| Método e caminho | Finalidade |
|---|---|
| `GET /health` | Informa disponibilidade e carregamento do modelo. |
| `GET /model/features` | Lista as features esperadas pelo modelo. |
| `GET /customers/{customer_id}/features` | Recupera as features de um cliente para edição. |
| `POST /predict/features` | Calcula o score a partir das features fornecidas. |
| `POST /predict/customer/{customer_id}` | Recupera o cliente na ABT e calcula o score. |

### Estrutura da requisição por features

O endpoint recebe um objeto `features` com todas as entradas listadas por `GET /model/features`:

```json
{
  "features": {
    "ext_source_1": 0.50,
    "ext_source_2": 0.62,
    "ext_source_3": 0.48,
    "ext_source_mean": 0.53,
    "age": 35.0,
    "occupation_type": "Laborers"
  }
}
```

O exemplo é abreviado para leitura; uma chamada válida deve incluir todas as features retornadas pelo endpoint de metadados.

### Estrutura da resposta

```json
{
  "source": "provided_features",
  "customer_id": null,
  "risk_score": 0.55,
  "predicted_class": 1,
  "model_decision_threshold": 0.5,
  "policy": {
    "recommendation": "manual_review",
    "reason": "Score na faixa intermediária; requer análise humana.",
    "policy_version": "demo-v1",
    "approve_max_score": 0.50,
    "manual_review_max_score": 0.60
  }
}
```

`source` informa se a pontuação veio do formulário ou do banco. Quando a consulta parte de um cliente armazenado, `customer_id` permite associar o resultado à origem.

## Jornadas do frontend

O Streamlit implementa três formas de demonstração:

### Preencher todos os dados

Renderiza as features agrupadas por contexto. Campos categóricos usam opções controladas, flags usam seleção binária e valores numéricos respeitam limites e passos definidos em `field_config.py`.

### Buscar cliente e editar

Recupera as features com `GET /customers/{id}/features`, mantém o cliente no `session_state`, preenche um novo formulário e permite simular mudanças antes da predição. Essa jornada evidencia como alterações cadastrais ou financeiras afetam o score sem modificar a ABT.

### Consultar cliente do banco

Envia apenas o identificador para `POST /predict/customer/{id}`. A API recupera a ABT e calcula a recomendação sem edição manual.

Em todas as jornadas, o frontend exibe score, classe, origem, threshold do modelo, limites da política, justificativa e resposta JSON completa. Uma mensagem fixa reforça que o score não é probabilidade calibrada.

## Empacotamento

### API

`Dockerfile.api` instala somente as dependências da API, copia `MLOps` e os artefatos de `Model/artifacts`, define `MODEL_PATH` e inicia Uvicorn na porta 8000.

### Frontend

`Dockerfile.frontend` instala Streamlit e Requests, copia a aplicação e inicia o servidor na porta 8501. A comunicação interna usa o DNS do Compose: `http://credit-api:8000`.

Como o código é copiado durante o build, alterações locais exigem reconstrução da imagem correspondente.

## Execução local

Com PostgreSQL e artefato disponíveis:

```bash
cd data-platform
python3 -m venv MLOps/.venv
MLOps/.venv/bin/python -m pip install -r MLOps/app/api/requirements.txt
MLOps/.venv/bin/python -m uvicorn MLOps.app.api.main:app --reload
```

Em outro terminal:

```bash
cd data-platform
MLOps/.venv/bin/python -m pip install -r MLOps/app/frontend/requirements.txt
CREDIT_API_URL=http://localhost:8000 \
  MLOps/.venv/bin/python -m streamlit run MLOps/app/frontend/app.py
```

## Testes

```bash
cd data-platform
MLOps/.venv/bin/python -m pip install -r MLOps/test-requirements.txt
MLOps/.venv/bin/python -m pip install -r MLOps/app/frontend/requirements.txt
MLOps/.venv/bin/python -m unittest discover -s MLOps/tests -v
```

### Cobertura dos testes existentes

| Arquivo | Responsabilidade validada |
|---|---|
| `test_credit_policy.py` | Faixas de aprovação, revisão, rejeição e limites inválidos. |
| `test_model_service.py` | Score válido e rejeição de features ausentes. |
| `test_predict.py` | Inferência pelo script local e contrato do resultado. |
| `test_health.py` | Carga do artefato pelo health check e estado anterior ao treinamento. |
| `test_frontend.py` | Inicialização da aplicação Streamlit. |
| `test_configuration.py` | Estrutura esperada e coerência entre configuração e artefato. |

## Limitações conhecidas

- a política usa limites demonstrativos;
- o score não está calibrado como probabilidade;
- não há autenticação ou autorização nos endpoints;
- requisições e respostas não são persistidas em armazenamento de auditoria;
- a API depende da disponibilidade da ABT no PostgreSQL;
- o artefato é empacotado na imagem e não obtido de um model registry;
- não há monitoramento contínuo de drift, latência ou performance pós-deploy.

## Próximos passos

Além de calibração do score, autenticação e adoção de um *model registry*, dois eixos completam a proposta de arquitetura (itens iii e iv do escopo individual).

### iii. Monitoramento em produção

O objetivo é detectar **falhas, perda de performance e mudança de comportamento dos dados** antes que afetem a decisão de crédito. Como a base é **transversal (sem datas absolutas de originação)**, o desenho usa **lotes de novas aplicações comparados ao baseline versionado de treino**. Monitoramento por safra passa a ser adotado quando a produção registrar datas de originação e maturidade do contrato.

#### Dados necessários

Cada inferência deve persistir, com acesso controlado: `prediction_id`, cliente anonimizado, timestamp, versão do modelo, versão da política, features ou estatísticas permitidas, score, recomendação, latência e código HTTP. Quando o resultado real amadurecer, a inadimplência observada deve ser associada ao `prediction_id`. O baseline de treino deve armazenar distribuições, schema e as métricas oficiais do modelo.

As métricas calculadas por lote seriam gravadas em uma tabela `model_monitoring_metrics`, permitindo dashboard no Metabase, histórico de alertas e auditoria. Métricas de infraestrutura poderiam ser coletadas por Prometheus/Grafana em uma evolução produtiva.

#### Plano operacional proposto

| Dimensão | Indicador e fonte | Frequência | Alerta inicial proposto | Ação | Responsável |
|---|---|---|---|---|---|
| Disponibilidade | `/health`, estado dos containers e disponibilidade do PostgreSQL | Contínua, a cada 1 min | Duas falhas consecutivas ou modelo indisponível após o treino | Reiniciar serviço; verificar volume, artefato e banco; escalar incidente | MLOps |
| Erros da API | Percentual de respostas `5xx` nos logs | Janela de 5 min | `5xx > 2%` | Bloquear automações, preservar revisão humana e investigar dependências | MLOps |
| Latência | p95 de `/predict/*` | Janela de 15 min | `p95 > 1 s` | Verificar banco, pool e recursos; aplicar degradação segura | MLOps |
| Pipeline | Estado e duração das tasks no Airflow | Por execução | Qualquer task falha ou duração `> 150%` da mediana histórica | Não publicar novo artefato; repetir etapa idempotente e abrir incidente | Engenharia de Dados |
| Contrato dos dados | Colunas, tipos, nulos, categorias desconhecidas e volume | Em cada lote, antes do scoring | Coluna obrigatória ausente; tipo incompatível; nulos `> 5 p.p.` do baseline; volume fora de `±30%` | Rejeitar ou colocar o lote em quarentena e acionar Engenharia de Dados | Engenharia de Dados |
| Drift de dados | PSI das principais features contra o treino | Em cada lote | `0,10 ≤ PSI < 0,25`: atenção; `PSI ≥ 0,25`: crítico | Investigar origem; revisar regras; iniciar avaliação de retreino | Dados + Risco |
| Drift do score | PSI da distribuição de `risk_score` | Diário ou por lote | Mesmas faixas de PSI | Revisar mix de clientes, qualidade das features e política | MLOps + Risco |
| Desempenho | AUC e KS com rótulos maduros; baseline atual AUC `0,7593` e KS `0,3980` | Mensal, após maturidade mínima | Queda absoluta `> 0,05` em AUC ou KS | Suspender decisão automática, analisar segmentos e avaliar challenger ou retreino | Ciência de Dados + Risco |
| Calibração | Brier e curva de calibração; baseline Brier `0,1908` | Mensal, com rótulos maduros | Brier piora `> 10%` relativo ou desvio sistemático da curva | Recalibrar score; não o comunicar como probabilidade até validação | Ciência de Dados |
| Decisão e negócio | Aprovação, revisão, rejeição e inadimplência dos aprovados | Diária; inadimplência mensal | Variação relativa `> 20%` contra baseline ou política | Validar mudança populacional e revisar limites com o negócio | Risco/Crédito |
| Fairness | AUC, TPR/FPR e taxa de decisão por subgrupo permitido | Mensal | Diferença entre grupos `> 10 p.p.` ou degradação persistente | Revisão de governança, análise de causa e supervisão humana reforçada | Risco + Governança |

Os limites acima são **hipóteses operacionais iniciais**, não regras regulatórias nem valores definitivamente aprovados. Eles devem ser ajustados com testes de carga, apetite de risco, custos reais, volume produtivo e validação das áreas de Crédito e Governança.

#### Fluxo de alerta e resposta

```text
Predições + logs da API + Airflow + desfechos reais
                         │
                         ▼
           jobs de qualidade e monitoramento por lote
                         │
                         ▼
       model_monitoring_metrics → dashboard Metabase
                         │
               normal / atenção / crítico
                         │
                         ▼
       alerta + ticket + responsável + evidências
                         │
       corrigir dados / ajustar serviço / avaliar retreino
```

Um alerta pode abrir automaticamente uma execução de avaliação ou treinar um modelo *challenger*, mas **não deve promover sozinho um novo modelo**. A publicação exige comparação com o modelo vigente, testes de contrato, desempenho e fairness, registro da versão e aprovação humana de Risco.

### iv. Ações automatizadas a partir das previsões

O componente `CreditPolicy` já implementa o primeiro mecanismo: transforma o score nas faixas `approve`, `manual_review` e `reject`, mantendo a regra de negócio fora do modelo. Em uma evolução produtiva, cada resposta da API produziria um evento durável para acionar o restante da jornada sem aumentar a latência da predição.

#### Fluxo proposto

```text
FastAPI calcula score + CreditPolicy define recomendação
                         │
                         ▼
         PredictionEvent persistido em outbox transacional
                         │
                         ▼
              worker/orquestrador de decisões
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
     approve            manual_review            reject
   esteira rápida       fila priorizada       revisão/justificativa
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
          agente gera resumo explicável para o analista
                            │
                            ▼
              decisão humana + trilha de auditoria
```

Para a demonstração acadêmica, a resposta HTTP e o log JSON representam esse evento. A primeira implementação produtiva pode usar uma tabela de *outbox* no PostgreSQL e um worker idempotente. Em maior escala, a outbox pode publicar em Kafka ou RabbitMQ sem alterar o contrato do modelo.

#### Ações por faixa da política

| Recomendação | Ação automatizada | Verificação obrigatória | Resultado |
|---|---|---|---|
| `approve` | Encaminhar para esteira rápida e solicitar validações cadastrais, antifraude e capacidade de pagamento. | Todas as regras mandatórias e a alçada de crédito precisam ser satisfeitas. | Proposta pronta para confirmação ou exceção enviada ao analista. |
| `manual_review` | Criar caso na fila humana, priorizado por risco, valor e tempo de espera. | Analista revisa documentos, drivers e regras não representadas pelo modelo. | Decisão humana registrada com justificativa. |
| `reject` | Criar caso com os fatores de risco permitidos e minuta de justificativa. | Revisão de regras, conformidade e possibilidade de contestação; o agente não comunica nem efetiva sozinho. | Confirmação humana ou devolução para nova análise. |
| Alerta crítico de monitoramento | Desabilitar ações automáticas e direcionar todas as propostas à revisão. | MLOps e Risco investigam o alerta. | Operação em modo seguro até liberação formal. |

#### Contrato do evento

O evento deve carregar somente o necessário para executar e auditar a automação:

```json
{
  "event_id": "uuid",
  "event_type": "credit_prediction_completed",
  "occurred_at": "2026-07-15T18:00:00Z",
  "prediction_id": "uuid",
  "customer_ref": "identificador_anonimizado",
  "model_version": "1.0.0",
  "policy_version": "demo-v1",
  "risk_score": 0.55,
  "recommendation": "manual_review",
  "top_drivers": [],
  "requested_action": "create_review_case"
}
```

`event_id` é a chave de idempotência: reprocessar o mesmo evento não pode duplicar casos, mensagens ou decisões. `top_drivers` somente é preenchido por um serviço de explicabilidade aprovado; o agente não pode inventar causas a partir do score.

#### Agente de apoio ao analista

O agente de IA tem função **redacional e assistiva**, sem autoridade para aprovar, rejeitar, alterar score ou mudar limites. Ele recebe:

- score e faixa da política;
- versões do modelo e da política;
- principais drivers calculados por SHAP ou outro explicador validado;
- regras aplicadas, documentos disponíveis e dados cadastrais estritamente necessários;
- base de conhecimento versionada com políticas e textos autorizados.

A resposta deve obedecer a um schema estruturado:

```json
{
  "summary": "Resumo objetivo para o analista.",
  "risk_drivers": ["driver calculado e sua direção"],
  "applied_rules": ["regra e versão"],
  "missing_information": ["documento ou verificação pendente"],
  "suggested_next_action": "review_documents",
  "requires_human_review": true
}
```

O resumo explica evidências já calculadas; ele não deve afirmar causalidade, criar motivos de recusa nem apresentar o score como probabilidade calibrada.

#### Guardrails e auditoria

- **decisão humana preservada:** `requires_human_review` não pode ser removido pelo agente;
- **minimização de dados:** mascarar identificadores e não enviar atributos sensíveis sem base e finalidade aprovadas;
- **saída validada:** rejeitar respostas fora do schema, com drivers inexistentes ou regras sem versão;
- **ferramentas permitidas:** o agente consulta apenas fontes internas autorizadas e não executa pagamentos, contratos ou alterações cadastrais;
- **rastreabilidade:** registrar prompt versionado, contexto permitido, resposta, modelo de IA utilizado, timestamps e decisão posterior do analista;
- **segregação:** mudanças no modelo de risco, na política e no prompt seguem aprovações independentes;
- **modo seguro:** indisponibilidade do agente, baixa qualidade ou alerta crítico gera resumo determinístico e encaminhamento humano, nunca aprovação automática.

#### Falhas e recuperação

| Falha | Tratamento automático | Fallback seguro |
|---|---|---|
| Evento não publicado | Outbox repete com *backoff* e limite de tentativas. | Caso permanece pendente e alerta MLOps. |
| Evento duplicado | Worker verifica `event_id`. | Retorna o resultado anterior sem repetir ações. |
| Agente indisponível | Repetição limitada e circuit breaker. | Template determinístico com score, política e encaminhamento humano. |
| Resposta inválida do agente | Validação de schema e drivers contra a fonte. | Descartar texto e criar revisão sem resumo gerativo. |
| Política ou modelo sem versão | Bloquear processamento. | Revisão humana e incidente de configuração. |
| Drift ou queda de desempenho crítica | Abrir avaliação e treinar challenger. | Manter modelo vigente ou modo manual; nunca promover automaticamente. |

#### Sequência de evolução

1. Persistir predições e versões em trilha de auditoria.
2. Implementar outbox e worker idempotente para criar a fila de revisão.
3. Adicionar explicações locais validadas e versionadas.
4. Integrar o agente em modo sombra, comparando seus resumos com os dos analistas.
5. Liberar uso assistivo após testes de qualidade, segurança, viés e aprovação de Governança.
6. Conectar alertas ao treinamento de challenger, mantendo a promoção sob aprovação humana.

Essa proposta conecta ML, automação e agente de IA sem transferir a decisão de crédito para o modelo generativo.

## Componentes relacionados

- [Modelo](../Model/README.md)
- [PostgreSQL](../postgres/README.md)
- [Airflow](../airflow/README.md)
- [Arquitetura da plataforma](../README.md)
