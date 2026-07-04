# Guia do notebook de treinamento da ABT

Este documento explica o notebook `model_training_abt.ipynb` de ponta a ponta. O objetivo é registrar não apenas o que o código faz, mas também por que cada etapa existe.

## Visão geral

O notebook transforma a ABT em um modelo que estima a probabilidade de inadimplência de um cliente.

```text
abt.csv
   ↓
Separação entre características e resposta
   ↓
Separação entre treino e teste
   ↓
Tratamento de dados numéricos e categóricos
   ↓
Treinamento da regressão logística
   ↓
Probabilidade de inadimplência
   ↓
Avaliação do modelo
   ↓
Persistência do modelo treinado
```

## 1. O que é a ABT?

ABT significa *Analytical Base Table*. Cada linha representa um cliente e cada coluna representa uma informação conhecida sobre ele.

Exemplo simplificado:

| Cliente | Idade | Renda | Score | Contratos recusados | Target |
|---|---:|---:|---:|---:|---:|
| 100002 | 25 | 202.500 | 0,26 | 0 | 1 |
| 100003 | 45 | 270.000 | 0,62 | 0 | 0 |

O `target` é aquilo que queremos prever:

- `0`: cliente adimplente;
- `1`: cliente inadimplente.

As demais colunas são as variáveis explicativas, também chamadas de *features*.

## 2. Carregamento dos dados

O notebook procura o arquivo `DataPipeline/abt.csv` e o carrega em um DataFrame:

```python
df = pd.read_csv(ABT_PATH, encoding="utf-8")
```

A ABT utilizada possui aproximadamente 307 mil clientes e 34 colunas.

## 3. Separação entre X e y

O notebook executa:

```python
X = df_model.drop(columns=["target", "sk_id_curr"])
y = df_model["target"]
```

- `X` contém as características utilizadas para fazer a previsão;
- `y` contém a resposta correta que o modelo precisa aprender.

O `sk_id_curr` é removido porque é apenas um identificador. Seu valor numérico não representa risco de crédito.

### Exemplo de X e y

Considere esta ABT reduzida:

| sk_id_curr | age | amt_income_total | ext_source_2 | occupation_type | target |
|---:|---:|---:|---:|---|---:|
| 100001 | 25 | 3.000 | 0,30 | Laborers | 1 |
| 100002 | 42 | 8.000 | 0,75 | Managers | 0 |
| 100003 | 31 | 4.500 | 0,55 | Sales staff | 0 |

O conteúdo de `X` será:

| age | amt_income_total | ext_source_2 | occupation_type |
|---:|---:|---:|---|
| 25 | 3.000 | 0,30 | Laborers |
| 42 | 8.000 | 0,75 | Managers |
| 31 | 4.500 | 0,55 | Sales staff |

O conteúdo de `y` será:

```text
0    1
1    0
2    0
```

Durante o treinamento, o modelo aprende a relação:

```text
X                                        y
características do cliente  ──────────► resultado conhecido
idade, renda, score, profissão           pagou ou não pagou
```

Para um cliente novo, temos apenas suas características:

```python
novo_cliente = {
    "age": 28,
    "amt_income_total": 3500,
    "ext_source_2": 0.35,
    "occupation_type": "Laborers",
}
```

Ainda não existe um `y` conhecido para esse cliente. O modelo utilizará `X` para estimar a probabilidade de `y = 1`.

Uma forma simples de memorizar:

- `X` = pistas;
- `y` = resposta;
- treinamento = aprender a relação entre as pistas e a resposta.

## 4. Separação entre treino e teste

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)
```

O notebook utiliza:

- 80% dos clientes para treinar;
- 20% dos clientes para testar.

O conjunto de teste fica isolado durante o treinamento. Ele simula clientes que o modelo nunca viu.

`stratify=y` preserva aproximadamente a mesma proporção de adimplentes e inadimplentes nos dois conjuntos.

Avaliar o modelo com os mesmos dados utilizados no treinamento produziria um resultado artificialmente otimista.

## 5. Variáveis numéricas e categóricas

O notebook separa automaticamente as colunas:

```python
numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number", "bool"]
).columns.tolist()
```

Exemplos numéricos:

- idade;
- renda;
- valor do crédito;
- quantidade de contratos;
- scores.

Exemplos categóricos:

- profissão;
- tipo de renda;
- escolaridade;
- organização;
- gênero.

Cada tipo de variável exige um tratamento diferente.

## 6. Tratamento das variáveis numéricas

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

### Imputação

Valores ausentes são preenchidos com a mediana calculada somente sobre o conjunto de treino. A mediana é mais resistente a valores extremos do que a média.

### Padronização

O `StandardScaler` coloca as variáveis em escalas comparáveis.

Sem a padronização, uma renda de `200000` poderia parecer numericamente mais importante do que um score entre `0` e `1` apenas por causa da unidade utilizada.

## 7. Tratamento das variáveis categóricas

```python
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=20,
    )),
])
```

Valores ausentes são preenchidos com a categoria mais frequente.

Depois, o One-Hot Encoding transforma categorias em colunas binárias. Por exemplo:

```text
code_gender = F
```

torna-se algo semelhante a:

```text
code_gender_F = 1
code_gender_M = 0
```

`handle_unknown="ignore"` permite que o modelo receba uma categoria futura que não apareceu no treino sem quebrar.

`min_frequency=20` agrupa o tratamento de categorias muito raras, reduzindo dimensionalidade e instabilidade.

## 8. Prevenção de vazamento de dados

Todo o pré-processamento está dentro de um `Pipeline`:

```python
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(...)),
])
```

Isso garante que medianas, escalas e categorias sejam aprendidas somente com os dados de treino.

Se esses valores fossem calculados utilizando também o conjunto de teste, informações do teste vazariam para o treinamento. Esse problema é chamado de *data leakage*.

## 9. Regressão logística

A regressão logística estima uma probabilidade entre `0` e `1`.

```text
Cliente A → 0,08 → 8% de risco
Cliente B → 0,71 → 71% de risco
```

O modelo aprende um peso para cada variável. De maneira simplificada:

```text
pontuação =
    peso_idade × idade
  + peso_score × score
  + peso_renda × renda
  + ...
```

Essa pontuação é convertida em probabilidade pela função logística.

## 10. Classes desbalanceadas

Na ABT, aproximadamente:

- 92% dos clientes são adimplentes: `target = 0`;
- 8% dos clientes são inadimplentes: `target = 1`.

Em 100 clientes, teríamos algo semelhante a:

| Classe | Quantidade |
|---|---:|
| Adimplentes (`0`) | 92 |
| Inadimplentes (`1`) | 8 |

O modelo recebe muito mais exemplos de adimplentes. Isso é chamado de desbalanceamento de classes.

### Por que isso é um problema?

Imagine um modelo que responda sempre que todos os clientes são adimplentes.

Ele acertaria 92 clientes e erraria 8, atingindo:

```text
92 acertos / 100 clientes = 92% de acurácia
```

Apesar da acurácia alta, ele não encontraria nenhum inadimplente:

```text
Recall dos inadimplentes = 0%
```

Por isso, a acurácia isolada pode ser enganosa em bases desbalanceadas.

### Como o modelo aprende?

Durante o treinamento, o modelo:

1. faz uma previsão;
2. compara a previsão com a resposta correta;
3. calcula o erro;
4. ajusta seus coeficientes para reduzir os erros.

Sem pesos, errar um adimplente e errar um inadimplente têm aproximadamente a mesma importância matemática. Como existem muitos mais adimplentes, os erros da classe `0` dominam o treinamento.

### O que faz class_weight="balanced"?

```python
LogisticRegression(class_weight="balanced")
```

Essa configuração aumenta o peso da classe rara e diminui o peso da classe frequente.

O peso é calculado aproximadamente por:

```text
peso da classe =
    total de registros
    ───────────────────────────────
    quantidade de classes × registros da classe
```

Em uma base com 100 clientes:

```text
peso da classe 0 = 100 / (2 × 92) ≈ 0,54
peso da classe 1 = 100 / (2 × 8)  = 6,25
```

| Classe | Peso aproximado |
|---|---:|
| Adimplente (`0`) | 0,54 |
| Inadimplente (`1`) | 6,25 |

Assim, errar um inadimplente gera uma penalização muito maior.

```text
6,25 / 0,54 ≈ 11,5
```

Errar um inadimplente pesa aproximadamente 11,5 vezes mais porque existem aproximadamente 11,5 adimplentes para cada inadimplente.

### Isso duplica os inadimplentes?

Não. `class_weight="balanced"` não altera a quantidade de linhas da ABT.

É diferente de:

- oversampling, que duplica ou cria exemplos da classe minoritária;
- undersampling, que remove exemplos da classe majoritária;
- SMOTE, que cria exemplos sintéticos.

O peso apenas muda quanto cada erro influencia o treinamento.

### Efeito esperado

O modelo tende a marcar mais clientes como potencialmente inadimplentes. Normalmente isso provoca:

- aumento do recall dos inadimplentes;
- redução dos falsos negativos;
- aumento dos falsos positivos;
- possível redução da acurácia;
- possível redução da precision.

É uma troca entre encontrar mais inadimplentes e gerar mais falsos alarmes.

### Resultado obtido

O modelo apresentou aproximadamente:

```text
Recall dos inadimplentes:    66,34%
Precision dos inadimplentes: 15,37%
```

O recall significa que, de cada 100 inadimplentes reais, aproximadamente 66 foram identificados.

A precision significa que, de cada 100 clientes marcados como inadimplentes, aproximadamente 15 realmente eram inadimplentes.

O modelo foi incentivado a encontrar a classe rara e, por isso, aceita mais falsos alarmes.

### Falso positivo e falso negativo

Falso positivo:

```text
O cliente pagaria corretamente,
mas o modelo o considera arriscado.
```

Possíveis consequências:

- crédito recusado desnecessariamente;
- perda de receita;
- perda de um bom cliente.

Falso negativo:

```text
O cliente ficará inadimplente,
mas o modelo o considera seguro.
```

Possíveis consequências:

- crédito aprovado indevidamente;
- perda financeira;
- custos de cobrança;
- aumento da inadimplência da carteira.

Normalmente, o falso negativo é mais caro para o banco, mas isso depende do valor do empréstimo e da política de crédito.

### Peso de classe não é o mesmo que limiar

Existem duas decisões diferentes:

```python
class_weight="balanced"
```

Muda como o modelo aprende.

```python
DECISION_THRESHOLD = 0.50
```

Muda como a probabilidade é convertida em decisão.

| Probabilidade | Limiar 0,50 | Limiar 0,30 |
|---:|---|---|
| 0,20 | Adimplente | Adimplente |
| 0,35 | Adimplente | Inadimplente |
| 0,60 | Inadimplente | Inadimplente |

Diminuir o limiar tende a aumentar o recall porque mais clientes são classificados como risco.

Forma resumida:

```text
class_weight → altera o aprendizado
threshold    → altera a decisão final
```

### Ideia principal

Sem balanceamento:

```text
Quero acertar o maior número total de clientes.
```

Com balanceamento:

```text
Não quero que a classe rara seja ignorada.
Errar um inadimplente deve pesar mais.
```

## 11. Limiar de decisão

O modelo produz uma probabilidade:

```python
y_probability = model.predict_proba(X_test)[:, 1]
```

Depois, o notebook aplica o limiar:

```python
y_prediction = (y_probability >= 0.50).astype(int)
```

- probabilidade menor que 50%: classe `0`;
- probabilidade igual ou maior que 50%: classe `1`.

O limiar não precisa permanecer em `0,50`. Ele deve ser escolhido considerando os custos de falsos positivos e falsos negativos para o negócio.

## 12. Métricas

O resultado inicial do modelo foi:

```text
ROC AUC: 0,7301
Average Precision: 0,2113
Recall dos inadimplentes: 0,6634
```

### ROC AUC

Mede a capacidade de ordenar clientes por risco:

- `0,50`: comportamento aleatório;
- `1,00`: separação perfeita;
- `0,73`: baseline razoável.

### Recall

Dos inadimplentes reais, aproximadamente 66% foram identificados.

### Precision

Entre os clientes classificados como inadimplentes, aproximadamente 15% realmente eram inadimplentes.

### Average Precision

Resume o equilíbrio entre precision e recall da classe minoritária. É especialmente relevante para este problema devido ao desbalanceamento.

## 13. Matriz de confusão

| Situação | Significado |
|---|---|
| Verdadeiro negativo | Adimplente corretamente classificado |
| Falso positivo | Adimplente considerado arriscado |
| Falso negativo | Inadimplente considerado seguro |
| Verdadeiro positivo | Inadimplente corretamente identificado |

No crédito, falsos negativos costumam ser caros porque o banco aprova crédito para alguém que posteriormente não paga.

## 14. Coeficientes

O notebook apresenta as variáveis com maiores coeficientes absolutos.

- coeficiente positivo: aumenta a pontuação associada à inadimplência;
- coeficiente negativo: reduz essa pontuação;
- valor absoluto alto: maior influência no modelo.

Os coeficientes representam associação, não causalidade. Uma variável pode acompanhar o risco sem necessariamente provocá-lo.

## 15. Persistência do modelo

O arquivo `.joblib` guarda:

```python
artifact = {
    "model": model,
    "decision_threshold": DECISION_THRESHOLD,
    "input_features": X.columns.tolist(),
    "metrics": {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    },
}
```

O artefato contém:

- tratamento de valores ausentes;
- padronização;
- One-Hot Encoding;
- regressão logística;
- limiar de decisão;
- lista de colunas esperadas;
- métricas registradas.

Para prever um novo cliente, não é necessário treinar novamente:

```python
artifact = joblib.load("logistic_regression_abt.joblib")
probabilidade = artifact["model"].predict_proba(novo_cliente)[:, 1]
```

## Resumo final

O notebook aprende padrões históricos que relacionam as características dos clientes à inadimplência. Depois, utiliza esses padrões para estimar a probabilidade de risco de novos clientes.

O pipeline completo garante que o mesmo tratamento aplicado durante o treinamento seja reutilizado durante futuras previsões.

---

# Aprofundamentos e integração com a API

As seções seguintes registram as dúvidas levantadas após a construção inicial do notebook e conectam o treinamento à utilização do modelo em uma aplicação.

## 16. Configurações gerais do experimento

No início do notebook são definidas estas constantes:

```python
RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50
```

### RANDOM_STATE

Controla a aleatoriedade do processo. A separação entre treino e teste envolve um embaralhamento dos clientes. Com um valor fixo, os mesmos clientes são selecionados em todas as execuções.

```text
Execução 1 → mesmos clientes no treino e no teste
Execução 2 → mesmos clientes no treino e no teste
Execução 3 → mesmos clientes no treino e no teste
```

O número `42` não possui significado matemático especial. Poderia ser qualquer inteiro. O importante é mantê-lo fixo para comparar experimentos de maneira justa.

```text
RANDOM_STATE = torna o sorteio reproduzível
```

### TEST_SIZE

Define a proporção reservada para teste:

```python
TEST_SIZE = 0.20
```

Isso significa:

```text
80% para treinamento
20% para teste
```

Com aproximadamente 307.511 clientes:

```text
Treino: aproximadamente 246.008 clientes
Teste:  aproximadamente 61.503 clientes
```

O conjunto de teste é a prova final do modelo. Ele não participa do aprendizado.

### DECISION_THRESHOLD

A regressão logística produz um score entre `0` e `1`. O limiar transforma o score em classe:

```python
y_prediction = (y_probability >= DECISION_THRESHOLD).astype(int)
```

Com limiar `0,50`:

| Score | Classe prevista |
|---:|---|
| 0,12 | `0` — adimplente |
| 0,43 | `0` — adimplente |
| 0,67 | `1` — inadimplente |
| 0,91 | `1` — inadimplente |

Um limiar menor tende a encontrar mais inadimplentes, aumentando recall e falsos positivos. Um limiar maior tende a marcar menos clientes como risco, reduzindo recall e falsos positivos.

Resumo:

```text
RANDOM_STATE       = torna o experimento reproduzível
TEST_SIZE          = define o tamanho da avaliação
DECISION_THRESHOLD = transforma score em classe
```

## 17. Recall em profundidade

Recall responde:

> De todos os casos que realmente pertenciam a uma classe, quantos o modelo encontrou?

```text
Recall = acertos da classe / total real da classe
```

### Recall dos inadimplentes

Para a classe `1`:

```text
Recall = verdadeiros positivos
         ─────────────────────────────────
         verdadeiros positivos + falsos negativos
```

- verdadeiro positivo: inadimplente corretamente identificado;
- falso negativo: inadimplente classificado como seguro.

O resultado foi:

```text
Recall da classe 1 = 0,6634 = 66,34%
```

Havia 4.965 inadimplentes no teste. Aproximadamente:

```text
4.965 inadimplentes reais
        │
        ├── 3.293 identificados corretamente
        └── 1.672 não identificados
```

Um falso negativo pode provocar aprovação de crédito para alguém que posteriormente não paga.

### Recall dos adimplentes

```text
Recall da classe 0 = 0,6792 = 67,92%
```

Havia 56.538 adimplentes no teste. Aproximadamente:

```text
56.538 adimplentes reais
        │
        ├── 38.400 reconhecidos corretamente
        └── 18.138 marcados como risco
```

Os clientes do segundo grupo são falsos positivos: bons clientes que poderiam ter o crédito recusado ou encaminhado para análise manual.

### Recall não considera falsos positivos

Um modelo pode obter recall de 100% marcando muitas pessoas como risco. Por isso, recall deve ser analisado junto com precision.

```text
Recall:
Dos inadimplentes reais, quantos encontramos?

Precision:
Dos clientes marcados como inadimplentes, quantos realmente eram?
```

No modelo:

```text
Recall da classe 1:    66,34%
Precision da classe 1: 15,37%
```

O modelo lança uma rede larga: encontra uma parte relevante dos inadimplentes, mas também gera falsos alarmes.

### Macro e weighted recall

O macro recall dá o mesmo peso às classes:

```text
(0,6792 + 0,6634) / 2 = 0,6713
```

O weighted recall considera a quantidade de registros de cada classe. Como há mais adimplentes, a classe `0` influencia mais. Em classificação de classe única, o weighted recall coincide com a acurácia:

```text
Weighted recall ≈ Accuracy ≈ 67,79%
```

## 18. O arquivo Joblib

O `.joblib` é um arquivo binário utilizado para serializar objetos Python. Neste projeto, ele funciona como uma fotografia do modelo depois do treinamento.

```text
Treinamento
    ↓
Modelo aprendido
    ↓
joblib.dump(...)
    ↓
logistic_regression_abt.joblib
```

Depois:

```text
logistic_regression_abt.joblib
    ↓
joblib.load(...)
    ↓
Modelo pronto para prever
```

O arquivo contém:

### Modelo

```python
artifact["model"]
```

Inclui todo o fluxo:

```text
Dados do cliente
    ↓
Tratamento dos valores ausentes
    ↓
Padronização
    ↓
One-Hot Encoding
    ↓
Regressão logística
    ↓
Score de risco
```

### Limiar

```python
artifact["decision_threshold"]
```

Registra o limiar usado para transformar score em classe.

### Features esperadas

```python
artifact["input_features"]
```

Registra os nomes e a ordem das colunas esperadas pelo modelo.

### Métricas

```python
artifact["metrics"]
```

Documenta os resultados da versão salva, incluindo ROC AUC e Average Precision.

### Carregamento

```python
import joblib

artifact = joblib.load("artifacts/logistic_regression_abt.joblib")
model = artifact["model"]
```

Não se deve carregar um `.joblib` de fonte desconhecida. Arquivos serializados podem executar código durante o carregamento. Também é importante preservar as versões de Python, Scikit-learn, NumPy e Joblib utilizadas.

## 19. O modelo aprova crédito?

O modelo estima risco; ele não deveria tomar sozinho a decisão final de crédito.

```text
Dados do cliente
      ↓
Modelo
      ↓
Score de risco
      ↓
Política de crédito
      ↓
Aprovar, analisar ou recusar
```

Por exemplo, um score de `0,72` indica risco elevado dentro da escala do modelo. Como foi utilizado `class_weight="balanced"`, esse valor deve ser tratado como score de ordenação, e não imediatamente como exatamente 72% de probabilidade real. Para essa interpretação, seria necessário calibrar o modelo.

Uma decisão real também pode considerar:

- valor solicitado;
- capacidade de pagamento;
- regras cadastrais;
- prevenção a fraude;
- documentação;
- exigências legais;
- garantias;
- política interna e apetite de risco.

No projeto, foi criada uma política demonstrativa:

| Score | Recomendação |
|---:|---|
| Menor que 0,35 | `approve` |
| De 0,35 até menos de 0,65 | `manual_review` |
| A partir de 0,65 | `reject` |

Esses limites são configuráveis e precisam ser validados com custos reais do negócio.

## 20. O conceito de Pipeline

Pipeline significa uma sequência ordenada de etapas:

```text
Entrada → Etapa 1 → Etapa 2 → Etapa 3 → Saída
```

O significado específico depende do contexto.

### Pipeline no Scikit-learn

```python
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
```

Fluxo:

```text
Dados numéricos
      ↓
Preenchimento de valores ausentes
      ↓
Padronização
      ↓
Dados preparados
```

O pipeline categórico executa:

```text
Categorias
      ↓
Preenchimento dos nulos
      ↓
One-Hot Encoding
      ↓
Colunas binárias
```

O pipeline principal contém pré-processamento e classificador:

```python
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(...)),
])
```

Quando `fit()` é chamado, cada etapa aprende o necessário. Quando `predict_proba()` é chamado, o novo cliente atravessa as mesmas transformações antes de chegar ao classificador.

Isso reduz riscos como:

- executar etapas na ordem errada;
- esquecer uma transformação na previsão;
- aprender novamente sobre dados novos;
- tratar treino e produção de formas diferentes;
- provocar vazamento de dados;
- salvar o classificador sem seus transformadores.

### Data pipeline

```text
Dados brutos
    ↓
Ingestão
    ↓
Limpeza
    ↓
Engenharia de features
    ↓
ABT
```

### Pipeline DevOps

```text
Código enviado ao Git
    ↓
Instalação de dependências
    ↓
Testes
    ↓
Build
    ↓
Publicação
    ↓
Deploy
```

Comparação:

| Aspecto | Scikit-learn | DevOps |
|---|---|---|
| Entrada | Dados | Código-fonte |
| Etapas | Imputação, escala, encoding, modelo | Testes, build, deploy |
| Saída | Previsão | Software publicado |
| Objetivo | Repetir tratamento e inferência | Automatizar entrega |

Para identificar o tipo de pipeline, pergunte:

1. Qual é a entrada?
2. Quais são as etapas?
3. Qual é a saída?

## 21. Exposição do modelo com FastAPI

Foi criada uma aplicação FastAPI única com duas formas de previsão:

```text
FastAPI
├── POST /predict/features
│      recebe as 32 features prontas
│
└── POST /predict/customer/{customer_id}
       consulta o PostgreSQL e monta as features
```

As duas rotas reutilizam:

- o mesmo arquivo Joblib;
- a mesma validação de entrada;
- o mesmo serviço de previsão;
- a mesma política de crédito;
- o mesmo formato de resposta.

Uma aplicação única é preferível a duas APIs independentes porque evita duplicação e divergência de regras.

### Componentes

```text
api/main.py
    Endpoints HTTP e ciclo de vida da aplicação

api/model_service.py
    Carregamento do Joblib e geração do score

api/feature_service.py
    Consulta ao banco e reprodução das features da ABT

api/credit_policy.py
    Conversão do score em recomendação

api/schemas.py
    Contratos de entrada e saída

api/config.py
    Caminhos, conexão e limiares configuráveis
```

### Fluxo por features prontas

```text
JSON com features
      ↓
POST /predict/features
      ↓
Validação das 32 features
      ↓
Pipeline do Joblib
      ↓
Score
      ↓
Política de crédito
      ↓
Resposta JSON
```

### Fluxo por cliente no banco

```text
sk_id_curr
      ↓
POST /predict/customer/{customer_id}
      ↓
Consultas a application_train, previous_application e bureau
      ↓
Criação das mesmas 32 features da ABT
      ↓
Pipeline do Joblib
      ↓
Score
      ↓
Política de crédito
      ↓
Resposta JSON
```

### Por que a criação idêntica das features importa?

O modelo precisa receber em produção as mesmas definições utilizadas no treinamento. Se `age`, `prev_refused_rate` ou qualquer outra feature for calculada de forma diferente, ocorre *training-serving skew*.

A implementação foi validada comparando as 32 features produzidas pela API para o cliente `100002` com a linha correspondente da ABT. Não houve divergências.

### Camada de política

O modelo e a política possuem responsabilidades diferentes:

```text
Modelo   → mede risco
Política → recomenda uma ação
```

Exemplo de resposta:

```json
{
  "source": "database",
  "customer_id": 100002,
  "risk_score": 0.8173685737,
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

### Endpoints auxiliares

```text
GET /health
```

Verifica se a API e o modelo estão carregados.

```text
GET /model/features
```

Lista todas as features esperadas pelo artefato.

### Execução

Na pasta `data-platform/Model`:

```bash
../.venv/bin/python -m uvicorn api.main:app --reload
```

Documentação interativa:

```text
http://localhost:8000/docs
```

### Configurações da política

```bash
export CREDIT_APPROVE_MAX_SCORE="0.35"
export CREDIT_MANUAL_REVIEW_MAX_SCORE="0.65"
export CREDIT_POLICY_VERSION="demo-v1"
```

### Validações realizadas

- seis testes automatizados aprovados;
- endpoint de health respondeu HTTP 200;
- as duas formas de previsão responderam HTTP 200;
- ambas produziram score `0,8174` para o cliente `100002`;
- as 32 features montadas pelo banco coincidiram com a ABT;
- a política recomendou `reject` para esse exemplo.

## 22. Visão ponta a ponta do projeto

```text
Fontes brutas
application_train
previous_application
bureau
      ↓
Pipeline de dados
      ↓
abt.csv
      ↓
Notebook de treinamento
      ↓
Pipeline Scikit-learn
      ↓
logistic_regression_abt.joblib
      ↓
FastAPI
      ↓
Score de risco
      ↓
Política de crédito
      ↓
approve | manual_review | reject
```

O modelo não substitui toda a operação de crédito. Ele adiciona uma avaliação de risco reproduzível, que deve ser combinada com regras cadastrais, prevenção a fraude, exigências legais e supervisão humana.
