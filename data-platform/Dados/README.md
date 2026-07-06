# Dados do projeto

O item C solicita os artefatos `raw_data.csv`, `clean_data.csv` e `abt.csv`.
Como os arquivos completos têm dezenas ou centenas de megabytes, eles não são
versionados no Git. Esta pasta preserva a estrutura solicitada e documenta a
origem de cada artefato.

| Arquivo solicitado | Origem/geração no projeto |
|---|---|
| `raw_data.csv` | `data/csv/application_train.csv`, carregado pela DAG `loadfile_csv_to_postgres` |
| `clean_data.csv` | tabela PostgreSQL `application_clean`, produzida por `DataPipeline/data_sanitization.py` |
| `abt.csv` | salvo diretamente nesta pasta por `DataPipeline/exp_analysis.ipynb` |

## Materialização local

Execute, a partir de `data-platform`:

```bash
python3 Dados/materialize.py
```

O comando cria a referência local `Dados/raw_data.csv`. Para exportar
`clean_data.csv` do PostgreSQL, informe `--export-clean`. Essa exportação usa o
`psql` do próprio container e não exige `pandas` no Python local. A `abt.csv` não
depende deste utilitário: ela é gravada diretamente pelo notebook de análise.

```bash
python3 Dados/materialize.py --export-clean
```

Os CSVs são ignorados pelo Git intencionalmente. O código, as configurações e
as instruções necessárias para reproduzi-los permanecem versionados.
