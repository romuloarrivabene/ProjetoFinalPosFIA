# Backup do Metabase

O arquivo `metabase_2026-07-16.dump` e um dump PostgreSQL em formato customizado
do banco interno `metabase`. Ele contem usuarios, colecoes, perguntas, cartoes,
dashboards, filtros e layouts do Metabase.

O backup nao inclui o banco analitico `data`, a ABT ou os artefatos do modelo.

## Restauracao

> A restauracao substitui o estado atual do Metabase. Pare o servico antes de
> executar os comandos.

Na raiz do projeto:

```bash
docker compose -f data-platform/docker-compose.yml stop metabase
docker exec -i postgres pg_restore \
  -U airflow \
  -d metabase \
  --clean \
  --if-exists \
  --no-owner \
  --no-privileges \
  < backup-metabase/metabase_2026-07-16.dump
docker compose -f data-platform/docker-compose.yml up -d metabase
```

