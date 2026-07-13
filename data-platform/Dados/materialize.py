"""Materializa os nomes de dados exigidos pelo item C sem duplicar arquivos."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DATA_PLATFORM_DIR = Path(__file__).resolve().parents[1]
DADOS_DIR = Path(__file__).resolve().parent


def link(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Fonte não encontrada: {source}")
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(Path(os.path.relpath(source, destination.parent)))


def export_table(table: str, destination: Path) -> None:
    """Exporta uma tabela do PostgreSQL para CSV usando o container Docker."""
    command = [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-v",
        "ON_ERROR_STOP=1",
        "-U",
        "airflow",
        "-d",
        "data",
        "--command",
        f"\\copy public.{table} TO STDOUT WITH (FORMAT CSV, HEADER TRUE)",
    ]

    if destination.exists() or destination.is_symlink():
        destination.unlink()

    try:
        with destination.open("wb") as output:
            subprocess.run(
                command,
                cwd=DATA_PLATFORM_DIR,
                stdout=output,
                stderr=subprocess.PIPE,
                check=True,
            )
    except FileNotFoundError as error:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "Docker não encontrado. Inicie o Docker Desktop e tente novamente."
        ) from error
    except subprocess.CalledProcessError as error:
        destination.unlink(missing_ok=True)
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"Não foi possível exportar {table}. Execute primeiro as DAGs "
            f"de ingestão e pipeline. Detalhe: {detail}"
        ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-clean",
        action="store_true",
        help=(
            "Exporta application_clean, application_train e application_abt "
            "do PostgreSQL para clean_data.csv, raw_data.csv e abt.csv."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_path = DADOS_DIR / "clean_data.csv"
    raw_path = DADOS_DIR / "raw_data.csv"
    abt_path = DADOS_DIR / "abt.csv"

    if args.export_clean:
        export_table("application_clean", clean_path)
        export_table("application_train", raw_path)
        export_table("application_abt", abt_path)
    else:
        if not clean_path.exists():
            print(
                "clean_data.csv depende da tabela application_clean; "
                "use --export-clean após executar o pipeline."
            )
        if not raw_path.exists():
            print(
                "raw_data.csv depende da tabela application_train; "
                "use --export-clean para gerar."
            )
        if not abt_path.exists():
            print(
                "abt.csv depende da tabela application_abt; "
                "use --export-clean para gerar."
            )

    print(f"Dados materializados em: {DADOS_DIR}")


if __name__ == "__main__":
    main()
