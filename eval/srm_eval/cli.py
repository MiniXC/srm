""""srm-eval" CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True)


@app.command()
def download(
    years: Annotated[str | None, typer.Option("--years", help="Comma-separated years, e.g. 2008,2009,2013")] = None,
    cache_dir: Annotated[str, typer.Option("--cache-dir", help="Cache directory")] = "~/.cache/srm-eval/blizzard",
) -> None:
    """Download the Blizzard Challenge MOS dataset."""
    from srm_eval.data.blizzard import DEFAULT_YEARS, load_blizzard

    if years:
        year_list = [int(y.strip()) for y in years.split(",")]
    else:
        year_list = list(DEFAULT_YEARS)

    data = load_blizzard(years=year_list, cache_dir=cache_dir, download=True)
    print(f"Loaded {len(data.df)} rows, {len(data.system_mos())} systems")
    print("Subtasks:")
    for year, subtask in sorted(data.subtasks()):
        n_sys = len(data.system_mos()[
            (data.system_mos().year == year) & (data.system_mos().subtask == subtask)
        ])
        print(f"  {year}/{subtask}: {n_sys} systems")


@app.command()
def extract(
    extractor: Annotated[str, typer.Option("--extractor", "-e", help="Extractor name")] = "whisper-l20",
    years: Annotated[str | None, typer.Option("--years", help="Comma-separated years")] = None,
    cache_dir: Annotated[str, typer.Option("--cache-dir", help="Cache directory")] = "~/.cache/srm-eval",
    device: Annotated[str, typer.Option("--device")] = "cuda",
) -> None:
    """Extract features for all systems and cache them."""
    from srm_eval.data.blizzard import DEFAULT_YEARS, load_blizzard
    from srm_eval.extractors.whisper import WhisperExtractor

    if years:
        year_list = [int(y.strip()) for y in years.split(",")]
    else:
        year_list = list(DEFAULT_YEARS)

    data = load_blizzard(years=year_list, download=False)
    ext = WhisperExtractor(device=device, cache_dir=f"{cache_dir}/features")

    sys_mos = data.system_mos()
    for (year, subtask), grp in sys_mos.groupby(["year", "subtask"]):
        for _, row in grp.iterrows():
            files_df = data.files_for(year, subtask, row["system"])
            print(f"{year}/{subtask}/{row['system']}: {len(files_df)} files")
            feats = ext.extract_system(files_df, show_progress=True)
            print(f"  -> {feats.shape[0]} chunks x {feats.shape[1]} dims")


@app.command()
def distance(
    extractor: Annotated[str, typer.Option("--extractor", "-e")] = "whisper-l20",
    years: Annotated[str | None, typer.Option("--years")] = None,
    distance_kind: Annotated[str, typer.Option("--distance", "-d")] = "w2_perdim",
    cache_dir: Annotated[str, typer.Option("--cache-dir")] = "~/.cache/srm-eval",
    device: Annotated[str, typer.Option("--device")] = "cuda",
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Compute Wasserstein distances from cached features."""
    from srm_eval.data.blizzard import DEFAULT_YEARS, load_blizzard
    from srm_eval.run import compute_distances

    if years:
        year_list = [int(y.strip()) for y in years.split(",")]
    else:
        year_list = list(DEFAULT_YEARS)

    data = load_blizzard(years=year_list, download=False)

    if extractor == "whisper-l20":
        from srm_eval.extractors.whisper import WhisperExtractor
        ext = WhisperExtractor(device=device, cache_dir=f"{cache_dir}/features")
    else:
        print(f"unknown extractor: {extractor}")
        raise typer.Exit(1)

    df = compute_distances(ext, data, distance_kind=distance_kind)
    if output:
        df.to_csv(output, index=False)
        print(f"wrote {len(df)} rows to {output}")

    if not df.empty:
        print(df.groupby(["year", "subtask"])["distance"].agg(["mean", "count"]))


@app.command()
def correlate(
    distances_path: Annotated[str, typer.Argument(help="Path to distances CSV")],
    cache_dir: Annotated[str, typer.Option("--cache-dir")] = "~/.cache/srm-eval/blizzard",
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Compute MOS correlations from a distances CSV."""
    import pandas as pd
    from srm_eval.data.blizzard import load_blizzard
    from srm_eval.correlate import correlate as compute_corr

    distances = pd.read_csv(distances_path)
    data = load_blizzard(download=False, cache_dir=cache_dir)
    corr = compute_corr(distances, data.system_mos()[["year", "subtask", "system", "mean_mos"]])
    corr["rho_mos"] = -corr["rho"]

    if output:
        corr.to_csv(output, index=False)
        print(f"wrote to {output}")

    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="MOS Correlation (positive = distance predicts MOS)")
    table.add_column("Year")
    table.add_column("Subtask")
    table.add_column("N")
    table.add_column("rho", justify="right")
    table.add_column("CI 95%")

    for _, row in corr.iterrows():
        table.add_row(
            str(int(row["year"])),
            row["subtask"],
            str(int(row["n_systems"])),
            f'{row["rho_mos"]:.3f}',
            f'[{row["ci_lo"]:.3f}, {row["ci_hi"]:.3f}]',
        )
    console.print(table)


@app.command()
def run(
    extractor: Annotated[str, typer.Option("--extractor", "-e")] = "whisper-l20",
    years: Annotated[str | None, typer.Option("--years")] = None,
    distance_kind: Annotated[str, typer.Option("--distance", "-d")] = "w2_perdim",
    cache_dir: Annotated[str, typer.Option("--cache-dir")] = "~/.cache/srm-eval",
    device: Annotated[str, typer.Option("--device")] = "cuda",
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Full pipeline: extract -> distance -> correlate."""
    from srm_eval.run import run_pipeline

    if years:
        year_list = [int(y.strip()) for y in years.split(",")]
    else:
        year_list = None

    result = run_pipeline(
        extractor_names=[extractor],
        years=year_list,
        distance_kinds=[distance_kind],
        cache_dir=cache_dir,
        device=device,
    )

    if output:
        result.to_csv(output, index=False)
        print(f"wrote {len(result)} rows to {output}")

    if not result.empty:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="MOS Correlation (positive rho = distance predicts MOS)")
        table.add_column("Year")
        table.add_column("Subtask")
        table.add_column("N")
        table.add_column("rho", justify="right")
        table.add_column("CI 95%")

        for _, row in result.iterrows():
            table.add_row(
                str(int(row["year"])),
                row["subtask"],
                str(int(row["n_systems"])),
                f'{row["rho_mos"]:.3f}',
                f'[{row["ci_lo"]:.3f}, {row["ci_hi"]:.3f}]',
            )
        console.print(table)


@app.command()
def ttsds(
    years: Annotated[str | None, typer.Option("--years")] = None,
    cache_dir: Annotated[str, typer.Option("--cache-dir")] = "~/.cache/srm-eval",
    output: Annotated[str | None, typer.Option("--output", "-o")] = None,
) -> None:
    """Run TTSDS evaluation on Blizzard systems."""
    from srm_eval.data.blizzard import DEFAULT_YEARS, load_blizzard
    from srm_eval.ttsds_runner import run_ttsds

    if years:
        year_list = [int(y.strip()) for y in years.split(",")]
    else:
        year_list = list(DEFAULT_YEARS)

    data = load_blizzard(years=year_list, cache_dir=f"{cache_dir}/blizzard", download=False)
    result = run_ttsds(data, cache_dir=Path(f"{cache_dir}/ttsds").expanduser(), years=year_list)

    if output:
        result.to_csv(output, index=False)
        print(f"wrote {len(result)} rows to {output}")
    else:
        if not result.empty:
            from rich.console import Console
            console = Console()
            console.print(result.to_string())


if __name__ == "__main__":
    app()
