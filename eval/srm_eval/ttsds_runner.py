"""TTSDS wrapper: run the full BenchmarkSuite on Blizzard Challenge systems.

TTSDS is an end-to-end evaluation library — it extracts features, builds
distributions, computes distances, and normalizes scores internally.
We feed it per-system DirectoryDataset objects and interpret the output.

Because TTSDS has a heavy/conflicting dep tree (numpy<2, fairseq fork),
we shell out to a separate venv via subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from srm_eval.data.blizzard import BlizzardData


def _ttsds_venv(venv_path: Path) -> Path:
    """Create or reuse a dedicated venv for TTSDS."""
    venv_python = venv_path / "bin" / "python"
    if not venv_python.exists():
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_path)])
        # Install TTSDS.  numpy<2 is required; install it first.
        subprocess.check_call(
            [str(venv_path / "bin" / "pip"), "install", "--quiet",
             "numpy<2", "ttsds", "pandas", "pyyaml"],
        )
    return venv_python


def _write_ttsds_script(
    system_dirs: list[Path],
    ref_dirs: list[Path],
    output_json: Path,
    benchmarks: list[str] | None = None,
) -> Path:
    """Write a self-contained Python script that runs TTSDS and writes results to JSON."""
    script_lines = [
        "import json, sys",
        "from pathlib import Path",
        "from ttsds import BenchmarkSuite",
        "from ttsds.util.dataset import DirectoryDataset",
        "",
        f"system_dirs = {[str(d) for d in system_dirs]}",
        f"ref_dirs = {[str(d) for d in ref_dirs]}",
        f"output = Path({str(output_json)!r})",
        "",
        "datasets = []",
        "for d in system_dirs:",
        "    name = Path(d).name",
        "    datasets.append(DirectoryDataset(d, name=name))",
        "",
        "refs = []",
        "for d in ref_dirs:",
        "    name = Path(d).name",
        "    refs.append(DirectoryDataset(d, name=name))",
        "",
    ]
    if benchmarks:
        script_lines.append(
            f"suite = BenchmarkSuite(datasets=datasets, reference_datasets=refs, "
            f"benchmarks={json.dumps(benchmarks)}, skip_errors=True)"
        )
    else:
        script_lines.append(
            "suite = BenchmarkSuite(datasets=datasets, reference_datasets=refs, skip_errors=True)"
        )
    script_lines += [
        "",
        "result = suite.run()",
        "# Convert to dict for JSON.",
        "output_json = result.to_dict(orient='records')",
        "output.write_text(json.dumps(output_json, indent=2, default=str))",
        "print(f'wrote {len(output_json)} rows to {output}')",
    ]

    script_path = output_json.with_suffix(".py")
    script_path.write_text("\n".join(script_lines))
    return script_path


def run_ttsds(
    data: BlizzardData,
    *,
    venv_path: Path = Path("~/.cache/srm-eval/.venv-ttsds").expanduser(),
    cache_dir: Path = Path("~/.cache/srm-eval/ttsds").expanduser(),
    years: Iterable[int] = (2008, 2009, 2010, 2011, 2012, 2013, 2016, 2019, 2020, 2021, 2023),
    benchmarks: list[str] | None = None,
) -> pd.DataFrame:
    """Run TTSDS on every system in every (year, subtask) of the Blizzard data.

    Returns a DataFrame with columns:
        benchmark_name, benchmark_category, dataset (system name),
        score, noise_dataset, reference_dataset
    """
    venv_python = _ttsds_venv(venv_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    sys_mos = data.system_mos()

    all_frames: list[pd.DataFrame] = []

    for (year, subtask), grp in sys_mos.groupby(["year", "subtask"]):
        if year not in years:
            continue

        # Reference = system A (natural speech).
        ref_system = grp[grp["system"] == "A"]
        if ref_system.empty:
            # Fallback: highest-MOS system.
            ref_system = grp.loc[grp["mean_mos"].idxmax()]

        # Collect all system filepaths into per-system directories.
        system_dirs: list[Path] = []
        ref_dirs: list[Path] = []

        for _, row in grp.iterrows():
            system = row["system"]
            files_df = data.files_for(year, subtask, system)
            paths = [Path(p) for p in files_df["filepath"]]

            # Create a symlink directory for TTSDS.
            sys_dir = cache_dir / f"{year}_{subtask}_{system}"
            sys_dir.mkdir(parents=True, exist_ok=True)
            for p in paths:
                if p.exists():
                    link = sys_dir / p.name
                    if not link.exists():
                        link.symlink_to(p.resolve())
            system_dirs.append(sys_dir)

            if system == ref_system["system"].iloc[0]:
                ref_dirs.append(sys_dir)

        output_json = cache_dir / f"{year}_{subtask}.json"
        if output_json.exists():
            with open(output_json) as f:
                rows = json.load(f)
        else:
            script = _write_ttsds_script(system_dirs, ref_dirs, output_json, benchmarks)
            subprocess.check_call([str(venv_python), str(script)], timeout=3600)

            if not output_json.exists():
                continue
            with open(output_json) as f:
                rows = json.load(f)

        df = pd.DataFrame(rows)
        df["year"] = year
        df["subtask"] = subtask
        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame()
    return pd.concat(all_frames, ignore_index=True)
