"""TTSDS wrapper: run BenchmarkSuite on Blizzard systems via subprocess."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

from srm_eval.data.blizzard import BlizzardData, DEFAULT_YEARS

# Path to the standalone runner script (distributed with the package).
_RUNNER_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_ttsds.py"


def run_ttsds(
    data: BlizzardData,
    *,
    venv_python: str | None = None,
    cache_dir: Path = Path("~/.cache/srm-eval/ttsds").expanduser(),
    years: Iterable[int] = DEFAULT_YEARS,
) -> pd.DataFrame:
    """Run TTSDS BenchmarkSuite and return aggregated scores DataFrame."""
    if venv_python is None:
        venv_python = str(Path("~/.cache/srm-eval/.venv-ttsds/bin/python").expanduser())

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build task list: (year, subtask, system, [wav_paths]).
    sys_mos = data.system_mos()
    tasks: list[dict] = []
    for (year, subtask), grp in sys_mos.groupby(["year", "subtask"]):
        if int(year) not in years:
            continue
        for _, row in grp.iterrows():
            files_df = data.files_for(int(year), str(subtask), str(row["system"]))
            paths = [str(Path(p).resolve()) for p in files_df["filepath"] if Path(p).exists()]
            if paths:
                tasks.append({
                    "year": int(year),
                    "subtask": str(subtask),
                    "system": str(row["system"]),
                    "files": paths,
                })

    config_path = cache_dir / "_config.json"
    config_path.write_text(json.dumps(
        {"tasks": tasks, "cache_dir": str(cache_dir)}, indent=2
    ))

    print(f"running TTSDS via {venv_python} ({len(tasks)} tasks)...")
    subprocess.check_call(
        [venv_python, str(_RUNNER_SCRIPT), str(config_path)],
        timeout=7200,
    )

    # Collect results.
    frames = []
    for json_file in sorted(cache_dir.glob("*.json")):
        if json_file.name.startswith("_"):
            continue
        df = pd.read_json(json_file)
        parts = json_file.stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            df["year"] = int(parts[0])
            df["subtask"] = parts[1]
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
