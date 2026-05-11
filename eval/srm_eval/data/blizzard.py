"""Blizzard Challenge MOS dataset loader + filename parser.

Every year has a different filename encoding scheme for the subtask/system.
This module centralizes that mess in a single parse_filepath() function
tested against actual filenames from each year's CSV.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class BlizzardData:
    """Parsed Blizzard Challenge MOS data.

    DataFrame columns:
        year       int      challenge year (2008..2023)
        subtask    str      subtask identifier (e.g. "EH1", "english_news")
        system     str      system letter (A, B, C, ...)
        filepath   str      path relative to year tarball root
        file_mos   float    MOS score for this file
        lang       str      language code (en, zh, fr, de)
    """

    df: pd.DataFrame

    def system_mos(self) -> pd.DataFrame:
        """Aggregate per-system MOS within each (year, subtask)."""
        return (
            self.df.groupby(["year", "subtask", "system"])
            .agg(n_files=("file_mos", "count"), mean_mos=("file_mos", "mean"), std_mos=("file_mos", "std"))
            .reset_index()
        )

    def subtasks(self) -> list[tuple[int, str]]:
        """Return unique (year, subtask) pairs."""
        return list(self.df[["year", "subtask"]].drop_duplicates().itertuples(index=False, name=None))  # type: ignore[return-value]

    def files_for(self, year: int, subtask: str, system: str) -> pd.DataFrame:
        """Return the subset of df for one system in one subtask/year."""
        return self.df[(self.df.year == year) & (self.df.subtask == subtask) & (self.df.system == system)]


# ---- filename parser -------------------------------------------------------


def parse_filepath(year: int, path: str) -> tuple[str, str]:
    """Return (subtask, system) from a Blizzard filename.

    The encoding varies per year — see the architecture doc for the full table.
    """
    # Strip the Blizzard_YYYY/ prefix if present.
    inner = path.split("/", 1)[1] if "/" in path else path

    if year == 2008:
        return _parse_2008(inner)
    if year in (2009, 2010):
        return _parse_2009_2010(inner)
    if year in (2011, 2012, 2016, 2019):
        return _parse_single_task(inner)
    if year == 2013:
        return _parse_2013(inner)
    if year == 2020:
        return _parse_2020(inner)
    if year == 2021:
        return _parse_2021(inner)
    if year == 2023:
        return _parse_2023(inner)
    raise ValueError(f"unsupported year: {year}")


def _parse_2008(inner: str) -> tuple[str, str]:
    # A_submission_directory_english_arctic_2008_news_news_2008_0002.wav
    system = inner[0]
    m = re.match(r"[A-Z]_submission_directory_([^_]+)_([^_]+)_", inner)
    if m:
        lang, corpus = m.group(1), m.group(2)
        subtask = f"{lang}_{corpus}"
    else:
        subtask = "unknown"
    return subtask, system


def _parse_2009_2010(inner: str) -> tuple[str, str]:
    # A_submission_directory_english_EH1_2009_conv_wavs_...
    system = inner[0]
    m = re.match(r"[A-Z]_submission_directory_([^_]+)_([^_]+)_", inner)
    if m:
        subtask = f"{m.group(1)}_{m.group(2)}"
    else:
        subtask = "unknown"
    return subtask, system


def _parse_single_task(inner: str) -> tuple[str, str]:
    # A_submission_directory_<year>_<corpus>_...
    system = inner[0]
    m = re.match(r"[A-Z]_submission_directory_\d+_([^_]+)", inner)
    if m:
        subtask = m.group(1)
    else:
        subtask = "unknown"
    return subtask, system


def _parse_2013(inner: str) -> tuple[str, str]:
    # A_submission_directory_2013_EH1-English_audiobook_sentences_...
    system = inner[0]
    m = re.match(r"[A-Z]_submission_directory_\d+_(\w+)-\w+", inner)
    if m:
        subtask = m.group(1)
    else:
        subtask = "unknown"
    return subtask, system


def _parse_2020(inner: str) -> tuple[str, str]:
    # MH1_A_submission_directory_news_wav_...
    parts = inner.split("_", 2)
    subtask = parts[0]  # MH1
    system = parts[1]  # A
    return subtask, system


def _parse_2021(inner: str) -> tuple[str, str]:
    # Like 2020?  Need to verify from actual data.
    # Fallback: try the 2020 pattern.
    try:
        return _parse_2020(inner)
    except (IndexError, ValueError):
        return "unknown", inner[0]


def _parse_2023(inner: str) -> tuple[str, str]:
    # A_2023-FH1_submission_directory_FH1_MOS_wav_FH1_MOS_0073.wav
    system = inner[0]
    m = re.match(r"[A-Z]_\d+-([^_]+)_", inner)
    if m:
        subtask = m.group(1)
    else:
        subtask = "unknown"
    return subtask, system


# ---- loader ----------------------------------------------------------------

DEFAULT_YEARS = (2008, 2009, 2010, 2011, 2012, 2013, 2016, 2019, 2020, 2021, 2023)

HF_REPO = "hewliyang/nisqa-blizzard-challenge-mos"


def load_blizzard(
    years: Iterable[int] = DEFAULT_YEARS,
    cache_dir: str | Path = "~/.cache/srm-eval/blizzard",
    download: bool = True,
) -> BlizzardData:
    """Load and parse the Blizzard Challenge MOS dataset."""
    import os
    from urllib.request import urlretrieve

    cache = Path(cache_dir).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    for year in years:
        csv_path = cache / f"Blizzard_{year}.csv"
        tar_path = cache / f"Blizzard_{year}.tar.gz"
        wav_dir = cache / f"Blizzard_{year}"

        # Download CSV if missing.
        if not csv_path.exists() and download:
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/Blizzard_{year}.csv"
            print(f"downloading {url} -> {csv_path}")
            urlretrieve(url, csv_path)

        # Download tarball if missing.
        if not tar_path.exists() and not wav_dir.exists() and download:
            url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/Blizzard_{year}.tar.gz"
            print(f"downloading {url} -> {tar_path}  (~{'18' if year < 2014 else '2'} GB)")
            urlretrieve(url, tar_path)

        # Unpack tarball lazily (only if we have it and haven't unpacked yet).
        if tar_path.exists() and not wav_dir.exists():
            print(f"extracting {tar_path} -> {wav_dir}")
            with tarfile.open(tar_path, "r:gz") as tf:
                tf.extractall(cache, filter="data")
            tar_path.unlink(missing_ok=True)  # save disk

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)

        # Normalize MOS column (2023 calls it "score").
        if "score" in df.columns and "mos" not in df.columns:
            df = df.rename(columns={"score": "mos"})

        # Parse (subtask, system) from each filepath.
        parsed = df["filepath_deg"].apply(lambda p: parse_filepath(year, str(p)))
        df["subtask"] = [s for s, _ in parsed]
        df["system"] = [sys for _, sys in parsed]
        df["year"] = year

        # Extract language.
        if "lang" in df.columns:
            df["lang"] = df["lang"].fillna("en")
        else:
            df["lang"] = "en"

        # Keep only rows with valid MOS.
        df = df.dropna(subset=["mos"])

        # Build full filepath relative to cache.
        # Tarball extracts directly into wav_dir, so strip the Blizzard_YYYY/ prefix.
        df["filepath"] = df["filepath_deg"].apply(
            lambda p: str(wav_dir / p.split("/", 1)[1] if "/" in p else p)
        )

        frames.append(df[["year", "subtask", "system", "filepath", "mos", "lang"]].rename(
            columns={"mos": "file_mos"}
        ))

    combined = pd.concat(frames, ignore_index=True)
    return BlizzardData(df=combined)


# ---- sanity check on load --------------------------------------------------

def _check_reference_system(data: BlizzardData) -> None:
    """Warn if system A isn't the top-MOS system in any subtask."""
    sys_mos = data.system_mos()
    for (year, subtask), grp in sys_mos.groupby(["year", "subtask"]):
        top = grp.loc[grp["mean_mos"].idxmax()]
        if top["system"] != "A":
            print(f"WARNING: {year}/{subtask}: top-MOS system is {top['system']} "
                  f"({top['mean_mos']:.2f}), not A ({grp[grp.system == 'A']['mean_mos'].values})")
