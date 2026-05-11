"""Main pipeline: extract → distance → correlate for Blizzard evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from srm_eval.correlate import correlate
from srm_eval.data.blizzard import BlizzardData, load_blizzard
from srm_eval.distance import frechet_distance, sliced_wasserstein, wasserstein_2_perdim
from srm_eval.extractors.base import Extractor
from srm_eval.extractors.whisper import WhisperExtractor


DISTANCE_FNS = {
    "w2_perdim": wasserstein_2_perdim,
    "frechet": frechet_distance,
    "sliced": sliced_wasserstein,
}


def compute_distances(
    extractor: Extractor,
    data: BlizzardData,
    *,
    distance_kind: str = "w2_perdim",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Extract features and compute Wasserstein distance to reference (system A)
    for every system in every (year, subtask).
    """
    dist_fn = DISTANCE_FNS[distance_kind]
    sys_mos = data.system_mos()
    rows = []

    for (year, subtask), grp in sys_mos.groupby(["year", "subtask"]):
        # Reference distribution = system A.
        ref_system = grp[grp["system"] == "A"]
        if ref_system.empty:
            ref_system = grp.loc[grp["mean_mos"].idxmax()]

        ref_row = ref_system.iloc[0]
        ref_files = data.files_for(year, subtask, ref_row["system"])
        print(f"{year}/{subtask}: extracting reference (system {ref_row['system']}, "
              f"{len(ref_files)} files)")
        ref_dist = extractor.extract_system(ref_files, show_progress=False)
        if ref_dist.shape[0] < 2:
            print(f"  WARNING: reference has only {ref_dist.shape[0]} chunks, skipping")
            continue

        for _, row in grp.iterrows():
            if row["system"] == ref_row["system"]:
                continue
            sys_files = data.files_for(year, subtask, row["system"])
            if len(sys_files) == 0:
                continue

            print(f"  system {row['system']} ({len(sys_files)} files)", end="")
            sys_dist = extractor.extract_system(sys_files, show_progress=False)
            if sys_dist.shape[0] < 2:
                print(" — too few chunks, skipping")
                continue

            distance = dist_fn(sys_dist, ref_dist)
            print(f" → {distance:.4f}")

            rows.append({
                "year": year,
                "subtask": subtask,
                "system": row["system"],
                "distance": distance,
            })

    return pd.DataFrame(rows)


def run_pipeline(
    extractor_names: list[str],
    *,
    years: list[int] | None = None,
    distance_kinds: list[str] | None = None,
    cache_dir: str = "~/.cache/srm-eval",
    device: str = "cuda",
) -> pd.DataFrame:
    """Full pipeline: load data → extract → distance → correlate → report."""
    if years is None:
        years = [2008, 2009, 2010, 2011, 2012, 2013, 2016, 2019, 2020, 2021, 2023]
    if distance_kinds is None:
        distance_kinds = ["w2_perdim"]

    data = load_blizzard(years=years)

    all_corr: list[pd.DataFrame] = []

    for ext_name in extractor_names:
        if ext_name == "whisper-l20":
            extractor = WhisperExtractor(device=device, cache_dir=f"{cache_dir}/features")
        else:
            print(f"unknown extractor: {ext_name}, skipping")
            continue

        for dist_kind in distance_kinds:
            print(f"\n=== {ext_name} / {dist_kind} ===")
            distances_df = compute_distances(
                extractor, data, distance_kind=dist_kind,
            )
            if distances_df.empty:
                print("  no distances computed")
                continue

            corr_df = correlate(
                distances_df,
                data.system_mos()[["year", "subtask", "system", "mean_mos"]],
            )
            corr_df["extractor"] = ext_name
            corr_df["distance_kind"] = dist_kind

            # Flip sign so positive = distance predicts MOS.
            corr_df["rho_mos"] = -corr_df["rho"]

            all_corr.append(corr_df)

    if not all_corr:
        return pd.DataFrame()
    return pd.concat(all_corr, ignore_index=True)
