# `srm-eval` — distribution-level speech representation evaluation

A small library for the **system-level MOS correlation** evaluation: given a
speech representation, compute the Wasserstein distance between each TTS
system's feature distribution and a reference distribution, then correlate
those distances with human MOS ratings.

This is the standard "TTSDS-style" benchmark: a *representation* is good for
TTS evaluation if its Wasserstein distance to a reference correlates highly
(negatively) with system MOS. Higher MOS → closer to reference → smaller
distance.

The goal of v1 is to reproduce this evaluation for two baselines on the
Blizzard Challenge MOS dataset:

1. **TTSDS** — the multi-factor benchmark from
   [github.com/ttsds/ttsds](https://github.com/ttsds/ttsds).
2. **Whisper-large encoder, layer 20** — single-factor baseline.

Our own representations will plug into the same interface once they exist.

---

## 1. The dataset and its subtask structure

We use [`hewliyang/nisqa-blizzard-challenge-mos`](https://huggingface.co/datasets/hewliyang/nisqa-blizzard-challenge-mos),
a packaging of the Blizzard Challenge listening-test MOS scores from 2008–2023.
The dataset is *several datasets in one* — within each year there are multiple
subtasks with different speakers, languages, and listener pools, so MOS scores
are only comparable within a single `(year, subtask)`.

### Subtask inventory (from the NISQA paper, fig. cited in dataset README)

| Year | Subtasks | Speaker(s) | Lang | # Systems range | Notes |
|---|---|---|---|---|---|
| 2008 | A, B | Roger / CAS | en, zh | 13–21 | A=English audiobook, B=English news, C=Mandarin |
| 2009 | EH1, EH2, ES1/ES2, MH, MS1/MS2 | Roger / iFLYTEK | en, zh | 12–21 | English Hub/Spoke + Mandarin |
| 2010 | EH1, EH2, ES1/ES3, MH1, MH2, MS1 | Roger / RJS / CAS | en, zh | 6–18 | |
| 2011 | EH1 | Nancy | en | 13 | |
| 2012 | EH1 | J. Greenman | en | 11 | |
| 2013 | EH1, EH2 | C. Byers | en | 10–15 | EH2 is German |
| 2016 | EH | L. Sims | en | 17 | Audiobook |
| 2019 | MH | Z. Luo | zh | 26 | Mandarin |
| 2020 | MH1, … | ? | zh | ~10–15 | |
| 2021 | MH, … | ? | zh | ~10 | |
| 2023 | FH1, FH2, … | ? | fr, … | ~10 | French Hub |

The HF dataset omits 2014, 2015, and 2017–2018 (no file-level MOS available).
2008 in this packaging combines subtasks A/B/C into one CSV; we split them
from the filename.

### Filename → (subtask, system) parsing rules

Filename patterns vary by year. The parser lives in `eval/data/blizzard.py`
as a single function `parse_filepath(year, path) -> (subtask, system)`,
covered by table tests against the actual filenames in each CSV.

| Year | Pattern (after `Blizzard_YYYY/`) | system | subtask | example |
|---|---|---|---|---|
| 2008 | `<S>_submission_directory_<lang>_<corpus>_<year>_<task>_<task>_<year>_<id>` | first char | `<lang>_<corpus>` | `A_submission_directory_english_arctic_2008_news_news_2008_0002.wav` → (`english_news`, A) |
| 2009 | `<S>_submission_directory_<lang>_<task>_<year>_<corpus>_…` | first char | `<lang>_<task>` | `A_submission_directory_english_EH1_2009_conv_wavs_conv_2009_0003.wav` → (`english_EH1`, A) |
| 2010 | same as 2009 | first char | `<lang>_<task>` | `A_submission_directory_english_EH1_2010_news_wavs_…` → (`english_EH1`, A) |
| 2011 | `<S>_submission_directory_<year>_<corpus>_…` | first char | `<corpus>` (single-task year) | `A_submission_directory_2011_news_wav_news_2011_0005.wav` → (`news`, A) |
| 2012 | same as 2011 | first char | `<corpus>` | |
| 2013 | `<S>_submission_directory_<year>_<task>-<lang>_<corpus>_…` | first char | `<task>` | `A_submission_directory_2013_EH1-English_audiobook_sentences_…` → (`EH1`, A) |
| 2016 | `<S>_submission_directory_<year>_<corpus>_…` | first char | `<corpus>` | `A_submission_directory_2016_audiobook_wav_TwelfthNight_0004.wav` → (`audiobook`, A) |
| 2019 | `<S>_submission_directory_<year>_<corpus>_…` | first char | `<corpus>` | `A_submission_directory_2019_celebrity_wav_…` → (`celebrity`, A) |
| 2020 | `<task>_<S>_submission_directory_…` | second token | first token | `MH1_A_submission_directory_news_…` → (`MH1`, A) |
| 2021 | (TBD — inspect CSV) | | | |
| 2023 | `<S>_<year>-<task>_submission_directory_<task>_…` | first char | first `<task>` | `A_2023-FH1_submission_directory_FH1_MOS_…` → (`FH1`, A) |

The CSV `track` column, where present, is **the same as the `system` field**
(despite the dataset README calling it a "subtask" — that's a doc error;
inspection shows `track` always equals the first character of the filename).
We do not rely on it for system extraction — we derive system from the
filename so 2008 (which has no `track` column) works identically.

### Reference system

Within each `(year, subtask)`, the natural-speech reference is conventionally
**system A**, which consistently has the highest MOS (typically >4.5). We
verify this per-subtask in a sanity check at load time and warn if A is not
the top-MOS system (e.g. when A was dropped from a year).

We will compute two reference-distance variants:

1. **Natural-A**: reference distribution = system A's files within the same
   `(year, subtask)`. Tests the representation's "within-subtask"
   discriminability.
2. **External**: reference distribution = a held-out real-speech set
   (LibriTTS `test-clean` for English, AISHELL-3 dev for Mandarin,
   matched by language). Tests "absolute" quality estimation.

For the primary correlation number we use variant 1 (natural-A), because the
listener panels never compared across subtasks. Variant 2 is a secondary
diagnostic.

---

## 2. Top-level layout

```
eval/
├── ARCHITECTURE.md            # this file
├── pyproject.toml             # `srm-eval` distribution
├── srm-eval.yaml              # default config: paths, extractors, distances
├── srm_eval/
│   ├── __init__.py
│   ├── data/
│   │   ├── blizzard.py        # CSV + tarball loader, filename parser
│   │   ├── reference.py       # external reference datasets (LibriTTS etc.)
│   │   └── cache.py           # on-disk WAV cache (~/.cache/srm-eval/)
│   ├── extractors/
│   │   ├── base.py            # Extractor ABC: extract(path) -> np.ndarray
│   │   ├── whisper.py         # Whisper-large encoder, layer 20
│   │   ├── ttsds.py           # wraps ttsds BenchmarkSuite
│   │   └── registry.py        # name -> Extractor constructor
│   ├── distance.py            # wasserstein_2 (TTSDS-style), sliced, frechet
│   ├── correlate.py           # pearson / spearman + bootstrap CIs
│   ├── run.py                 # main pipeline (extract -> distance -> correlate)
│   ├── report.py              # human + machine-readable result tables
│   └── cli.py                 # typer commands: extract, distance, correlate, run
├── scripts/
│   └── download.sh            # one-time data fetch
└── tests/
    ├── test_blizzard_parser.py  # table tests for filename parsing per year
    ├── test_distance.py
    └── test_correlate.py
```

This is a separate Python package from `srm-tpu` (and from any future
representation-training code) so it can be used standalone to evaluate any
representation.

---

## 3. The pipeline

```
            +-------------+
            | Blizzard    |   year, subtask, system, file, mos
            | CSVs+WAVs   |
            +------+------+
                   |
       group_by (year, subtask, system)
                   |
                   v
            +--------------+
            | per-system   |
            | file lists   |
            +------+-------+
                   |
                   v   (for each Extractor in registry)
            +--------------+
            |  Extractor   |   .extract(path) -> np.ndarray[D]
            |  cache: pkl  |   (mean-pooled feature vector per file)
            +------+-------+
                   |
                   v
            per-system distributions  (N_files x D)
                   |
       distance(system_dist, reference_dist)  [W2, sliced, frechet]
                   |
                   v
       per-system Wasserstein distances (a scalar per system)
                   |
       pearson/spearman(distances, system_MOS) within each (year, subtask)
                   |
                   v
            +-------------+
            |  report.    |   per (extractor, year, subtask, distance_kind)
            |  csv/md     |   → rho_pearson, rho_spearman, n_systems, ci_95
            +-------------+
```

### 3.1 Data layer (`data/blizzard.py`)

Single public function:

```python
def load_blizzard(
    years: Iterable[int] = (2008, 2009, 2010, 2011, 2012, 2013, 2016, 2019, 2020, 2021, 2023),
    cache_dir: Path = Path("~/.cache/srm-eval/blizzard").expanduser(),
    download: bool = True,
) -> BlizzardData
```

`BlizzardData` is a frozen dataclass holding a pandas DataFrame with columns:

```
year:int  subtask:str  system:str  filepath:Path  file_mos:float  lang:str
```

and three convenience accessors:

```python
.system_mos() -> DataFrame[year, subtask, system, n_files, mean_mos, std_mos]
.subtasks() -> list[tuple[year, subtask]]
.files_for(year, subtask, system) -> list[Path]
```

Loader steps:

1. For each year in `years`, ensure `cache_dir/Blizzard_<year>.csv` is present
   (download from HF if missing).
2. Ensure `cache_dir/Blizzard_<year>/` is unpacked (extract from
   `Blizzard_<year>.tar.gz` lazily, only on first access of any file).
3. Parse the CSV; rename `score` → `mos` for 2023; apply `parse_filepath` to
   each row to add `subtask` and `system` columns.
4. Drop rows where MOS is NaN or `commercial_ok == "no"` (optional flag).
5. Concatenate across years.

Tarballs total ~18 GB; we unpack lazily and provide a `--prefetch` CLI to
download everything up-front.

### 3.2 Extractor layer (`extractors/`)

```python
class Extractor(ABC):
    name: str             # unique key for the cache
    device: str = "cuda"

    @abstractmethod
    def extract(self, path: Path) -> np.ndarray:
        """Return a (T, D) or (D,) feature array for one audio file."""
```

Two concrete implementations for v1:

#### `extractors/whisper.py`

- Model: `openai/whisper-large` (or `large-v3`; configurable).
- Forward the audio through the encoder; capture
  `model.encoder.layers[19].forward(...)` output (zero-indexed layer 20).
- Mean-pool over the time axis → `(D,)` vector per file.
- Batched: process files in batches of 8 with padding/truncation to 30 s.
- 16 kHz mono, log-mel spectrogram (whisper's built-in preprocessor).

Output shape: `(1280,)` per file (Whisper-large encoder hidden size).

#### `extractors/ttsds.py`

Wraps the TTSDS library. TTSDS evaluates 10–13 individual benchmarks
(prosody/pitch, mpm, hubert, wav2vec2, whisper-activations, dvector,
wespeaker, voicerestore, wada_snr, …) and combines them into a multi-factor
score. We expose two modes:

- `ttsds-combined`: use `BenchmarkSuite.run()`'s final aggregated score.
  Returns a scalar per (system, reference) pair — bypasses our own distance
  computation entirely.
- `ttsds-<benchmark>` (e.g. `ttsds-hubert`): use a single TTSDS benchmark's
  feature `get_distribution(dataset)`, fall back to our generic distance
  computation. Lets us isolate factors and compare against whisper-l20 on
  equal footing.

The wrapper installs `ttsds` lazily on first use (`pip install ttsds`) since
its dep tree is heavy (pyannote, fairseq, etc.).

### 3.3 Feature cache

Every extractor's per-file output is cached at
`cache_dir/features/<extractor>/<file_hash>.npy`. Hash is
`md5(extractor.name + str(path) + extractor.version)`. This lets us re-run
correlations without re-extracting.

### 3.4 Distance layer (`distance.py`)

Three implementations, each taking two `(N, D)` arrays and returning a
scalar:

- `wasserstein_2_perdim(x, y)`: TTSDS's method — compute the 1-D 2-Wasserstein
  distance on each feature dim independently (using the sorted-samples trick
  with subsampling to align sizes), then average. Cheap, stable, what TTSDS
  uses by default.
- `frechet_distance(x, y)`: Gaussian-Wasserstein — fits each as a
  multivariate Gaussian, closed-form W2 distance. This is FID/FAD-style.
- `sliced_wasserstein(x, y, n_projections=128)`: random 1-D projections,
  W2 distance on each, average.

Primary is `wasserstein_2_perdim` for parity with TTSDS. The other two are
diagnostic — we'll report all three.

### 3.5 Correlation layer (`correlate.py`)

```python
def correlate(
    distances: pd.DataFrame,    # cols: year, subtask, system, distance
    mos: pd.DataFrame,          # cols: year, subtask, system, mean_mos
    *,
    method: Literal["pearson", "spearman"] = "spearman",
    bootstrap: int = 1000,
) -> pd.DataFrame                # cols: year, subtask, n_sys, rho, ci_lo, ci_hi
```

Computes `(year, subtask)`-level correlations between Wasserstein distance
and mean MOS. Bootstrap 95% CIs over system resamples.

We **always** flip the sign for reporting so that a positive number means
"distance correctly predicts MOS" (since the raw correlation is negative —
lower distance ↔ higher MOS).

We also report aggregated numbers:

- **Macro**: mean of per-`(year, subtask)` correlations, weighted by number
  of systems.
- **Micro / pooled**: a single correlation over all systems across all
  subtasks. Less meaningful (different listener panels), but matches the
  TTSDS paper's headline number — useful for direct comparison.

### 3.6 Pipeline / CLI

Top-level command:

```bash
srm-eval run --extractor whisper-l20 --years 2008-2023 --reference natural-A
srm-eval run --extractor ttsds-combined
srm-eval run --extractor ttsds-hubert,whisper-l20  --reference both
```

Outputs:

- `runs/<timestamp>/distances.parquet` — long-form, one row per
  `(extractor, year, subtask, system)`
- `runs/<timestamp>/correlations.csv` — one row per `(extractor, year,
  subtask, distance_kind)`
- `runs/<timestamp>/summary.md` — human-readable comparison table
- stdout: rich table of headline numbers

Sub-commands (for partial / debugging runs):

- `srm-eval extract --extractor X --years Y` — extract features only.
- `srm-eval distance --extractor X` — compute distances from cached features.
- `srm-eval correlate <distances.parquet>` — recompute correlations.
- `srm-eval report runs/<timestamp>` — re-render markdown.

---

## 4. Expected reference numbers

For a sanity check the implementation should reproduce roughly:

- **TTSDS-combined**: Spearman ρ ≈ 0.6–0.7 against Blizzard system MOS
  (from the TTSDS paper, table 1). Some subtasks higher, some near 0.
- **Whisper-large layer 20**: in the TTSDS paper this is one of the
  individual "intelligibility" features; expect ρ ≈ 0.3–0.5 alone. Our
  layer-20 mean-pooled version may differ.

If headline numbers come out below ρ=0.2 on most subtasks, something is
broken (likely subtask parsing or system-grouping). The TTSDS authors
publish per-subtask numbers we can diff against during dev.

---

## 5. Implementation order

A suggested 5-step incremental implementation:

1. **`data/blizzard.py` + tests** — get the dataframe right first. Verify
   per `(year, subtask)`: # systems matches the table in §1.
2. **`distance.py` + tests** — port TTSDS's `wasserstein_distance` exactly
   so we can later sanity-check on shared inputs.
3. **`extractors/whisper.py`** — single GPU, batched, with feature caching.
   Smoke-test on one subtask (e.g. Blizzard 2013 EH1, 15 systems): compute
   distances, eyeball MOS correlation.
4. **`extractors/ttsds.py`** — call out to the existing library;
   no original logic.
5. **`correlate.py` + `cli.py` + `run.py`** — wire it all up, generate the
   summary table.

Each step has its own tests and produces a usable artifact, so we can
parallelise or pause between.

---

## 6. Open design questions

- **Layer choice**: the user picked Whisper-large layer 20 specifically.
  Layer 20 is near the top of Whisper-large's 32-layer encoder, where
  features are highly semantic. Should we also report layer 4 / 12 / 24
  for context? Easy to add, but expands the table.
- **Pooling**: mean-pool over time is the default; max-pool and CLS-style
  (first token) are alternatives. Start with mean-pool only.
- **GPU memory**: Whisper-large is ~3 GB; with batch 8 + 30s audio we're
  comfortably under the RTX 5070's 12 GB. No sharding needed.
- **ttsds install**: their pin-heavy dep tree (numpy<2, fairseq fork) will
  collide with `torchax`/`jax[tpu]`. Solution: install `srm-eval[ttsds]`
  into its own venv (`.venv-ttsds`) and shell out from the wrapper. The
  alternative — vendoring TTSDS feature code — is more work than it's worth
  for v1.
- **Reproducibility**: TTSDS's W2 implementation subsamples randomly with
  `np.random.seed(0)`. We seed identically. If we go multi-thread, we'll
  need to thread-local the RNG.
