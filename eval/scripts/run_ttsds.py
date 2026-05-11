"""TTSDS runner — executed standalone in the TTSDS venv."""

import json
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

import torchaudio

# ---- monkey-patches for compatibility ----

# 1. torchaudio 2.7+ removed sox_effects, etc.
for mod in [
    "torchaudio.sox_effects",
    "torchaudio.compliance",
    "torchaudio.functional",
]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
sys.modules["torchaudio.sox_effects"].apply_effects_tensor = (
    lambda *a, **kw: (torchaudio.load(a[0])[0], 16000)
)

# 2. voicerestore needs a fake class that also provides bigvgan submodule.
import sys as _sys
import types as _types

class _FakeRestorer:
    BIGVGAN_SR = 24000
    model = None
    def __init__(self, *a, **kw): pass
    def _load_model(self): pass
    def estimate_quality(self, audio, sr): return 0.0
    def get_quality(self, audio, sr): return 0.0

_vr_mock = _types.ModuleType("voicerestore")
_vr_mock.__path__ = []  # make it a package
_vr_mock_restore = _types.ModuleType("voicerestore.restore")
_vr_mock_restore.ShortAudioRestorer = _FakeRestorer
_vr_mock_restore.load_bigvgan_model = lambda device: None
_vr_mock_bigvgan = _types.ModuleType("voicerestore.bigvgan")
def _fake_mel(*a, **kw):
    import numpy as np
    return np.zeros((1, 80, 100))
_vr_mock_bigvgan.get_mel_spectrogram = _fake_mel
_vr_mock.restore = _vr_mock_restore
_vr_mock.bigvgan = _vr_mock_bigvgan
_sys.modules["voicerestore"] = _vr_mock
_sys.modules["voicerestore.restore"] = _vr_mock_restore
_sys.modules["voicerestore.bigvgan"] = _vr_mock_bigvgan

from ttsds import BenchmarkSuite
from ttsds.util.dataset import WavListDataset


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python run_ttsds.py <config.json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        config = json.load(f)

    tasks = config["tasks"]
    cache = Path(config["cache_dir"])

    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for t in tasks:
        groups[(t["year"], t["subtask"])].append(t)

    for (year, subtask), group in sorted(groups.items()):
        print(f"\n=== {year}/{subtask} ({len(group)} systems) ===", flush=True)

        datasets = []
        refs = []
        for t in group:
            paths = [Path(p) for p in t["files"]]
            ds = WavListDataset(wavs=paths, name=t["system"], sample_rate=16000)
            datasets.append(ds)
            if t["system"] == "A":
                refs.append(ds)

        if not refs:
            refs = [datasets[0]]

        suite = BenchmarkSuite(
            datasets=datasets,
            reference_datasets=refs,
            skip_errors=True,
            include_environment=False,
            cache_dir=str(cache / f"{year}_{subtask}_cache"),
        )
        result = suite.run()

        out = cache / f"{year}_{subtask}.json"
        out.write_text(
            result.to_json(orient="records", indent=2, default_handler=str)
        )
        print(f"  wrote {len(result)} rows to {out.name}", flush=True)

    print("\nall done.", flush=True)


if __name__ == "__main__":
    main()
