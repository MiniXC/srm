"""WeSpeaker extractor — VoxBlink2 SimAMResNet100-ft (multilingual) via ONNX.

Uses onnxruntime with pre-extracted fbank features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from srm_eval.extractors.base import Extractor

SAMPLE_RATE = 16000
FBANK_DIM = 80


def _extract_fbank(audio: np.ndarray, sr: int) -> np.ndarray:
    """80-dim log-mel fbank (25ms window, 10ms shift) via torchaudio."""
    import torch
    import torchaudio

    if sr != SAMPLE_RATE:
        audio_t = torchaudio.functional.resample(
            torch.from_numpy(audio).unsqueeze(0), sr, SAMPLE_RATE
        )
    else:
        audio_t = torch.from_numpy(audio).unsqueeze(0)

    fbank = torchaudio.compliance.kaldi.fbank(
        audio_t,
        num_mel_bins=FBANK_DIM,
        sample_frequency=SAMPLE_RATE,
        frame_length=25.0,
        frame_shift=10.0,
        window_type="hamming",
        use_energy=False,
    )
    return fbank.numpy().astype(np.float32)


class WeSpeakerExtractor(Extractor):
    name = "wespeaker-voxblink2"
    version = "3"

    # ResNet100 has output dim 256, ResNet34 has 256. Both are 256.
    EMBEDDING_DIM = 256

    def __init__(
        self,
        device: str = "cpu",
        cache_dir: str = "~/.cache/srm-eval/features",
        model_path: str | None = None,
    ):
        super().__init__(device=device, cache_dir=cache_dir)
        self._model_path = model_path
        self._session = None
        self._input_name = None
        self._output_name = None

    def _load_model(self):
        if self._session is None:
            import onnxruntime as ort

            if self._model_path is None:
                # Auto-discover in repo root.
                candidates = sorted(Path(__file__).resolve().parents[3].glob("voxblink2_samresnet*.onnx"))
                if candidates:
                    self._model_path = str(candidates[-1])  # prefer larger (100 > 34)
                else:
                    raise RuntimeError(
                        "No WeSpeaker ONNX model found. Place voxblink2_samresnet*.onnx "
                        "in the repo root."
                    )

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self._session = ort.InferenceSession(
                self._model_path, providers=providers
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

    def extract(self, path: Path) -> np.ndarray:
        """Return (T_chunks, 256) — one speaker embedding per 1-second chunk."""
        import soundfile as sf

        self._load_model()

        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        feats = _extract_fbank(audio, sr)  # (T_frames, 80)

        # 100 frames = 1 second (10ms shift).
        chunk_size = 100
        n_chunks = max(1, feats.shape[0] // chunk_size)

        embeddings = []
        for i in range(n_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk = feats[start:end][np.newaxis, :, :]  # (1, 100, 80)
            emb = self._session.run(
                [self._output_name], {self._input_name: chunk}
            )[0]  # (1, 256)
            embeddings.append(emb[0])

        return np.stack(embeddings, axis=0)
