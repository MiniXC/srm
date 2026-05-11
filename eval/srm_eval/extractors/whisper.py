"""Whisper-large encoder extractor — layer 20, 1-second pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from srm_eval.extractors.base import Extractor

WHISPER_FRAME_RATE = 50
WHISPER_SAMPLE_RATE = 16000
WHISPER_LAYER = 20


def _load_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    import resampy

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != WHISPER_SAMPLE_RATE:
        audio = resampy.resample(audio, sr, WHISPER_SAMPLE_RATE)
    return audio.astype(np.float32)


class WhisperExtractor(Extractor):
    name = "whisper-l20"
    version = "5"

    def __init__(self, device: str = "cuda", cache_dir: str = "~/.cache/srm-eval/features"):
        super().__init__(device=device, cache_dir=cache_dir)
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model("large", device=self.device)
        return self._model

    def extract(self, path: Path) -> np.ndarray:
        import whisper

        model = self._load_model()
        audio_np = _load_audio(path)
        mel = whisper.log_mel_spectrogram(audio_np, n_mels=128)

        if isinstance(mel, np.ndarray):
            mel = torch.from_numpy(mel)
        mel = whisper.pad_or_trim(mel, 3000).unsqueeze(0).to(self.device)

        with torch.no_grad():
            x = model.encoder.conv1(mel)
            x = model.encoder.conv2(x)
            x = x.permute(0, 2, 1)
            x = x + model.encoder.positional_embedding

            hidden = x
            for i, block in enumerate(model.encoder.blocks):
                hidden = block(hidden)
                if i == WHISPER_LAYER - 1:
                    h20 = hidden.clone()

            h20_np = h20.squeeze(0).cpu().numpy()

        n_frames = h20_np.shape[0]
        chunk_size = WHISPER_FRAME_RATE
        n_chunks = max(1, n_frames // chunk_size)
        pooled = np.zeros((n_chunks, h20_np.shape[1]), dtype=np.float32)

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_frames)
            pooled[i] = h20_np[start:end].mean(axis=0)

        return pooled
