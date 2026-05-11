"""Whisper-large encoder extractor — layer 20, 1-second pooling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchaudio

from srm_eval.extractors.base import Extractor

# Whisper-large frame rate (input 30ms stride, output ≈50 Hz for 25ms hop).
# The encoder uses 25ms window with 20ms stride → 50 frames/sec.
# Actually Whisper uses mel spectrogram with 25ms window, 10ms hop = 100 Hz.
# Then 2x stride via conv1d = 50 Hz. Let's find empirically.
# With a 30-second clip, Whisper's encoder outputs ~1500 frames.
# So 1500 / 30 = 50 frames/sec.  Pool 50 frames per 1-second chunk.

WHISPER_FRAME_RATE = 50  # encoder output frames per second
WHISPER_SAMPLE_RATE = 16000
WHISPER_LAYER = 20  # 0-indexed, so layer 20 in the paper = index 19


class WhisperExtractor(Extractor):
    name = "whisper-l20"
    version = "2"  # bumped for layer-20 + 1s-pooling

    def __init__(self, device: str = "cuda", cache_dir: str = "~/.cache/srm-eval/features"):
        super().__init__(device=device, cache_dir=cache_dir)
        self._model = None

    def _load_model(self):
        if self._model is None:
            import whisper  # type: ignore[import-untyped]
            self._model = whisper.load_model("large", device=self.device)
        return self._model

    def extract(self, path: Path) -> np.ndarray:
        """Return (T_chunks, 1280) — one vector per 1-second window."""
        import whisper

        model = self._load_model()

        # Load + resample to 16kHz.
        audio, sr = torchaudio.load(path)
        if sr != WHISPER_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, WHISPER_SAMPLE_RATE)
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio_np = audio.squeeze(0).numpy()

        # Get log-mel spectrogram (Whisper's internal preprocessing).
        mel = whisper.log_mel_spectrogram(audio_np, n_mels=128 if "large" in self._model else 80)

        # Pad to 30s max for memory safety, then do the full forward.
        mel = whisper.pad_or_trim(mel)

        # Feed through encoder, capture layer-20 hidden states.
        with torch.no_grad():
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(self.device)
            # Whisper encoder layers are in model.encoder.blocks (nn.ModuleList)
            # We need to run all layers and capture the output of layer 20 (index 19).
            x = model.encoder.conv1(mel_t)
            x = model.encoder.conv2(x)
            x = x.permute(0, 2, 1)
            x = x + model.encoder.positional_embedding

            hidden = x
            for i, block in enumerate(model.encoder.blocks):
                hidden = block(hidden)
                if i == WHISPER_LAYER - 1:
                    h20 = hidden.clone()

            # h20 shape: (1, n_frames, 1280)
            h20_np = h20.squeeze(0).cpu().numpy()  # (n_frames, 1280)

        # Pool into 1-second chunks: 50 frames per chunk.
        n_frames = h20_np.shape[0]
        chunk_size = WHISPER_FRAME_RATE  # 50 frames = ~1 second
        n_chunks = max(1, n_frames // chunk_size)
        pooled = np.zeros((n_chunks, h20_np.shape[1]), dtype=np.float32)

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, n_frames)
            pooled[i] = h20_np[start:end].mean(axis=0)

        return pooled
