"""Common STT provider typing contracts and supported provider enum values."""
from __future__ import annotations
from typing import Literal, Optional, Protocol
from ..audio_capture import AudioChunk
STTProvider = Literal['whisperx', 'whispercpp']
SUPPORTED_STT_PROVIDERS: tuple[STTProvider, ...] = ('whisperx', 'whispercpp')

class STTTranscriber(Protocol):

    def has_enough_signal(self, chunk: AudioChunk, threshold: float=0.008, channel_mode: str='mono') -> bool:
        ...

    def transcribe(self, chunk: AudioChunk, language: Optional[str]=None, channel_mode: str='mono') -> str:
        ...
