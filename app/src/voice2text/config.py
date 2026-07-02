"""Runtime configuration model shared by capture, WhisperX STT, translation, and overlay modules."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RuntimeConfig:
    model_size: str = 'small'
    model_device: str = 'cuda'
    compute_type: str = 'float16'
    cpu_threads: int = 0
    stt_provider: str = 'whisperx'
    stt_variant: str = 'auto'
    stt_auto_download: bool = True
    stt_model_path: str = ''
    stt_whispercpp_model_path: str = ''
    stt_whispercpp_model_size: str = 'medium'
    stt_whispercpp_binary_path: str = ''
    stt_whispercpp_server_path: str = ''
    stt_whispercpp_mode: str = 'server'
    stt_whispercpp_server_vad: bool = False
    stt_whispercpp_vad_model_path: str = ''
    stt_whispercpp_vad_model: str = 'ggml-silero-v5.1.2.bin'
    stt_whispercpp_server_max_len: int = 0
    stt_whispercpp_request_timeout_seconds: float = 30.0
    stt_whispercpp_no_speech_threshold: float = 0.85
    stt_whispercpp_avg_logprob_min: float = -1.2
    stt_whispercpp_repetition_similarity: float = 0.92
    stt_whispercpp_boilerplate_phrases: str = '请不吝点赞|訂閱|订阅|轉發|转发|打賞|打赏'
    whisperx_enable_phoneme_asr: bool = True
    whisperx_enable_forced_alignment: bool = True
    whisperx_enable_vad: bool = True
    whisperx_vad_method: str = 'silero-vad'
    whisperx_enable_diarization: bool = False
    whisperx_alignment_model: str = ''
    whisperx_english_align_large: bool = True
    whisperx_zh_align_wbbbbb: bool = False
    whisperx_alignment_language: str = 'auto'
    whisperx_alignment_device: str = 'auto'
    # Alignment CUDA safety guard: 'safe' (default) downgrades CUDA alignment to CPU on Windows
    # (known torchaudio/wav2vec2 access-violation); 'unsafe-cuda' forces CUDA with a warning;
    # 'probe' runs an isolated subprocess probe and caches the verdict.
    # The legacy env var VOICE2TEXT_WHISPERX_ALLOW_UNSAFE_CUDA_ALIGN still works for safe mode.
    whisperx_align_guard: str = 'safe'
    whisperx_diarization_device: str = 'auto'
    whisperx_diarization_model: str = 'pyannote/speaker-diarization-3.1'
    whisperx_hf_token: str = ''
    whisperx_speaker_profile_enabled: bool = True
    whisperx_speaker_profile_backend: str = 'pyannote'
    whisperx_speaker_profile_model: str = 'pyannote/embedding'
    whisperx_speaker_speechbrain_model: str = 'speechbrain/spkrec-ecapa-voxceleb'
    whisperx_speaker_nemo_model: str = 'nvidia/speakerverification_en_titanet_large'
    # Round 0045 Fix 2: set backend='wespeaker' to use this (diar-3.1's own embedding,
    # separates zh where pyannote/embedding collapses cross-window identity).
    whisperx_speaker_wespeaker_model: str = 'pyannote/wespeaker-voxceleb-resnet34-lm'
    whisperx_speaker_profile_match_threshold: float = 0.72
    whisperx_speaker_profile_min_seconds: float = 2.0
    whisperx_speaker_profile_reconcile_threshold: float = 0.52
    whisperx_speaker_profile_store_path: str = ''
    # Rolling-window (realtime) speaker maturity floors. These gate how much repeated
    # evidence a diarized speaker needs before it becomes a candidate profile
    # (candidate_*) and then a stable visible identity (visible_*). They apply ONLY to
    # the realtime rolling path (window <= 15s); long direct chunks keep a conservative
    # fixed policy so reference/export speaker counts stay clean (decoupled by design).
    # Defaults reproduce the shipped behavior (candidate 6s/8x, visible 24s/16x at the
    # standard min_seconds=2.0 operating point). Lower the realtime floors (e.g. 4/5,
    # 6/10) to recover real minority speakers at the cost of more noise splits — validate
    # attribution against ground truth, not speaker count.
    whisperx_speaker_realtime_candidate_seconds: float = 6.0
    whisperx_speaker_realtime_candidate_samples: int = 8
    # 0.0 = derive from match_threshold-0.05; lower (e.g. 0.55) reduces candidate fragmentation
    # so minority speakers' windows concentrate and reach the promotion floor (realtime only).
    whisperx_speaker_realtime_candidate_match_threshold: float = 0.0
    # 0.0 = blend into a profile centroid at the assign gate (legacy); higher (e.g. 0.85) only
    # blends on a strong match so centroids stay pure and the dominant profile cannot drift to
    # absorb other speakers (realtime only).
    whisperx_speaker_realtime_update_match_threshold: float = 0.0
    whisperx_speaker_realtime_visible_seconds: float = 24.0
    whisperx_speaker_realtime_visible_samples: int = 16
    whisperx_speaker_realtime_refresh_seconds: float = 0.0
    whisperx_speaker_realtime_refresh_alpha: float = 0.5
    whisperx_speaker_realtime_refresh_assign_threshold: float = 0.55
    whisperx_speaker_realtime_refresh_min_cluster_seconds: float = 4.0
    whisperx_speaker_realtime_refresh_merge: bool = True
    whisperx_speaker_realtime_refresh_match_mode: str = 'argmax'
    # Round 0023 learn-path quality gate: when on, gibberish / music-tail / degenerate /
    # low-confidence clips can still match an existing profile for display but never update or
    # create a centroid. Default off until the harness A/B confirms it is CER-neutral.
    whisperx_speaker_profile_quality_gate_enabled: bool = False
    whisperx_speaker_profile_quality_min_confidence: float = 0.45
    speaker_marker_style: str = 'spk'
    speaker_pause_break_seconds: float = 1.8
    # Delayed-freeze speaker re-anchoring: hold committed words this long before
    # baking the speaker marker, so a late cross-window profile identity (warmup
    # ~24s) can back-date a new turn's marker to its true onset instead of where the
    # profile finally confirmed (~12-27s late). 0.0 = disabled (legacy immediate
    # freeze, byte-identical). ~26-30 covers the warmup; trades commit latency for
    # marker accuracy. `stabilization` = 'consecutive' (legacy gate) or 'majority'.
    subtitle_commit_hold_seconds: float = 0.0
    subtitle_reanchor_stabilization: str = 'consecutive'
    subtitle_reanchor_majority_window_seconds: float = 2.0
    subtitle_reanchor_majority_min_ratio: float = 0.6
    # Round 0048: pre-commit local diarization relabel for the LIVE overlay (Plan B from 0046).
    # A pending batch's speaker label is resolved from a short local re-diarization pass over its
    # own (not-yet-frozen) audio span right before it freezes -- never rewrites already-committed
    # text (that was 0046's proven failure mode). Read-only against the profile store (no EMA, no
    # merge, no new-profile creation). Default off = byte-identical (no rolling audio buffer, no
    # resolver call). Feasibility-spike-validated at window=20s + sliver_floor=1.5s
    # (app/src/tests/claude_output/plan_b_spike/, 2026-07-02); assign_threshold is NOT
    # spike-validated, a starting point pending A/B.
    subtitle_relabel_enabled: bool = False
    subtitle_relabel_window_seconds: float = 20.0
    subtitle_relabel_sliver_floor_seconds: float = 1.5
    subtitle_relabel_assign_threshold: float = 0.65
    # Round 0052: turn-aware overwrite gate -- a resolved profile only replaces a word's existing
    # non-empty label when its cosine beats the incumbent label's cosine by this margin.
    subtitle_relabel_margin: float = 0.05
    # Final display-script fold for the visible/exported subtitle: '' (off, keep
    # per-word original script), 'hant', or 'hans'. Comparison/CER unaffected.
    subtitle_display_script: str = 'hant'
    cpu_fallback_on_cuda_error: bool = True
    cuda_compat_source_dll: str = 'D:\\CUDA\\bin\\x64\\cublas64_13.dll'
    ffmpeg_dll_dir: str = 'D:\\FFmpeg\\ffmpeg-7.1.1-full_build-shared\\bin'
    # Measured best operating point (round 0014 Phase B): seg 10 / hop 2 (overlap 5)
    # gives the lowest CER and keeps up in realtime (rtf ~0.93 vs the old 6/1.5's
    # ~1.48); startup is unaffected (prefill fires the first window after ~hop).
    segment_seconds: float = 10.0
    hop_seconds: float = 2.0
    # Active runtime preset name ('' = none). Presets seed the model/compute/beam/
    # seg-hop/alignment/diarization/speaker-profile bundle; explicit knobs override.
    runtime_preset: str = ''
    source_language: Optional[str] = None
    cjk_no_space_gap_seconds: float = 0.6
    source_mode: str = 'loopback'
    source_file_path: str = ''
    source_file_replay_speed: float = 0.0
    source_file_chunk_seconds: float = 0.25
    ui_language: str = 'zh'
    source_device_indices: list[int] = field(default_factory=list)
    source_mix_weights: list[float] = field(default_factory=list)
    source_app_name: str = ''
    source_app_names: list[str] = field(default_factory=list)
    source_channel_mode: str = 'mono'
    overlap_merge_method: str = 'stable-tail'
    preprocess_enabled: bool = True
    preprocess_modules: str = 'auto'
    whisper_max_context: Optional[int] = None
    whisper_entropy_thold: Optional[float] = None
    whisper_logprob_thold: Optional[float] = None
    whisper_no_speech_thold: Optional[float] = None
    whisper_temperature: Optional[float] = None
    whisper_beam_size: Optional[int] = 5
    whisper_batch_size: int = 4
    whisper_best_of: Optional[int] = None
    # Rolling per-window initial_prompt: feed this many recent committed chars as
    # decode context for cross-window continuity (code-switch / proper nouns).
    # 0 disables (byte-identical to no prompt).
    whisperx_rolling_prompt_chars: int = 0
    # Round 0049: surgical overrides for faster-whisper's temperature-fallback re-decode
    # (the hard-window latency source). Empty/None = library defaults (byte-identical).
    # NOTE: the legacy whisper_logprob_thold/whisper_no_speech_thold/whisper_temperature
    # fields above are pre-WhisperX dead knobs (no stt/ consumer) — these are the wired ones.
    whisperx_asr_temperatures: str = ""
    whisperx_asr_log_prob_threshold: Optional[float] = None
    whisperx_asr_compression_ratio_threshold: Optional[float] = None
    whisperx_asr_no_speech_threshold: Optional[float] = None
    max_lines: int = 10
    overlay_width: int = 1200
    overlay_height: int = 320
    overlay_x: int = 40
    overlay_y: int = 700
    overlay_opacity: float = 0.8
    font_size: int = 18
    text_color: str = '#F0F2F5'
    source_text_color: str = '#F0F2F5'
    translated_text_color: str = '#FFD98A'
    status_color: str = '#78D7FF'
    error_color: str = '#FF7878'
    background_color: str = '#0A101A'
    translation_enabled: bool = False
    translation_from: str = 'auto'
    translation_to: str = 'zh'
    # Round 0026/0030 pluggable translation backend + off-thread engine policy.
    # backend: 'argos' (default) or 'nllb'; 'llm'/'cloud' are reserved (disabled stubs).
    translation_backend: str = 'argos'
    translation_nllb_model_path: str = ''
    translation_nllb_model_repo: str = 'facebook/nllb-200-distilled-600M'
    translation_nllb_auto_download: bool = True
    translation_nllb_auto_convert: bool = True
    translation_nllb_device: str = 'cpu'
    translation_nllb_compute_type: str = 'int8'
    # Engine policy. queue_max <= 0 keeps the engine in inline-passthrough mode (byte-identical
    # to the historical direct backend call); > 0 moves translation onto a bounded background
    # worker with a per-request timeout + bounded retry so a slow backend never stalls the loop.
    translation_queue_max: int = 0
    translation_request_timeout_seconds: float = 8.0
    translation_max_retries: int = 0
    translation_retry_backoff_seconds: float = 0.25
    bilingual_style: str = 'stacked'
    device_index: Optional[int] = None
    log_dir: str = 'logs'
    debug_mode: bool = False
    # Round 0020: record the live session (exact PCM -> WAV + manifest) for
    # deterministic replay/bug-repro. Default off; ignored for source_mode=file.
    session_record_enabled: bool = False
    # Round 0047: after a genuine session end (not a settings restart / mode switch), run one
    # whole-file direct-quality transcription+diarization pass over the just-recorded session WAV
    # on a background thread and write it as an ADDITIONAL export (never touches the live overlay
    # or the incremental export). Requires session_record_enabled=True to have any effect. Default
    # off -> byte-identical to pre-0047.
    session_finalize_direct_relabel_enabled: bool = False
    transcript_export_enabled: bool = False
    transcript_export_formats: str = 'txt,srt,json'
    transcript_export_include_timestamps: bool = True
    transcript_export_include_speaker: bool = True
    transcript_export_display_text_only: bool = False
    transcript_export_include_confidence: bool = True
    transcript_export_dir: str = ''
    import_direct_path: str = ''
    import_direct_chunk_seconds: float = 0.0
    import_direct_language_subchunk_seconds: float = 30.0
    # Round 0045: direct/import diarization runs once on the whole file (globally
    # consistent labels, no weaker profile re-cluster) instead of per-chunk.
    import_direct_whole_file_diarization: bool = True
