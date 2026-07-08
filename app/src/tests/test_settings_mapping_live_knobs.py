from __future__ import annotations

import unittest

from voice2text.settings.mapping import SettingsPayloadInput, build_settings_updates


def _payload(**overrides) -> SettingsPayloadInput:
    base = dict(
        ui_language="zh",
        source_mode="loopback",
        stt_provider="whisperx",
        stt_variant="auto",
        compute_type="float16",
        stt_model_path="",
        stt_auto_download=True,
        whisperx_enable_phoneme_asr=True,
        whisperx_enable_forced_alignment=True,
        whisperx_enable_vad=True,
        whisperx_vad_method="silero-vad",
        whisperx_enable_diarization=False,
        whisperx_alignment_model="",
        whisperx_alignment_language="auto",
        whisperx_alignment_device="auto",
        whisperx_align_guard="safe",
        whisperx_diarization_device="auto",
        whisperx_diarization_model="pyannote/speaker-diarization-3.1",
        whisperx_diarization_expected_speakers=0,
        whisperx_hf_token="",
        whisperx_speaker_profile_backend="pyannote",
        source_language="auto",
        translation_to="en",
        segment_seconds=10.0,
        hop_seconds=2.0,
        selected_loopback_indices=[],
        selected_app_names=[],
        overlap_merge_method="stable-tail",
        preprocess_enabled=True,
        preprocess_modules="auto",
        translation_enabled=False,
        translation_backend="argos",
        bilingual_style="stacked",
        font_size=20,
        overlay_opacity=0.8,
        source_text_color="#ffffff",
        translated_text_color="#ffffff",
        background_color="#000000",
        debug_mode=False,
        transcript_export_enabled=False,
        transcript_export_formats="txt,srt,json",
        transcript_export_include_timestamps=True,
        transcript_export_include_speaker=True,
    )
    base.update(overrides)
    return SettingsPayloadInput(**base)


def _build(payload: SettingsPayloadInput) -> dict[str, object]:
    return build_settings_updates(payload, lang="zh", hop_gt_segment_message="hop>segment")


class LiveKnobMappingTests(unittest.TestCase):
    def test_defaults_map_to_byte_identical_values(self) -> None:
        updates = _build(_payload())
        self.assertEqual(updates["whisperx_alignment_model_defaults"], {})
        self.assertEqual(updates["whisperx_asr_temperatures"], "")
        self.assertEqual(updates["subtitle_commit_hold_seconds"], 0.0)
        self.assertEqual(updates["whisperx_diarization_min_speakers"], 0)
        self.assertEqual(updates["whisperx_diarization_max_speakers"], 0)

    def test_values_pass_through(self) -> None:
        updates = _build(
            _payload(
                whisperx_alignment_model_defaults={"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"},
                whisperx_asr_temperatures="0.0,0.2,0.4",
                subtitle_commit_hold_seconds=20.0,
                whisperx_diarization_expected_speakers=3,
            )
        )
        self.assertEqual(
            updates["whisperx_alignment_model_defaults"],
            {"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"},
        )
        self.assertEqual(updates["whisperx_asr_temperatures"], "0.0,0.2,0.4")
        self.assertEqual(updates["subtitle_commit_hold_seconds"], 20.0)
        self.assertEqual(updates["whisperx_diarization_min_speakers"], 3)
        self.assertEqual(updates["whisperx_diarization_max_speakers"], 3)

    def test_invalid_temperatures_raise_with_dialog_message(self) -> None:
        payload = _payload(
            whisperx_asr_temperatures="not,a,schedule",
            asr_temperatures_invalid_message="INVALID-TEMPS",
        )
        with self.assertRaises(ValueError) as ctx:
            _build(payload)
        self.assertEqual(str(ctx.exception), "INVALID-TEMPS")

    def test_out_of_range_temperatures_raise(self) -> None:
        with self.assertRaises(ValueError):
            _build(_payload(whisperx_asr_temperatures="0.0,1.5"))

    def test_commit_hold_clamped(self) -> None:
        updates = _build(_payload(subtitle_commit_hold_seconds=999.0))
        self.assertEqual(updates["subtitle_commit_hold_seconds"], 120.0)
        updates = _build(_payload(subtitle_commit_hold_seconds=-5.0))
        self.assertEqual(updates["subtitle_commit_hold_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
