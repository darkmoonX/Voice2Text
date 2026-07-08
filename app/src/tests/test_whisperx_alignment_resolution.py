"""Unit tests for WhisperX alignment model cache resolution."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt.whisperx_provider import WhisperXTranscriber


def _touch_model_files(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    # Content is irrelevant for cache-detection tests.
    (root / "pytorch_model.bin").write_bytes(b"ok")


class WhisperXAlignmentResolutionTests(unittest.TestCase):
    def _new_stub(self, model_root: Path) -> WhisperXTranscriber:
        inst = WhisperXTranscriber.__new__(WhisperXTranscriber)
        inst._model_root = model_root
        inst._alignment_model = ""
        inst._english_align_large = True
        inst._zh_align_wbbbbb = False
        inst._alignment_model_defaults = {}
        inst._alignment_language = "auto"
        inst._source_language_hint = None
        return inst

    def test_alignment_repo_resolution_lazy_imports_whisperx_alignment_mapping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._whisperx = object()

            repo_id = transcriber._resolve_alignment_repo_id("pt")

            self.assertEqual(repo_id, "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese")

    def test_auto_mode_reuses_language_scoped_cache_when_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            _touch_model_files(root / "align" / "hf" / "zh")
            transcriber = self._new_stub(root)
            transcriber._resolve_alignment_repo_id = lambda _lang: ""  # type: ignore[attr-defined]

            resolved = transcriber._resolve_alignment_model_name_for_load("zh")

            self.assertEqual(
                Path(resolved),
                root / "align" / "hf" / "zh",
            )

    def test_auto_mode_prefers_language_folder_even_if_repo_slug_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            repo_slug_dir = root / "align" / "hf" / "jonatasgrosman-wav2vec2-large-xlsr-53-chinese-zh-cn"
            lang_dir = root / "align" / "hf" / "zh"
            _touch_model_files(repo_slug_dir)
            _touch_model_files(lang_dir)

            transcriber = self._new_stub(root)
            transcriber._resolve_alignment_repo_id = (  # type: ignore[attr-defined]
                lambda _lang: "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            )

            resolved = transcriber._resolve_alignment_model_name_for_load("zh")

            self.assertEqual(Path(resolved), lang_dir)

    def test_follow_source_uses_stt_source_language_folder(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            transcriber = self._new_stub(root)
            transcriber._alignment_language = "follow-source"
            transcriber._source_language_hint = "en"
            transcriber._resolve_alignment_repo_id = (  # type: ignore[attr-defined]
                lambda _lang: "facebook/wav2vec2-large-960h-lv60-self"
            )
            _touch_model_files(root / "align" / "hf" / "en")

            resolved = transcriber._resolve_alignment_model_name_for_load("zh")

            self.assertEqual(Path(resolved), root / "align" / "hf" / "en")

    def test_explicit_repo_is_keyed_by_repo_id_not_language(self) -> None:
        # Regression: explicit HF repos must cache under a repo-keyed folder, not
        # a shared language folder, so two repos for the same language don't
        # collide and silently reuse the first-cached model.
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            transcriber = self._new_stub(root)
            transcriber._alignment_language = "follow-source"
            transcriber._source_language_hint = "zh"
            transcriber._alignment_model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"

            resolved_dir = transcriber._resolve_alignment_local_dir(
                "zh",
                "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            )

            self.assertEqual(
                resolved_dir,
                root / "align" / "hf" / "jonatasgrosman-wav2vec2-large-xlsr-53-chinese-zh-cn",
            )
            self.assertNotEqual(resolved_dir, root / "align" / "hf" / "zh")

    def test_two_explicit_english_repos_get_distinct_cache_dirs(self) -> None:
        # The language-scoped-cache bug: two different English HF align repos
        # used to collide into align/hf/en. They must now be isolated.
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            transcriber = self._new_stub(root)

            transcriber._alignment_model = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            dir_a = transcriber._resolve_alignment_local_dir(
                "en", "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            )
            transcriber._alignment_model = "facebook/wav2vec2-large-960h-lv60-self"
            dir_b = transcriber._resolve_alignment_local_dir(
                "en", "facebook/wav2vec2-large-960h-lv60-self"
            )

            self.assertNotEqual(dir_a, dir_b)
            self.assertNotEqual(dir_a, root / "align" / "hf" / "en")
            self.assertNotEqual(dir_b, root / "align" / "hf" / "en")

    def test_changing_explicit_repo_does_not_reuse_stale_language_cache(self) -> None:
        # A stale align/hf/en (e.g. a different pre-cached model) must NOT shadow
        # a newly-configured explicit repo via the load-name fallback.
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            _touch_model_files(root / "align" / "hf" / "en")
            transcriber = self._new_stub(root)
            transcriber._alignment_model = "facebook/wav2vec2-large-960h-lv60-self"
            transcriber._resolve_alignment_repo_id = (  # type: ignore[attr-defined]
                lambda _lang: "facebook/wav2vec2-large-960h-lv60-self"
            )

            resolved = transcriber._resolve_alignment_model_name_for_load("en")

            # Falls through to the explicit repo id, not the stale hf/en cache.
            self.assertEqual(resolved, "facebook/wav2vec2-large-960h-lv60-self")
            self.assertNotEqual(Path(resolved), root / "align" / "hf" / "en")

    def test_auto_default_repo_still_uses_language_folder(self) -> None:
        # Auto-default per-language repos keep the flat language folder so
        # existing caches stay valid (no surprise re-download).
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            transcriber = self._new_stub(root)
            transcriber._alignment_model = ""  # no explicit repo => auto-default

            resolved_dir = transcriber._resolve_alignment_local_dir(
                "zh", "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            )

            self.assertEqual(resolved_dir, root / "align" / "hf" / "zh")

    def test_explicit_model_used_when_no_local_cache_ready(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            transcriber = self._new_stub(root)
            transcriber._alignment_model = "WAV2VEC2_ASR_LARGE_960H"
            transcriber._resolve_alignment_repo_id = lambda _lang: ""  # type: ignore[attr-defined]

            resolved = transcriber._resolve_alignment_model_name_for_load("zh")

            self.assertEqual(resolved, "WAV2VEC2_ASR_LARGE_960H")

    def test_english_default_upgrades_to_large_bundle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            self.assertEqual(
                transcriber._effective_alignment_model("en"),
                "WAV2VEC2_ASR_LARGE_LV60K_960H",
            )

    def test_english_upgrade_routes_to_custom_dir_bypassing_stale_hf_en(self) -> None:
        # A stale align/hf/en cache must NOT shortcut the English large bundle.
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            _touch_model_files(root / "align" / "hf" / "en")
            transcriber = self._new_stub(root)
            transcriber._resolve_alignment_repo_id = lambda _lang: ""  # type: ignore[attr-defined]
            resolved = transcriber._resolve_alignment_model_name_for_load("en")
            self.assertEqual(resolved, "WAV2VEC2_ASR_LARGE_LV60K_960H")
            local_dir = transcriber._resolve_alignment_local_dir("en", "")
            self.assertNotEqual(Path(local_dir), root / "align" / "hf" / "en")

    def test_english_upgrade_off_keeps_base_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._english_align_large = False
            self.assertEqual(transcriber._effective_alignment_model("en"), "")

    def test_cjk_unaffected_by_english_upgrade(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            self.assertEqual(transcriber._effective_alignment_model("zh"), "")

    def test_alignment_model_defaults_map_wins_over_legacy_booleans(self) -> None:
        # Round 0077: the generalized per-language map takes priority over the legacy
        # english_align_large/zh_align_wbbbbb booleans when it has an entry for the language.
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._alignment_model_defaults = {"en": "WAV2VEC2_ASR_BASE_960H"}
            self.assertEqual(transcriber._effective_alignment_model("en"), "WAV2VEC2_ASR_BASE_960H")

    def test_alignment_model_defaults_covers_language_with_no_legacy_flag(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._alignment_model_defaults = {
                "ja": "patrickvonplaten/wav2vec2-large-xlsr-53-japanese",
            }
            self.assertEqual(
                transcriber._effective_alignment_model("ja"),
                "patrickvonplaten/wav2vec2-large-xlsr-53-japanese",
            )

    def test_alignment_model_defaults_falls_back_to_legacy_boolean_when_no_entry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._zh_align_wbbbbb = True
            transcriber._alignment_model_defaults = {"en": "WAV2VEC2_ASR_BASE_960H"}
            self.assertEqual(
                transcriber._effective_alignment_model("zh"),
                "wbbbbb/wav2vec2-large-chinese-zh-cn",
            )

    def test_explicit_pin_wins_over_alignment_model_defaults(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._alignment_model = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
            transcriber._alignment_model_defaults = {"zh": "wbbbbb/wav2vec2-large-chinese-zh-cn"}
            self.assertEqual(
                transcriber._effective_alignment_model("zh"),
                "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            )

    def test_explicit_model_wins_over_english_upgrade(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            transcriber = self._new_stub(Path(td))
            transcriber._alignment_model = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
            self.assertEqual(
                transcriber._effective_alignment_model("en"),
                "jonatasgrosman/wav2vec2-large-xlsr-53-english",
            )

    def test_external_download_monitor_uses_expected_total_when_available(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-") as td:
            root = Path(td)
            messages: list[str] = []
            transcriber = self._new_stub(root)
            transcriber._progress_callback = messages.append
            transcriber._download_probe_roots = [root]
            transcriber._external_download_monitor_lock = threading.Lock()
            transcriber._external_download_monitor_suppress_epoch = 0
            transcriber._external_download_monitor_suppress_count = 0

            def write_downloaded_file() -> str:
                (root / "download.bin").write_bytes(b"x" * (5 * 1024 * 1024))
                return "ok"

            result = transcriber._run_with_external_download_progress(
                "align-pt",
                write_downloaded_file,
                expected_total_bytes=10 * 1024 * 1024,
            )

            self.assertEqual(result, "ok")
            joined = "\n".join(messages)
            self.assertIn("50%", joined)
            self.assertIn("(5.0/10.0 MB)", joined)
            self.assertNotIn("total unknown", joined)


if __name__ == "__main__":
    unittest.main()
