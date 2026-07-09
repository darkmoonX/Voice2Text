"""Round 0081 integration test: the writer side (stt/whisperx_provider.py's
_write_alignment_candidate_tag) and the reader side (settings/presenter.py's
discover_custom_alignment_candidates) agree on the same on-disk tag format/layout.

The two unit test files (test_whisperx_alignment_resolution.py,
test_settings_presenter_alignment_discovery.py) each test their own side in isolation with
hand-written JSON; this test wires them together for real so a schema drift between the two
would actually fail a test."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.settings.presenter import discover_custom_alignment_candidates
from voice2text.stt.whisperx_provider import WhisperXTranscriber


class AlignmentDynamicDiscoveryIntegrationTests(unittest.TestCase):
    def _new_stub(self, model_root: Path) -> WhisperXTranscriber:
        inst = WhisperXTranscriber.__new__(WhisperXTranscriber)
        inst._model_root = model_root
        return inst

    def test_repo_tagged_by_provider_is_discovered_by_presenter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-integration-") as td:
            root = Path(td)
            align_root = root / "align"
            local_dir = align_root / "hf" / "some-org-custom-zh-repo"
            local_dir.mkdir(parents=True)
            transcriber = self._new_stub(root)

            transcriber._write_alignment_candidate_tag(
                "zh-hant", local_dir, "some-org/custom-zh-repo", ""
            )
            discovered = discover_custom_alignment_candidates(align_root)

            self.assertEqual(discovered, {"zh": ["some-org/custom-zh-repo"]})

    def test_bundle_tagged_by_provider_is_discovered_by_presenter(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-integration-") as td:
            root = Path(td)
            align_root = root / "align"
            local_dir = align_root / "custom" / "some-custom-bundle"
            local_dir.mkdir(parents=True)
            transcriber = self._new_stub(root)

            transcriber._write_alignment_candidate_tag("en", local_dir, "SOME_CUSTOM_BUNDLE", "")
            discovered = discover_custom_alignment_candidates(align_root)

            self.assertEqual(discovered, {"en": ["SOME_CUSTOM_BUNDLE"]})

    def test_generic_per_language_folder_never_gets_tagged_or_discovered(self) -> None:
        # The generic align/hf/{lang} auto-default folder must stay untagged (see
        # _write_alignment_candidate_tag's docstring) so it never gets misread as "a custom
        # repo literally named 'zh'".
        with tempfile.TemporaryDirectory(prefix="v2t-align-integration-") as td:
            root = Path(td)
            align_root = root / "align"
            local_dir = align_root / "hf" / "zh"
            local_dir.mkdir(parents=True)
            transcriber = self._new_stub(root)

            transcriber._write_alignment_candidate_tag("zh", local_dir, "", "")
            discovered = discover_custom_alignment_candidates(align_root)

            self.assertEqual(discovered, {})


if __name__ == "__main__":
    unittest.main()
