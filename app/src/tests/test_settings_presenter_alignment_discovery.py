"""Round 0081: Settings presenter's disk-based discovery of custom alignment models.

`discover_custom_alignment_candidates()` scans `.v2t_align_meta.json` sidecar tags that
`stt/whisperx_provider.py::_write_alignment_candidate_tag` writes once a repo/bundle is
actually loaded for a language, so the Settings dialog can show custom candidates without a
separately-persisted registry."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.settings.presenter import discover_custom_alignment_candidates


def _write_tag(root: Path, subdir: str, dirname: str, language: str, model: str) -> None:
    target = root / subdir / dirname
    target.mkdir(parents=True, exist_ok=True)
    (target / ".v2t_align_meta.json").write_text(
        json.dumps({"language": language, "model": model}), encoding="utf-8"
    )


class AlignmentCandidateDiscoveryTests(unittest.TestCase):
    def test_empty_root_returns_empty_dict(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            self.assertEqual(discover_custom_alignment_candidates(Path(td)), {})

    def test_discovers_tagged_repo_under_hf(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            _write_tag(root, "hf", "some-org-some-repo", "zh", "some-org/some-repo")

            self.assertEqual(
                discover_custom_alignment_candidates(root), {"zh": ["some-org/some-repo"]}
            )

    def test_discovers_tagged_bundle_under_custom(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            _write_tag(root, "custom", "some-bundle-name", "en", "SOME_CUSTOM_BUNDLE")

            self.assertEqual(
                discover_custom_alignment_candidates(root), {"en": ["SOME_CUSTOM_BUNDLE"]}
            )

    def test_multiple_languages_and_multiple_models_per_language(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            _write_tag(root, "hf", "repo-a", "zh", "org/repo-a")
            _write_tag(root, "hf", "repo-b", "zh", "org/repo-b")
            _write_tag(root, "hf", "repo-c", "en", "org/repo-c")

            result = discover_custom_alignment_candidates(root)

            self.assertEqual(sorted(result["zh"]), ["org/repo-a", "org/repo-b"])
            self.assertEqual(result["en"], ["org/repo-c"])

    def test_missing_tag_file_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            (root / "hf" / "untagged-dir").mkdir(parents=True)

            self.assertEqual(discover_custom_alignment_candidates(root), {})

    def test_malformed_tag_file_is_skipped_not_raised(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            target = root / "hf" / "broken"
            target.mkdir(parents=True)
            (target / ".v2t_align_meta.json").write_text("{not valid json", encoding="utf-8")

            self.assertEqual(discover_custom_alignment_candidates(root), {})

    def test_tag_missing_language_or_model_field_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            target = root / "hf" / "partial"
            target.mkdir(parents=True)
            (target / ".v2t_align_meta.json").write_text(
                json.dumps({"language": "zh"}), encoding="utf-8"
            )

            self.assertEqual(discover_custom_alignment_candidates(root), {})

    def test_duplicate_model_within_same_language_is_deduped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="v2t-align-discover-") as td:
            root = Path(td)
            _write_tag(root, "hf", "dir-a", "zh", "org/same-repo")
            _write_tag(root, "custom", "dir-b", "zh", "org/same-repo")

            result = discover_custom_alignment_candidates(root)

            self.assertEqual(result["zh"], ["org/same-repo"])


if __name__ == "__main__":
    unittest.main()
