"""Round 0022 Phase A: headless model/alignment cache scan + guarded delete."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.stt.model_cache import (
    cache_summary,
    delete_cache_entry,
    human_size,
    scan_model_cache,
)


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00" * size)


def _build_cache(root: Path) -> None:
    # Base ASR model.
    _write(root / "stt" / "medium" / "model.bin", 100)
    # Alignment: nested model under align/hf/zh/<model>.
    _write(root / "align" / "hf" / "zh" / "wav2vec2-zh" / "pytorch_model.bin", 200)
    # Alignment: flat lang dir holding files directly under align/torch/en.
    _write(root / "align" / "torch" / "en" / "weights.pt", 50)
    # An empty (not-ready) model dir.
    (root / "stt" / "empty-model").mkdir(parents=True, exist_ok=True)


class HumanSizeTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(human_size(0), "0 B")
        self.assertEqual(human_size(512), "512 B")
        self.assertEqual(human_size(1536), "1.5 KB")
        self.assertEqual(human_size(1024 * 1024), "1.0 MB")
        self.assertEqual(human_size(3 * 1024 ** 3), "3.0 GB")


class ScanTests(unittest.TestCase):
    def test_scan_reports_entries_sizes_and_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_cache(root)
            scan = scan_model_cache(root)
            by_name = {e.name: e for e in scan.entries}

            self.assertIn("medium", by_name)
            self.assertEqual(by_name["medium"].kind, "stt")
            self.assertEqual(by_name["medium"].size_bytes, 100)
            self.assertTrue(by_name["medium"].ready)

            self.assertIn("wav2vec2-zh", by_name)
            self.assertEqual(by_name["wav2vec2-zh"].kind, "align")
            self.assertEqual(by_name["wav2vec2-zh"].lang, "zh")
            self.assertEqual(by_name["wav2vec2-zh"].size_bytes, 200)

            # Flat lang dir (align/torch/en) becomes a single entry named after the lang.
            self.assertIn("en", by_name)
            self.assertEqual(by_name["en"].kind, "align")
            self.assertEqual(by_name["en"].size_bytes, 50)

            # Empty model dir is listed but not ready.
            self.assertIn("empty-model", by_name)
            self.assertFalse(by_name["empty-model"].ready)
            self.assertEqual(by_name["empty-model"].size_bytes, 0)

            self.assertEqual(scan.total_bytes, 350)
            self.assertEqual(scan.bucket_totals().get("stt"), 100)
            self.assertEqual(scan.bucket_totals().get("align"), 250)

    def test_scan_missing_root_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan = scan_model_cache(Path(tmp) / "does-not-exist")
            self.assertEqual(scan.entries, [])
            self.assertEqual(scan.total_bytes, 0)

    def test_cache_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_cache(root)
            info = cache_summary(root)
            self.assertEqual(info["total_bytes"], 350)
            self.assertEqual(info["entry_count"], 4)
            self.assertEqual(info["bucket_totals"].get("align"), 250)


class HuggingFaceLayoutTests(unittest.TestCase):
    def test_skips_hf_hub_internals_and_dedupes_blob_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # align/cache/models--org--name/ in HF-hub layout: blobs hold the real data, snapshots
            # symlink/copy it (would double-count), plus .no_exist/refs internals.
            hub = root / "align" / "cache" / "models--org--wav2vec2-thing"
            _write(hub / "blobs" / "abc123", 1000)
            _write(hub / "snapshots" / "rev" / "pytorch_model.bin", 1000)  # duplicate of the blob
            _write(hub / "refs" / "main", 40)
            (hub / ".no_exist").mkdir(parents=True, exist_ok=True)

            entries = {(e.name, e.lang): e for e in scan_model_cache(root).entries}
            # Exactly one entry for the hub model; no blobs/refs/snapshots/.no_exist rows.
            self.assertIn(("org/wav2vec2-thing", ""), entries)
            self.assertNotIn(("blobs", "models--org--wav2vec2-thing"), entries)
            for bad in ("blobs", "refs", "snapshots", ".no_exist"):
                self.assertFalse(any(e.name == bad for e in scan_model_cache(root).entries), bad)
            # Size counts blobs only (1000), not blobs+snapshots (2000).
            self.assertEqual(entries[("org/wav2vec2-thing", "")].size_bytes, 1000)

    def test_flat_lang_with_kenlm_subdir_is_one_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # align/hf/ro/ holds the model at root plus a language_model/ (kenlm) subdir + .cache.
            _write(root / "align" / "hf" / "ro" / "config.json", 50)
            _write(root / "align" / "hf" / "ro" / "pytorch_model.bin", 400)
            _write(root / "align" / "hf" / "ro" / "language_model" / "lm.bin", 200)
            _write(root / "align" / "hf" / "ro" / ".cache" / "x", 10)
            entries = [e for e in scan_model_cache(root).entries if e.kind == "align"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].name, "ro")
            self.assertEqual(entries[0].lang, "ro")
            # language_model + .cache are not separate entries, but are counted in the size.
            self.assertEqual(entries[0].size_bytes, 660)

    def test_lang_with_extra_model_subdirs_aggregates_to_one_entry(self):
        # Intentional: a lang dir with a primary model at root + alternate model subdirs is ONE
        # entry sized over everything, so a delete frees exactly the displayed size (predictable).
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "align" / "hf" / "zh" / "config.json", 50)
            _write(root / "align" / "hf" / "zh" / "pytorch_model.bin", 300)
            _write(root / "align" / "hf" / "zh" / "alt-model" / "model.safetensors", 500)
            align = [e for e in scan_model_cache(root).entries if e.kind == "align"]
            self.assertEqual([e.name for e in align], ["zh"])
            self.assertEqual(align[0].size_bytes, 850)  # root + alt-model both counted


class DeleteTests(unittest.TestCase):
    def test_delete_in_root_returns_freed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _build_cache(root)
            target = root / "align" / "hf" / "zh" / "wav2vec2-zh"
            freed = delete_cache_entry(target, root=root)
            self.assertEqual(freed, 200)
            self.assertFalse(target.exists())

    def test_delete_outside_root_refused(self):
        with tempfile.TemporaryDirectory() as tmp_root, tempfile.TemporaryDirectory() as tmp_other:
            outside = Path(tmp_other) / "victim"
            outside.mkdir(parents=True)
            (outside / "f").write_bytes(b"x")
            with self.assertRaises(ValueError):
                delete_cache_entry(outside, root=Path(tmp_root))
            self.assertTrue(outside.exists())  # not deleted

    def test_delete_root_itself_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                delete_cache_entry(root, root=root)
            self.assertTrue(root.exists())

    def test_delete_missing_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            freed = delete_cache_entry(root / "stt" / "nope", root=root)
            self.assertEqual(freed, 0)


if __name__ == "__main__":
    unittest.main()
