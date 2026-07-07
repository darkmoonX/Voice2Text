from __future__ import annotations

from pathlib import Path
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from voice2text.config import RuntimeConfig
from voice2text.translation.nllb_backend import NllbTranslator, _map_app_code_to_flores
from voice2text.translation.registry import build_backend


class NllbBackendTests(unittest.TestCase):
    def _config(self, **overrides: object) -> RuntimeConfig:
        cfg = RuntimeConfig()
        cfg.translation_enabled = True
        cfg.translation_backend = "nllb"
        cfg.translation_from = "auto"
        cfg.translation_to = "zh-hant"
        cfg.translation_nllb_auto_download = False
        cfg.translation_nllb_auto_convert = False
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return cfg

    def test_missing_ctranslate2_disables_without_raise(self) -> None:
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, *args, **kwargs):
            if name == "ctranslate2":
                return None
            return real_find_spec(name, *args, **kwargs)

        with patch("importlib.util.find_spec", side_effect=fake_find_spec):
            backend = NllbTranslator.from_config(self._config())
        self.assertFalse(backend.enabled)
        self.assertFalse(backend.state.active)
        self.assertIn("ctranslate2", backend.state.message)
        self.assertIsNone(backend.translate("hello", "en"))

    def test_language_code_mapping(self) -> None:
        self.assertEqual(_map_app_code_to_flores("en"), "eng_Latn")
        self.assertEqual(_map_app_code_to_flores("zh"), "zho_Hans")
        self.assertEqual(_map_app_code_to_flores("zh-hant", target=True), "zho_Hant")
        self.assertEqual(_map_app_code_to_flores("ja"), "jpn_Jpan")
        self.assertIsNone(_map_app_code_to_flores("xx"))

    def test_unsupported_source_returns_none(self) -> None:
        backend = NllbTranslator(enabled=True, source_code="auto", target_code="zh", auto_download=False)
        self.assertIsNone(backend.translate("hello", "xx"))

    def test_registry_builds_nllb_backend(self) -> None:
        backend = build_backend("nllb", self._config())
        self.assertIsInstance(backend, NllbTranslator)
        self.assertEqual(backend.name, "nllb")

    def test_registry_import_is_heavy_free(self) -> None:
        for name in ("ctranslate2", "transformers"):
            sys.modules.pop(name, None)
        from voice2text.translation import registry  # noqa: F401

        self.assertNotIn("ctranslate2", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_disabled_construction_does_not_import_heavy_modules(self) -> None:
        for name in ("ctranslate2", "transformers"):
            sys.modules.pop(name, None)
        backend = build_backend("nllb", self._config(translation_enabled=False))
        self.assertIsInstance(backend, NllbTranslator)
        self.assertFalse(backend.enabled)
        self.assertNotIn("ctranslate2", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_ready_translate_uses_stubbed_translator(self) -> None:
        class FakeTokenizer:
            src_lang = ""

            def encode(self, text: str):
                return [10, 11]

            def convert_ids_to_tokens(self, ids):
                return [f"id{item}" for item in ids]

            def convert_tokens_to_ids(self, tokens):
                return [20 for _ in tokens]

            def decode(self, ids, skip_special_tokens: bool = True):
                return "翻譯結果"

        class FakeTranslator:
            def translate_batch(self, source, target_prefix=None, beam_size=4):
                self.source = source
                self.target_prefix = target_prefix
                return [type("Result", (), {"hypotheses": [["zho_Hant", "tok_a"]]})()]

        backend = NllbTranslator(enabled=True, source_code="auto", target_code="zh-hant", auto_download=False)
        backend._translator = FakeTranslator()
        backend._tokenizer = FakeTokenizer()
        backend._ready = True
        backend._enabled = True
        self.assertEqual(backend.translate("hello", "en"), "翻譯結果")

    def test_auto_convert_invoked_when_ct2_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "nllb-ct2"
            with patch.object(NllbTranslator, "_start_warmup", return_value=None):
                backend = NllbTranslator(
                    enabled=True,
                    source_code="auto",
                    target_code="zh",
                    model_dir=model_dir,
                    model_repo=str(Path(tmp) / "pytorch-source"),
                    auto_download=False,
                    auto_convert=True,
                )

            def fake_convert() -> None:
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "config.json").write_text("{}", encoding="utf-8")
                (model_dir / "model.bin").write_bytes(b"ct2")

            with patch.object(
                backend, "_convert_pytorch_to_ct2",
                side_effect=fake_convert,
            ) as convert:
                backend._prepare_model_dir()
            convert.assert_called_once()
            self.assertTrue(backend._is_ct2_model_ready(model_dir))

    def test_warmup_fake_converter_enables_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "nllb-ct2"
            source_dir = root / "pytorch-source"
            source_dir.mkdir()
            (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

            class FakeConverter:
                def __init__(self, model_name_or_path: str, copy_files=None):
                    self.model_name_or_path = model_name_or_path
                    self.copy_files = copy_files

                def convert(self, output_dir: str, quantization=None, force=False, vmap=None):
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "config.json").write_text("{}", encoding="utf-8")
                    (out / "model.bin").write_bytes(b"ct2")

            class FakeCtranslate2:
                class converters:
                    TransformersConverter = FakeConverter

                class Translator:
                    def __init__(self, *_args, **_kwargs):
                        pass

            class FakeTransformers:
                class AutoTokenizer:
                    @staticmethod
                    def from_pretrained(*_args, **_kwargs):
                        return object()

            def fake_import(name: str):
                if name == "ctranslate2":
                    return FakeCtranslate2
                if name == "transformers":
                    return FakeTransformers
                return __import__(name)

            with patch.object(NllbTranslator, "_start_warmup", return_value=None):
                backend = NllbTranslator(
                    enabled=True,
                    source_code="auto",
                    target_code="zh",
                    model_dir=model_dir,
                    model_repo=str(source_dir),
                    auto_download=False,
                    auto_convert=True,
                )
            with patch("importlib.import_module", side_effect=fake_import):
                backend._warmup()

            self.assertTrue(backend.enabled)
            self.assertTrue(backend.state.active)
            self.assertTrue((model_dir / "config.json").exists())
            self.assertTrue((model_dir / "model.bin").exists())

    def test_auto_convert_false_skips_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "nllb-ct2"
            with patch.object(NllbTranslator, "_start_warmup", return_value=None):
                backend = NllbTranslator(
                    enabled=True,
                    source_code="auto",
                    target_code="zh",
                    model_dir=model_dir,
                    model_repo=str(Path(tmp) / "pytorch-source"),
                    auto_download=False,
                    auto_convert=False,
                )
            with patch.object(
                backend, "_convert_pytorch_to_ct2",
            ) as convert:
                backend._prepare_model_dir()
            convert.assert_not_called()
            self.assertFalse(backend._is_ct2_model_ready(model_dir))

    def test_conversion_failure_stays_disabled_without_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(NllbTranslator, "_start_warmup", return_value=None):
                backend = NllbTranslator(
                    enabled=True,
                    source_code="auto",
                    target_code="zh",
                    model_dir=Path(tmp) / "nllb-ct2",
                    model_repo=str(Path(tmp) / "pytorch-source"),
                    auto_download=False,
                    auto_convert=True,
                )
            with patch.object(
                backend,
                "_convert_pytorch_to_ct2",
                side_effect=RuntimeError("boom"),
            ):
                backend._warmup()
            self.assertFalse(backend.enabled)
            self.assertFalse(backend.state.active)
            self.assertIn("boom", backend.state.message)

    def test_downloaded_pytorch_cache_removed_after_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "nllb-ct2"
            with patch.object(NllbTranslator, "_start_warmup", return_value=None):
                backend = NllbTranslator(
                    enabled=True,
                    source_code="auto",
                    target_code="zh",
                    model_dir=model_dir,
                    # Not a local path -> conversion source is downloaded into the _pytorch cache.
                    model_repo="facebook/nllb-200-distilled-600M",
                    auto_download=False,
                    auto_convert=True,
                )
            raw_dir = backend._raw_pytorch_cache_dir()

            def fake_download(*, output_dir, **_kwargs):
                out = Path(output_dir)
                out.mkdir(parents=True, exist_ok=True)
                (out / "config.json").write_text("{}", encoding="utf-8")
                (out / "pytorch_model.bin").write_bytes(b"pt")

            class FakeConverter:
                def __init__(self, model_name_or_path: str, copy_files=None):
                    self.model_name_or_path = model_name_or_path
                    self.copy_files = copy_files

                def convert(self, output_dir: str, quantization=None, force=False, vmap=None):
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "config.json").write_text("{}", encoding="utf-8")
                    (out / "model.bin").write_bytes(b"ct2")

            class FakeCtranslate2:
                class converters:
                    TransformersConverter = FakeConverter

            def fake_import(name: str):
                if name == "ctranslate2":
                    return FakeCtranslate2
                return __import__(name)

            with patch("voice2text.translation.nllb_backend.download_hf_files_with_progress", side_effect=fake_download), \
                    patch("importlib.import_module", side_effect=fake_import):
                backend._convert_pytorch_to_ct2()

            self.assertTrue(backend._is_ct2_model_ready(model_dir))
            self.assertFalse(raw_dir.exists())

    def test_local_pytorch_source_is_not_deleted_after_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "nllb-ct2"
            source_dir = root / "pytorch-source"
            source_dir.mkdir()
            (source_dir / "config.json").write_text("{}", encoding="utf-8")
            (source_dir / "pytorch_model.bin").write_bytes(b"pt")
            with patch.object(NllbTranslator, "_start_warmup", return_value=None):
                backend = NllbTranslator(
                    enabled=True,
                    source_code="auto",
                    target_code="zh",
                    model_dir=model_dir,
                    model_repo=str(source_dir),
                    auto_download=False,
                    auto_convert=True,
                )

            class FakeConverter:
                def __init__(self, model_name_or_path: str, copy_files=None):
                    self.model_name_or_path = model_name_or_path
                    self.copy_files = copy_files

                def convert(self, output_dir: str, quantization=None, force=False, vmap=None):
                    out = Path(output_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    (out / "config.json").write_text("{}", encoding="utf-8")
                    (out / "model.bin").write_bytes(b"ct2")

            class FakeCtranslate2:
                class converters:
                    TransformersConverter = FakeConverter

            with patch("importlib.import_module", side_effect=lambda n: FakeCtranslate2 if n == "ctranslate2" else __import__(n)):
                backend._convert_pytorch_to_ct2()

            self.assertTrue(backend._is_ct2_model_ready(model_dir))
            self.assertTrue(source_dir.exists())  # user-supplied source must be preserved


if __name__ == "__main__":
    unittest.main()
