from __future__ import annotations

import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gemini_pool
import llm_translator
import llm_transport


class GeminiKeyPoolTests(unittest.TestCase):
    def _pool(self, keys=("TEST_KEY_ALPHA_111111", "TEST_KEY_BRAVO_222222")):
        pool = gemini_pool.GeminiKeyPool()
        pool.keys = list(keys)
        pool.cooldowns = {}
        pool.current_idx = 0
        return pool

    def test_ready_keys_round_robin(self) -> None:
        pool = self._pool()
        self.assertEqual("TEST_KEY_ALPHA_111111", pool.get_key())
        self.assertEqual("TEST_KEY_BRAVO_222222", pool.get_key())
        self.assertEqual("TEST_KEY_ALPHA_111111", pool.get_key())

    def test_429_cools_only_failed_key_and_next_key_is_selected(self) -> None:
        pool = self._pool()
        first = pool.get_key()
        pool.report_error(first, 429)
        self.assertEqual("TEST_KEY_BRAVO_222222", pool.get_key())

    def test_cooling_key_is_never_returned_while_another_is_ready(self) -> None:
        pool = self._pool()
        pool.cooldowns["TEST_KEY_ALPHA_111111"] = 200.0
        with patch("gemini_pool.time.time", return_value=100.0):
            self.assertEqual("TEST_KEY_BRAVO_222222", pool.get_key())

    def test_all_cooling_keys_return_none(self) -> None:
        pool = self._pool()
        pool.cooldowns = {key: 200.0 for key in pool.keys}
        with patch("gemini_pool.time.time", return_value=100.0):
            self.assertIsNone(pool.get_key())

    def test_expired_cooldown_restores_eligibility(self) -> None:
        pool = self._pool(("TEST_KEY_ALPHA_111111",))
        pool.cooldowns["TEST_KEY_ALPHA_111111"] = 100.0
        with patch("gemini_pool.time.time", return_value=100.0):
            self.assertEqual("TEST_KEY_ALPHA_111111", pool.get_key())

    def test_duplicate_environment_keys_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(gemini_pool, "ENV_FILE", Path(directory) / "missing.env"), patch.dict(os.environ, {"GEMINI_API_KEYS": "TEST_KEY_ALPHA_111111, TEST_KEY_ALPHA_111111,TEST_KEY_BRAVO_222222"}, clear=False):
            self.assertEqual(["TEST_KEY_ALPHA_111111", "TEST_KEY_BRAVO_222222"], gemini_pool.GeminiKeyPool().keys)

    def test_plural_environment_pool_has_precedence_over_single_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.object(gemini_pool, "ENV_FILE", Path(directory) / "missing.env"), patch.dict(os.environ, {"GEMINI_API_KEYS": "TEST_KEY_ALPHA_111111,TEST_KEY_BRAVO_222222", "GEMINI_API_KEY": "TEST_SINGLE_333333"}, clear=False):
            self.assertEqual(["TEST_KEY_ALPHA_111111", "TEST_KEY_BRAVO_222222"], gemini_pool.GeminiKeyPool().keys)

    def test_safe_stats_and_logs_do_not_contain_key_material(self) -> None:
        pool = self._pool()
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = gemini_pool.logger
        logger.addHandler(handler)
        try:
            pool.report_error("TEST_KEY_ALPHA_111111", 403)
            rendered = str(pool.get_stats()) + output.getvalue()
        finally:
            logger.removeHandler(handler)
        for secret in ("TEST_KEY_ALPHA_111111", "TEST_KEY_BRAVO_222222", "111111", "222222"):
            self.assertNotIn(secret, rendered)
        self.assertEqual({"total": 2, "active": 1, "cooling": 1}, pool.get_stats())

    def test_403_cools_failed_key_but_keeps_other_key_available(self) -> None:
        pool = self._pool()
        pool.report_error("TEST_KEY_ALPHA_111111", 403)
        self.assertEqual("TEST_KEY_BRAVO_222222", pool.get_key())

    def test_translator_does_not_retry_or_fallback_when_configured_pool_has_no_ready_key(self) -> None:
        original_keys = list(llm_translator.gemini_pool.keys)
        original_cooldowns = dict(llm_translator.gemini_pool.cooldowns)
        original_index = llm_translator.gemini_pool.current_idx
        try:
            llm_translator.gemini_pool.keys = ["TEST_KEY_ALPHA_111111"]
            llm_translator.gemini_pool.cooldowns = {"TEST_KEY_ALPHA_111111": float("inf")}
            llm_translator.gemini_pool.current_idx = 0
            with patch.object(llm_translator, "translate_with_gemini_single_chunk") as request:
                result = llm_translator.translate_with_gemini([{"position": 0, "startMs": 0, "endMs": 1000, "sourceTextZh": "x"}], api_key="TEST_SINGLE_333333")
            request.assert_not_called()
            self.assertNotIn("translatedTextVi", result[0])
        finally:
            llm_translator.gemini_pool.keys = original_keys
            llm_translator.gemini_pool.cooldowns = original_cooldowns
            llm_translator.gemini_pool.current_idx = original_index

    def test_structured_transport_retries_with_next_ready_pool_key_after_429(self) -> None:
        pool = self._pool()
        limited = type("Response", (), {"status_code": 429})()
        succeeded = type("Response", (), {"status_code": 200, "json": lambda self: {"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}]}})()
        with patch.object(llm_transport, "gemini_pool", pool), patch.object(llm_transport.requests, "post", side_effect=[limited, succeeded]) as post:
            result = llm_transport.gemini_json_transport(model="fake").call({"contents": []}, required_keys={"ok"})
        self.assertEqual({"ok": True}, result)
        self.assertEqual(2, post.call_count)
        self.assertEqual(1, pool.get_stats()["cooling"])

    def test_structured_transport_fails_typed_without_network_when_pool_has_no_ready_key(self) -> None:
        pool = self._pool(("TEST_KEY_ALPHA_111111",))
        pool.cooldowns["TEST_KEY_ALPHA_111111"] = float("inf")
        with patch.object(llm_transport, "gemini_pool", pool), patch.object(llm_transport.requests, "post") as post:
            with self.assertRaisesRegex(llm_transport.ProviderError, "NO_READY_GEMINI_KEY"):
                llm_transport.gemini_json_transport(model="fake").call({"contents": []}, required_keys={"ok"})
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
