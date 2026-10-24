import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llm_translator
import vn_localizer


class _Response:
    def __init__(self, translations):
        self.translations = translations

    def raise_for_status(self):
        pass

    def json(self):
        return {"choices": [{"message": {"content": self.translations}}]}


class DeepSeekTranslationTests(unittest.TestCase):
    def test_unresolved_text_never_becomes_empty_provider_input(self):
        calls = []

        def request(payload):
            calls.append(payload)
            return {"translations": []}

        result = vn_localizer.localize(
            [{
                "utteranceId": "u-av-conflict",
                "resolvedZh": "甲",
                "source_confidence": "high",
                "classification": "RESOLVED_DIALOGUE",
                "kind": "dialogue",
                "flags": ["AV_CONFLICT", "UNRESOLVED"],
            }],
            scene_memory={"terminologyEvidence": []},
            request=request,
            profile="neutral_faithful",
        )

        self.assertEqual([], calls)
        self.assertEqual([], result["accepted"])
        self.assertEqual([{"utteranceId": "u-av-conflict", "reason": "UNRESOLVED_SOURCE"}], result["review_required"])

    def test_retries_only_invalid_chinese_result_and_keeps_valid_positions(self):
        cues = [
            {"position": 4, "startMs": 0, "endMs": 1200, "sourceTextZh": "老婆生了龙凤胎"},
            {"position": 9, "startMs": 1200, "endMs": 2400, "sourceTextZh": "上面记得要留空"},
        ]
        replies = [
            _Response('{"translations":[{"position":9,"translatedTextVi":"上面记得要留空"},{"position":4,"translatedTextVi":"Vợ sinh đôi trai gái"}]}'),
            _Response('{"translations":[{"position":9,"translatedTextVi":"Phía trên nhớ chừa khoảng trống"}]}'),
        ]
        with patch.object(llm_translator.requests, "post", side_effect=replies) as post:
            result = llm_translator.translate_with_deepseek(cues, "test-key")

        self.assertEqual(["Vợ sinh đôi trai gái", "Phía trên nhớ chừa khoảng trống"],
                         [cue["translatedTextVi"] for cue in result])
        self.assertEqual(2, post.call_count)
        self.assertEqual("deepseek-flash", post.call_args_list[0].kwargs["json"]["model"])
        self.assertEqual({"type": "disabled"}, post.call_args_list[0].kwargs["json"]["thinking"])

    def test_exhausted_retry_is_left_blank_and_marked_for_review(self):
        cues = [{"position": 0, "startMs": 0, "endMs": 1000, "sourceTextZh": "留空"}]
        bad = _Response('{"translations":[{"position":0,"translatedTextVi":"留空"}]}')
        with patch.object(llm_translator.requests, "post", return_value=bad):
            result = llm_translator.translate_with_deepseek(cues, "test-key")
        self.assertFalse(result[0].get("translatedTextVi"))
        self.assertTrue(result[0]["needsReview"])


if __name__ == "__main__":
    unittest.main()
