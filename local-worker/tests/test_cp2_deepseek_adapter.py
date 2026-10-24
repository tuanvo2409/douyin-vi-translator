import json
import unittest
from unittest.mock import patch

from cp2_deepseek_adapter import CP2DeepSeekAdapter


class _Response:
    def __init__(self, document):
        self.document = document

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": json.dumps(self.document, ensure_ascii=False)}}]}


class DeepSeekCP2Tests(unittest.TestCase):
    def test_native_uses_source_and_semantic_and_preserves_identity(self):
        payload = {"mode": "native", "utterances": [
            {"utteranceId": "u9", "resolvedZh": "不如这样改", "semanticVi": "Thà cải tạo thế này", "neighbors": []},
            {"utteranceId": "u2", "resolvedZh": "上面记得要留空", "semanticVi": "Phía trên nhớ để trống", "neighbors": []},
        ], "localization_profile": "neutral_faithful"}
        reply = _Response({"translations": [
            {"utteranceId": "u2", "nativeVi": "Phía trên nhớ chừa khoảng trống"},
            {"utteranceId": "u9", "nativeVi": "Hay là sửa thế này"},
        ]})
        with patch("cp2_deepseek_adapter.requests.post", return_value=reply) as post:
            output = CP2DeepSeekAdapter(api_key="test-key")(payload)
        self.assertEqual(["u9", "u2"], [item["utteranceId"] for item in output["translations"]])
        self.assertEqual("Hay là sửa thế này", output["translations"][0]["nativeVi"])
        sent = post.call_args.kwargs["json"]
        self.assertEqual("deepseek-flash", sent["model"])
        self.assertEqual({"type": "disabled"}, sent["thinking"])
        self.assertIn("不如这样改", sent["messages"][1]["content"])
        self.assertIn("Thà cải tạo thế này", sent["messages"][1]["content"])

    def test_wrong_identity_fails_closed(self):
        payload = {"mode": "semantic", "utterances": [
            {"utteranceId": "u9", "resolvedZh": "不如这样改"},
        ]}
        bad = _Response({"translations": [{"utteranceId": "other", "semanticVi": "Hay là sửa thế này"}]})
        with patch("cp2_deepseek_adapter.requests.post", return_value=bad):
            with self.assertRaisesRegex(ValueError, "identity"):
                CP2DeepSeekAdapter(api_key="test-key")(payload)

    def test_retries_only_chinese_leak_without_retranslating_valid_id(self):
        payload = {"mode": "semantic", "utterances": [
            {"utteranceId": "u1", "resolvedZh": "上面记得要留空"},
            {"utteranceId": "u2", "resolvedZh": "那就手机安个住小帮"},
        ]}
        first = _Response({"translations": [
            {"utteranceId": "u1", "semanticVi": "Phía trên nhớ chừa khoảng trống"},
            {"utteranceId": "u2", "semanticVi": "Cài 住小帮 trên điện thoại"},
        ]})
        second = _Response({"translations": [
            {"utteranceId": "u2", "semanticVi": "Cài ứng dụng Zhuxiaobang trên điện thoại"},
        ]})
        with patch("cp2_deepseek_adapter.requests.post", side_effect=[first, second]) as post:
            result = CP2DeepSeekAdapter(api_key="test-key")(payload)
        self.assertEqual("Cài ứng dụng Zhuxiaobang trên điện thoại", result["translations"][1]["semanticVi"])
        self.assertEqual(2, post.call_count)
        retry_rows = json.loads(post.call_args_list[1].kwargs["json"]["messages"][1]["content"])
        self.assertEqual(["u2"], [row["utteranceId"] for row in retry_rows])

    def test_stray_id_is_discarded_and_missing_id_retried(self):
        payload = {"mode": "semantic", "utterances": [
            {"utteranceId": "u1", "resolvedZh": "空间太小"},
            {"utteranceId": "u2", "resolvedZh": "不如这样改"},
        ]}
        first = _Response({"translations": [
            {"utteranceId": "u1", "semanticVi": "Không gian quá chật"},
            {"utteranceId": "wrong", "semanticVi": "Đoán bừa"},
        ]})
        second = _Response({"translations": [
            {"utteranceId": "u2", "semanticVi": "Hay sửa thế này"},
        ]})
        with patch("cp2_deepseek_adapter.requests.post", side_effect=[first, second]) as post:
            result = CP2DeepSeekAdapter(api_key="test-key")(payload)
        self.assertEqual(["u1", "u2"], [row["utteranceId"] for row in result["translations"]])
        self.assertEqual("Không gian quá chật", result["translations"][0]["semanticVi"])
        retry_rows = json.loads(post.call_args_list[1].kwargs["json"]["messages"][1]["content"])
        self.assertEqual(["u2"], [row["utteranceId"] for row in retry_rows])


if __name__ == "__main__":
    unittest.main()
