from __future__ import annotations

import unittest

from canonical_profile import CanonicalProfileConfigurationError, resolve_canonical_profile


class CanonicalProfileTests(unittest.TestCase):
    def test_page_profiles_resolve_distinct_persona_voice_and_safe_subtitle_policy(self) -> None:
        page_a = resolve_canonical_profile({"localization_profile": "page_goc_tro_bat_on", "target_platform": "tiktok"})
        page_b = resolve_canonical_profile({"localization_profile": "page_giai_cuu_chuong_lon", "target_platform": "tiktok"})
        self.assertEqual("BV007_streaming", page_a.voice)
        self.assertEqual("BV421_vivn_streaming", page_b.voice)
        self.assertNotEqual(page_a.persona, page_b.persona)
        self.assertEqual("source_aware_measured", page_a.subtitle_presentation["layout"])
        self.assertEqual(["cyan", "yellow"], page_a.subtitle_presentation["brand_color_options"])
        self.assertTrue(page_a.subtitle_presentation["brand_configuration_required"])
        self.assertEqual(2, page_b.subtitle_presentation["max_visible_lines"])
        self.assertEqual(
            {"badge_side": "left", "background": "white", "text": "black"},
            page_b.subtitle_presentation["brand_overrides"],
        )

    def test_unknown_profile_fails_closed_with_configuration_required(self) -> None:
        with self.assertRaises(CanonicalProfileConfigurationError) as raised:
            resolve_canonical_profile({"localization_profile": "unknown", "target_platform": "tiktok"})
        self.assertIn("localization_profile", str(raised.exception))

    def test_existing_legacy_profile_remains_an_explicit_non_page_default(self) -> None:
        profile = resolve_canonical_profile({"localization_profile": "loc-v1", "target_platform": "tiktok"})
        self.assertEqual("loc-v1", profile.profile_id)
        self.assertIsNone(profile.persona)
        self.assertEqual("BV421_vivn_streaming", profile.voice)


if __name__ == "__main__":
    unittest.main()
