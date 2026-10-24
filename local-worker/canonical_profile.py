"""Canonical Translator page-profile resolution.

Only explicit inbound localization profiles are accepted.  Legacy callers may
continue using their own channel defaults; this resolver is the single source
of canonical page semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


class CanonicalProfileConfigurationError(ValueError):
    """A canonical profile is missing or cannot be resolved safely."""


@dataclass(frozen=True)
class CanonicalLocalizationProfile:
    profile_id: str
    persona: str | None
    voice: str
    tts_provider: str
    subtitle_presentation: Mapping[str, object]


_SAFE_SUBTITLE_PRESENTATION = {
    "layout": "source_aware_measured",
    "max_visible_lines": 2,
    "brand_overrides": None,
}

_PROFILES = {
    # Existing P1C fixtures and legacy callers use this opaque default route.
    # It intentionally carries no inferred page persona or presentation brand.
    "loc-v1": CanonicalLocalizationProfile(
        "loc-v1",
        None,
        "BV421_vivn_streaming",
        "capcut",
        dict(_SAFE_SUBTITLE_PRESENTATION),
    ),
    "page_goc_tro_bat_on": CanonicalLocalizationProfile(
        "page_goc_tro_bat_on",
        "sarcastic_dramatic_roommate_landlord_student_housing",
        "BV075_streaming",
        "capcut",
        {
            **_SAFE_SUBTITLE_PRESENTATION,
            "brand_overrides": None,
            "color": "#00FFFF",
            "brand_color_options": ["cyan", "yellow"],
            "configuration_required_reason": "USER_PRODUCT_DECISION: Page A recovery evidence conflicts on cyan versus yellow",
            "brand_configuration_required": True,
        },
    ),
    "page_giai_cuu_chuong_lon": CanonicalLocalizationProfile(
        "page_giai_cuu_chuong_lon",
        "humorous_self_deprecating_cleaning_home_appliance_review",
        "BV421_vivn_streaming",
        "capcut",
        {
            **_SAFE_SUBTITLE_PRESENTATION,
            "brand_overrides": {"badge_side": "left", "background": "white", "text": "black"},
            "color": "#FFE600",
            "badge_side": "left",
            "background": "white",
            "text": "black",
            "brand_configuration_required": False,
        },
    ),
}


def resolve_canonical_profile(document: Mapping[str, object]) -> CanonicalLocalizationProfile:
    """Resolve an explicit supported-platform profile; never infer from a path or slug."""

    if not isinstance(document, Mapping):
        raise CanonicalProfileConfigurationError("canonical localization_profile is required")
    platform = document.get("target_platform")
    if platform not in {"tiktok", "facebook-reels"}:
        raise CanonicalProfileConfigurationError("canonical target_platform must be tiktok or facebook-reels")
    profile_id = document.get("localization_profile")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise CanonicalProfileConfigurationError("canonical localization_profile is required")
    profile = _PROFILES.get(profile_id.strip())
    if profile is None:
        raise CanonicalProfileConfigurationError(
            f"canonical localization_profile {profile_id!r} requires configuration"
        )
    return profile
