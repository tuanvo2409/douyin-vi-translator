"""Gemini API Key Pool Manager - Xoay tua nhiều API Key thông minh.

Tính năng:
- Round-Robin tự động luân phiên giữa các key
- Tự động phát hiện 429 (Rate Limit) và đưa key vào hàng đợi nghỉ ngơi (cooldown 60s)
- Tự động nhảy sang key kế tiếp ngay lập tức mà không làm gián đoạn pipeline
- Đọc trực tiếp file .env để luôn cập nhật key mới nhất
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("dubvi.gemini_pool")

ENV_FILE = Path(__file__).resolve().parent / ".env"


class GeminiKeyPool:
    def __init__(self):
        self.keys: List[str] = []
        self.cooldowns: Dict[str, float] = {}
        self.current_idx: int = 0
        self.load_keys()

    def load_keys(self) -> List[str]:
        """Đọc danh sách key trực tiếp từ file .env."""
        raw = ""
        if ENV_FILE.is_file():
            text = ENV_FILE.read_text(encoding="utf-8")
            for line in text.splitlines():
                line = line.strip()
                if line.startswith("GEMINI_API_KEYS="):
                    raw = line.split("=", 1)[1].strip()
                    break
                elif line.startswith("GEMINI_API_KEY=") and not raw:
                    raw = line.split("=", 1)[1].strip()
        if not raw:
            raw = os.getenv("GEMINI_API_KEYS", "") or os.getenv("GEMINI_API_KEY", "")

        keys = []
        for part in raw.replace("\n", ",").replace(";", ",").split(","):
            k = part.strip().strip('"').strip("'")
            if k and k not in keys:
                keys.append(k)

        self.keys = keys
        self.current_idx = 0
        return self.keys

    def set_keys(self, new_keys: List[str], save_env: bool = True) -> None:
        cleaned = []
        for k in new_keys:
            ck = k.strip().strip('"').strip("'")
            if ck and ck not in cleaned:
                cleaned.append(ck)
        self.keys = cleaned
        self.current_idx = 0

        if save_env and ENV_FILE.is_file():
            try:
                lines = ENV_FILE.read_text(encoding="utf-8").splitlines()
                new_lines = []
                found_keys = False
                joined_keys = ",".join(self.keys)
                for line in lines:
                    if line.startswith("GEMINI_API_KEYS="):
                        new_lines.append(f"GEMINI_API_KEYS={joined_keys}")
                        found_keys = True
                    elif line.startswith("GEMINI_API_KEY="):
                        first_key = self.keys[0] if self.keys else ""
                        new_lines.append(f"GEMINI_API_KEY={first_key}")
                    else:
                        new_lines.append(line)
                if not found_keys:
                    new_lines.append(f"GEMINI_API_KEYS={joined_keys}")
                ENV_FILE.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            except Exception as e:
                logger.warning(f"Lỗi khi lưu .env: {e}")

    def get_key(self) -> Optional[str]:
        if not self.keys:
            self.load_keys()
        if not self.keys:
            return None

        now = time.time()
        n = len(self.keys)

        for i in range(n):
            idx = (self.current_idx + i) % n
            candidate = self.keys[idx]
            if self.cooldowns.get(candidate, 0) <= now:
                self.current_idx = (idx + 1) % n
                return candidate

        earliest_key = min(self.keys, key=lambda k: self.cooldowns.get(k, 0))
        return earliest_key

    def report_error(self, key: str, status_code: int = 429) -> None:
        if not key:
            return
        if status_code == 429:
            self.cooldowns[key] = time.time() + 60.0
            logger.warning(f"🔑 Key '...{key[-6:]}' bị 429 Rate Limit. Đưa vào cooldown 60s.")
        elif status_code in (400, 403):
            self.cooldowns[key] = time.time() + 3600.0
            logger.warning(f"🔑 Key '...{key[-6:]}' lỗi {status_code}. Đưa vào cooldown 1h.")

    def get_stats(self) -> Dict[str, Any]:
        now = time.time()
        active = sum(1 for k in self.keys if self.cooldowns.get(k, 0) <= now)
        cooling = len(self.keys) - active
        return {
            "total": len(self.keys),
            "active": active,
            "cooling": cooling,
            "keys": [
                {
                    "masked": f"...{k[-6:]}" if len(k) > 6 else k,
                    "is_ready": self.cooldowns.get(k, 0) <= now,
                    "cooldown_remaining": max(0, int(self.cooldowns.get(k, 0) - now)),
                }
                for k in self.keys
            ],
        }


# Singleton instance
gemini_pool = GeminiKeyPool()
