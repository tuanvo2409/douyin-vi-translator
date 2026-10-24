"""Centralized, testable multilingual content-policy signals."""
from __future__ import annotations

from typing import Iterable


HARD_COMMERCE_TERMS = (
    "小黄车", "带货", "下单", "优惠券", "商品链接", "看橱窗", "点击左下角",
    "mua ngay", "giỏ hàng", "gio hang", "mã giảm giá", "ma giam gia", "link mua",
)

SOFT_COMMERCE_TERMS = (
    "好物", "推荐", "购物", "链接", "橱窗", "同款", "拍下", "包邮", "优惠", "价格",
    "性价比", "测评", "开箱", "种草", "安利", "入手", "神器", "商品",
    "khuyến mãi", "khuyen mai", "giá ưu đãi", "gia uu dai", "review sản phẩm",
)

PHOTO_SLIDESHOW_TERMS = (
    "图文", "圖文", "组图", "照片", "壁纸", "截图", "相册", "plog", "九宫格", "图集",
    "album ảnh", "album anh", "trình chiếu ảnh", "trinh chieu anh", "ảnh tĩnh", "anh tinh",
)

TRUST_PATTERNS = (
    "室友", "舍友", "同居", "男生宿舍", "女生宿舍", "寝室", "大学生活", "房东", "押金",
    "邋遢", "吵架", "吐槽", "独居日记", "恋爱日常", "合租",
    "bạn cùng phòng", "ban cung phong", "ở ghép", "o ghep", "phòng trọ", "phong tro",
    "chủ trọ", "chu tro", "tiền cọc", "tien coc", "drama ký túc xá", "drama ky tuc xa",
)


def normalize_text(parts: Iterable[str | None]) -> str:
    return " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip()).casefold()


def matched_terms(text: str, terms: Iterable[str]) -> tuple[str, ...]:
    normalized = text.casefold()
    return tuple(term for term in terms if term.casefold() in normalized)
