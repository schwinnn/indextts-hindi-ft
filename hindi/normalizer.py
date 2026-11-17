"""
印地语文本前端：normalize_hindi(text) -> str

覆盖：
  - Unicode NFC 规范化
  - 天城文数字 → 阿拉伯数字
  - 数字展开（整数、小数、序数、百分号、货币 ₹、时间、单位）
  - 常见缩写展开
  - 句末 danda（।）→ 句点
  - 控制字符清理

在 itts25ft 里注册方式：
    --normalizer module:hindi.normalizer:normalize_hindi
    --case none
"""
from __future__ import annotations
import re
import unicodedata
from typing import Callable

# ── 天城文数字映射 ────────────────────────────────────────────────
_DEVA_DIGIT = str.maketrans("०१२३४५६७८९", "0123456789")

# ── 数字转文字表 ──────────────────────────────────────────────────
# 印地语 1-99 有独立词形，不能用规则拼
_HI_ONES = [
    "", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ",
    "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह",
    "अठारह", "उन्नीस", "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस",
    "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस",
    "इकतीस", "बत्तीस", "तैंतीस", "चौंतीस", "पैंतीस", "छत्तीस",
    "सैंतीस", "अड़तीस", "उनतालीस", "चालीस", "इकतालीस", "बयालीस",
    "तैंतालीस", "चौवालीस", "पैंतालीस", "छियालीस", "सैंतालीस",
    "अड़तालीस", "उनचास", "पचास", "इक्यावन", "बावन", "तिरपन",
    "चौवन", "पचपन", "छप्पन", "सत्तावन", "अट्ठावन", "उनसठ", "साठ",
    "इकसठ", "बासठ", "तिरसठ", "चौंसठ", "पैंसठ", "छियासठ", "सड़सठ",
    "अड़सठ", "उनहत्तर", "सत्तर", "इकहत्तर", "बहत्तर", "तिहत्तर",
    "चौहत्तर", "पचहत्तर", "छिहत्तर", "सतहत्तर", "अठहत्तर",
    "उनासी", "अस्सी", "इक्यासी", "बयासी", "तिरासी", "चौरासी",
    "पचासी", "छियासी", "सत्तासी", "अट्ठासी", "नवासी", "नब्बे",
    "इक्यानबे", "बानबे", "तिरानबे", "चौरानबे", "पचानबे",
    "छियानबे", "सत्तानबे", "अट्ठानबे", "निन्यानबे",
]
_HI_HUNDREDS = ["", "एक सौ", "दो सौ", "तीन सौ", "चार सौ",
                "पाँच सौ", "छह सौ", "सात सौ", "आठ सौ", "नौ सौ"]
_HI_SCALES   = ["", "हज़ार", "लाख", "करोड़", "अरब", "खरब"]


def _hi_int(n: int) -> str:
    if n < 0:
        return "माइनस " + _hi_int(-n)
    if n == 0:
        return "शून्य"
    if n <= 99:
        return _HI_ONES[n]
    if n <= 999:
        h, r = divmod(n, 100)
        base = _HI_HUNDREDS[h]
        return (base + " " + _HI_ONES[r]).strip() if r else base
    # भारतीय अंक प्रणाली: हज़ार / लाख / करोड़
    breakpoints = [
        (10_00_00_00_000, 4),  # खरब
        ( 1_00_00_00_000, 3),  # अरब
        (    1_00_00_000, 2),  # करोड़
        (       1_00_000, 1),  # लाख
        (          1_000, 0),  # हज़ार
    ]
    for threshold, idx in breakpoints:
        if n >= threshold:
            q, r = divmod(n, threshold)
            scale_word = _HI_SCALES[idx + 1]
            part = _hi_int(q) + " " + scale_word
            return (part + " " + _hi_int(r)).strip() if r else part
    return str(n)  # fallback


def _hi_ordinal(n: int) -> str:
    """序数词：पहला / दूसरा / तीसरा / … वाँ"""
    _special = {1: "पहला", 2: "दूसरा", 3: "तीसरा", 4: "चौथा", 5: "पाँचवाँ"}
    return _special.get(n, _hi_int(n) + "वाँ")


# ── 单位 ──────────────────────────────────────────────────────────
_UNITS: list[tuple[str, str]] = [
    ("km/h", "किलोमीटर प्रति घंटा"),
    ("km",   "किलोमीटर"),
    ("cm",   "सेंटीमीटर"),
    ("mm",   "मिलीमीटर"),
    ("kg",   "किलोग्राम"),
    ("g",    "ग्राम"),
    ("mg",   "मिलीग्राम"),
    ("ml",   "मिलीलीटर"),
    ("l",    "लीटर"),
    ("m²",   "वर्ग मीटर"),
    ("m2",   "वर्ग मीटर"),
    ("gb",   "गीगाबाइट"),
    ("mb",   "मेगाबाइट"),
    ("kb",   "किलोबाइट"),
    ("tb",   "टेराबाइट"),
    ("hz",   "हर्ट्ज़"),
    ("khz",  "किलोहर्ट्ज़"),
    ("mhz",  "मेगाहर्ट्ज़"),
    ("ghz",  "गीगाहर्ट्ज़"),
    ("°c",   "डिग्री सेल्सियस"),
    ("°f",   "डिग्री फ़ारेनहाइट"),
    ("%",    "प्रतिशत"),
]

# ── 通货 ─────────────────────────────────────────────────────────
_CURRENCY: list[tuple[str, str]] = [
    ("₹",  "रुपये"),
    ("rs.", "रुपये"),
    ("rs", "रुपये"),
    ("$",  "डॉलर"),
    ("€",  "यूरो"),
    ("£",  "पाउंड"),
]

# ── 缩写 ─────────────────────────────────────────────────────────
_ABBREV: list[tuple[str, str]] = [
    ("डॉ.",  "डॉक्टर"),
    ("प्रो.", "प्रोफ़ेसर"),
    ("श्री", "श्री"),   # already spoken form
    ("श्रीमती", "श्रीमती"),
    ("सं.",  "संख्या"),
    ("नं.",  "नंबर"),
    ("पृ.",  "पृष्ठ"),
    ("अ.",   "अध्याय"),
    ("इ.",   "इस्वी"),
    ("ई.",   "ईसवी"),
]

_NUM = r"(\d+(?:[.,]\d+)*)"


def _expand_number_token(raw: str) -> str:
    """'1,23,456' or '1.5' → Hindi words."""
    raw = raw.strip(".,")
    if "." in raw and "," not in raw:
        parts = raw.split(".", 1)
        return (_hi_int(int(parts[0])) + " दशमलव " +
                " ".join(_HI_ONES[int(d)] or "शून्य" for d in parts[1] if d.isdigit()))
    # remove Indian grouping commas
    raw_clean = raw.replace(",", "")
    if raw_clean.isdigit():
        return _hi_int(int(raw_clean))
    return raw


def normalize_hindi(text: str) -> str:
    """Main normalizer entry point for the itts25ft text frontend."""
    text = unicodedata.normalize("NFC", text)

    # 天城文数字 → 阿拉伯数字
    text = text.translate(_DEVA_DIGIT)

    # danda → period
    text = text.replace("।", ".").replace("॥", ".")

    # 控制字符
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # 缩写
    for abbr, full in _ABBREV:
        text = re.sub(r"(?<!\w)" + re.escape(abbr), full, text)

    # 通货（前缀型：₹500 / $25）
    for sym, word in _CURRENCY:
        text = re.sub(
            re.escape(sym) + r"\s*" + _NUM,
            lambda m, w=word: _expand_number_token(m.group(1)) + " " + w,
            text, flags=re.IGNORECASE,
        )
    # 通货（后缀型：500 रुपये 不常见，但处理 "500 rs"）
    for sym, word in _CURRENCY:
        text = re.sub(
            _NUM + r"\s*" + re.escape(sym) + r"(?!\w)",
            lambda m, w=word: _expand_number_token(m.group(1)) + " " + w,
            text, flags=re.IGNORECASE,
        )

    # 百分号：50% / %50
    text = re.sub(r"(\d[\d,.]*)\s*%",
                  lambda m: _expand_number_token(m.group(1)) + " प्रतिशत", text)
    text = re.sub(r"%\s*(\d[\d,.]*)",
                  lambda m: _expand_number_token(m.group(1)) + " प्रतिशत", text)

    # 时间：14:30 → चौदह बजकर तीस मिनट
    def _time(m: re.Match) -> str:
        h, mi = int(m.group(1)), int(m.group(2))
        base = _hi_int(h) + " बजकर"
        return base + " " + _hi_int(mi) + " मिनट" if mi else _hi_int(h) + " बजे"
    text = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", _time, text)

    # 单位（最长优先）
    for unit, word in sorted(_UNITS, key=lambda kv: -len(kv[0])):
        text = re.sub(
            r"(\d)\s*" + re.escape(unit) + r"(?!\w)",
            lambda m, w=word: m.group(1) + " " + w,
            text, flags=re.IGNORECASE,
        )

    # 序数："3." 在数字 + "." + 空格 + 词时 → तीसरा
    def _ordinal_or_keep(m: re.Match) -> str:
        following = (m.group(2) or "").strip()
        # 如果后面跟的是数字或行尾，当作普通句点保留
        if not following or following[0].isdigit():
            return m.group(0)
        return _hi_ordinal(int(m.group(1)))
    text = re.sub(r"\b(\d+)\.(?=\s+(\S+)|\s*$)", _ordinal_or_keep, text)

    # 小数（逗号分隔）：1,5 → एक दशमलव पाँच（印度不常用，但防止漏掉）
    text = re.sub(r"\b(\d+),(\d+)\b",
                  lambda m: _hi_int(int(m.group(1))) + " दशमलव " +
                            " ".join(_HI_ONES[int(d)] or "शून्य" for d in m.group(2)),
                  text)

    # 整数（含印度千分位逗号格式 1,23,456）
    text = re.sub(r"\b\d[\d,]*\d\b|\b\d\b",
                  lambda m: _expand_number_token(m.group(0)), text)

    # 多余空白
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── 供 itts25ft 直接 import ───────────────────────────────────────
if __name__ == "__main__":
    import sys
    test_cases = [
        "मेरे पास ₹1,500 हैं।",
        "तापमान 38.5°C है।",
        "वह 14:30 बजे आएगा।",
        "यह 5वाँ अध्याय है।",
        "दूरी 42km है।",
        "डॉ. शर्मा आए।",
        "०१२३४५ नंबर",
        "50% छूट",
    ]
    for t in test_cases:
        print(f"  IN : {t}")
        print(f"  OUT: {normalize_hindi(t)}\n")
