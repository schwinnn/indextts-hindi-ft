"""
天城文字符到拉丁音标的映射，用于替换 itts25ft/modeling.py 里的
_SEED_TRANSLIT，使 init_text_rows_from_subwords() 能为天城文 token
找到已训练的子词片段做初始化。

在 modeling.py 里的用法（patches/modeling_seed_translit.patch 会自动替换）：
    from hindi.devanagari_translit import DEVANAGARI_SEED_TRANSLIT
    _SEED_TRANSLIT = str.maketrans(DEVANAGARI_SEED_TRANSLIT)
"""

# 独立辅音 → 近似拉丁拼写
# 目标：让 tiktoken 能把拼写切成已训练的英文/拉丁子词片段
_CONSONANTS = {
    "क": "k",  "ख": "kh", "ग": "g",  "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh","ज": "j",  "झ": "jh", "ञ": "ny",
    "ट": "t",  "ठ": "th", "ड": "d",  "ढ": "dh", "ण": "n",
    "त": "t",  "थ": "th", "द": "d",  "ध": "dh", "न": "n",
    "प": "p",  "फ": "ph", "ब": "b",  "भ": "bh", "म": "m",
    "य": "y",  "र": "r",  "ल": "l",  "व": "v",
    "श": "sh", "ष": "sh", "स": "s",  "ह": "h",
    # nukta variants
    "क़": "q",  "ख़": "kh", "ग़": "g",  "ज़": "z",
    "ड़": "r",  "ढ़": "rh", "फ़": "f",  "य़": "y",
}

# स्वर (स्वतंत्र vowels)
_VOWELS = {
    "अ": "a",  "आ": "aa", "इ": "i",  "ई": "ee",
    "उ": "u",  "ऊ": "oo", "ए": "e",  "ऐ": "ai",
    "ओ": "o",  "औ": "au", "ऋ": "ri", "अं": "an",
    "अः": "ah",
}

# मात्राएँ (dependent vowel signs) → same as independent vowels
_MATRAS = {
    "ा": "aa", "ि": "i",  "ी": "ee", "ु": "u",
    "ू": "oo", "े": "e",  "ै": "ai", "ो": "o",
    "ौ": "au", "ृ": "ri", "ं": "n",  "ः": "h",
    "ँ": "n",
}

# विराम (halant) — suppresses inherent vowel; map to empty so the
# consonant cluster reads as a single consonant block
_MISC = {
    "्": "",   # halant / virama
    "ऽ": "",   # avagraha
    "।": ".",
    "॥": ".",
    "‌": "",  # ZWNJ
    "‍": "",  # ZWJ
}

# Merge all mappings
DEVANAGARI_SEED_TRANSLIT: dict[str, str] = {
    **_CONSONANTS,
    **_VOWELS,
    **_MATRAS,
    **_MISC,
}
