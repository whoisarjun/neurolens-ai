"""Language normalization helpers used across multilingual pipeline stages."""

from __future__ import annotations

import re

_EN_ALIASES = {
    'en',
    'eng',
    'english',
    'en-us',
    'en_us',
    'en-gb',
    'en_gb',
}

_ZH_ALIASES = {
    'zh',
    'zho',
    'chi',
    'cmn',
    'mandarin',
    'chinese',
    'zh-cn',
    'zh_cn',
    'zh-hans',
    'zh_hans',
}


def normalize_language(language: str | None, default: str = 'en') -> str:
    if language is None:
        return default

    raw = str(language).strip().lower()
    raw = re.sub(r'\s+', '', raw)
    raw = raw.replace('_', '-')

    if raw in _EN_ALIASES or raw.startswith('en-'):
        return 'en'
    if raw in _ZH_ALIASES or raw.startswith('zh-'):
        return 'zh'
    return default
