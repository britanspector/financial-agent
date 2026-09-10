"""Chinese-aware tokenization for lexical retrieval."""

from __future__ import annotations

import logging
import re

import jieba


_TOKEN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+(?:[._-][a-z0-9]+)*", re.IGNORECASE)
jieba.setLogLevel(logging.WARNING)
_JIEBA = jieba.Tokenizer()


class ChineseTokenizer:
    """Use Jieba search-mode words and retain normalized alphanumeric terms."""

    def __init__(self) -> None:
        self._tokenizer = _JIEBA

    def tokenize(self, text: str) -> list[str]:
        normalized = text.casefold()
        tokens: list[str] = []
        for piece in self._tokenizer.cut_for_search(normalized, HMM=False):
            tokens.extend(match.group(0) for match in _TOKEN.finditer(piece))
        return tokens
