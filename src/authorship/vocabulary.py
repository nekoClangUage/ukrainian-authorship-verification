import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import torch

from authorship.config import Config

logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(
        self,
        config: Config,
        word2idx: dict[str, int] | None = None,
        idx2word: dict[int, str] | None = None,
        vocab_size: int = 2,
    ) -> None:
        if word2idx is None or idx2word is None:
            self.word2idx: dict[str, int] = {
                "<PAD>": 0,
                "<UNK>": 1,
            }
            self.idx2word: dict[int, str] = {
                0: "<PAD>",
                1: "<UNK>",
            }
        elif len(word2idx) != len(idx2word):
            raise ValueError(
                "Length word2idx != idx2word; indexes and words has different length"
            )
        else:
            self.word2idx = word2idx
            self.idx2word = idx2word

        self.vocab_size = vocab_size
        self._configuration = config

        # Regular expression for tokenization:
        # 1. Words with apostrophes and hyphenated forms: \w+(?:[''ʼ-]\w+)*
        # 2. Or any punctuation marks (excluding spaces): [^\w\s]
        self.tokenizer_pattern = re.compile(
            r"(?u)\w+(?:[''ʼ-]\w+)*|[^\w\s]"
        )

        # List of common Ukrainian word endings for "easy" stemming
        # Order is important: from long to short
        self.endings = [
            # Verbs and adverbs
            "ати",
            "яти",
            "учи",
            "ючи",
            "вши",
            "ши",
            "сь",
            "ся",
            # Adjectives
            "ий",
            "ій",
            "а",
            "я",
            "е",
            "є",
            "ого",
            "ього",
            "ому",
            "ьому",
            "им",
            "ім",
            "их",
            "іх",
            "ими",
            "іми",
            # Nouns (various cases)
            "ів",
            "їв",
            "ом",
            "ем",
            "єм",
            "ові",
            "еві",
            "єві",
            "ами",
            "ями",
            "ою",
            "ею",
            "єю",
            # Trivial (simple) endings
            "у",
            "ю",
            "і",
            "ї",
            "о",
            "е",
            "є",
        ]

    def _simple_stem(self, word: str) -> str:
        """
        Heuristic stemming: removes known endings if the word is long enough.
        Example: inflected forms of one Ukrainian root collapse to a single stem.
        """
        # Do not touch very short words
        if len(word) <= 3:
            return word

        for ending in self.endings:
            if (
                word.endswith(ending)
                and len(word) - len(ending) >= 2
            ):
                return word[: -len(ending)]
                # Return immediately after the first match (the longest one)
                # so as not to cut off anything unnecessary
        return word

    def tokenize(self, text: str) -> list[str]:
        """
        Breaks text into tokens, preserving punctuation and applying stemming to words.
        """
        text = text.lower()
        matches = self.tokenizer_pattern.findall(text)

        processed_tokens = []
        for token in matches:
            # If the token consists of letters (or contains an apostrophe/hyphen inside), it is a word.
            # Simple check: are there any letters in the token?
            if re.search(r"\w", token):
                stemmed_word = self._simple_stem(token)
                processed_tokens.append(stemmed_word)
            else:
                # Punctuation
                processed_tokens.append(token)

        return processed_tokens

    def build(self, texts_list: list[str]) -> None:
        logger.info(
            "Building vocabulary (Custom Regex + Stemming)..."
        )
        counter: Counter = Counter()

        total = len(texts_list)
        for i, text in enumerate(texts_list):
            if i % 200 == 0:
                logger.debug(
                    "Processing text %d/%d...", i, total
                )
            counter.update(self.tokenize(text))
        logger.info(
            "Vocabulary processing complete: %d texts.",
            total,
        )

        for word, count in counter.items():
            if count >= self._configuration.data.min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1
        logger.info("Vocabulary size: %d", self.vocab_size)

    def save(
        self,
        path: str | Path,
        fingerprint: str | None = None,
    ) -> None:
        """Serialize vocabulary to a JSON file, optionally storing a data fingerprint."""
        data: dict = {
            "word2idx": self.word2idx,
            "vocab_size": self.vocab_size,
        }
        if fingerprint is not None:
            data["fingerprint"] = fingerprint
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        logger.info("Vocabulary saved to %s", path)

    @classmethod
    def load(
        cls, path: str | Path, config: Config
    ) -> "Vocabulary":
        """Restore vocabulary from a JSON file produced by save()."""
        with Path(path).open(encoding="utf-8") as f:
            data = json.load(f)
        word2idx: dict[str, int] = data["word2idx"]
        idx2word: dict[int, str] = {
            int(v): k for k, v in word2idx.items()
        }
        vocab_size: int = data["vocab_size"]
        logger.info(
            "Vocabulary loaded from %s (%d tokens)",
            path,
            vocab_size,
        )
        return cls(
            config,
            word2idx=word2idx,
            idx2word=idx2word,
            vocab_size=vocab_size,
        )

    @staticmethod
    def is_cache_valid(
        path: str | Path, fingerprint: str
    ) -> bool:
        """Return True if the cache file exists and was built from the same data fingerprint."""
        try:
            with Path(path).open(encoding="utf-8") as f:
                data = json.load(f)
            return data.get("fingerprint") == fingerprint
        except (OSError, json.JSONDecodeError):
            return False

    def text_to_indices(self, text: str) -> torch.Tensor:
        tokens: list[str] = self.tokenize(text)

        indices: list[int] = [
            self.word2idx.get(t, 1) for t in tokens
        ]

        # Padding / Truncating
        if (
            len(indices)
            > self._configuration.data.max_seq_len
        ):
            indices = indices[
                : self._configuration.data.max_seq_len
            ]
        else:
            indices += [0] * (
                self._configuration.data.max_seq_len
                - len(indices)
            )

        return torch.tensor(indices, dtype=torch.long)


def compute_data_fingerprint(data_dir: Path) -> str:
    """
    Hash the .txt files in data_dir (names + sizes + mtimes) to detect directory changes.
    Use this to decide whether a vocabulary cache is still valid.
    """
    entries = sorted(data_dir.glob("*.txt"))
    hasher = hashlib.md5()
    for path in entries:
        stat = path.stat()
        hasher.update(
            f"{path.name}:{stat.st_size}:{stat.st_mtime:.6f}".encode()
        )
    return hasher.hexdigest()
