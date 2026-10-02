import pytest
import torch

from authorship.config import Config
from authorship.vocabulary import Vocabulary


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def vocab(config):
    return Vocabulary(config)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------
class TestTokenize:
    def test_simple_sentence(self, vocab):
        tokens = vocab.tokenize("Привіт світ")

        assert len(tokens) == 2
        assert all(t == t.lower() for t in tokens)

    def test_apostrophe_word(self, vocab):
        tokens = vocab.tokenize("м'ясо")

        assert len(tokens) == 1

    def test_hyphen_word(self, vocab):
        tokens = vocab.tokenize("будь-ласка")

        assert len(tokens) == 1

    def test_punctuation_separate(self, vocab):
        tokens = vocab.tokenize("Привіт!")

        assert "!" in tokens

    def test_empty_string(self, vocab):
        assert vocab.tokenize("") == []

    def test_case_insensitivity(self, vocab):
        upper = vocab.tokenize("ПРИВІТ")
        lower = vocab.tokenize("привіт")

        assert upper == lower


# ---------------------------------------------------------------------------
# _simple_stem
# ---------------------------------------------------------------------------
class TestSimpleStem:
    def test_strips_known_ending(self, vocab):
        result = vocab._simple_stem("зеленого")

        assert result == "зелен"

    def test_short_word_unchanged(self, vocab):
        assert vocab._simple_stem("до") == "до"

    def test_preserves_root_min_two(self, vocab):
        """Stemming must not reduce the root below 2 characters."""
        result = vocab._simple_stem("яку")
        # "яку" has length 3, ending "у" would leave "як" (2 chars) — allowed
        assert len(result) >= 2


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
class TestBuild:
    def test_vocab_size_grows(self, config):
        vocab = Vocabulary(config)
        texts = [
            "слово один слово два",
            "слово три слово чотири",
        ]
        vocab.build(texts)

        assert vocab.vocab_size > 2

    def test_special_tokens(self, config):
        vocab = Vocabulary(config)
        vocab.build(["тест тест"])

        assert vocab.word2idx["<PAD>"] == 0
        assert vocab.word2idx["<UNK>"] == 1

    def test_below_min_freq_excluded(self, config):
        vocab = Vocabulary(config)
        # "рідкісне" appears only once — below min_freq=2
        texts = ["рідкісне часте часте"]
        vocab.build(texts)

        stemmed_rare = vocab._simple_stem("рідкісне")
        assert stemmed_rare not in vocab.word2idx

    def test_at_min_freq_included(self, config):
        vocab = Vocabulary(config)
        # "часте" appears twice — meets min_freq=2
        texts = ["часте часте"]
        vocab.build(texts)

        stemmed = vocab._simple_stem("часте")
        assert stemmed in vocab.word2idx


# ---------------------------------------------------------------------------
# text_to_indices
# ---------------------------------------------------------------------------
class TestTextToIndices:
    def test_returns_tensor_of_max_seq_len(self, config):
        vocab = Vocabulary(config)
        vocab.build(["привіт світ привіт світ"])
        result = vocab.text_to_indices("привіт світ")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (config.data.max_seq_len,)

    def test_unknown_token_maps_to_unk(self, config):
        vocab = Vocabulary(config)
        vocab.build(["привіт привіт"])
        result = vocab.text_to_indices("невідоме")

        # First token should be UNK (1)
        assert result[0].item() == 1

    def test_short_text_zero_padded(self, config):
        config.data.max_seq_len = 10
        vocab = Vocabulary(config)
        vocab.build(["слово слово"])
        result = vocab.text_to_indices("слово")

        # Trailing positions should be 0 (PAD)
        assert result[-1].item() == 0

    def test_long_text_truncated(self):
        config = Config()
        config.data.max_seq_len = 3
        vocab = Vocabulary(config)
        vocab.build(["а б в г д а б в г д"])
        result = vocab.text_to_indices("а б в г д е ж з")

        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# save / load / is_cache_valid
# ---------------------------------------------------------------------------
class TestSaveLoad:
    def test_roundtrip(self, config, tmp_path):
        vocab = Vocabulary(config)
        vocab.build(["привіт світ привіт світ"])
        path = tmp_path / "vocab.json"
        vocab.save(path, fingerprint="abc123")

        loaded = Vocabulary.load(path, config)

        assert loaded.word2idx == vocab.word2idx
        assert loaded.vocab_size == vocab.vocab_size

    def test_cache_valid_matching_fingerprint(
        self, config, tmp_path
    ):
        vocab = Vocabulary(config)
        vocab.build(["тест тест"])
        path = tmp_path / "vocab.json"
        vocab.save(path, fingerprint="fp1")

        assert (
            Vocabulary.is_cache_valid(path, "fp1") is True
        )

    def test_cache_invalid_different_fingerprint(
        self, config, tmp_path
    ):
        vocab = Vocabulary(config)
        vocab.build(["тест тест"])
        path = tmp_path / "vocab.json"
        vocab.save(path, fingerprint="fp1")

        assert (
            Vocabulary.is_cache_valid(path, "fp_different")
            is False
        )
