"""Custom exceptions for the authorship module.

Hierarchy:
    AuthorshipError
    ├── InsufficientAuthorsError  (fewer than 2 authors for pair generation)
    ├── VocabularyCacheError      (vocabulary cache is corrupt or unreadable)
    ├── CheckpointLoadError       (model checkpoint could not be loaded)
    └── TextTooShortError         (text is too short for processing)
"""


class AuthorshipError(Exception):
    """Base exception for authorship identification errors."""


class InsufficientAuthorsError(AuthorshipError, ValueError):
    """Raised when fewer than 2 authors are provided for pair generation.

    Example:
        raise InsufficientAuthorsError(
            "At least 2 authors required, got 1"
        )
    """


class VocabularyCacheError(AuthorshipError, OSError):
    """Raised when the vocabulary cache file is corrupt or unreadable.

    Example:
        raise VocabularyCacheError(
            f"Cache file {path} is corrupt: {reason}"
        )
    """


class CheckpointLoadError(AuthorshipError, RuntimeError):
    """Raised when a model checkpoint cannot be loaded.

    Example:
        raise CheckpointLoadError(
            f"Failed to load checkpoint: {path}"
        )
    """


class TextTooShortError(AuthorshipError, ValueError):
    """Raised when a text is too short for processing.

    Example:
        raise TextTooShortError(
            f"Text contains {n} tokens, minimum required: {min_len}"
        )
    """
