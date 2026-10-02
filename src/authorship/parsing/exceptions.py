"""Custom exceptions for the parsing module.

Hierarchy:
    ParsingError
    ├── BookDownloadError        (book could not be downloaded from a source)
    ├── BookFormatError          (book format is unrecognised or malformed)
    └── AuthorTextTooShortError  (author's combined text is too short to chunk)
"""


class ParsingError(Exception):
    """Base exception for data parsing pipeline errors."""


class BookDownloadError(ParsingError, RuntimeError):
    """Raised when a book cannot be downloaded from a source.

    Example:
        raise BookDownloadError(
            f"HTTP {status}: failed to download {url}"
        )
    """


class BookFormatError(ParsingError, ValueError):
    """Raised when a book's format is unrecognised or malformed.

    Example:
        raise BookFormatError(
            "Gutenberg start marker not found in file"
        )
    """


class AuthorTextTooShortError(ParsingError, ValueError):
    """Raised when an author's combined text is too short to produce enough chunks.

    Example:
        raise AuthorTextTooShortError(
            f"Author '{name}': {length} chars, minimum required: {required}"
        )
    """
