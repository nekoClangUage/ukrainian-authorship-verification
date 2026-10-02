from pathlib import Path
from typing import Final

_PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]

# === DATA FOLDERS ===
SAVE_FOLDER_UKRLIB: Final = (
    _PROJECT_ROOT / "content" / "raw" / "ukr_data"
)

SAVE_FOLDER_UKRLIB_TEST: Final = (
    SAVE_FOLDER_UKRLIB / "test_data"
)

SAVE_FOLDER_GUTENBERG: Final = (
    _PROJECT_ROOT / "content" / "raw" / "gutenberg"
)

DOWNLOAD_CACHE: Final = (
    SAVE_FOLDER_UKRLIB.parent / "download_cache.json"
)

RENAME_CACHE: Final = (
    SAVE_FOLDER_UKRLIB.parent / "rename_cache.json"
)
# ====================


def ensure_directories() -> None:
    """Create all directories required for storing downloaded data.

    Call this function explicitly before file download or processing
    operations. Importing this module has no side effects.
    """
    SAVE_FOLDER_UKRLIB.mkdir(parents=True, exist_ok=True)
    SAVE_FOLDER_UKRLIB_TEST.mkdir(
        parents=True, exist_ok=True
    )
    SAVE_FOLDER_GUTENBERG.mkdir(parents=True, exist_ok=True)
