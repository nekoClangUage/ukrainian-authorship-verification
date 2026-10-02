import json
import logging
from pathlib import Path

import authorship.parsing.constants as constants
from authorship.parsing.book_renamer import (
    UkrainianBookRenamer,
)
from authorship.parsing.book_sources import (
    TRAINING_URLS,
    VALIDATION_URLS,
)
from authorship.parsing.cleaner import Cleaner
from authorship.parsing.downloader import Downloader

logger = logging.getLogger(__name__)


class Parser:
    def __init__(self):
        self.__downloader = Downloader()
        self.__cleaner = Cleaner()

    def __load_cache(
        self, cache_path: Path
    ) -> dict[str, bool]:
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(
                    "Failed to load download cache: %s", e
                )
        return {}

    def __save_cache(
        self, cache: dict[str, bool], cache_path: Path
    ) -> None:
        try:
            with cache_path.open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    cache, f, ensure_ascii=False, indent=2
                )
        except Exception as e:
            logger.warning(
                "Failed to save download cache: %s", e
            )

    def __parseURLs(
        self,
        URLs: list[str],
        gutenberg_folder: Path,
        ukrlib_folder: Path,
        cache: dict[str, bool],
        force: bool = False,
    ) -> dict[str, bool]:
        parsed_URLs: dict[str, bool] = {}

        for url_string in URLs:
            if not force and cache.get(url_string):
                logger.info(
                    "Skipping (already downloaded): %s",
                    url_string,
                )
                parsed_URLs[url_string] = True
                continue

            try:
                # Gutenberg
                if url_string.startswith(
                    "https://www.gutenberg.org/"
                ):
                    self.__downloader.parseURLGutenberg(
                        url_string,
                        gutenberg_folder,
                        force=force,
                    )
                    parsed_URLs[url_string] = True
                # Ukrlib
                elif "printitzip.php" in url_string:
                    tid = self.__downloader.extract_tid_from_url(
                        url_string
                    )
                    theme_domain = self.__downloader.extract_theme_domain_from_url(
                        url_string
                    )
                    if tid and theme_domain:
                        url = f"https://www.ukrlib.com.ua/{theme_domain}/getfile.php?tid={tid}&type=6"
                        logger.info(
                            "Converted to download link: %s",
                            url,
                        )
                        self.__downloader.parseURLUkrLib(
                            url, ukrlib_folder, force=force
                        )
                        parsed_URLs[url_string] = True
                    else:
                        raise ValueError(
                            "Wrong URL format (no tid or theme_domain)"
                        )
            except Exception as e:
                logger.error(
                    "Failed to parse URL %s: %s",
                    url_string,
                    e,
                )
                parsed_URLs[url_string] = False

        return parsed_URLs

    def __cleanBooks(
        self, gutenberg_folder: Path, ukrlib_folder: Path
    ) -> dict[str, bool]:
        cleaned_books: dict[str, bool] = {}

        if gutenberg_folder and gutenberg_folder.exists():
            for txt_book in gutenberg_folder.glob("*.txt"):
                cleaned_books[txt_book.name] = (
                    self.__cleaner.cleanGutenbergFile(
                        str(txt_book)
                    )
                )
        if ukrlib_folder and ukrlib_folder.exists():
            for html_file in ukrlib_folder.glob("*.html"):
                cleaned_books[html_file.name] = (
                    self.__cleaner.cleanUkrLibFile(
                        str(html_file)
                    )
                )

        return cleaned_books

    def __clean_directory(self, path: Path) -> None:
        """Delete all files in *path* (non-recursive) before downloading."""
        if not (path.exists() and path.is_dir()):
            return
        removed = 0
        for f in path.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        logger.info(
            "Cleaned %d file(s) from %s", removed, path
        )

    def __removeHTMLFilesFromDir(self, path: Path) -> bool:
        if path.exists() and path.is_dir():
            for html_file in path.glob("*.html"):
                html_file.unlink(missing_ok=True)

            return True

        return False

    def parse_list_of_books(
        self,
        URLs: list[str],
        gutenberg_folder: Path = constants.SAVE_FOLDER_GUTENBERG,
        ukrlib_folder: Path = constants.SAVE_FOLDER_UKRLIB,
        force: bool = False,
        clean: bool = False,
        cache_path: Path = constants.DOWNLOAD_CACHE,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        constants.ensure_directories()

        if clean:
            logger.info(
                "Cleaning download directories before downloading..."
            )
            self.__clean_directory(gutenberg_folder)
            self.__clean_directory(ukrlib_folder)

        cache = self.__load_cache(cache_path)
        if clean:
            # Files are gone — drop cache entries for URLs in this run
            # so they are re-downloaded unconditionally.
            for url in URLs:
                cache.pop(url, None)
            logger.debug(
                "Cleared %d cache entries for cleaned URLs",
                len(URLs),
            )

        parsed_URLs: dict[str, bool] = self.__parseURLs(
            URLs,
            gutenberg_folder,
            ukrlib_folder,
            cache,
            force,
        )
        cleaned_books: dict[str, bool] = self.__cleanBooks(
            gutenberg_folder, ukrlib_folder
        )
        self.__removeHTMLFilesFromDir(ukrlib_folder)

        for url, success in parsed_URLs.items():
            if success:
                cache[url] = True
        self.__save_cache(cache, cache_path)

        return parsed_URLs, cleaned_books


def training_books(
    force: bool = False, clean: bool = False
):
    parser = Parser()
    parser.parse_list_of_books(
        TRAINING_URLS, force=force, clean=clean
    )

    renamer = UkrainianBookRenamer()
    renamer.rename_books(
        constants.SAVE_FOLDER_UKRLIB, force=force
    )


def validation_books(
    force: bool = False, clean: bool = False
):
    parser = Parser()
    parser.parse_list_of_books(
        VALIDATION_URLS,
        constants.SAVE_FOLDER_GUTENBERG,
        constants.SAVE_FOLDER_UKRLIB_TEST,
        force=force,
        clean=clean,
    )

    renamer = UkrainianBookRenamer()
    renamer.rename_books(
        constants.SAVE_FOLDER_UKRLIB_TEST, force=force
    )
