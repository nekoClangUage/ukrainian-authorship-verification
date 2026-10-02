import logging
import re
import urllib.parse
from pathlib import Path

import requests

import authorship.parsing.constants as constants

logger = logging.getLogger(__name__)


class Downloader:
    def __init__(self) -> None:
        pass

    def __get_filename_from_cd(self, response):
        cd = response.headers.get("content-disposition")

        if not cd:
            return None
        fname = re.findall("filename=?([^;]+)", cd)

        if fname:
            # Strip quotes and whitespace
            return fname[0].strip("\"' ")

        return None

    def extract_theme_domain_from_url(self, url):
        """Extract domain of theme of literature from URL"""
        parsed = urllib.parse.urlparse(url)
        theme_domain: str = parsed.path.strip("/")

        if theme_domain:
            catalog = theme_domain.split("/", 1)
            return catalog[0]
        return None

    def extract_tid_from_url(self, url):
        """Extracts tid from URLs of the form ...?tid=1212&..."""
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        return query.get("tid", [None])[0]

    def parseURLUkrLib(
        self,
        URL: str,
        folder=constants.SAVE_FOLDER_UKRLIB,
        force: bool = False,
    ):
        logger.info("Downloading: %s", URL)

        HEADERS = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,*/*",
        }

        with requests.get(
            URL, stream=True, headers=HEADERS
        ) as r:
            r.raise_for_status()

            filename = self.__get_filename_from_cd(r)

            if (
                not filename
                or not filename.lower().endswith(".html")
            ):
                tid = (
                    self.extract_tid_from_url(URL)
                    or "unknown"
                )
                filename = f"ukrlib_book_{tid}.html"

            filepath = folder / filename

            if filepath.exists():
                if force:
                    filepath.unlink()
                else:
                    raise FileExistsError(
                        "Path is exist (can't save the book; book already exists)"
                    )

            logger.info("Save as: %s", filepath.name)

            with filepath.open("wb") as f:
                for chunk in r.iter_content(
                    1024 * 1024 * 10
                ):  # 10 MB chunks
                    f.write(chunk)

            logger.info("Data is ready!")
            return str(filepath)

    def parseURLGutenberg(
        self,
        URL: str,
        folder=constants.SAVE_FOLDER_GUTENBERG,
        force: bool = False,
    ):
        raw_book = requests.get(URL).text
        book_start_reg_exp: str = "\\*\\*\\* START OF THE PROJECT GUTENBERG EBOOK ([^*]+) \\*\\*\\*"
        first_match = re.search(
            book_start_reg_exp, raw_book
        )

        if first_match is None:
            raise RuntimeError(
                "Error: not found start of the book _parseURLGutenberg"
            )

        btitle = first_match.group(1)
        btitle = btitle.replace(" ", "_")

        destination_writing_path = Path(
            folder / (btitle + ".txt")
        )

        if destination_writing_path.exists() and not force:
            raise FileExistsError(
                f"Path is exist (can't save the book; book already exists): {destination_writing_path.name}"
            )

        with destination_writing_path.open(
            mode="w"
        ) as raw_file:
            raw_file.write(raw_book)

        return str(destination_writing_path)
