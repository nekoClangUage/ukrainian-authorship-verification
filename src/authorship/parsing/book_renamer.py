"""LLM-based renaming of downloaded books.

ukrlib serves files named after the full "author-title-tid" slug. The
training pipeline (``Divider``) expects ``surname-name-N.txt`` so that
books of one author group together. This module asks a local Ollama
model to normalize filenames to that format. Renaming is cached in
``constants.RENAME_CACHE`` so repeated downloads do not re-query the
model for files that were already processed.
"""

import json
import logging
from pathlib import Path

import ollama

import authorship.parsing.constants as constants

logger = logging.getLogger(__name__)


class UkrainianBookRenamer:
    def __init__(
        self,
        model_name: str = "nemotron-3-ultra:cloud",
        timeout=None,
    ):
        self.model_name = model_name
        self._client = ollama.Client(timeout=timeout)

    def __load_cache(self, cache_path: Path) -> set[str]:
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as f:
                    return set(json.load(f).keys())
            except Exception as e:
                logger.warning(
                    "Failed to load rename cache: %s", e
                )
        return set()

    def __save_cache(
        self, cache: set[str], cache_path: Path
    ) -> None:
        try:
            with cache_path.open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {name: True for name in sorted(cache)},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.warning(
                "Failed to save rename cache: %s", e
            )

    def rename_books(
        self,
        directory: Path | str,
        extension: str = ".txt",
        force: bool = False,
        rename_cache_path: Path = constants.RENAME_CACHE,
    ) -> None:
        """Rename every ``*{extension}`` file in *directory* to the
        ``surname-name-N.txt`` format using the configured model.

        Files already present in the rename cache are skipped unless
        *force* is set. On model or transport failure the error is
        logged and processing stops; already renamed files stay as
        they are.
        """
        directory = Path(directory)
        cache = self.__load_cache(rename_cache_path)

        all_files = [
            f.name for f in directory.glob(f"*{extension}")
        ]
        new_files = (
            all_files
            if force
            else [f for f in all_files if f not in cache]
        )

        if not new_files:
            logger.info(
                "No new files to rename in %s", directory
            )
            return

        logger.info(
            "Renaming %d new file(s) via %s...",
            len(new_files),
            self.model_name,
        )

        file_list = json.dumps(
            new_files, ensure_ascii=False
        )

        prompt = f"""
        You are a deterministic filename normalizer for a corpus of Ukrainian literature.

        Input: a JSON array of raw filenames from ukrlib.com.ua. Each raw filename has the form
        'surname-name[-patronymic]-title<tid>.txt', where <tid> is a numeric book id glued to the
        end of the title, e.g. 'karpenko-karyy-ivan-suieta2345.txt'.

        Convert EVERY filename to 'author-slug-N.txt' following these rules exactly:
        1. author-slug = the author's transliterated name taken from the START of the filename:
           surname, given name, and patronymic if the input contains one — lowercase,
           hyphen-separated. Keep the patronymic whenever it is present; never invent one.
           Strip the title and the numeric tid from the output.
        2. A double surname (e.g. 'karpenko-karyy', 'nechuy-levytskyy') is one unit: keep all of
           its parts.
        3. N is a 1-based sequence number per author, assigned in the order the files appear in
           the input array: the first book of each author gets 1, that author's next book gets 2,
           and so on. N is NOT the tid and NOT the index in the input list.
        4. If a filename already matches 'author-slug-N.txt', return it unchanged.
        5. Output nothing but a valid JSON object: every input filename appears exactly once as a
           key, mapped to its new filename. No markdown, no code fences, no commentary.

        Example input:
        ["karpenko-karyy-ivan-suieta2345.txt", "karpenko-karyy-ivan-borulia1009.txt", "fedoriv-roman-u-pushchakh-karpatskykh4758.txt"]
        Example output:
        {{"karpenko-karyy-ivan-suieta2345.txt": "karpenko-karyy-ivan-1.txt", "karpenko-karyy-ivan-borulia1009.txt": "karpenko-karyy-ivan-2.txt", "fedoriv-roman-u-pushchakh-karpatskykh4758.txt": "fedoriv-roman-1.txt"}}

        Filenames to process: {file_list}
        """

        try:
            response = self._client.generate(
                model=self.model_name,
                prompt=prompt,
                format="json",
                options={
                    "temperature": 0,
                    "top_p": 0.1,
                },
            )

            new_names_map = json.loads(response["response"])

            for old_name, new_name in new_names_map.items():
                old_path = directory / old_name
                new_path = directory / new_name

                if old_path.exists():
                    if not new_path.exists():
                        old_path.rename(new_path)
                        logger.info(
                            "Renamed: %s -> %s",
                            old_name,
                            new_name,
                        )
                    else:
                        logger.info(
                            "Skip (already exists): %s",
                            new_name,
                        )
                    cache.add(new_name)
                else:
                    logger.warning(
                        "Not found: %s", old_name
                    )

            self.__save_cache(cache, rename_cache_path)

        except Exception as e:
            logger.error("Renaming failed: %s", e)
