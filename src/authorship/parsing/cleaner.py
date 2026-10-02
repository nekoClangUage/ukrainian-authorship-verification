import logging
import re
from pathlib import Path

import bs4

logger = logging.getLogger(__name__)


class Cleaner:
    def __init__(self):
        pass

    def __html_to_clean_text(  # noqa: C901
        self,
        html_content: str,
        include_title: bool = True,
        include_author: bool = True,
        paragraph_separator: str = "\n\n",
    ) -> str:
        """
        Convert HTML content (in particular the <body>) into clean paragraph text.

        Args:
        - html_content (str): Full HTML of the file or a part of it.
        - include_title (bool): Whether to prepend the book title (<h1>).
        - include_author (bool): Whether to add the author (<h2>) after the title.
        - paragraph_separator (str): Separator between paragraphs (usually "\n\n").

        Returns:
        str: Clean text with the paragraph structure preserved.
        """
        soup = bs4.BeautifulSoup(
            html_content, "html.parser"
        )

        text_parts = []

        # Extract the title and the author if requested
        if include_title:
            title_tag = soup.find("h1")
            if title_tag:
                text_parts.append(
                    title_tag.get_text(strip=True)
                )

        if include_author:
            author_tag = soup.find("h2")
            if author_tag:
                text_parts.append(
                    author_tag.get_text(strip=True)
                )

        # Main content usually lives in <div class="prose"> or simply in <body>
        # Look for div.prose first, fall back to the whole body
        content_container = soup.find("div", class_="prose")
        if not content_container:
            content_container = soup.body

        if content_container:
            direct_paragraphs = content_container.find_all(
                "p", recursive=False
            )
            if direct_paragraphs:
                # Normal format: content wrapped in <p> tags
                for p_tag in direct_paragraphs:
                    paragraph = p_tag.get_text(
                        separator="\n", strip=True
                    )
                    if paragraph:
                        text_parts.append(paragraph)
            else:
                # Bad format: content separated by <br> tags instead of <p> tags
                for br_tag in content_container.find_all(
                    "br"
                ):
                    br_tag.replace_with("\n")
                raw_text = content_container.get_text()
                lines = [
                    line.strip()
                    for line in raw_text.split("\n")
                ]
                non_empty = [line for line in lines if line]
                if non_empty:
                    text_parts.append("\n".join(non_empty))

        # Join everything together
        final_text = paragraph_separator.join(text_parts)

        # When title/author were added — separate them with an extra line break
        if (include_title or include_author) and len(
            text_parts
        ) > (1 if include_title else 0) + (
            1 if include_author else 0
        ):
            final_text = final_text.replace(
                "\n\n", "\n\n\n", 1
            )  # larger gap after the metadata

        return final_text

    def cleanUkrLibFile(self, path: str):
        try:
            path_transformed = Path(path)

            with path_transformed.open(
                encoding="utf-8"
            ) as f:
                text_processed = f.read()

            with path_transformed.with_suffix(".txt").open(
                mode="w+",
                encoding="utf-8",
            ) as f:
                f.write(
                    self.__html_to_clean_text(
                        text_processed
                    )
                )

        except Exception as e:
            logger.error(
                "cleanUkrLibFile failed for %s: %s", path, e
            )
            return False

        return True

    def cleanGutenbergFile(self, path: str):
        try:
            with Path(path).open(encoding="utf-8") as f:
                raw_book = f.read()

            book_start_reg_exp: str = "\\*\\*\\* START OF THE PROJECT GUTENBERG EBOOK ([^*]+) \\*\\*\\*"
            book_end_reg_exp: str = "\\*\\*\\* END OF THE PROJECT GUTENBERG EBOOK ([^*]+) \\*\\*\\*"
            start_match: re.Match[str] | None = re.search(
                book_start_reg_exp, raw_book
            )
            end_match: re.Match[str] | None = re.search(
                book_end_reg_exp, raw_book
            )

            if (
                start_match is not None
                and end_match is not None
            ):
                with Path(path).open(
                    mode="w", encoding="utf-8"
                ) as wf:
                    wf.write(
                        raw_book[
                            start_match.end() : end_match.start()
                        ]
                    )
            else:
                return False

        except Exception as e:
            logger.error(
                "cleanGutenbergFile failed for %s: %s",
                path,
                e,
            )
            return False

        return True
