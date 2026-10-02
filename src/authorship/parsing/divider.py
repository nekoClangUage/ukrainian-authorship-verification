import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Divider:
    def __init__(
        self, target_count: int = 5000, min_length: int = 50
    ) -> None:
        """
        Initialization of the Divider class.

        Args:
            target_count (int): The number of texts into which the author's TOTAL volume of texts should be divided.
            min_length (int): The minimum length of each piece of text (in characters).
        """
        self.target_count = target_count
        self.min_length = min_length

    def parse_directory(
        self, directory_path: str | Path
    ) -> dict[str, list[str]]:
        """
        Parses .txt files, groups them by author, and splits them into parts.

        Args:
            directory_path: Path to the folder with text files.
            The file format must be: surname-name-index.txt
            (for example: london-jack-1.txt, karpenko-karyy-ivan-2.txt)

        Returns:
            A dictionary where the key is the author's name (str) and the value is a list of texts (list[str]).
        """
        directory = Path(directory_path)

        if not directory.exists():
            raise FileNotFoundError(
                f"Error: Directory not found {directory}"
            )

        file_paths = sorted(directory.glob("*.txt"))

        # Structure: { "london-jack": ["text_book_1", "text_book_2"], ... }
        authors_raw_content: dict[str, list[str]] = {}

        logger.info(
            "Start of reading and grouping of files"
        )
        for file_path in file_paths:
            author_name = self._get_author_from_filename(
                file_path.name
            )
            try:
                content = file_path.read_text(
                    encoding="utf-8", errors="ignore"
                )
                if author_name not in authors_raw_content:
                    authors_raw_content[author_name] = []
                authors_raw_content[author_name].append(
                    content
                )

            except OSError as e:
                logger.error(
                    "Cannot read file %s: %s", file_path, e
                )

        logger.info(
            "Handling %d authors...",
            len(authors_raw_content),
        )

        result: dict[str, list[str]] = {}

        # We sort authors alphabetically so that the order in the result is always the same.
        sorted_authors = sorted(authors_raw_content.keys())

        for author in sorted_authors:
            # Combine all books by the author into one long string separated by spaces
            full_text = " ".join(
                authors_raw_content[author]
            )

            min_required = (
                self.target_count * self.min_length
            )
            if len(full_text) < min_required:
                logger.warning(
                    "Author '%s' has too little text (%d symbols, min %d). Skipped.",
                    author,
                    len(full_text),
                    min_required,
                )
                continue

            chunks = self._split_text(full_text)

            if len(chunks) == self.target_count:
                result[author] = chunks
            else:
                logger.error(
                    "_split_text returned wrong chunk count for author '%s'.",
                    author,
                )

        return result

    def _get_author_from_filename(
        self, filename: str
    ) -> str:
        """
        Extracts the author's name from the file name in the format 'surname-name-index.txt'.

        Example:
            'london-jack-1.txt' -> 'london-jack'

            'karpenko-karyy-ivan-2.txt' -> 'karpenko-karyy-ivan'
        """
        stem = Path(filename).stem
        parts = stem.rsplit("-", 1)

        if len(parts) == 2 and parts[1].isdigit():
            # If the format is correct (there is a dash and a number at the end), we take the first part.
            return parts[0]

        # If there are no numbers at the end (e.g., 'london-jack.txt'), we consider the entire name to be the author.
        return stem

    def _split_text(self, text: str) -> list[str]:
        """
        An internal method for splitting a string into an even number of parts.
        """
        text_len = len(text)
        step = text_len // self.target_count

        chunks = []
        for i in range(self.target_count):
            start_index = i * step
            end_index = (
                text_len
                if i == self.target_count - 1
                else (i + 1) * step
            )
            chunks.append(text[start_index:end_index])

        return chunks
