import pytest

from authorship.parsing.cleaner import Cleaner


@pytest.fixture
def cleaner():
    return Cleaner()


# ---------------------------------------------------------------------------
# HTML edge cases
# ---------------------------------------------------------------------------
class TestHtmlEdgeCases:
    def test_missing_h1(self, cleaner, tmp_path):
        html = """\
<html><body>
<div class="prose"><p>Текст без заголовка.</p></div>
</body></html>"""
        f = tmp_path / "no_h1.html"
        f.write_text(html, encoding="utf-8")

        result = cleaner.cleanUkrLibFile(str(f))

        assert result is True
        content = (tmp_path / "no_h1.txt").read_text(
            encoding="utf-8"
        )
        assert "Текст без заголовка" in content

    def test_missing_h2(self, cleaner, tmp_path):
        html = """\
<html><body>
<h1>Заголовок</h1>
<div class="prose"><p>Контент книги.</p></div>
</body></html>"""
        f = tmp_path / "no_h2.html"
        f.write_text(html, encoding="utf-8")

        result = cleaner.cleanUkrLibFile(str(f))

        assert result is True
        content = (tmp_path / "no_h2.txt").read_text(
            encoding="utf-8"
        )
        assert "Контент книги" in content

    def test_empty_prose_div(self, cleaner, tmp_path):
        html = """\
<html><body>
<h1>Заголовок</h1>
<div class="prose"></div>
</body></html>"""
        f = tmp_path / "empty_prose.html"
        f.write_text(html, encoding="utf-8")

        result = cleaner.cleanUkrLibFile(str(f))

        assert result is True

    def test_inline_tags_stripped(self, cleaner, tmp_path):
        html = """\
<html><body>
<div class="prose">
  <p>Це <b>жирний</b> і <i>курсивний</i> та <a href="#">посилання</a> текст.</p>
</div>
</body></html>"""
        f = tmp_path / "inline.html"
        f.write_text(html, encoding="utf-8")
        cleaner.cleanUkrLibFile(str(f))

        content = (tmp_path / "inline.txt").read_text(
            encoding="utf-8"
        )

        assert "жирний" in content
        assert "курсивний" in content
        assert "посилання" in content
        assert "<b>" not in content
        assert "<i>" not in content
        assert "<a" not in content

    def test_ukrainian_special_chars_preserved(
        self, cleaner, tmp_path
    ):
        html = """\
<html><body>
<div class="prose"><p>Ґрунт, єдність, інтер'єр, їжак.</p></div>
</body></html>"""
        f = tmp_path / "ukr_chars.html"
        f.write_text(html, encoding="utf-8")
        cleaner.cleanUkrLibFile(str(f))

        content = (tmp_path / "ukr_chars.txt").read_text(
            encoding="utf-8"
        )

        for char in "ґєії":
            assert char in content.lower()


# ---------------------------------------------------------------------------
# Gutenberg cleaner
# ---------------------------------------------------------------------------
GUTENBERG_TEMPLATE = """\
The Project Gutenberg eBook of Test Book

*** START OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

{body}

*** END OF THE PROJECT GUTENBERG EBOOK TEST BOOK ***

End of the Project Gutenberg EBook
"""


class TestGutenbergCleaner:
    def test_valid_format(self, cleaner, tmp_path):
        f = tmp_path / "gutenberg.txt"
        f.write_text(
            GUTENBERG_TEMPLATE.format(
                body="Основний текст книги."
            ),
            encoding="utf-8",
        )

        result = cleaner.cleanGutenbergFile(str(f))

        assert result is True
        content = f.read_text(encoding="utf-8")
        assert "Основний текст книги" in content

    def test_missing_start_marker(self, cleaner, tmp_path):
        f = tmp_path / "no_start.txt"
        f.write_text(
            "Some preamble\n*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n",
            encoding="utf-8",
        )

        assert cleaner.cleanGutenbergFile(str(f)) is False

    def test_missing_end_marker(self, cleaner, tmp_path):
        f = tmp_path / "no_end.txt"
        f.write_text(
            "*** START OF THE PROJECT GUTENBERG EBOOK TEST ***\nContent here\n",
            encoding="utf-8",
        )

        assert cleaner.cleanGutenbergFile(str(f)) is False

    def test_both_markers_missing(self, cleaner, tmp_path):
        f = tmp_path / "no_markers.txt"
        f.write_text(
            "Just some plain text without markers.",
            encoding="utf-8",
        )

        assert cleaner.cleanGutenbergFile(str(f)) is False


# ---------------------------------------------------------------------------
# Private __html_to_clean_text options (accessed via name mangling)
# ---------------------------------------------------------------------------
class TestCleanTextOptions:
    def test_include_title_false(self, cleaner):
        html = """\
<html><body>
<h1>Мій Заголовок</h1>
<div class="prose"><p>Основний текст.</p></div>
</body></html>"""

        result = cleaner._Cleaner__html_to_clean_text(
            html, include_title=False
        )

        assert "Мій Заголовок" not in result
        assert "Основний текст" in result

    def test_include_author_false(self, cleaner):
        html = """\
<html><body>
<h1>Назва</h1>
<h2>Автор Прізвищенко</h2>
<div class="prose"><p>Текст книги.</p></div>
</body></html>"""

        result = cleaner._Cleaner__html_to_clean_text(
            html, include_author=False
        )

        assert "Автор Прізвищенко" not in result
        assert "Текст книги" in result

    def test_custom_paragraph_separator(self, cleaner):
        html = """\
<html><body>
<div class="prose">
  <p>Перший абзац.</p>
  <p>Другий абзац.</p>
</div>
</body></html>"""

        result = cleaner._Cleaner__html_to_clean_text(
            html,
            include_title=False,
            include_author=False,
            paragraph_separator="|||",
        )

        assert "|||" in result
