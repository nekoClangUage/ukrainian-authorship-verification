import pytest

from authorship.parsing.cleaner import Cleaner

# HTML with <p> tags in div.prose (normal UkrLib format)
GOOD_FORMAT_HTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html>
<head><title>Мартин Боруля</title></head>
<body>
<div class="wrap">
 <div class="page-title">
  <h1>Мартин Боруля</h1>
  <h2>Іван Карпенко-Карий</h2>
 </div>
 <div class="prose" id="content">
  <p>Комедія в 5 діях</p>
  <p>Перший абзац з діалогом.</p>
  <p>Другий абзац тексту.</p>
  <p>Третій абзац тексту.</p>
 </div>
</div>
</body>
</html>"""

# HTML with <br> tags in div.prose instead of <p> tags (bad format)
BAD_FORMAT_HTML = """\
<?xml version="1.0" encoding="utf-8"?>
<html>
<head><title>Невольник</title></head>
<body>
<div class="wrap">
 <div class="page-title">
  <h1>Невольник</h1>
  <h2>Іван Карпенко-Карий</h2>
 </div>
 <div class="prose" id="content">
<blockquote>Драма в 5-ох діях зі співами і танцями<br />
усценізував<br />
КАРПЕНКО-КАРИЙ</blockquote><br />
<br />
ДІЄВІ ОСОБИ:<br />
<br />
Коваль Василь, старий запорожець<br />
Ярина, його дочка<br />
Стефан, його годованець<br />
<br />
ДІЯ ПЕРША<br />
<br />
Недобитий. (Грає на бандурі і співає думу):<br />
<br />
Коваль. Еге... То мабуть як не думай, як не марикуй, а такой треба буде...<br />
 </div>
</div>
</body>
</html>"""

# HTML with no div.prose and no <p> tags — fallback to body with <br>
NO_PROSE_DIV_HTML = """\
<html>
<head><title>Проста книга</title></head>
<body>
<h1>Проста книга</h1>
<h2>Автор Тестовий</h2>
Рядок перший.<br />
Рядок другий.<br />
Рядок третій.<br />
</body>
</html>"""


@pytest.fixture
def cleaner():
    return Cleaner()


class TestGoodFormat:
    def test_returns_true_on_success(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "good.html"
        html_file.write_text(
            GOOD_FORMAT_HTML, encoding="utf-8"
        )

        result = cleaner.cleanUkrLibFile(str(html_file))

        assert result is True

    def test_creates_txt_file(self, cleaner, tmp_path):
        html_file = tmp_path / "good.html"
        html_file.write_text(
            GOOD_FORMAT_HTML, encoding="utf-8"
        )

        cleaner.cleanUkrLibFile(str(html_file))

        assert (tmp_path / "good.txt").exists()

    def test_includes_title_and_author(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "good.html"
        html_file.write_text(
            GOOD_FORMAT_HTML, encoding="utf-8"
        )
        cleaner.cleanUkrLibFile(str(html_file))

        content = (tmp_path / "good.txt").read_text(
            encoding="utf-8"
        )

        assert "Мартин Боруля" in content
        assert "Іван Карпенко-Карий" in content

    def test_includes_body_paragraphs(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "good.html"
        html_file.write_text(
            GOOD_FORMAT_HTML, encoding="utf-8"
        )
        cleaner.cleanUkrLibFile(str(html_file))

        content = (tmp_path / "good.txt").read_text(
            encoding="utf-8"
        )

        assert "Перший абзац з діалогом" in content
        assert "Другий абзац тексту" in content
        assert "Третій абзац тексту" in content


class TestBadFormat:
    def test_returns_true_on_success(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "bad.html"
        html_file.write_text(
            BAD_FORMAT_HTML, encoding="utf-8"
        )

        result = cleaner.cleanUkrLibFile(str(html_file))

        assert result is True

    def test_creates_txt_file(self, cleaner, tmp_path):
        html_file = tmp_path / "bad.html"
        html_file.write_text(
            BAD_FORMAT_HTML, encoding="utf-8"
        )

        cleaner.cleanUkrLibFile(str(html_file))

        assert (tmp_path / "bad.txt").exists()

    def test_includes_title_and_author(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "bad.html"
        html_file.write_text(
            BAD_FORMAT_HTML, encoding="utf-8"
        )
        cleaner.cleanUkrLibFile(str(html_file))

        content = (tmp_path / "bad.txt").read_text(
            encoding="utf-8"
        )

        assert "Невольник" in content
        assert "Іван Карпенко-Карий" in content

    def test_includes_br_separated_body_text(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "bad.html"
        html_file.write_text(
            BAD_FORMAT_HTML, encoding="utf-8"
        )
        cleaner.cleanUkrLibFile(str(html_file))

        content = (tmp_path / "bad.txt").read_text(
            encoding="utf-8"
        )

        assert "ДІЄВІ ОСОБИ" in content
        assert "Коваль Василь" in content
        assert "ДІЯ ПЕРША" in content

    def test_body_is_more_than_just_title_and_author(
        self, cleaner, tmp_path
    ):
        """Bad format files must extract body text, not only title/author."""
        html_file = tmp_path / "bad.html"
        html_file.write_text(
            BAD_FORMAT_HTML, encoding="utf-8"
        )
        cleaner.cleanUkrLibFile(str(html_file))

        content = (tmp_path / "bad.txt").read_text(
            encoding="utf-8"
        )
        non_empty_lines = [
            ln for ln in content.split("\n") if ln.strip()
        ]

        assert len(non_empty_lines) > 2, (
            f"Expected more than title+author, got {len(non_empty_lines)} lines"
        )


class TestFallbackToBody:
    def test_no_prose_div_still_extracts_text(
        self, cleaner, tmp_path
    ):
        html_file = tmp_path / "no_prose.html"
        html_file.write_text(
            NO_PROSE_DIV_HTML, encoding="utf-8"
        )
        cleaner.cleanUkrLibFile(str(html_file))

        content = (tmp_path / "no_prose.txt").read_text(
            encoding="utf-8"
        )

        assert "Рядок перший" in content
        assert "Рядок другий" in content
        assert "Рядок третій" in content


class TestErrorHandling:
    def test_returns_false_on_missing_file(
        self, cleaner, tmp_path
    ):
        result = cleaner.cleanUkrLibFile(
            str(tmp_path / "nonexistent.html")
        )

        assert result is False
