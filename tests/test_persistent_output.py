from __future__ import annotations

import io

from ss_image_processor.output.persistent import PersistentStream


def get_output_after_write(chunks: list[str]) -> str:
    buf = io.StringIO()
    ps = PersistentStream(buf)
    for c in chunks:
        ps.write(c)
    ps.close()
    return buf.getvalue()


def test_cr_becomes_newlines_simple():
    out = get_output_after_write(["a\r", "b\r", "c\n"])
    assert out == "a\nb\nc\n"


def test_progress_like_updates():
    chunks = [
        "Downloading 1%\r",
        "Downloading 2%\r",
        "Done\n",
    ]
    out = get_output_after_write(chunks)
    assert out == "Downloading 1%\nDownloading 2%\nDone\n"


def test_crlf_normalized_to_lf():
    out = get_output_after_write(["line1\r\n", "line2\r\n"])
    assert out == "line1\nline2\n"


def test_partial_line_emitted_on_close():
    buf = io.StringIO()
    ps = PersistentStream(buf)
    ps.write("partial line without newline")
    ps.close()
    assert buf.getvalue() == "partial line without newline\n"
