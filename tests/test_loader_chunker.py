import pytest

from src.chunker import chunk_document, split_text
from src.loader import DocumentError, load_document


def test_loads_a_text_file(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world", encoding="utf-8")
    doc = load_document(f)
    assert doc == {"source": "a.txt", "text": "hello world"}


def test_unsupported_type_is_rejected(tmp_path):
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    with pytest.raises(DocumentError):
        load_document(f)


def test_empty_file_is_rejected(tmp_path):
    f = tmp_path / "a.md"
    f.write_text("   \n", encoding="utf-8")
    with pytest.raises(DocumentError):
        load_document(f)


def test_split_keeps_every_word_with_overlap():
    text = " ".join(str(i) for i in range(10))
    chunks = split_text(text, size=4, overlap=2)
    assert chunks[0] == "0 1 2 3"
    assert chunks[1] == "2 3 4 5"
    assert chunks[-1].endswith("9")


def test_short_text_gives_one_chunk():
    assert split_text("a b c", size=10, overlap=2) == ["a b c"]


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        split_text("a b c", size=3, overlap=3)


def test_chunks_keep_source_and_position():
    doc = {"source": "a.txt", "text": " ".join(["w"] * 30)}
    chunks = chunk_document(doc, size=10, overlap=0)
    assert [c["position"] for c in chunks] == [0, 1, 2]
    assert all(c["source"] == "a.txt" for c in chunks)
