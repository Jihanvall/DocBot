import numpy as np
import pytest

from src.embedding_cache import cache_key, cached_embed


def fake_embed(texts, task_type=None):
    return np.array([[float(len(t)), 1.0] for t in texts])


def test_second_call_does_not_call_the_embedder(tmp_path):
    texts = ["aa", "bbb"]
    first = cached_embed(fake_embed, texts, "m", "DOC", cache_dir=tmp_path)

    def boom(texts, task_type=None):
        raise AssertionError("embedder must not be called")

    second = cached_embed(boom, texts, "m", "DOC", cache_dir=tmp_path)
    assert np.array_equal(first, second)


def test_changed_chunks_are_recomputed(tmp_path):
    cached_embed(fake_embed, ["aa"], "m", "DOC", cache_dir=tmp_path)
    calls = []

    def counting(texts, task_type=None):
        calls.append(1)
        return fake_embed(texts)

    cached_embed(counting, ["aa", "new chunk"], "m", "DOC", cache_dir=tmp_path)
    assert calls == [1]


def test_key_depends_on_model_task_and_texts():
    base = cache_key(["a"], "m", "DOC")
    assert base != cache_key(["a"], "m2", "DOC")
    assert base != cache_key(["a"], "m", "QUERY")
    assert base != cache_key(["b"], "m", "DOC")
    assert base == cache_key(["a"], "m", "DOC")


def test_cache_file_is_created(tmp_path):
    cached_embed(fake_embed, ["aa"], "m", "DOC", cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_empty_text_list_is_not_confused_with_other_keys():
    with pytest.raises(Exception):
        cached_embed(lambda t, k=None: (_ for _ in ()).throw(ValueError()), [], "m", "DOC")
