from app.services.embeddings import split_text


def test_split_text_stays_below_conservative_limit() -> None:
    chunks = split_text("胸部影像说明。" * 500, max_chars=280, overlap=40)
    assert chunks
    assert all(0 < len(item.content) <= 280 for item in chunks)


def test_split_text_preserves_short_content() -> None:
    chunks = split_text("短文本")
    assert [item.content for item in chunks] == ["短文本"]

