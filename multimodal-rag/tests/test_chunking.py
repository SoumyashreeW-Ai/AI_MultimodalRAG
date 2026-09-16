from app.ingestion.chunking import chunk_document_pages, chunk_text


def test_short_document_chunking():
    chunks = chunk_text(
        document_id="doc-short",
        filename="short.txt",
        source_path="/tmp/short.txt",
        page=1,
        text="Hello world",
        chunk_size=50,
        chunk_overlap=10,
    )

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-short"
    assert chunks[0].filename == "short.txt"
    assert chunks[0].page == 1
    assert chunks[0].chunk_id == "doc-short:page-1:chunk-1"
    assert chunks[0].content == "Hello world"


def test_long_document_chunking():
    words = [f"word-{i}" for i in range(1, 101)]
    text = " ".join(words)

    chunks = chunk_text(
        document_id="doc-long",
        filename="long.txt",
        source_path="/tmp/long.txt",
        page=1,
        text=text,
        chunk_size=20,
        chunk_overlap=5,
    )

    assert len(chunks) > 1
    assert all(chunk.document_id == "doc-long" for chunk in chunks)
    assert all(chunk.page == 1 for chunk in chunks)
    assert all(chunk.content.strip() for chunk in chunks)


def test_overlap_is_preserved():
    text = " ".join([f"token-{i}" for i in range(1, 31)])

    chunks = chunk_text(
        document_id="doc-overlap",
        filename="overlap.txt",
        source_path="/tmp/overlap.txt",
        page=1,
        text=text,
        chunk_size=12,
        chunk_overlap=4,
    )

    assert len(chunks) > 1
    first_words = chunks[0].content.split()
    second_words = chunks[1].content.split()
    assert first_words[-4:] == second_words[:4]


def test_empty_content_returns_no_chunks():
    chunks = chunk_text(
        document_id="doc-empty",
        filename="empty.txt",
        source_path="/tmp/empty.txt",
        page=1,
        text="   \n\t  ",
        chunk_size=25,
        chunk_overlap=5,
    )

    assert chunks == []


def test_multiple_pages_are_chunked_independently():
    pages = [
        (1, " ".join([f"page1-{i}" for i in range(1, 21)])),
        (2, " ".join([f"page2-{i}" for i in range(1, 21)])),
    ]

    chunks = chunk_document_pages(
        document_id="doc-pages",
        filename="pages.txt",
        source_path="/tmp/pages.txt",
        pages=pages,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert len(chunks) > 2
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert all(chunk.document_id == "doc-pages" for chunk in chunks)
    assert any(chunk.chunk_id.startswith("doc-pages:page-1:") for chunk in chunks)
    assert any(chunk.chunk_id.startswith("doc-pages:page-2:") for chunk in chunks)
