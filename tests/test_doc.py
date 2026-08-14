"""Tests for the doc tool (deterministic long-document store)."""

from tools import doc


def test_docs_are_deterministic_and_long():
    assert sorted(doc.list_docs()) == ["deltas", "haven", "meridian", "nexus"]
    for doc_id in doc.list_docs():
        # long enough to force paging at the read cap
        assert len(doc.DOCS[doc_id]) >= 8000, doc_id
        # every fact keyword appears in its own document (case-insensitive,
        # matching the eval keyword_correct convention)
        for keyword, _ in doc._DOC_FACTS[doc_id]["facts"]:
            assert keyword.lower() in doc.DOCS[doc_id].lower()


def test_read_pages_cover_whole_doc():
    doc_id = "nexus"
    full = doc.DOCS[doc_id]
    start = 0
    chunks = []
    while True:
        page = doc.read_doc(doc_id, start, 2000)
        if not page:
            break
        chunks.append(page)
        start += len(page)
    assert "".join(chunks) == full


def test_read_caps_length():
    page = doc.read_doc("deltas", 0, 10**6)
    assert len(page) == 2500  # _PAGE_CAP


def test_tool_unknown_doc():
    tool = doc.DocTool()
    out = tool.run({"action": "read", "doc": "nope", "start": 0, "length": 100})
    assert out.startswith("Error: unknown doc")


def test_tool_list():
    tool = doc.DocTool()
    out = tool.run({"action": "list"})
    assert "deltas" in out and "nexus" in out


def test_tool_end_of_doc():
    tool = doc.DocTool()
    out = tool.run({"action": "read", "doc": "deltas", "start": 10**6, "length": 100})
    assert out == "(end of deltas)"


def test_tool_bad_args():
    tool = doc.DocTool()
    assert tool.run({"action": "read", "doc": "deltas", "start": "x", "length": 10}).startswith("Error: start")
    assert tool.run({"action": "read", "doc": "deltas", "start": 0, "length": "x"}).startswith("Error: length")
    assert tool.run({"action": "bogus"}).startswith("Error: action")
