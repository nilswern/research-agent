from src.services.validation import normalize_url, split_report, validate_report

CLEAN = """# RAG
## Findings
It combines retrieval and generation [1]. Trade-offs are discussed in [2].
## References
[1] RAG paper - https://arxiv.org/abs/2005.11401
[2] Overview - https://en.wikipedia.org/wiki/RAG
"""

FABRICATED = """# RAG
## Findings
A benchmark shows 40% fewer hallucinations [1].
## References
[1] Invented study - https://example.com/definitely-made-up
"""

DANGLING = """# RAG
## Findings
Claim one [1]. Claim two [3].
## References
[1] RAG paper - https://arxiv.org/abs/2005.11401
"""

ALLOWED = ["https://arxiv.org/abs/2005.11401", "https://en.wikipedia.org/wiki/RAG"]


def test_clean_report_passes() -> None:
    result = validate_report(CLEAN, ALLOWED)
    assert result.is_valid
    assert result.as_issues() == []


def test_fabricated_url_is_detected() -> None:
    result = validate_report(FABRICATED, ALLOWED)
    assert result.unsupported_urls == ["https://example.com/definitely-made-up"]
    assert not result.is_valid
    assert "never returned" in result.as_issues()[0]


def test_dangling_citation_is_detected() -> None:
    result = validate_report(DANGLING, ALLOWED)
    assert result.dangling_citations == [3]
    assert not result.is_valid


def test_trailing_slash_and_case_are_ignored() -> None:
    report = "See https://ArXiv.org/abs/2005.11401/ for details."
    assert validate_report(report, ALLOWED).unsupported_urls == []


def test_url_in_trailing_punctuation_is_matched() -> None:
    report = "Details at https://arxiv.org/abs/2005.11401."
    assert validate_report(report, ALLOWED).unsupported_urls == []


def test_split_report_separates_references() -> None:
    body, references = split_report(CLEAN)
    assert "## References" not in body
    assert "arxiv.org" in references


def test_report_without_reference_section() -> None:
    result = validate_report("Claim [1] with no list.", ALLOWED)
    assert result.dangling_citations == [1]


def test_normalize_url() -> None:
    assert normalize_url("https://A.test/path/") == "https://a.test/path"
