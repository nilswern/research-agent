"""HTTP fetch and text extraction for arbitrary web pages."""

from __future__ import annotations

import ipaddress
import socket
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
import trafilatura

from src.config.logging_config import get_logger
from src.models.document import ContentType, SourceDocument, SourceTool

logger = get_logger(__name__)


MAX_REDIRECTS = 5


class ScrapeError(RuntimeError):
    """The page could not be fetched or yielded no usable text."""


def ensure_public_url(url: str) -> None:
    """Reject anything that points into the machine or the local network.

    The model chooses the URL, and a scraped page can suggest the next one, so
    the agent must not become a way to read `http://localhost:8000/api` or a
    router's admin page. Every hop of a redirect chain is checked before it is
    requested, not afterwards.
    """
    host = urlsplit(url).hostname
    if not host:
        raise ScrapeError(f"No host in URL '{url}'")

    try:
        resolved = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise ScrapeError(f"Could not resolve {host}: {exc}") from exc

    for *_, sockaddr in resolved:
        address = ipaddress.ip_address(str(sockaddr[0]).split("%")[0])
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ScrapeError(f"Refusing to fetch a non-public address: {host} ({address})")


def fetch_html(url: str, *, timeout: int, user_agent: str) -> str:
    headers = {"User-Agent": user_agent, "Accept-Language": "en,de;q=0.8"}
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False, headers=headers) as client:
            for _ in range(MAX_REDIRECTS):
                ensure_public_url(url)
                response = client.get(url)
                if response.is_redirect and response.next_request is not None:
                    url = str(response.next_request.url)
                    continue
                break
            else:
                raise ScrapeError(f"Too many redirects for {url}")
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise ScrapeError(f"HTTP {exc.response.status_code} for {url}") from exc
    except httpx.HTTPError as exc:
        raise ScrapeError(f"Request failed for {url}: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "html" not in content_type and "xml" not in content_type:
        raise ScrapeError(f"Unsupported content type '{content_type}' for {url}")
    return response.text


def extract_document(
    html: str,
    url: str,
    *,
    max_chars: int,
    query: str | None = None,
) -> SourceDocument:
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not text or len(text.strip()) < 200:
        raise ScrapeError(f"No meaningful content extracted from {url}")

    title, author, published, language = url, None, None, "en"
    try:
        metadata = trafilatura.extract_metadata(html, default_url=url)
        if metadata is not None:
            title = metadata.title or url
            author = metadata.author
            published = metadata.date
            language = metadata.language or "en"
    except Exception as exc:  # metadata is optional
        logger.debug("Metadata extraction failed for %s: %s", url, exc)

    return SourceDocument(
        source_tool=SourceTool.WEB_SCRAPER,
        url=url,
        title=title,
        text=text.strip()[:max_chars],
        author=author,
        published_date=published,
        query=query,
        content_type=ContentType.WEBPAGE,
        language=language,
        retrieved_at=datetime.now(UTC),
    )


def scrape_url(
    url: str,
    *,
    timeout: int,
    max_chars: int,
    user_agent: str,
    query: str | None = None,
) -> SourceDocument:
    html = fetch_html(url, timeout=timeout, user_agent=user_agent)
    return extract_document(html, url, max_chars=max_chars, query=query)
