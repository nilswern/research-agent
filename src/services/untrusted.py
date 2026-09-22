"""Fencing for text that came from the internet.

Anything a tool fetched - page bodies, snippets, abstracts, stored passages -
is data written by someone else and may try to give the agent instructions.
Such text is wrapped in delimiters here, the system prompts declare wrapped
text to be data only, and the graph strips the wrapped blocks before it reads
anything machine-relevant (URLs) out of a tool result.
"""

from __future__ import annotations

import re

OPEN = "<untrusted_content>"
CLOSE = "</untrusted_content>"

_BLOCK = re.compile(re.escape(OPEN) + r".*?" + re.escape(CLOSE), re.DOTALL)
_LOOKALIKE = re.compile(r"</?\s*untrusted_content[^>]*>", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def wrap_untrusted(text: str) -> str:
    """Fence fetched text, neutralising delimiters the text itself contains."""
    return f"{OPEN}\n{_LOOKALIKE.sub('[removed]', text)}\n{CLOSE}"


def strip_untrusted(text: str) -> str:
    """Drop every fenced block, leaving only what the tool itself wrote."""
    without_blocks = _BLOCK.sub(" ", text)
    # A block without its closing delimiter must not leak the rest of the text.
    return without_blocks.split(OPEN)[0]


def sanitize_line(text: str, limit: int = 200) -> str:
    """Make attacker-controlled metadata (titles, authors) a single short line."""
    collapsed = _WHITESPACE.sub(" ", _LOOKALIKE.sub("", text)).strip()
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "..."
