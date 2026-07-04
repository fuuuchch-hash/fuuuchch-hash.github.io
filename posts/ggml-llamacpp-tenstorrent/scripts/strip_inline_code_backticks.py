#!/usr/bin/env python3
"""
Strip Markdown inline code backticks (`code`) while preserving fenced code blocks (``` ... ``` / ~~~ ... ~~~).

This is intentionally narrow: it only removes inline code span delimiters and leaves all other Markdown intact.
It also handles the case where an inline code span accidentally crosses a newline (it will be normalized to
single spaces, per CommonMark's "line endings are converted to spaces" rule for code spans).
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass


_FENCE_OPEN_RE = re.compile(r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<rest>.*)$")


@dataclass
class FenceState:
    in_fence: bool = False
    fence_char: str = ""
    fence_len: int = 0


@dataclass
class CodeSpanState:
    in_span: bool = False
    tick_len: int = 0
    buf: list[str] | None = None


def _is_fence_close_line(line: str, fence_char: str, fence_len: int) -> bool:
    # Closing fence: up to 3 leading spaces, then >= fence_len of the same char, then optional spaces.
    if fence_char not in ("`", "~") or fence_len < 3:
        return False
    pat = r"^ {0,3}" + re.escape(fence_char) + r"{" + str(fence_len) + r",}\s*$"
    return re.match(pat, line) is not None


def _normalize_codespan_content(s: str) -> str:
    # CommonMark: within code spans, line endings become spaces, and whitespace is trimmed by one space at each end
    # if both ends are spaces. We keep it simple: collapse all whitespace (incl newlines) to single spaces,
    # then strip a single leading/trailing space if present.
    out = re.sub(r"\s+", " ", s)
    if out.startswith(" ") and out.endswith(" ") and len(out) >= 2:
        out = out[1:-1]
    return out


def _strip_inline_backticks_from_text(text: str) -> tuple[str, bool]:
    """
    Strip inline code span delimiters from `text` (which must not include fenced code blocks).
    Returns (processed_text, had_unclosed_span).
    """
    st = CodeSpanState(in_span=False, tick_len=0, buf=None)
    out: list[str] = []

    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "`":
            if st.in_span:
                assert st.buf is not None
                st.buf.append(ch)
            else:
                out.append(ch)
            i += 1
            continue

        # Count run of backticks.
        j = i
        while j < n and text[j] == "`":
            j += 1
        run_len = j - i

        if not st.in_span:
            # Start a new code span.
            st.in_span = True
            st.tick_len = run_len
            st.buf = []
            i = j
            continue

        # We're inside a code span: close only if run matches opening length.
        if run_len == st.tick_len:
            assert st.buf is not None
            out.append(_normalize_codespan_content("".join(st.buf)))
            st.in_span = False
            st.tick_len = 0
            st.buf = None
            i = j
            continue

        # Otherwise it's literal backticks inside the code span.
        assert st.buf is not None
        st.buf.append("`" * run_len)
        i = j

    if st.in_span:
        # Unclosed span: leave it as literal original text by re-inserting the opening delimiter and buffered text.
        # This avoids silently eating content if a stray backtick exists.
        assert st.buf is not None
        out.append("`" * st.tick_len + "".join(st.buf))
        return ("".join(out), True)

    return ("".join(out), False)


def process_markdown_preserving_fences(src: str) -> tuple[str, bool]:
    """
    Remove inline code span backticks from markdown, but do not touch fenced code blocks.
    Returns (processed_text, had_unclosed_span).
    """
    fence = FenceState()
    had_unclosed = False

    out_lines: list[str] = []
    pending_nonfence_chunk: list[str] = []

    def flush_pending() -> None:
        nonlocal had_unclosed
        if not pending_nonfence_chunk:
            return
        chunk = "".join(pending_nonfence_chunk)
        pending_nonfence_chunk.clear()
        processed, unclosed = _strip_inline_backticks_from_text(chunk)
        had_unclosed = had_unclosed or unclosed
        out_lines.append(processed)

    # Keep original newlines by splitting with keepends=True, but we need to treat fence lines discretely.
    for line in src.splitlines(keepends=True):
        if fence.in_fence:
            flush_pending()
            out_lines.append(line)
            if _is_fence_close_line(line.rstrip("\r\n"), fence.fence_char, fence.fence_len):
                fence.in_fence = False
                fence.fence_char = ""
                fence.fence_len = 0
            continue

        m = _FENCE_OPEN_RE.match(line.rstrip("\r\n"))
        if m:
            flush_pending()
            fence_token = m.group("fence")
            fence.in_fence = True
            fence.fence_char = fence_token[0]
            fence.fence_len = len(fence_token)
            out_lines.append(line)
            continue

        pending_nonfence_chunk.append(line)

    flush_pending()
    return ("".join(out_lines), had_unclosed)


def _read_text_preserve_bom(path: str) -> tuple[str, bool, str]:
    data = open(path, "rb").read()
    has_bom = data.startswith(b"\xef\xbb\xbf")
    text = data.decode("utf-8-sig")
    newline = "\r\n" if b"\r\n" in data else "\n"
    # splitlines(keepends=True) preserves actual line endings, but if file is single-line it won't matter.
    return text, has_bom, newline


def _write_text_preserve_bom(path: str, text: str, has_bom: bool) -> None:
    b = text.encode("utf-8")
    if has_bom:
        b = b"\xef\xbb\xbf" + b
    open(path, "wb").write(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Markdown file to edit in-place")
    ap.add_argument("--backup", action="store_true", default=True, help="Create a .bak backup (default: on)")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="Do not create a backup")
    args = ap.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        raise SystemExit(f"not a file: {path}")

    src, has_bom, _newline = _read_text_preserve_bom(path)
    processed, had_unclosed = process_markdown_preserving_fences(src)

    if processed == src:
        return 0

    if args.backup:
        bak = path + ".bak"
        if os.path.exists(bak):
            k = 1
            while os.path.exists(f"{bak}.{k}"):
                k += 1
            bak = f"{bak}.{k}"
        with open(bak, "wb") as f:
            b = src.encode("utf-8")
            if has_bom:
                b = b"\xef\xbb\xbf" + b
            f.write(b)

    # Refuse to overwrite if we detect an unclosed code span outside fences; that's almost always a typo.
    if had_unclosed:
        raise SystemExit("found unclosed inline code span outside fenced code blocks; refusing to overwrite")

    _write_text_preserve_bom(path, processed, has_bom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

