"""
Parses a .docx file containing app store reviews.
Each review block is separated by "Did you find this helpful?".
Returns a list of (name, date, review_text) tuples.

Expected block format:
    Reviewer Name
    Date (e.g. 15 March 2026)
    Review text...
    [optional] X people found this review helpful
    [optional] Developer Reply Name
    [optional] Reply Date
    [optional] Reply text...
    Did you find this helpful?

Lines that are metadata (helpful counts, developer replies, dates)
are stripped so only the reviewer's own words are kept.
"""

import re
from docx import Document
from pathlib import Path


# Lines to strip from review body — these are app-store UI artefacts, not review text
_NOISE_PATTERNS = [
    re.compile(r"^\d+\s+(?:person|people)\s+found\s+this\s+review\s+helpful", re.I),
    re.compile(r"^(?:1\s+)?person\s+found\s+this\s+helpful", re.I),
    re.compile(r"^\d+\s+found\s+this\s+helpful", re.I),
    re.compile(r"^did\s+you\s+find\s+this\s+helpful", re.I),
    # Developer reply date line (e.g. "15 March 2026", "Mar 15, 2026")
    re.compile(r"^\d{1,2}\s+\w+\s+\d{4}$"),
    re.compile(r"^\w+\s+\d{1,2},\s+\d{4}$"),
]

# If a line matches this, everything from here on is a developer reply
_DEVELOPER_REPLY_RE = re.compile(
    r"\b(response from developer|developer response|reply from|emergent labs?)\b", re.I
)

# Block names that are developer/company accounts, not real users — skip entirely
_DEVELOPER_NAMES = re.compile(
    r"^(emergent\s*labs?|emergent\s*ai|developer|app\s*team|support\s*team)$", re.I
)


def _is_noise(line: str) -> bool:
    return any(p.match(line.strip()) for p in _NOISE_PATTERNS)


def parse_docx_reviews(doc_path: str) -> list[tuple[str, str, str]]:
    """
    Extract reviews from a Word document.

    Returns:
        List of (name, date, review_text) tuples where review_text contains
        only the reviewer's own words — metadata and developer replies stripped.
    """
    path = Path(doc_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    doc = Document(str(path))
    full_text = "\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())

    reviews = []
    separator = "Did you find this helpful?"

    # Split on the separator regardless of surrounding whitespace
    raw_blocks = re.split(r"\nDid you find this helpful\?\n?", full_text.strip(), flags=re.I)

    for block in raw_blocks:
        block = block.strip()
        if not block:
            continue
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if len(lines) < 2:
            continue

        name = lines[0]
        date = lines[1]

        # Skip developer/company reply blocks entirely
        if _DEVELOPER_NAMES.match(name.strip()):
            continue

        # Collect only the reviewer's text — stop at developer reply or noise
        review_lines = []
        for line in lines[2:]:
            if _is_noise(line):
                continue
            if _DEVELOPER_REPLY_RE.search(line):
                break  # everything after is developer reply
            review_lines.append(line)

        review_text = " ".join(review_lines).strip()
        if review_text:
            reviews.append((name, date, review_text))

    return reviews
