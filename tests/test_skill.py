from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "skills" / "brain-engine" / "SKILL.md"


def _prose(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.S)


def test_skill_frontmatter_names_it() -> None:
    text = SKILL.read_text(encoding="utf-8")

    frontmatter = text.split("---\n")[1]
    assert re.search(r"^name: brain-engine$", frontmatter, re.M)
    assert re.search(r"^description: Use when ", frontmatter, re.M)


def test_every_quote_in_the_skill_exists() -> None:
    # The skill quotes log and error messages, and names README sections. Each quote must still
    # match the code or a README heading, so a reworded message can't leave the skill stale.
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src" / "brain_engine").glob("*.py"))
    headings = set(re.findall(r"^#+ (.+)$", (ROOT / "README.md").read_text(encoding="utf-8"), re.M))
    quotes = re.findall(r'"([^"]+)"', _prose(SKILL.read_text(encoding="utf-8")))

    assert quotes
    for quote in quotes:
        quote = quote.split("<")[0].rstrip(": ")
        assert quote in headings or quote in source, quote
