"""
One-off utility: remove emoji and pictographic characters from every .md
file in the repository so the documentation reads in a neutral, academic
tone. Run from the project root:

    py scripts/strip_emojis.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Unicode ranges that cover virtually all emoji / pictographs / dingbats
# without touching ordinary punctuation or accented letters.
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # Misc symbols & pictographs, emoticons,
                              # transport, supplemental symbols, etc.
    "\U00002600-\U000027BF"   # Misc symbols + dingbats (sun, check, arrows)
    "\U0001F100-\U0001F1FF"   # Enclosed alphanumerics / regional indicators
    "\U00002300-\U000023FF"   # Misc technical (gear, hourglass, etc.)
    "\U0001F900-\U0001F9FF"   # Supplemental symbols & pictographs
    "\U00002B00-\U00002BFF"   # Misc symbols & arrows
    "\U0000FE0F"              # Variation Selector-16 (emoji presentation)
    "\U0000200D"              # Zero Width Joiner (used in compound emoji)
    "]+",
    flags=re.UNICODE,
)

# After removing emoji, collapse the leftover artefacts:
#   "- **3.** **Loss Functions**"  <- stray bullet + double space
#   "##  Heading"                  <- double space after #
DOUBLE_SPACE = re.compile(r"  +")
TRAILING_SPACES = re.compile(r"[ \t]+$", flags=re.MULTILINE)
EMPTY_BOLD = re.compile(r"\*\*\s*\*\*")


def clean_text(text: str) -> str:
    text = EMOJI_PATTERN.sub("", text)
    text = EMPTY_BOLD.sub("", text)
    text = DOUBLE_SPACE.sub(" ", text)
    text = TRAILING_SPACES.sub("", text)
    # Tidy "list item that was just an emoji"
    text = re.sub(r"^\s*[-*]\s*$\n?", "", text, flags=re.MULTILINE)
    return text


def main(root: Path) -> int:
    changed = 0
    for md in root.rglob("*.md"):
        # Skip git internals and any external venvs
        if any(part in {".git", "venv", "env", ".venv"} for part in md.parts):
            continue
        original = md.read_text(encoding="utf-8")
        cleaned = clean_text(original)
        if cleaned != original:
            md.write_text(cleaned, encoding="utf-8", newline="\n")
            changed += 1
            print(f"cleaned: {md.relative_to(root)}")
    print(f"\nDone. {changed} file(s) modified.")
    return 0


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    sys.exit(main(root))
