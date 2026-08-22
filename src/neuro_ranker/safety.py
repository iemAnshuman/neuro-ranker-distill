import re


def filter_banned(texts: list[str], banned_terms: list[str]) -> list[bool]:
    patterns = [
        re.compile(rf"(?<!\w){re.escape(term.strip())}(?!\w)", re.IGNORECASE)
        for term in banned_terms
        if term.strip()
    ]
    return [any(pattern.search(text) for pattern in patterns) for text in texts]
