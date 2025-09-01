from typing import List

def filter_banned(texts: List[str], banned_terms: List[str]) -> List[bool]:
    flags = []
    for t in texts:
        low = t.lower()
        flags.append(any(b in low for b in banned_terms))
    return flags