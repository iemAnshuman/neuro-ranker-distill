def filter_banned(texts: list[str], banned_terms: list[str]) -> list[bool]:
    normalized_terms = [term.lower() for term in banned_terms]
    return [any(term in text.lower() for term in normalized_terms) for text in texts]
