import re

def choose_retriever(claim: str):
    words = re.findall(r"\b[\w'-]+\b", claim)
    word_count = len(words)

    numbers = re.findall(r"\b\d{2,4}\b", claim)

    proper_names = re.findall(
        r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        claim
    )

    quoted = bool(re.search(r'["“”].+?["“”]', claim))

    lexical_score = 0
    semantic_score = 0

    # Experimental heuristic rules
    if word_count <= 8:
        lexical_score += 1
    elif word_count >= 15:
        semantic_score += 2

    if numbers:
        lexical_score += 2

    if proper_names:
        lexical_score += 2

    if quoted:
        lexical_score += 2

    if word_count >= 9:
        semantic_score += 1

    if lexical_score > semantic_score:
        return "TF-IDF", "Distinctive keywords, names or numbers"

    return "Semantic (FAISS)", "Semantic context and broader wording"