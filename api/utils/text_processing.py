import re
from data.label_keywords import LABEL_KEYWORDS

def simple_stem(word):
    """Apply simple stemming to a word."""
    word = word.lower()
    if word.endswith('ing'):
        return word[:-3]
    if word.endswith('ed'):
        return word[:-2]
    if word.endswith('s'):
        return word[:-1]
    if word.endswith('es'):
        return word[:-2]
    return word

def is_valid_case_text(case_text):
    """Check if the case text is meaningful or a valid single-word case type."""
    if not case_text or not isinstance(case_text, str):
        return False

    cleaned_text = re.sub(r'[^\w\s]', '', case_text.lower()).strip()
    words = cleaned_text.split()

    if len(words) == 1:
        word = simple_stem(words[0])
        for label, keywords in LABEL_KEYWORDS.items():
            stemmed_keywords = set(simple_stem(kw) for kw in keywords)
            if word in stemmed_keywords:
                return True
        return False

    min_words = 2
    min_word_length = 3
    if len(words) < min_words:
        return False

    valid_word_count = sum(1 for word in words if len(word) >= min_word_length)
    return valid_word_count >= len(words) * 0.5

def map_single_word_to_case_type(word):
    """Map a single word to its corresponding case type, if any."""
    stemmed_word = simple_stem(word.lower())
    for label, keywords in LABEL_KEYWORDS.items():
        stemmed_keywords = set(simple_stem(kw) for kw in keywords)
        if stemmed_word in stemmed_keywords:
            return label
    return None